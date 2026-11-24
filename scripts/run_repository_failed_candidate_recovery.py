"""Real-model conditional failure recovery and from-mutant diagnostic runner.

The operator prefix replays a hash-verified prior model edit and reproduces its
public failure. All subsequent decisions come from the real provider. The
prefix consumes tool steps, not physical model calls. No hidden tests, oracle
values, or scripted corrections are exposed to the live model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from gpu_agent.agent.models import AgentBudget, ProviderError
from gpu_agent.agent.policy import LLMCallGate, RepositoryCallBudget
from gpu_agent.agent.provider import (
    DevelopmentCallPolicy,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
    provider_invocations,
)
from gpu_agent.harness.check_metrics import public_check_lifecycle
from gpu_agent.harness.journal import read_verified_journal
from gpu_agent.harness.metrics import summarize_repository_trajectory
from gpu_agent.harness.models import (
    HarnessAction,
    HarnessDecision,
    RepoCompileCandidate,
    RepoInspectSyntax,
    RepoReplaceSyntax,
    RepoRunPublicChecks,
    SessionSnapshot,
)
from gpu_agent.harness.repo_audit import read_repository_tool_events
from gpu_agent.harness.repo_runner import run_repository_coding_session
from gpu_agent.harness.repo_tools import RepositoryWorkspaceTools
from gpu_agent.store import RunStore, read_regular, reject_symlinks

GOAL = (
    "Repair the registered CUDA kernel to satisfy its functional requirement. "
    "Use approved public GPU checks before finish; do not inspect hidden tests."
)
_SHA = re.compile(r"source_sha256=([0-9a-f]{64})\.")


def load_prior(raw: bytes, case: str, failure_kind: str) -> dict[str, Any]:
    """Accept a prior single-step report or full audited model-action ledger."""
    data = json.loads(raw)
    failure = "BUILD_FAILED" if failure_kind == "build" else "NUMERIC_MISMATCH"
    if isinstance(data, dict):
        if (
            data.get("case_id") != case
            or data.get("model_physical_calls") != 1
            or data.get("legal_non_noop") is not True
            or data.get("checker_reached") is not True
            or data.get("public_check_passed") is not False
            or data.get("provider_error") is not None
        ):
            raise ValueError("prior report is not a failed accepted model edit")
        feedback = json.loads(data["checker_feedback"])
        selected = {
            "action": data["action"],
            "candidate_sha256": data["candidate_sha256"],
            "checker_feedback": feedback,
        }
    elif isinstance(data, list):
        selected = None
        last_edit = None
        for event in data:
            if event["tool"] == "repo_replace_syntax" and event["status"] == "OK":
                match = _SHA.search(event["observation"])
                if match is None:
                    raise ValueError("accepted edit has no source hash")
                last_edit = {"action": event["action"], "candidate_sha256": match[1]}
            elif event["tool"] == "repo_run_public_checks" and event["status"] == "FAILED":
                feedback = json.loads(event["observation"])
                if (
                    last_edit is not None
                    and feedback["failure_code"] == failure
                    and feedback["candidate_sha256"] == last_edit["candidate_sha256"]
                ):
                    selected = {**last_edit, "checker_feedback": feedback}
                    break
        if selected is None:
            raise ValueError("prior ledger has no matching failed candidate")
    else:
        raise ValueError("unsupported prior evidence format")
    edit = RepoReplaceSyntax.model_validate(selected["action"])
    if (
        edit.path != f"benchmarks/hardset_v2/{case}/mutant/kernel.cu"
        or feedback["failure_code"] != failure
        or feedback["candidate_sha256"] != selected["candidate_sha256"]
        or feedback["passed"] is not False
        or re.fullmatch(r"[0-9a-f]{64}", selected["candidate_sha256"]) is None
    ):
        raise ValueError("prior failure candidate or case differs")
    return selected


class PriorFailureRecovery:
    def __init__(
        self,
        host: RepositoryWorkspaceTools,
        prior: dict[str, Any],
        *,
        failure_kind: str,
        settings: OpenAIProviderSettings,
        max_steps: int,
        max_model_calls: int,
        compile_gate: bool,
        allow_revert: bool,
        public_diagnostics: bool,
        ast_target_binding: bool = False,
    ) -> None:
        self.host = host
        self.prior = prior
        self.edit = RepoReplaceSyntax.model_validate(prior["action"])
        self.failure_kind = failure_kind
        self.compile_gate = compile_gate
        self.allow_revert = allow_revert
        self.public_diagnostics = public_diagnostics
        self.ast_target_binding = ast_target_binding
        self.prefix_turns = 0
        self.prefix_complete = False
        self.failure_feedback: dict[str, Any] | None = None
        self.provider = OpenAIResponsesProvider(
            settings,
            LLMCallGate(
                AgentBudget(max_llm_calls=max_model_calls, max_wall_time_seconds=600),
                reserve_repair_stages=False,
            ),
            host.store,
            host.run_id,
            call_policy=DevelopmentCallPolicy(max_llm_calls=max_model_calls),
            repository_budget=RepositoryCallBudget.from_session(
                max_steps=max_steps, max_model_calls=max_model_calls, hierarchical=False
            ),
        )
        self.provider.ensure_available()

    def invocations(self):
        return self.provider.invocations()

    def _prefix_action(self, snapshot: SessionSnapshot, action: HarnessAction) -> HarnessDecision:
        if action.tool not in snapshot.allowed_tools:
            raise ValueError("prefix action unavailable under frozen capability configuration")
        self.prefix_turns += 1
        return HarnessDecision(action=action, reason="Operator reproduces prior public failure")

    def choose_harness_v1(self, snapshot: SessionSnapshot) -> HarnessDecision:
        if self.prefix_complete:
            return self.provider.choose_harness_v1(snapshot)
        events = read_repository_tool_events(
            self.host.store,
            self.host.run_id,
            expected_count=self.host.tool_calls,
            require_observation=True,
        )
        if events and events[-1]["status"] == "REJECTED":
            raise ValueError("operator failure prefix rejected")
        state = snapshot.syntax_state
        if snapshot.current_candidate_sha256 is None:
            observed = {n.id for n in state.observed_nodes} if state else set()
            if state is None or not {e.node_id for e in self.edit.edits}.issubset(observed):
                if state is not None and state.next_offset is None:
                    raise ValueError("prior candidate targets not available")
                return self._prefix_action(
                    snapshot,
                    RepoInspectSyntax(
                        path=self.edit.path,
                        offset=state.next_offset if state else 0,
                        limit=8,
                    ),
                )
            if state.source_sha256 != self.edit.expected_source_sha256:
                raise ValueError("prior edit refers to a different mutant SHA")
            edit = self.edit
            if self.ast_target_binding:
                # Historical model actions predate old-code binding. The
                # operator prefix explicitly attests their actual original
                # targets, reproducing the same bad bytes, not correcting them.
                by_id = {node.id: node for node in state.observed_nodes}
                edit = edit.model_copy(update={"edits": [
                    item.model_copy(update={"expected_old_code": by_id[item.node_id].code})
                    for item in edit.edits
                ]})
            return self._prefix_action(snapshot, edit)

        actual = hashlib.sha256(
            read_regular(self.host.workspace.root / self.edit.path, 1024 * 1024)
        ).hexdigest()
        if actual != self.prior["candidate_sha256"]:
            raise ValueError("replayed candidate differs from prior model candidate")
        view = snapshot.candidate_recovery
        if self.compile_gate:
            if view is None or not view.compile_gate or view.source_sha256 != actual:
                raise ValueError("compile gate not active in actual snapshot")
            if view.compile_status == "NOT_CHECKED":
                return self._prefix_action(
                    snapshot, RepoCompileCandidate(expected_source_sha256=actual)
                )
        if self.failure_kind == "build":
            if view is None or view.compile_status != "BUILD_FAILED":
                raise ValueError("build recovery precondition did not reproduce")
            if self.allow_revert and (
                not view.revert_available or "repo_revert_last_edit" not in snapshot.allowed_tools
            ):
                raise ValueError("rollback capability missing at handoff")
            expected_tool = "repo_compile_candidate"
        else:
            if self.compile_gate and (view is None or view.compile_status != "BUILD_PASSED"):
                raise ValueError("numeric recovery prefix must compile")
            if self.host.public_checks_attempted == 0:
                return self._prefix_action(snapshot, RepoRunPublicChecks())
            expected_tool = "repo_run_public_checks"
        last = events[-1]
        if last["tool"] != expected_tool or last["status"] != "FAILED":
            raise ValueError("first failure did not reproduce")
        feedback = json.loads(last["observation"])
        failure = (
            feedback.get("status") if self.failure_kind == "build" else feedback.get("failure_code")
        )
        if failure != self.prior["checker_feedback"]["failure_code"]:
            raise ValueError("reproduced failure class differs")
        if self.public_diagnostics and not feedback.get("numeric_diagnostic"):
            raise ValueError("numeric diagnostics not active at handoff")
        self.failure_feedback = feedback
        self.prefix_complete = True
        # Attest the exact capability list seen by the first live decision.
        self.host.store.put(
            self.host.run_id,
            "recovery/handoff.json",
            json.dumps(
                {
                    "operator_prefix_tool_turns": self.prefix_turns,
                    "failed_candidate_sha256": actual,
                    "allowed_tools": snapshot.allowed_tools,
                    "candidate_recovery": view.model_dump(mode="json") if view else None,
                    "failure_feedback": feedback,
                }
            ).encode(),
            "public",
        )
        return self.provider.choose_harness_v1(snapshot)


def summarize_trial(
    output: Path,
    *,
    config: dict[str, Any],
    model: PriorFailureRecovery | None,
    outcome: Any,
    error_code: str | None,
) -> dict[str, Any]:
    """Use audited events and candidate identities even if the model aborted."""
    agent = output / "agent"
    journal = read_verified_journal(agent / "session-events.jsonl")
    started = journal[0]["payload"]
    for key in (
        "compile_gate",
        "allow_revert",
        "public_diagnostics",
        "ast_target_binding",
        "state_scoped_actions",
        "max_compiles",
        "max_steps",
        "max_model_calls",
        "max_output_tokens",
    ):
        if started.get(key) != config[key]:
            raise ValueError(f"actual journal disagrees with frozen {key}")
    store = RunStore(agent / "public")
    run_id = started["public_run_id"]
    records = provider_invocations(store, run_id)
    if any(r.configured_max_output_tokens != config["max_output_tokens"] for r in records):
        raise ValueError("actual provider request cap differs from frozen trial")
    if len(records) > config["max_model_calls"]:
        raise ValueError("physical call cap exceeded")
    count = sum(
        ref.name.startswith("repo-harness/v1/tool-") and ref.name.endswith(".json")
        for ref in store.load(run_id).artifact_refs
    )
    events = read_repository_tool_events(
        store, run_id, expected_count=count, require_observation=True
    )
    lifecycle = public_check_lifecycle(store, run_id, journal, events)
    prefix = model.prefix_turns if model else 0
    live = [(i, e) for i, e in enumerate(events, 1) if i > prefix]
    live_edits = [
        (i, _SHA.search(e["observation"]))
        for i, e in live
        if e["tool"] == "repo_replace_syntax" and e["status"] == "OK"
    ]
    if any(match is None for _, match in live_edits):
        raise ValueError("live accepted edit lacks source hash")
    prior_sha = model.prior["candidate_sha256"] if model else None
    candidates = {match[1] for _, match in live_edits if match and match[1] != prior_sha}
    compiled = candidates.intersection(lifecycle.build_passed_candidates)
    admitted_by_turn = {}
    for ref in store.load(run_id).artifact_refs:
        if ref.name.startswith("repo-harness/v1/check-") and ref.name.endswith("-admitted.json"):
            admission = json.loads(store.read(ref))
            admitted_by_turn[admission["turn"]] = admission["candidate_sha256"]
    dispatched = {
        admitted_by_turn[t]
        for t in lifecycle.executed
        if t > prefix and admitted_by_turn.get(t) in candidates
    }
    passed = {
        admitted_by_turn[t]
        for t in lifecycle.passed
        if t > prefix and admitted_by_turn.get(t) in candidates
    }
    builds_failed = []
    numeric_failed = []
    for turn, event in enumerate(events, 1):
        if event["status"] != "FAILED":
            continue
        if event["tool"] == "repo_compile_candidate":
            if json.loads(event["observation"]).get("status") == "BUILD_FAILED":
                builds_failed.append(turn)
        elif event["tool"] == "repo_run_public_checks":
            failure = json.loads(event["observation"]).get("failure_code")
            if failure == "BUILD_FAILED":
                builds_failed.append(turn)
            if failure == "NUMERIC_MISMATCH":
                numeric_failed.append(turn)
    revert_selected = any(
        j["kind"] == "tool_started"
        and j["payload"].get("tool") == "repo_revert_last_edit"
        and j["payload"]["turn"] > prefix
        and any(t < j["payload"]["turn"] for t in builds_failed)
        for j in journal
    )
    revert_accepted = any(
        e["tool"] == "repo_revert_last_edit" and e["status"] == "OK" for _, e in live
    )
    usage = [r.usage.total_tokens for r in records if r.usage and r.usage.total_tokens is not None]
    incomplete_codes = [
        r.error_code for r in records if r.error_code and r.error_code.startswith("LLM_INCOMPLETE")
    ]
    terminal_reason = outcome.reason if outcome is not None else error_code
    metrics = summarize_repository_trajectory(agent).model_dump(mode="json") if outcome else None
    return {
        **config,
        "public_run_id": run_id,
        "operator_prefix_tool_turns": prefix,
        "live_tool_turns": len(live),
        "total_tool_turns": count,
        "first_failure_reproduced": model.prefix_complete if model else None,
        "initial_failure_feedback": model.failure_feedback if model else None,
        "build_failure_observed": bool(builds_failed),
        "numeric_failure_observed": bool(numeric_failed),
        "revert_selected_after_build_failure": revert_selected,
        "revert_accepted": revert_accepted,
        "second_ast_edit_accepted": bool(candidates) if model else None,
        "second_build_passed": bool(compiled) if model else None,
        "second_numeric_dispatched": bool(dispatched) if model else None,
        "second_public_check_passed": bool(passed) if model else None,
        "live_ast_edit_accepted_count": len(live_edits),
        "first_live_ast_edit_accepted_turn": min((i for i, _ in live_edits), default=None),
        "live_edited_candidate_shas": sorted(candidates),
        "live_build_passed_candidate_shas": sorted(compiled),
        "live_numeric_dispatched_candidate_shas": sorted(dispatched),
        "live_public_passed_candidate_shas": sorted(passed),
        "build_passed_candidate_count": len(compiled),
        "numeric_dispatched_candidate_count": len(dispatched),
        "public_check_passed_candidate_count": len(passed),
        "step_budget_exhausted": terminal_reason == "STEP_BUDGET_EXHAUSTED",
        "llm_incomplete": bool(incomplete_codes),
        "incomplete_codes": incomplete_codes,
        "model_physical_calls": len(records),
        "format_retries": sum(r.attempt == 1 for r in records),
        "model_tokens": sum(usage) if len(usage) == len(records) else None,
        "terminal_reason": terminal_reason,
        "public_pass": bool(passed),
        "finish_accepted": any(e["tool"] == "finish" and e["status"] == "OK" for _, e in live),
        "outcome": outcome.model_dump(mode="json") if outcome else None,
        "metrics": metrics,
        "tool_trace": [
            {
                "turn": i,
                "actor": "operator" if i <= prefix else "model",
                "tool": e["tool"],
                "status": e["status"],
                "rejection_code": e.get("rejection_code"),
            }
            for i, e in enumerate(events, 1)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New trial directory; agent/ contains session"
    )
    parser.add_argument("--case", choices=("h001", "h002", "h003"), required=True)
    parser.add_argument("--failure-kind", choices=("build", "numeric", "none"), required=True)
    parser.add_argument("--prior-result", type=Path)
    parser.add_argument("--prior-result-sha256")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--max-model-calls", type=int, default=18)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--compile-gate", action="store_true")
    parser.add_argument("--allow-revert", action="store_true")
    parser.add_argument("--public-diagnostics", action="store_true")
    parser.add_argument("--ast-target-binding", action="store_true")
    parser.add_argument("--state-scoped-actions", action="store_true")
    parser.add_argument("--max-compiles", type=int, default=6)
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--allow-real-gpu", action="store_true")
    args = parser.parse_args()
    if not (args.allow_paid_calls and args.allow_real_gpu):
        parser.error("explicit model and GPU approval required")
    if not 1 <= args.max_steps <= 38 or not args.max_steps + 2 <= args.max_model_calls <= 40:
        parser.error("invalid model/tool budget")
    if not 1 <= args.max_compiles <= 8 or not 256 <= args.max_output_tokens <= 16384:
        parser.error("invalid compile or output-token budget")
    prior = None
    if args.failure_kind != "none":
        if not args.prior_result or not args.prior_result_sha256:
            parser.error("hash-pinned prior evidence is required")
        raw = read_regular(args.prior_result, 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != args.prior_result_sha256:
            parser.error("prior result hash mismatch")
        prior = load_prior(raw, args.case, args.failure_kind)
        if args.failure_kind == "build" and not args.compile_gate:
            parser.error("build recovery requires compile gate")
    elif args.prior_result or args.prior_result_sha256:
        parser.error("from-mutant trial cannot have a prior candidate")
    reject_symlinks(args.repository)
    repo = args.repository.resolve(strict=True)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()
    if subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip():
        parser.error("experiment requires a clean frozen checkout")
    settings = OpenAIProviderSettings.from_environment().model_copy(
        update={"max_output_tokens": args.max_output_tokens}
    )
    if not settings.endpoint or not settings.model:
        parser.error("provider configuration missing")
    config = {
        "experiment": "conditional_recovery" if prior else "autonomous_from_mutant",
        "case_id": args.case,
        "failure_kind": args.failure_kind,
        "base_sha": sha,
        "prior_result_sha256": args.prior_result_sha256,
        "prior_candidate_sha256": prior["candidate_sha256"] if prior else None,
        "compile_gate": args.compile_gate,
        "allow_revert": args.allow_revert,
        "public_diagnostics": args.public_diagnostics,
        "ast_target_binding": args.ast_target_binding,
        "state_scoped_actions": args.state_scoped_actions,
        "extended_syntax": False,
        "hierarchical_mode": False,
        "max_compiles": args.max_compiles,
        "max_public_checks": 2,
        "max_steps": args.max_steps,
        "max_model_calls": args.max_model_calls,
        "max_output_tokens": args.max_output_tokens,
        "model": settings.model,
        "request_timeout_seconds": settings.timeout_seconds,
        "wall_budget_seconds": 600,
        "prefix_included_in_step_budget": True,
    }
    output = args.output.absolute()
    reject_symlinks(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "experiment-config.json").write_text(json.dumps(config, indent=2) + "\n")
    holder = []

    def factory(host: RepositoryWorkspaceTools):
        assert prior is not None
        model = PriorFailureRecovery(
            host,
            prior,
            failure_kind=args.failure_kind,
            settings=settings,
            max_steps=args.max_steps,
            max_model_calls=args.max_model_calls,
            compile_gate=args.compile_gate,
            allow_revert=args.allow_revert,
            public_diagnostics=args.public_diagnostics,
            ast_target_binding=args.ast_target_binding,
        )
        holder.append(model)
        return model

    outcome = None
    error_code = None
    try:
        outcome = run_repository_coding_session(
            repository=repo,
            output=output / "agent",
            base_sha=sha,
            goal=GOAL,
            public_case_id=args.case,
            syntax_mode=True,
            allow_batch_edits=False,
            context_mode="evidence_first",
            max_steps=args.max_steps,
            max_model_calls=args.max_model_calls,
            max_public_checks=2,
            compile_gate=args.compile_gate,
            allow_revert=args.allow_revert,
            max_compiles=args.max_compiles,
            public_diagnostics=args.public_diagnostics,
            ast_target_binding=args.ast_target_binding,
            state_scoped_actions=args.state_scoped_actions,
            provider_settings=settings,
            allow_paid_calls=True,
            model_factory=factory if prior else None,
        )
    except ProviderError as exc:
        error_code = exc.code
    except Exception as exc:
        error_code = type(exc).__name__
        print(f"EXPERIMENT_ABORTED: {error_code}", flush=True)
    try:
        report = summarize_trial(
            output,
            config=config,
            model=holder[0] if holder else None,
            outcome=outcome,
            error_code=error_code,
        )
    except Exception as exc:
        report = {
            **config,
            "terminal_reason": error_code,
            "audit_error": type(exc).__name__,
            "public_pass": None,
            "first_failure_reproduced": None,
        }
    (output / "controlled-recovery.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 1 if report.get("audit_error") or error_code else 0


if __name__ == "__main__":
    raise SystemExit(main())
