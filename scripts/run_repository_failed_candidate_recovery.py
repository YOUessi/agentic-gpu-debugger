"""Controlled real-model recovery from a prior public-check-failed model candidate.

The prefix replays a previously recorded *model* action, not a reference fix.
The actual current GPU Checker must reproduce failure before the live model
takes over. This is a conditional recovery experiment, not an autonomous
from-mutant trial. No hidden verifier or expected output is exposed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from gpu_agent.agent.models import AgentBudget
from gpu_agent.agent.policy import LLMCallGate, RepositoryCallBudget
from gpu_agent.agent.provider import (
    DevelopmentCallPolicy,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
)
from gpu_agent.harness.metrics import summarize_repository_trajectory
from gpu_agent.harness.models import (
    HarnessDecision,
    RepoInspectSyntax,
    RepoReplaceSyntax,
    RepoRunPublicChecks,
    SessionSnapshot,
)
from gpu_agent.harness.repo_runner import run_repository_coding_session
from gpu_agent.harness.repo_tools import RepositoryWorkspaceTools
from gpu_agent.store import read_regular


class PriorFailureRecovery:
    def __init__(
        self, host: RepositoryWorkspaceTools, prior: dict, *, max_steps: int,
        max_model_calls: int,
    ) -> None:
        self.host = host
        self.prior = prior
        self.edit = RepoReplaceSyntax.model_validate(prior["action"])
        self.prefix_turns = 0
        self.prefix_complete = False
        self.provider = OpenAIResponsesProvider(
            OpenAIProviderSettings.from_environment(),
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

    def invocations(self):
        return self.provider.invocations()

    def choose_harness_v1(self, snapshot: SessionSnapshot) -> HarnessDecision:
        if self.prefix_complete:
            return self.provider.choose_harness_v1(snapshot)
        state = snapshot.syntax_state
        if snapshot.current_candidate_sha256 is None:
            observed = {n.id for n in state.observed_nodes} if state else set()
            if state is None or not {e.node_id for e in self.edit.edits}.issubset(observed):
                if state is not None and state.next_offset is None:
                    raise ValueError("prior candidate targets are not available in current AST")
                action = RepoInspectSyntax(
                    path=self.edit.path,
                    offset=state.next_offset if state else 0,
                    limit=8,
                )
            else:
                if state.source_sha256 != self.edit.expected_source_sha256:
                    raise ValueError("prior candidate refers to a different mutant SHA")
                action = self.edit
            self.prefix_turns += 1
            return HarnessDecision(action=action, reason="Operator replays prior public evidence")
        if self.host.public_checks_attempted == 0:
            self.prefix_turns += 1
            actual = hashlib.sha256((self.host.workspace.root / self.edit.path).read_bytes())
            if actual.hexdigest() != self.prior["candidate_sha256"]:
                raise ValueError("replayed candidate differs from prior model-produced candidate")
            return HarnessDecision(
                action=RepoRunPublicChecks(), reason="Reproduce previous real public failure"
            )
        last = snapshot.history[-1]
        if last["tool"] != "repo_run_public_checks" or last["status"] != "FAILED":
            raise ValueError("recovery precondition failed: first checker did not fail")
        actual = json.loads(last["content"])
        prior = json.loads(self.prior["checker_feedback"])
        if actual["failure_code"] != prior["failure_code"]:
            raise ValueError("recovery precondition failed: failure class changed")
        self.prefix_complete = True
        return self.provider.choose_harness_v1(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--prior-result-sha256", required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--allow-real-gpu", action="store_true")
    args = parser.parse_args()
    if not (args.allow_paid_calls and args.allow_real_gpu):
        parser.error("explicit model and GPU approval required")
    raw = read_regular(args.prior_result, 100_000)
    if hashlib.sha256(raw).hexdigest() != args.prior_result_sha256:
        parser.error("prior result hash mismatch")
    prior = json.loads(raw)
    if (
        prior.get("model_physical_calls") != 1
        or prior.get("legal_non_noop") is not True
        or prior.get("checker_reached") is not True
        or prior.get("public_check_passed") is not False
        or prior.get("provider_error") is not None
    ):
        parser.error("prior result is not an accepted, publicly failed model edit")
    case = prior["case_id"]
    expected_path = f"benchmarks/hardset_v2/{case}/mutant/kernel.cu"
    edit = RepoReplaceSyntax.model_validate(prior["action"])
    if case not in ("h001", "h002", "h003") or edit.path != expected_path:
        parser.error("prior action is not the registered case")
    repo = args.repository.resolve(strict=True)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    if subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True, capture_output=True, text=True, timeout=20,
    ).stdout.strip():
        parser.error("controlled recovery requires a clean frozen checkout")
    holder = []

    def factory(host: RepositoryWorkspaceTools):
        model = PriorFailureRecovery(
            host, prior, max_steps=args.max_steps, max_model_calls=args.max_steps + 2
        )
        holder.append(model)
        return model

    outcome = run_repository_coding_session(
        repository=repo, output=args.output.absolute(), base_sha=sha,
        goal=(
            "Repair the registered CUDA kernel to satisfy its functional requirement. "
            "Use approved public GPU checks before finish; do not inspect hidden tests."
        ),
        public_case_id=case, syntax_mode=True, allow_batch_edits=False,
        context_mode="evidence_first", max_steps=args.max_steps,
        max_model_calls=args.max_steps + 2, max_public_checks=2,
        allow_paid_calls=True, model_factory=factory,
    )
    metrics = summarize_repository_trajectory(args.output)
    model = holder[0]
    revised = [
        step.turn for step in metrics.tool_trajectory
        if step.turn > model.prefix_turns
        and step.tool == "repo_replace_syntax" and step.status == "OK"
    ]
    report = {
        "experiment": "controlled_recovery_from_prior_model_candidate",
        "prior_result_sha256": args.prior_result_sha256,
        "base_sha": sha,
        "operator_prefix_tool_turns": model.prefix_turns,
        "first_failure_reproduced": model.prefix_complete,
        "real_model_first_edit_turn": min(revised, default=None),
        "real_model_recovery_pass": bool(
            model.prefix_complete and revised and outcome.public_checks_passed
            and metrics.first_public_check_pass_turn is not None
            and metrics.first_public_check_pass_turn > model.prefix_turns
        ),
        "outcome": outcome.model_dump(mode="json"),
        "metrics": metrics.model_dump(mode="json"),
    }
    (args.output / "controlled-recovery.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
