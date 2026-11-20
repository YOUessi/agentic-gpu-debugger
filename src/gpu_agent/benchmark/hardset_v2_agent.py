"""End-to-end, model-driven HardSet V2 CUDA repair Agent (development contract).

An actual LangGraph orchestrates evidence acquisition, LLM diagnosis, LLM patch
generation, controlled GPU public checks, bounded revisions and termination.
The private verifier is invoked by the operator controller ONLY after the
public loop has ended. It is never a model-selected tool or feedback source.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from gpu_agent.agent.models import AgentBudget
from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.agent.provider import (
    DevelopmentCallPolicy,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
)
from gpu_agent.benchmark.hardset_v2 import PUBLIC_REQUIREMENTS, load_registry
from gpu_agent.benchmark.hardset_v2_agent_models import (
    HardsetAgentContext,
    HardsetAgentOutcome,
    HardsetDiagnosis,
    HardsetPublicCheck,
)
from gpu_agent.benchmark.hardset_v2_graph import (
    ControlledProbe,
    Planner,
    run_hardset_probe_graph,
)
from gpu_agent.benchmark.hardset_v2_probe import (
    FunctionalProbeObservation,
    HardsetFunctionalProbe,
)
from gpu_agent.benchmark.hardset_v2_selfcheck import check_candidate_publicly
from gpu_agent.benchmark.hardset_v2_verifier import run_strict_hardset_verification
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.patching import (
    SourceSnapshot,
    apply_generated_candidate,
    materialize_candidate,
)
from gpu_agent.store import RunStore, read_regular, reject_symlinks


class HardsetRepairProvider(Protocol):
    """Only audited LLM capabilities; never arbitrary tools or evaluator access."""

    def plan_hardset_v2(self, snapshot: Any) -> Any: ...

    def diagnose_hardset_v2(self, context: HardsetAgentContext) -> HardsetDiagnosis: ...

    def propose_patch_hardset_v2(
        self,
        context: HardsetAgentContext,
        diagnosis: HardsetDiagnosis,
        public_feedback: list[HardsetPublicCheck],
        previous_candidate_hashes: list[str],
    ) -> str: ...


class AgentGraphState(TypedDict, total=False):
    observations: list[dict[str, Any]]
    diagnosis: dict[str, Any]
    feedback: list[dict[str, Any]]
    candidate_hashes: list[str]
    accepted_diff: str | None
    attempts: int
    pending_hash: str | None
    action: Literal["public_check", "retry", "stop"]
    stop_reason: str


def run_repair_graph(
    *,
    probe: ControlledProbe,
    mode: Literal["D", "E"],
    case_id: str,
    public_source: str,
    functional_requirement: str,
    provider: HardsetRepairProvider,
    validate_candidate: Callable[[str], tuple[str, bytes]],
    public_check: Callable[[bytes, int], HardsetPublicCheck],
    max_probes: int = 3,
    max_candidates: int = 3,
    audit: Callable[[dict[str, object]], None] | None = None,
) -> AgentGraphState:
    """Run the real agent decision loop, with injectable *controlled* tools.

    Injected tools permit deterministic tests; production passes the actual
    isolated CUDA backend and original provider worker. No evaluator is
    accepted by this graph, so private test outcomes cannot steer revisions.
    """
    if mode not in {"D", "E"}:
        raise ValueError("invalid research mode")
    if not 1 <= max_probes <= 15 or not 1 <= max_candidates <= 3:
        raise ValueError("unbounded agent controller")
    if case_id not in PUBLIC_REQUIREMENTS:
        raise ValueError("unregistered HardSet algorithm")
    if not public_source.strip():
        raise ValueError("agent requires original public source")
    source_hash = hashlib.sha256(public_source.encode()).hexdigest()

    def record(phase: str, data: dict[str, object]) -> None:
        if audit is not None:
            audit({"phase": phase, **data})

    def investigate(state: AgentGraphState) -> dict[str, Any]:
        selected: Planner | None = provider.plan_hardset_v2 if mode == "E" else None
        trace = run_hardset_probe_graph(
            probe,
            case_id=case_id,
            mode=mode,
            max_probes=max_probes,
            planner=selected,
            public_source=public_source,
            functional_requirement=functional_requirement,
        )
        observed = trace["observations"]
        record(
            "investigate",
            {
                "mode": mode,
                "probe_count": len(observed),
                "observations": observed,
                "model_planning": mode == "E",
            },
        )
        if not observed:
            return {"observations": [], "action": "stop", "stop_reason": "NO_PUBLIC_EVIDENCE"}
        return {"observations": observed, "action": "retry"}

    def context(state: AgentGraphState) -> HardsetAgentContext:
        return HardsetAgentContext(
            case_id=case_id,
            source_sha256=source_hash,
            public_source=public_source,
            functional_requirement=functional_requirement,
            observations=tuple(
                FunctionalProbeObservation.model_validate(item) for item in state["observations"]
            ),
        )

    def diagnose(state: AgentGraphState) -> dict[str, Any]:
        public_context = context(state)
        diagnosis = provider.diagnose_hardset_v2(public_context)
        reference_ids = {v.evidence_ref_id for v in public_context.observations}
        if (
            not diagnosis.cited_public_probe_ids
            or not set(diagnosis.cited_public_probe_ids) <= reference_ids
            or any(
                lineno > len(public_source.splitlines()) or lineno <= 0
                for lineno in diagnosis.suspected_lines
            )
        ):
            raise ValueError("LLM_DIAGNOSIS_CITATION_INVALID")
        record(
            "diagnose",
            {
                "root_cause": diagnosis.root_cause,
                "suspected_lines": diagnosis.suspected_lines,
                "cited_public_probe_ids": diagnosis.cited_public_probe_ids,
            },
        )
        return {"diagnosis": diagnosis.model_dump(mode="json")}

    def patch(state: AgentGraphState) -> dict[str, Any]:
        attempt = state.get("attempts", 0) + 1
        previous = list(state.get("candidate_hashes", []))
        history = [HardsetPublicCheck.model_validate(item) for item in state.get("feedback", [])]
        diff = provider.propose_patch_hardset_v2(
            context(state),
            HardsetDiagnosis.model_validate(state["diagnosis"]),
            history,
            previous,
        )
        if not isinstance(diff, str) or not diff:
            raise ValueError("LLM_PATCH_MISSING")
        record(
            "llm_patch",
            {
                "attempt": attempt,
                "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            },
        )
        try:
            candidate_hash, candidate_code = validate_candidate(diff)
        except (OSError, ValueError):
            rejected = HardsetPublicCheck(passed=False, failure_code="PATCH_SCOPE_REJECTED")
            record("patch_rejected", {"attempt": attempt, "reason": "PATCH_SCOPE_REJECTED"})
            return {
                "attempts": attempt,
                "feedback": [*state.get("feedback", []), rejected.model_dump(mode="json")],
                "action": "retry" if attempt < max_candidates else "stop",
                "stop_reason": "PATCH_SCOPE_REJECTED",
                "pending_hash": None,
            }
        if candidate_hash in previous:
            record("patch_rejected", {"attempt": attempt, "reason": "REPEATED_CANDIDATE"})
            return {
                "attempts": attempt,
                "action": "stop",
                "stop_reason": "REPEATED_CANDIDATE",
                "pending_hash": None,
            }
        candidates[candidate_hash] = (candidate_code, diff)
        record("scope_accepted", {"attempt": attempt, "candidate_sha256": candidate_hash})
        return {
            "attempts": attempt,
            "candidate_hashes": [*previous, candidate_hash],
            "pending_hash": candidate_hash,
            "action": "public_check",
        }

    def public_selfcheck(state: AgentGraphState) -> dict[str, Any]:
        value = state.get("pending_hash")
        if value is None or value not in candidates:
            raise ValueError("controller cannot execute unknown candidate")
        candidate_bytes, diff = candidates[value]
        feedback = public_check(candidate_bytes, state["attempts"])
        if feedback.candidate_sha256 != value:
            raise ValueError("public checker candidate provenance mismatch")
        record(
            "public_selfcheck",
            {
                "attempt": state["attempts"],
                "candidate_sha256": value,
                "passed": feedback.passed,
                "failure_code": feedback.failure_code,
                "first_failing_size": feedback.first_failing_size,
                "public_run_id": feedback.public_run_id,
            },
        )
        return {
            "feedback": [
                *state.get("feedback", []),
                feedback.model_dump(mode="json"),
            ],
            "accepted_diff": diff if feedback.passed else None,
            "action": (
                "stop" if feedback.passed or state["attempts"] >= max_candidates else "retry"
            ),
            "stop_reason": (
                "PUBLIC_CHECKS_PASSED"
                if feedback.passed
                else "CANDIDATE_LIMIT"
                if state["attempts"] >= max_candidates
                else feedback.failure_code
            ),
        }

    def finish(state: AgentGraphState) -> dict[str, object]:
        return {
            "stop_reason": state.get("stop_reason", "NO_CANDIDATE"),
        }

    candidates: dict[str, tuple[bytes, str]] = {}
    graph = StateGraph(AgentGraphState)
    graph.add_node("investigate", investigate)
    graph.add_node("diagnose", diagnose)
    graph.add_node("patch", patch)
    graph.add_node("public_selfcheck", public_selfcheck)
    graph.add_node("finish", finish)
    graph.add_edge(START, "investigate")
    graph.add_conditional_edges(
        "investigate",
        lambda s: "diagnose" if s.get("action") != "stop" else "finish",
        {"diagnose": "diagnose", "finish": "finish"},
    )
    graph.add_edge("diagnose", "patch")
    graph.add_conditional_edges(
        "patch",
        lambda s: s["action"],
        {
            "public_check": "public_selfcheck",
            "retry": "patch",
            "stop": "finish",
        },
    )
    graph.add_conditional_edges(
        "public_selfcheck",
        lambda s: s["action"],
        {"retry": "patch", "stop": "finish"},
    )
    graph.add_edge("finish", END)
    return cast(
        AgentGraphState,
        graph.compile().invoke(
            {"observations": [], "feedback": [], "candidate_hashes": [], "attempts": 0},
            config={"recursion_limit": (max_candidates + max_probes + 10) * 6},
        ),
    )


def run_autonomous_hardset(
    repository: Path,
    output: Path,
    case_id: str,
    *,
    mode: Literal["D", "E"],
    evaluator_secret: bytes,
    max_probes: int = 3,
    max_candidates: int = 3,
    max_llm_calls: int = 20,
    allow_paid_calls: bool = False,
    provider_factory: (
        Callable[[RunStore, str, Callable[[str], object]], HardsetRepairProvider] | None
    ) = None,
) -> HardsetAgentOutcome:
    """Execute *all* Agent stages against real CUDA, then one private verification.

    In production, the provider is the original killable, budgeted SDK worker.
    Explicit factory injection is for zero-cost unit tests only. Private strict
    validation happens after the decision graph and is never fed into a model.
    """
    if provider_factory is None and not allow_paid_calls:
        raise ValueError("real-model Agent requires explicit paid-call authorization")
    if not isinstance(evaluator_secret, bytes) or len(evaluator_secret) < 32:
        raise ValueError("independent evaluator secret is required")
    if not 1 <= max_candidates <= 3 or not 1 <= max_probes <= 15:
        raise ValueError("invalid bounded Agent experiment")
    if max_llm_calls < max_candidates + 3 or max_llm_calls > 40:
        raise ValueError("invalid model-call budget")
    reject_symlinks(repository)
    reject_symlinks(output)
    if output.exists():
        raise ValueError("output exists; do not overwrite evidence")
    root = repository.resolve()
    registry, _ = load_registry(root / "benchmarks")
    case = next((c for c in registry.cases if c.case_id == case_id), None)
    if case is None:
        raise ValueError("unregistered HardSet case")
    original = read_regular(
        root / "benchmarks" / "hardset_v2" / case_id / "mutant" / "kernel.cu",
        4 * 1024 * 1024,
    )
    if hashlib.sha256(original).hexdigest() != case.mutant_source_sha256:
        raise ValueError("registered HardSet source has changed")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    history: list[dict[str, object]] = []
    outcome: HardsetAgentOutcome | None = None
    with HardsetFunctionalProbe(
        root, output / "investigation", case_id, max_probes=max_probes
    ) as probe:
        assert probe.run_id is not None
        store = probe.public_store

        def audit(event: dict[str, object]) -> None:
            index = len(history) + 1
            history.append(event)
            store.put(
                probe.run_id,
                f"hardset/agent-trace/{index:03d}.json",
                json.dumps(event, ensure_ascii=False, sort_keys=True).encode(),
                "public",
            )

        with tempfile.TemporaryDirectory(prefix="hardset-v2-agent-base-") as source_dir:
            immutable = Path(source_dir)
            IsolatedGPUBackend._write_snapshot(immutable / "kernel.cu", original)
            snapshot = SourceSnapshot(
                parent_run_id=probe.run_id,
                root=immutable,
                hashes={"kernel.cu": case.mutant_source_sha256},
            )

            def validate_candidate(diff: str) -> tuple[str, bytes]:
                patch = apply_generated_candidate(snapshot, diff)
                candidate = materialize_candidate(snapshot, patch)["kernel.cu"]
                return hashlib.sha256(candidate).hexdigest(), candidate

            if provider_factory is None:
                settings = OpenAIProviderSettings.from_environment()
                provider: HardsetRepairProvider = OpenAIResponsesProvider(
                    settings,
                    LLMCallGate(
                        AgentBudget(
                            max_llm_calls=max_llm_calls,
                            max_wall_time_seconds=600,
                        )
                    ),
                    store,
                    probe.run_id,
                    diff_validator=lambda diff: validate_candidate(diff),
                    call_policy=DevelopmentCallPolicy(max_llm_calls=max_llm_calls),
                )
                provider.ensure_available()
            else:
                provider = provider_factory(store, probe.run_id, validate_candidate)

            def check(code: bytes, attempt: int) -> HardsetPublicCheck:
                return check_candidate_publicly(
                    store=store,
                    parent_run_id=probe.run_id,
                    repository=root,
                    case_id=case_id,
                    candidate=code,
                    output=output / f"public-candidate-{attempt:02d}",
                )

            state = run_repair_graph(
                probe=probe,
                mode=mode,
                case_id=case_id,
                public_source=probe.public_source,
                functional_requirement=PUBLIC_REQUIREMENTS[case_id],
                provider=provider,
                validate_candidate=validate_candidate,
                public_check=check,
                max_probes=max_probes,
                max_candidates=max_candidates,
                audit=audit,
            )
            accepted = state.get("accepted_diff")
            latest = state.get("candidate_hashes", [])
            selected = latest[-1] if accepted and latest else None
            invocations_fn = getattr(provider, "invocations", None)
            records = invocations_fn() if callable(invocations_fn) else []
            model_calls = len(records)
            token_counts = [
                item.usage.total_tokens
                for item in records
                if item.usage is not None and item.usage.total_tokens is not None
            ]
            tokens = sum(token_counts) if len(token_counts) == len(records) else None
            preliminary = {
                "schema_version": "hardset-v2-real-agent-1",
                "case_id": case_id,
                "mode": mode,
                "investigation_run_id": probe.run_id,
                "candidate_sha256": selected,
                "probes": len(state.get("observations", [])),
                "candidates": state.get("attempts", 0),
                "verified": False,
                "verdict": "INCONCLUSIVE",
                "reason": state.get("stop_reason", "AGENT_STOPPED"),
                "model_physical_calls": model_calls,
                "model_total_tokens": tokens,
            }
        # Graph completed and the provider cannot use private verifier state.
    if accepted and selected:
        projection = run_strict_hardset_verification(
            root,
            output / "strict",
            case_id=case_id,
            unified_diff=accepted,
            evaluator_secret=evaluator_secret,
        )
        preliminary.update(
            verdict=projection.verdict,
            verified=projection.verdict == "VERIFIED_FIXED",
            reason=projection.reason_code,
        )
    outcome = HardsetAgentOutcome.model_validate(preliminary)
    (output / "agent-report.json").write_text(
        outcome.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return outcome
