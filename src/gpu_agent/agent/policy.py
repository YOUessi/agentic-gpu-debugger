"""Controller authority over action phase, budget and evidence sufficiency."""

import threading
import time
from collections.abc import Callable
from typing import Literal

from gpu_agent.agent.models import (
    AgentAction,
    AgentBudget,
    DiagnosisResult,
    PolicyDecision,
    ProviderError,
    PublicEvidence,
)
from gpu_agent.contracts import CurrentPhase

CallKind = Literal["plan", "diagnose", "patch"]


class LLMCallGate:
    def __init__(
        self, budget: AgentBudget | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._budget = budget or AgentBudget()
        self._clock, self._start = clock, clock()
        self._lock = threading.Lock()
        self._calls = self._plans = 0
        self._diagnosed = self._patched = False
        self._format_retried = False

    def remaining(self) -> float:
        return max(0.0, self._budget.max_wall_time_seconds - (self._clock() - self._start))

    def snapshot(self) -> AgentBudget:
        return self._budget.model_copy(
            update={"llm_calls": self._calls, "remaining_seconds": self.remaining()}
        )

    def timeout(self, limit: float) -> float:
        remaining = self.remaining()
        if remaining <= 0:
            raise ProviderError("AGENT_BUDGET_EXHAUSTED")
        return min(limit, remaining)

    def reserve(self, kind: CallKind, *, attempt: int = 0) -> float:
        with self._lock:
            if attempt not in {0, 1} or (attempt == 1 and self._format_retried):
                raise ProviderError("LLM_INVALID_OUTPUT")
            remaining = self.remaining()
            reserve = (
                (int(not self._diagnosed) + int(not self._patched))
                if kind == "plan"
                else (int(not self._patched) if kind == "diagnose" else 0)
            )
            if (
                remaining <= 0
                or self._calls + reserve >= self._budget.max_llm_calls
                or (kind == "plan" and self._plans >= 4)
            ):
                raise ProviderError("AGENT_BUDGET_EXHAUSTED")
            self._calls += 1
            self._plans += int(kind == "plan")
            self._diagnosed |= kind == "diagnose"
            self._patched |= kind == "patch"
            self._format_retried |= attempt == 1
            return min(60.0, remaining)


SUPPORTED = {
    "inspect_source",
    "run_memcheck",
    "retrieve_official_docs",
    "finish_diagnosis",
    "declare_inconclusive",
}


def decide_action(
    action: AgentAction,
    evidence: PublicEvidence,
    budget: AgentBudget,
    phase: CurrentPhase,
    seen: set[str],
) -> PolicyDecision:
    mandatory = []
    if not evidence.tool_findings:
        mandatory.append("run_memcheck")
    if not evidence.documentation:
        mandatory.append("retrieve_official_docs")
    reason: str | None = None
    signature = action.action_type + action.typed_arguments.model_dump_json()
    if phase != CurrentPhase.DIAGNOSING:
        reason = "ACTION_PHASE_INVALID"
    elif budget.agent_steps >= budget.max_agent_steps or budget.remaining_seconds <= 0:
        reason = "AGENT_BUDGET_EXHAUSTED"
    elif action.action_type not in SUPPORTED:
        reason = "ACTION_UNSUPPORTED"
    elif signature in seen:
        reason = "DUPLICATE_NO_BENEFIT"
    elif action.action_type == "finish_diagnosis" and mandatory:
        reason = "MANDATORY_EVIDENCE_MISSING"
    elif (
        action.action_type == "run_memcheck"
        and budget.sanitizer_calls >= budget.max_sanitizer_calls
    ):
        reason = "AGENT_BUDGET_EXHAUSTED"
    elif (
        action.action_type == "retrieve_official_docs" and budget.rag_calls >= budget.max_rag_calls
    ):
        reason = "AGENT_BUDGET_EXHAUSTED"
    elif action.action_type == "inspect_source":
        args = action.typed_arguments
        if (
            args.source_id not in {s.source_id for s in evidence.sources}
            or args.start_line > args.end_line
        ):
            reason = "SOURCE_NOT_REGISTERED"
        elif budget.source_reads >= budget.max_source_reads:
            reason = "AGENT_BUDGET_EXHAUSTED"
    return PolicyDecision(
        allowed=reason is None,
        mandatory_actions=mandatory,
        prohibited_actions=sorted({"shell", "network", "private_files", "modify_source"}),
        reason_codes=[reason] if reason else [],
    )


def validate_diagnosis(result: DiagnosisResult, evidence: PublicEvidence) -> bool:
    if result.diagnostic_outcome != "DIAGNOSED":
        return True
    layers = [
        (result.observed_facts, {i for f in evidence.observed_facts for i in f.citation_ids}),
        (result.tool_findings, {f.artifact_id for f in evidence.tool_findings}),
        (result.documentation_evidence, {d.chunk_id for d in evidence.documentation}),
    ]
    if any(
        not claims or any(not set(c.citation_ids) <= allowed for c in claims)
        for claims, allowed in layers
    ):
        return False
    return bool(result.source_locations) and all(
        loc.path == "kernel.cu"
        and loc.line is not None
        and any(f.source_location == loc for f in evidence.tool_findings)
        and any(loc.line <= len(source.content.splitlines()) for source in evidence.sources)
        for loc in result.source_locations
    )
