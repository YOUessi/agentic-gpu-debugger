"""Controller authority over action phase, budget and evidence sufficiency."""

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from gpu_agent.agent.models import (
    AgentAction,
    AgentActionOutput,
    AgentBudget,
    DiagnosisResult,
    PolicyDecision,
    ProviderError,
    PublicEvidence,
)
from gpu_agent.contracts import CurrentPhase
from gpu_agent.execution.models import SanitizerTool

CallKind = Literal["plan", "diagnose", "patch"]


class BudgetExceeded(Exception):
    pass


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: int
    action: str
    attempt: int


class BudgetLedger:
    """Thread-safe pre-execution reservations with an append-only audit."""

    def __init__(
        self, budget: AgentBudget | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.budget = budget or AgentBudget()
        self._clock, self._start = clock, clock()
        self._lock = threading.Lock()
        self._counts = {"llm": 0, "planner": 0, "sanitizer": 0, "rag": 0, "source": 0}
        self._retried = False
        self._next_id = 0
        self._active: set[int] = set()
        self.audit: list[dict[str, object]] = []

    def remaining(self) -> float:
        return max(0.0, self.budget.max_wall_time_seconds - (self._clock() - self._start))

    def reserve(self, action: str, *, attempt: int = 0) -> BudgetReservation:
        with self._lock:
            self._next_id += 1
            reservation = BudgetReservation(self._next_id, action, attempt)
            self.audit.append(
                {"id": reservation.reservation_id, "action": action, "state": "ATTEMPTED"}
            )
            try:
                if self.remaining() <= 0:
                    raise BudgetExceeded("WALL_TIME_EXHAUSTED")
                if attempt not in {0, 1} or (attempt == 1 and self._retried):
                    raise BudgetExceeded("RETRY_EXHAUSTED")
                if action in {"planner_llm", "diagnosis_llm", "patch_llm"}:
                    reserve_final = 2 if action == "planner_llm" else int(action == "diagnosis_llm")
                    if self._counts["llm"] + reserve_final >= self.budget.max_llm_calls:
                        raise BudgetExceeded("LLM_BUDGET_EXHAUSTED")
                    if action == "planner_llm" and self._counts["planner"] >= 4:
                        raise BudgetExceeded("PLANNER_BUDGET_EXHAUSTED")
                    self._counts["llm"] += 1
                    self._counts["planner"] += int(action == "planner_llm")
                elif action.startswith("run_") and action.endswith("check"):
                    if self._counts["sanitizer"] >= self.budget.max_sanitizer_calls:
                        raise BudgetExceeded("SANITIZER_BUDGET_EXHAUSTED")
                    self._counts["sanitizer"] += 1
                elif action == "retrieve_official_docs":
                    if self._counts["rag"] >= self.budget.max_rag_calls:
                        raise BudgetExceeded("RAG_BUDGET_EXHAUSTED")
                    self._counts["rag"] += 1
                elif action == "inspect_source":
                    if self._counts["source"] >= self.budget.max_source_reads:
                        raise BudgetExceeded("SOURCE_BUDGET_EXHAUSTED")
                    self._counts["source"] += 1
                else:
                    raise BudgetExceeded("UNKNOWN_BUDGET_ACTION")
                self._retried |= attempt == 1
            except BudgetExceeded as error:
                self.audit.append(
                    {
                        "id": reservation.reservation_id,
                        "action": action,
                        "state": "REJECTED",
                        "reason": str(error),
                    }
                )
                raise
            self.audit.append(
                {"id": reservation.reservation_id, "action": action, "state": "STARTED"}
            )
            self._active.add(reservation.reservation_id)
            return reservation

    def settle(self, reservation: BudgetReservation, result: str = "COMPLETED") -> None:
        if result not in {"COMPLETED", "FAILED", "CANCELLED"}:
            raise ValueError("invalid settlement")
        with self._lock:
            if reservation.reservation_id not in self._active:
                raise ValueError("reservation is unknown or already settled")
            self._active.remove(reservation.reservation_id)
            self.audit.append(
                {"id": reservation.reservation_id, "action": reservation.action, "state": result}
            )


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
    "run_racecheck",
    "run_initcheck",
    "run_synccheck",
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
    if "memcheck" not in evidence.sanitizer_outcomes:
        mandatory.append("run_memcheck")
    if evidence.tool_findings and not evidence.documentation:
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
    elif action.action_type == "finish_diagnosis" and not evidence.tool_findings:
        reason = "MANDATORY_EVIDENCE_MISSING"
    elif action.action_type in {"run_racecheck", "run_initcheck", "run_synccheck"} and (
        evidence.sanitizer_outcomes.get(SanitizerTool.MEMCHECK) != "CLEAN"
    ):
        reason = "MEMCHECK_PRECHECK_REQUIRED"
    elif (
        action.action_type in {"run_memcheck", "run_racecheck", "run_initcheck", "run_synccheck"}
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
    action_content = AgentActionOutput(action=action).model_dump_json().encode()
    return PolicyDecision(
        action_type=action.action_type,
        action_hash=hashlib.sha256(action_content).hexdigest(),
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
