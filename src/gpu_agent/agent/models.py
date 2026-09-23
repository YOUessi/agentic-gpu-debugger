"""M1 model-facing contracts contain identifiers, never executable capabilities."""

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from gpu_agent.contracts import new_id
from gpu_agent.execution.models import CheckOutcome, ExecutionModel, SanitizerTool, SourceLocation
from gpu_agent.knowledge.models import DocumentChunk

Identifier = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]


class AgentBudget(ExecutionModel):
    budget_version: Literal["m1-v1"] = "m1-v1"
    max_agent_steps: int = Field(default=38, ge=0, le=38)
    max_sanitizer_calls: int = Field(default=4, ge=0, le=4)
    max_rag_calls: int = Field(default=3, ge=0, le=3)
    max_source_reads: int = Field(default=5, ge=0, le=5)
    max_llm_calls: int = Field(default=40, ge=0, le=40)
    max_wall_time_seconds: float = Field(default=600, gt=0, le=600)
    agent_steps: int = Field(default=0, ge=0)
    sanitizer_calls: int = Field(default=0, ge=0)
    rag_calls: int = Field(default=0, ge=0)
    source_reads: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    remaining_seconds: float = Field(default=600, ge=0)


class AcquisitionUsage(ExecutionModel):
    """Actual acquisition invocations, distinct from budget reservations and attempts."""

    schema_version: Literal[1] = 1
    sanitizer_calls: int = Field(strict=True, ge=0, le=4)
    retrieval_calls: int = Field(strict=True, ge=0, le=3)


class NoArguments(ExecutionModel):
    pass


class SourceArguments(ExecutionModel):
    source_id: Identifier
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=200, ge=1, le=10000)


class DocsArguments(ExecutionModel):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=5)


class EvidenceArguments(ExecutionModel):
    evidence_kind: Literal["source", "memcheck", "official_docs"]


class ActionBase(ExecutionModel):
    action_id: Identifier = Field(default_factory=new_id)
    rationale: str = Field(default="", max_length=300)
    expected_information_gain: Literal["low", "medium", "high"] = "medium"
    budget_snapshot: AgentBudget | None = None


class InspectSourceAction(ActionBase):
    action_type: Literal["inspect_source"] = "inspect_source"
    typed_arguments: SourceArguments


class InspectEnvironmentAction(ActionBase):
    action_type: Literal["inspect_environment"] = "inspect_environment"
    typed_arguments: NoArguments = Field(default_factory=NoArguments)


class RunProgramAction(ActionBase):
    action_type: Literal["run_program"] = "run_program"
    typed_arguments: NoArguments = Field(default_factory=NoArguments)


class MemcheckAction(ActionBase):
    action_type: Literal["run_memcheck"] = "run_memcheck"
    typed_arguments: NoArguments = Field(default_factory=NoArguments)


class RacecheckAction(ActionBase):
    action_type: Literal["run_racecheck"] = "run_racecheck"
    typed_arguments: NoArguments = Field(default_factory=NoArguments)


class InitcheckAction(ActionBase):
    action_type: Literal["run_initcheck"] = "run_initcheck"
    typed_arguments: NoArguments = Field(default_factory=NoArguments)


class SynccheckAction(ActionBase):
    action_type: Literal["run_synccheck"] = "run_synccheck"
    typed_arguments: NoArguments = Field(default_factory=NoArguments)


class RetrieveDocsAction(ActionBase):
    action_type: Literal["retrieve_official_docs"] = "retrieve_official_docs"
    typed_arguments: DocsArguments


class RequestEvidenceAction(ActionBase):
    action_type: Literal["request_more_evidence"] = "request_more_evidence"
    typed_arguments: EvidenceArguments


class FinishAction(ActionBase):
    action_type: Literal["finish_diagnosis"] = "finish_diagnosis"
    typed_arguments: NoArguments = Field(default_factory=NoArguments)


class InconclusiveAction(ActionBase):
    action_type: Literal["declare_inconclusive"] = "declare_inconclusive"
    typed_arguments: NoArguments = Field(default_factory=NoArguments)


ActionVariants = (
    InspectSourceAction
    | InspectEnvironmentAction
    | RunProgramAction
    | MemcheckAction
    | RacecheckAction
    | InitcheckAction
    | SynccheckAction
    | RetrieveDocsAction
    | RequestEvidenceAction
    | FinishAction
    | InconclusiveAction
)
AgentAction = Annotated[ActionVariants, Field(discriminator="action_type")]
ACTION_ADAPTER: TypeAdapter[AgentAction] = TypeAdapter(AgentAction)


class AgentActionOutput(ExecutionModel):
    # Responses strict schemas support nested anyOf. Preserve the exact variant set
    # on the wire; ACTION_ADAPTER applies discriminated domain validation on return.
    action: ActionVariants


# Model-facing planner contract. Controller-owned fields (action_id, budget_snapshot)
# are deliberately absent: under strict structured output every property becomes
# required, which previously forced the model to invent a 32-hex identifier and to
# echo the controller budget. The controller assigns those after validation.
PLANNER_RATIONALE_LIMIT = 300
PLANNER_QUERY_LIMIT = 500
PLANNER_MAX_K = 5


class _PlannerChoice(ExecutionModel):
    rationale: str = Field(default="", max_length=4000)
    expected_information_gain: Literal["low", "medium", "high"] = "medium"


class PlanNoArgumentAction(_PlannerChoice):
    action_type: Literal[
        "run_memcheck",
        "run_racecheck",
        "run_initcheck",
        "run_synccheck",
        "finish_diagnosis",
        "declare_inconclusive",
    ]
    typed_arguments: NoArguments = Field(default_factory=NoArguments)


class PlanDocsArguments(ExecutionModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=PLANNER_MAX_K, ge=1, le=50)


class PlanRetrieveDocsAction(_PlannerChoice):
    action_type: Literal["retrieve_official_docs"]
    typed_arguments: PlanDocsArguments


class PlanInspectSourceAction(_PlannerChoice):
    action_type: Literal["inspect_source"]
    typed_arguments: SourceArguments


PlannerChoice = PlanNoArgumentAction | PlanRetrieveDocsAction | PlanInspectSourceAction


class PlannerOutput(ExecutionModel):
    """What the planner model returns; converted to a full AgentAction by the controller."""

    action: PlannerChoice

    def to_action_output(self, action_id: str) -> "AgentActionOutput":
        choice = self.action
        arguments: dict[str, object]
        if isinstance(choice, PlanRetrieveDocsAction):
            arguments = {
                "query": choice.typed_arguments.query[:PLANNER_QUERY_LIMIT],
                "k": min(choice.typed_arguments.k, PLANNER_MAX_K),
            }
        else:
            arguments = choice.typed_arguments.model_dump()
        action = ACTION_ADAPTER.validate_python(
            {
                "action_id": action_id,
                "action_type": choice.action_type,
                "rationale": choice.rationale[:PLANNER_RATIONALE_LIMIT],
                "expected_information_gain": choice.expected_information_gain,
                "typed_arguments": arguments,
            }
        )
        return AgentActionOutput(action=action)


class PublicSource(ExecutionModel):
    source_id: Identifier
    path: Literal["kernel.cu"] = "kernel.cu"
    content: str = Field(max_length=4 * 1024 * 1024)


class EvidenceClaim(ExecutionModel):
    text: str = Field(min_length=1, max_length=2000)
    citation_ids: list[str] = Field(min_length=1, max_length=20)


class PublicFinding(ExecutionModel):
    artifact_id: Identifier
    category: str
    source_location: SourceLocation | None = None


class PublicEvidence(ExecutionModel):
    """Explicit export allowlist; backend commands, paths and harness stay controller-side."""

    sources: list[PublicSource] = Field(default_factory=list)
    observed_facts: list[EvidenceClaim] = Field(default_factory=list)
    tool_findings: list[PublicFinding] = Field(default_factory=list)
    documentation: list[DocumentChunk] = Field(default_factory=list)
    sanitizer_outcomes: dict[SanitizerTool, CheckOutcome] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


# Frozen before any evaluation run (evaluation/development-labels.json uses the same set).
# A closed vocabulary makes family accuracy a label comparison, not a spelling contest.
FailureFamily = Literal[
    "out_of_bounds",
    "misaligned_access",
    "shared_memory_race",
    "uninitialized_memory_read",
    "barrier_misuse",
    "other",
    "unknown",
]
FAILURE_FAMILIES: tuple[str, ...] = (
    "out_of_bounds",
    "misaligned_access",
    "shared_memory_race",
    "uninitialized_memory_read",
    "barrier_misuse",
    "other",
    "unknown",
)


class DiagnosisResult(ExecutionModel):
    diagnostic_outcome: Literal["DIAGNOSED", "INCONCLUSIVE", "LLM_UNAVAILABLE"]
    failure_family: FailureFamily = "unknown"
    root_cause: str = Field(default="", max_length=2000)
    source_locations: list[SourceLocation] = Field(default_factory=list)
    observed_facts: list[EvidenceClaim] = Field(default_factory=list)
    tool_findings: list[EvidenceClaim] = Field(default_factory=list)
    documentation_evidence: list[EvidenceClaim] = Field(default_factory=list)
    model_inferences: list[str] = Field(default_factory=list, max_length=10)
    recommended_change: str = Field(default="", max_length=2000)
    confidence_label: Literal["low", "medium", "high"] = "low"
    limitations: list[str] = Field(default_factory=list)

    @classmethod
    def inconclusive(cls, code: str) -> "DiagnosisResult":
        return cls(
            diagnostic_outcome="LLM_UNAVAILABLE" if code == "LLM_UNAVAILABLE" else "INCONCLUSIVE",
            limitations=[code],
        )


class PolicyDecision(ExecutionModel):
    action_type: str = Field(min_length=1)
    action_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    allowed: bool
    mandatory_actions: list[str] = Field(default_factory=list)
    prohibited_actions: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    policy_version: Literal["diagnosis-m1-v1"] = "diagnosis-m1-v1"


class PatchOutput(ExecutionModel):
    unified_diff: str = Field(min_length=1, max_length=4 * 1024 * 1024)


class ProviderError(Exception):
    """Only a fixed error code is printable; upstream messages never cross this boundary."""

    def __init__(self, code: str, *, state: str = "NOT_STARTED", retryable: bool = False) -> None:
        super().__init__(code)
        self.code, self.state, self.retryable = code, state, retryable
