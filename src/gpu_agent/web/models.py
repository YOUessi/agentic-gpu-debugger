"""Public web API contracts for the operator console."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ArtifactSummary(BaseModel):
    id: str
    name: str
    byte_count: int
    sha256: str


class CitationTarget(BaseModel):
    citation_id: str
    artifact_id: str
    artifact_name: str
    kind: Literal["artifact", "document"]
    label: str
    preview: str | None = None
    source_url: str | None = None


class CaseSummary(BaseModel):
    case_id: str
    algorithm: str
    requirement: str
    template_id: str | None = None
    mutation_id: str | None = None
    target_tool: str | None = None
    expected_finding: str | None = None
    repair_ready: bool


class RunSummary(BaseModel):
    id: str
    kind: str
    parent_run_id: str | None
    status: str
    phase: str | None
    last_event_at: datetime | None
    artifact_count: int
    diagnosis_outcome: str | None = None
    failure_family: str | None = None
    confidence: str | None = None
    repair_stop_reason: str | None = None
    verification_verdict: str | None = None


class RunListResponse(BaseModel):
    items: list[RunSummary]
    total: int
    page: int
    page_size: int


class RunStats(BaseModel):
    total_diagnoses: int
    active: int
    diagnosed: int
    verified_fixed: int
    needs_attention: int
    failure_families: dict[str, int]


class RunDetail(BaseModel):
    summary: RunSummary
    events: list[dict[str, Any]] = Field(default_factory=list)
    diagnosis: dict[str, Any] | None = None
    repair_summary: dict[str, Any] | None = None
    repair_rounds: list[dict[str, Any]] = Field(default_factory=list)
    candidate: dict[str, Any] | None = None
    verifications: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    citations: dict[str, CitationTarget] = Field(default_factory=dict)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)


class RepairJobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RepairJob(BaseModel):
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    status: RepairJobStatus
    case_id: str = Field(pattern=r"^case_\d{4}$")
    mode: Literal["D", "E"]
    created_at: datetime
    updated_at: datetime
    run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    verification_verdict: str | None = None
    error_code: str | None = None


class RepairRequest(BaseModel):
    case_id: str = Field(pattern=r"^case_\d{4}$")
    mode: Literal["D", "E"] = "E"
    max_candidates: int = Field(default=3, ge=1, le=20)
    max_llm_calls: int = Field(default=40, ge=1, le=40)
    allow_paid_calls: bool = False


class RepairResponse(BaseModel):
    run_id: str
    status: str
    verification_verdict: str | None = None


class VerifyRequest(BaseModel):
    strict: bool = True


class VerifyResponse(BaseModel):
    verification_run_id: str
    result: dict[str, Any]


class BatchCard(BaseModel):
    run_id: str
    status: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    case_count: int
    registered: int
    validated: int
    failed: int
    running: int
    not_run: int
    target_tools: dict[str, int] = Field(default_factory=dict)


class EvaluationModeSummary(BaseModel):
    mode: str
    record_count: int
    diagnosed: int
    verified_fixed: int
    verified_rate: float | None = None
    latency_mean_ms: float | None = None
    llm_calls_mean: float | None = None
    tokens_mean: float | None = None
    known_cost_usd: float | None = None


class EvaluationCard(BaseModel):
    run_id: str
    status: str
    split: str | None = None
    corpus_cutoff: int | None = None
    expected_units: int | None = None
    executed_units: int
    modes: list[str] = Field(default_factory=list)
    repeats: int | None = None
    verified_fixed: int
    verified_rate: float | None = None
    diagnosed: int
    latency_mean_ms: float | None = None
    llm_calls_mean: float | None = None
    tokens_mean: float | None = None
    known_cost_usd: float | None = None


class AnalyticsOverview(BaseModel):
    store_root: str
    batch_count: int
    evaluation_count: int
    projection_errors: list[str] = Field(default_factory=list)
    batches: list[BatchCard]
    evaluations: list[EvaluationCard]


class EvaluationRecordRow(BaseModel):
    ordinal: int
    record_id: str
    case_id: str
    template_id: str
    mode: str
    repeat: int
    status: str
    diagnosis_outcome: str | None = None
    failure_family: str | None = None
    verdict: str | None = None
    oracle_passed: bool | None = None
    latency_ms: float | None = None
    physical_calls: int | None = None
    sanitizer_calls: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    failure_reason: str | None = None


class EvaluationDetail(BaseModel):
    summary: EvaluationCard
    mode_metrics: list[EvaluationModeSummary]
    failure_families: dict[str, int]
    records: list[EvaluationRecordRow]
    total: int
    page: int
    page_size: int


class BatchCaseRow(BaseModel):
    case_id: str
    target_tool: str
    repetitions: int
    status: str
    clean_run_id: str | None = None
    clean_runtime_status: str | None = None
    clean_oracle_passed: bool | None = None
    clean_sanitizer_outcomes: list[str] = Field(default_factory=list)
    mutant_run_id: str | None = None
    mutant_runtime_status: str | None = None
    mutant_oracle_passed: bool | None = None
    mutant_sanitizer_outcomes: list[str] = Field(default_factory=list)
    target_detections: list[bool] = Field(default_factory=list)
    reason_code: str | None = None


class BatchDetail(BaseModel):
    summary: BatchCard
    register_requested: bool
    stopped_reason: str | None = None
    cases: list[BatchCaseRow]
