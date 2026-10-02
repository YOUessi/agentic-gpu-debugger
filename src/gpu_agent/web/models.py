"""Public web API contracts for the operator console."""

from datetime import datetime
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
    diagnosis: dict[str, Any] | None = None
    repair_summary: dict[str, Any] | None = None
    repair_rounds: list[dict[str, Any]] = Field(default_factory=list)
    candidate: dict[str, Any] | None = None
    verifications: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    citations: dict[str, CitationTarget] = Field(default_factory=dict)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)


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
