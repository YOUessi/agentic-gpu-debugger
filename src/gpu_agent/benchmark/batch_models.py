"""Operator-facing batch summaries; never accepted as corpus validation evidence."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.benchmark.models import AuthoritativeCaseSpec, CaseExecutionPlan
from gpu_agent.contracts import RepositorySnapshot, RunStatus, now
from gpu_agent.environment import ExpectedToolchain
from gpu_agent.execution.models import ExecutionModel, SanitizerTool


class SeedRecipe(ExecutionModel):
    case_id: str = Field(pattern=r"^case_[0-9]{4}$")
    clean_source: str = Field(pattern=r"^public/case_[0-9]{4}/public_input/kernel\.cu$")
    mutant_source: str = Field(pattern=r"^public/case_[0-9]{4}/public_input/kernel\.cu$")
    n: int = Field(ge=1, le=65536)
    a_value: float = 1.0
    b_value: float = 2.0


class SeedRecipes(ExecutionModel):
    schema_version: Literal[1] = 1
    cases: list[SeedRecipe] = Field(min_length=1, max_length=16)


class SeedDescription(ExecutionModel):
    case_id: str
    target_tool: SanitizerTool
    n: int
    repetitions: int
    source_hash: str
    clean_source_hash: str
    harness_hash: str
    input_set_hash: str


class PreflightReport(ExecutionModel):
    schema_version: Literal[1] = 1
    repository: str
    data_root: str
    repository_snapshot: RepositorySnapshot
    toolchain: ExpectedToolchain
    registry_hash: str
    recipe_hash: str
    cases: list[SeedDescription]
    gpu_executed: Literal[False] = False
    notes: list[str] = Field(
        default_factory=lambda: [
            "Static checks only; GPU/container readiness is checked when the batch starts.",
            "Public development seeds only; no private holdout or paid LLM evaluation.",
        ]
    )


@dataclass(frozen=True)
class PreparedSeed:
    spec: AuthoritativeCaseSpec
    clean_plan: CaseExecutionPlan
    mutant_plan: CaseExecutionPlan
    input_bytes: bytes


@dataclass(frozen=True)
class PreparedBatch:
    report: PreflightReport
    seeds: tuple[PreparedSeed, ...]


class RoleSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str | None = None
    run_status: RunStatus | None = None
    build_success: bool | None = None
    runtime_status: str | None = None
    oracle_passed: bool | None = None
    sanitizer_outcomes: list[str] = Field(default_factory=list)
    target_detections: list[bool] = Field(default_factory=list)
    instrumented_oracles: list[bool] = Field(default_factory=list)
    failure_stage: str | None = None
    reason_code: str | None = None
    error_type: str | None = None


CaseStatus = Literal["NOT_RUN", "RUNNING", "VALIDATED", "REGISTERED", "FAILED"]


class SeedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    target_tool: SanitizerTool
    repetitions: int
    status: CaseStatus = "NOT_RUN"
    clean: RoleSummary | None = None
    mutant: RoleSummary | None = None
    failure_stage: str | None = None
    reason_code: str | None = None
    error_type: str | None = None


class BatchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    batch_run_id: str
    status: RunStatus = RunStatus.RUNNING
    register_requested: bool = False
    started_at: datetime = Field(default_factory=now)
    finished_at: datetime | None = None
    cases: list[SeedResult]
    stopped_reason: str | None = None
    note: str = (
        "Operational summary only. Corpus membership requires BenchmarkBuilder's native "
        "validation and ledger registration; this is not release or evaluation evidence."
    )

    @property
    def all_passed(self) -> bool:
        return (
            self.status == RunStatus.COMPLETED
            and bool(self.cases)
            and all(case.status in {"VALIDATED", "REGISTERED"} for case in self.cases)
        )
