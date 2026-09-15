"""Immutable evidence contracts for benchmark case registration."""

from typing import Literal

from pydantic import Field, model_validator

from gpu_agent.contracts import ArtifactRef
from gpu_agent.execution.models import ExecutionModel, SanitizerTool
from gpu_agent.verification.models import OracleResult


class CaseExecution(ExecutionModel):
    """Legacy non-release summary; no production registration API accepts this model."""

    case_id: str = Field(pattern=r"^case_[0-9]{4}$")
    template_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    mutation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    split: Literal["public", "private"]
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    harness_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    toolchain_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_set_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    oracle_id: str
    target_tool: SanitizerTool
    expected_finding: str
    run_ids: list[str] = Field(min_length=1)
    oracle_passed: bool
    required_checks_clean: bool
    target_confirmed: bool = False
    timed_out: bool = False
    detection_outcomes: list[Literal["CLEAN", "FINDING", "TOOL_ERROR", "UNSUPPORTED"]]


class CaseValidation(ExecutionModel):
    """Legacy non-release summary retained only for reading historical test data."""

    clean: CaseExecution
    mutant: CaseExecution
    same_configuration: bool
    target_confirmed: bool
    clean_source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    mutant_source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def identities(self) -> "CaseValidation":
        if self.clean.case_id != self.mutant.case_id:
            raise ValueError("case identities differ")
        if self.clean_source_hash != self.clean.source_hash:
            raise ValueError("clean source hash mismatch")
        if self.mutant_source_hash != self.mutant.source_hash:
            raise ValueError("mutant source hash mismatch")
        return self


class CaseManifest(ExecutionModel):
    id: str = Field(pattern=r"^case_[0-9]{4}$")
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    harness_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    mutation_id: str
    template_id: str
    split: Literal["public", "private"]
    oracle_id: str
    target_tool: SanitizerTool
    expected_finding: str
    validation_run_ids: list[str] = Field(min_length=2)
    toolchain_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_set_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class CaseExecutionPlan(ExecutionModel):
    case_id: str = Field(pattern=r"^case_[0-9]{4}$")
    template_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    mutation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    role: Literal["clean", "mutant"]
    split: Literal["public", "private"]
    source_manifest: dict[str, str] = Field(min_length=4, max_length=4)
    oracle_id: Literal["vector-add-cpu-v1"] = "vector-add-cpu-v1"
    target_tool: SanitizerTool
    expected_finding: str = Field(min_length=1, max_length=256)
    sanitizer_repetitions: int = Field(default=1, ge=1, le=10)
    atol: float = Field(default=1e-5, ge=0)
    rtol: float = Field(default=1e-5, ge=0)

    @model_validator(mode="after")
    def role_matches_mutation(self) -> "CaseExecutionPlan":
        if (self.role == "clean") != (self.mutation_id == "clean"):
            raise ValueError("clean role and mutation identity disagree")
        return self


class CaseOracleObservation(ExecutionModel):
    oracle_id: Literal["vector-add-cpu-v1"]
    input_ref: ArtifactRef
    output_ref: ArtifactRef
    result: OracleResult


class CaseExecutionObservation(ExecutionModel):
    case_id: str = Field(pattern=r"^case_[0-9]{4}$")
    template_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    mutation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    role: Literal["clean", "mutant"]
    split: Literal["public", "private"]
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    harness_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_set_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    toolchain_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    oracle_id: Literal["vector-add-cpu-v1"]
    target_tool: SanitizerTool
    expected_finding: str = Field(min_length=1, max_length=256)
    build_ref: ArtifactRef
    runtime_ref: ArtifactRef
    sanitizer_refs: list[ArtifactRef] = Field(min_length=1, max_length=10)
    oracle_ref: ArtifactRef


class CaseValidationArtifact(ExecutionModel):
    clean_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    mutant_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    clean_observation_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    mutant_observation_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
