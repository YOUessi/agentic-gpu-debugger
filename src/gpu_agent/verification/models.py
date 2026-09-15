"""Typed verification observations and explicitly public aggregate results."""

from enum import StrEnum
from typing import Literal

from pydantic import Field

from gpu_agent.execution.models import ExecutionModel, Finding, SanitizerTool


class CheckRequirement(ExecutionModel):
    tool: SanitizerTool
    required: bool
    support: Literal["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE"]
    reason_code: str


class OracleResult(ExecutionModel):
    oracle_id: str = "vector-add-cpu-v1"
    oracle_type: str = "numeric"
    input_set_id: str = "controller"
    passed: bool
    expected_summary: str = "withheld"
    actual_summary: str = "withheld"
    atol: float
    rtol: float
    nan_policy: str
    inf_policy: str
    failure_reason: str | None = None


class VerificationObservation(ExecutionModel):
    build_ok: bool | None = None
    runtime_ok: bool | None = None
    original_finding_present: bool | None = None
    public_oracle_passed: bool | None = None
    private_holdout_passed: bool | None = None
    required_evidence_missing: bool = False
    new_blocking_findings: list[Finding] = Field(default_factory=list)
    check_requirements: list[CheckRequirement] = Field(default_factory=list)
    check_outcomes: dict[SanitizerTool, str] = Field(default_factory=dict)


class VerificationVerdict(StrEnum):
    VERIFIED_FIXED = "VERIFIED_FIXED"
    NOT_FIXED = "NOT_FIXED"
    REGRESSION_DETECTED = "REGRESSION_DETECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class VerificationResult(ExecutionModel):
    verdict: VerificationVerdict
    failure_stage: str | None
    reason_code: str
    original_finding_present: bool | None
    public_oracle_passed: bool | None
    private_holdout_passed: bool | None
    required_checks: dict[str, str]
    check_requirements: list[CheckRequirement] = Field(default_factory=list)
    check_outcomes: dict[SanitizerTool, str] = Field(default_factory=dict)
    not_run_reasons: dict[SanitizerTool, str] = Field(default_factory=dict)
    check_plan_version: Literal["verification-m4-v1"] = "verification-m4-v1"
    new_findings: int = 0
    candidate_hash: str
    binary_hashes: list[str] = Field(default_factory=list)
    public_passed_count: int = 0
    private_passed_count: int = 0
    not_run_count: int = 0
    suite_hash: str
    evaluator_audit_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    evaluator_observation_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    limitations: list[str] = Field(default_factory=list)
