"""Public contracts for a *real* HardSet V2 CUDA debugging Agent.

Only public functional probes and user-supplied CUDA source reach model requests.
No private reference or evaluator test input is represented by these classes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.benchmark.hardset_v2_probe import FunctionalProbeObservation


class HardsetDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    root_cause: str = Field(min_length=5, max_length=1800)
    suspected_lines: list[int] = Field(min_length=1, max_length=12)
    cited_public_probe_ids: list[str] = Field(min_length=1, max_length=15)
    repair_strategy: str = Field(min_length=5, max_length=1400)
    confidence: Literal["low", "medium", "high"]


class HardsetAgentContext(BaseModel):
    """The full public input to diagnosis and patch generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: Literal["h001", "h002", "h003"]
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    public_source: str = Field(min_length=1, max_length=4 * 1024 * 1024)
    functional_requirement: str = Field(min_length=5, max_length=2000)
    observations: tuple[FunctionalProbeObservation, ...] = Field(min_length=1, max_length=15)


class HardsetPublicCheck(BaseModel):
    """Untrusted feedback from an actual public candidate GPU execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    passed: bool
    failure_code: Literal[
        "PUBLIC_CHECKS_PASSED",
        "PATCH_SCOPE_REJECTED",
        "BUILD_FAILED",
        "NUMERIC_MISMATCH",
        "PROGRAM_FAILED",
        "SANITIZER_FAILED",
        "PUBLIC_EXECUTION_UNAVAILABLE",
    ]
    first_failing_size: int | None = Field(default=None, ge=1, le=65536)
    public_error_excerpt: str = Field(default="", max_length=1000)
    public_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class HardsetAgentOutcome(BaseModel):
    """Only a bounded final verdict; no private oracle is exported."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["hardset-v2-real-agent-1"] = "hardset-v2-real-agent-1"
    case_id: Literal["h001", "h002", "h003"]
    mode: Literal["D", "E"]
    investigation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    candidate_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    probes: int = Field(ge=0, le=15)
    candidates: int = Field(ge=0, le=3)
    verified: bool
    verdict: Literal["VERIFIED_FIXED", "NOT_VERIFIED", "INCONCLUSIVE"]
    reason: str = Field(max_length=120)
    model_physical_calls: int = Field(ge=0)
    model_total_tokens: int | None = Field(default=None, ge=0)
    api_cost_usd: None = None
