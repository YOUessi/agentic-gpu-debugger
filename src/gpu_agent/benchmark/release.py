"""Typed release evidence gate; documentation cannot substitute for missing runs."""

from typing import Literal

from pydantic import Field

from gpu_agent.execution.models import ExecutionModel


class TestCounts(ExecutionModel):
    expected: int = Field(ge=0)
    executed: int = Field(ge=0)
    skipped_required: int = Field(ge=0)
    failed: int = Field(ge=0)


class ReleaseManifest(ExecutionModel):
    schema_version: Literal[1] = 1
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    toolchain_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    test_counts: TestCounts
    public_case_count: int = Field(ge=0)
    private_case_count: int = Field(ge=0)
    evidence_run_ids: dict[str, list[str]]
    unresolved_items: list[str]


class ReleaseGateResult(ExecutionModel):
    passed: bool
    reason_codes: list[str]


class ReleaseGate:
    REQUIRED_EVIDENCE = {
        "four_tools",
        "isolation",
        "private_oracle",
        "live_llm",
        "five_mode_evaluation",
    }

    def check(self, manifest: ReleaseManifest) -> ReleaseGateResult:
        reasons = []
        counts = manifest.test_counts
        if counts.executed != counts.expected:
            reasons.append("TEST_COUNT_INCOMPLETE")
        if counts.skipped_required:
            reasons.append("REQUIRED_TEST_SKIPPED")
        if counts.failed:
            reasons.append("TEST_FAILURE")
        if manifest.public_case_count < 16 or manifest.private_case_count < 8:
            reasons.append("CORPUS_COUNT_INSUFFICIENT")
        missing = self.REQUIRED_EVIDENCE - {
            key for key, run_ids in manifest.evidence_run_ids.items() if run_ids
        }
        if missing:
            reasons.append("LIVE_EVIDENCE_MISSING")
        if manifest.unresolved_items:
            reasons.append("UNRESOLVED_ITEMS")
        return ReleaseGateResult(passed=not reasons, reason_codes=reasons)
