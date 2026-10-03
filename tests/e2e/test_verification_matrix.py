"""The four public verdicts remain explainable under frozen check requirements."""

import pytest


@pytest.mark.parametrize(
    "updates,want",
    [
        ({}, "VERIFIED_FIXED"),
        ({"original_finding_present": True}, "NOT_FIXED"),
        ({"public_oracle_passed": False}, "REGRESSION_DETECTED"),
        ({"required_evidence_missing": True}, "INCONCLUSIVE"),
    ],
)
def test_verification_verdict_matrix(updates, want):
    from gpu_agent.verification.models import CheckRequirement, VerificationObservation
    from gpu_agent.verification.policy import decide_verdict

    observation = VerificationObservation(
        build_ok=True,
        runtime_ok=True,
        original_finding_present=False,
        public_oracle_passed=True,
        private_holdout_passed=True,
        check_requirements=[
            CheckRequirement(
                tool="memcheck", required=True, support="SUPPORTED", reason_code="TARGET_TOOL"
            )
        ],
        check_outcomes={"memcheck": "CLEAN"},
    ).model_copy(update=updates)
    assert decide_verdict(observation).value == want
