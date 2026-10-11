import pytest


@pytest.mark.parametrize(
    "updates,want",
    [
        ({}, "VERIFIED_FIXED"),
        (
            {
                "build_ok": False,
                "runtime_ok": None,
                "original_finding_present": None,
                "public_oracle_passed": None,
                "private_holdout_passed": None,
            },
            "NOT_FIXED",
        ),
        ({"original_finding_present": True}, "NOT_FIXED"),
        ({"original_finding_present": None}, "INCONCLUSIVE"),
        ({"required_evidence_missing": True}, "INCONCLUSIVE"),
        ({"public_oracle_passed": False}, "REGRESSION_DETECTED"),
        ({"private_holdout_passed": False}, "REGRESSION_DETECTED"),
        ({"runtime_ok": False}, "REGRESSION_DETECTED"),
        ({"public_oracle_passed": None}, "INCONCLUSIVE"),
        ({"new_blocking_findings": True}, "REGRESSION_DETECTED"),
        ({"new_blocking_findings": True, "original_finding_present": True}, "NOT_FIXED"),
        ({"build_ok": False, "required_evidence_missing": True}, "INCONCLUSIVE"),
        ({"original_finding_present": None, "new_blocking_findings": True}, "INCONCLUSIVE"),
        ({"original_finding_present": None, "public_oracle_passed": False}, "INCONCLUSIVE"),
    ],
)
def test_verdict_priority(updates, want):
    from gpu_agent.execution.models import Finding
    from gpu_agent.verification.models import VerificationObservation
    from gpu_agent.verification.policy import decide_verdict

    base = dict(
        build_ok=True,
        runtime_ok=True,
        original_finding_present=False,
        public_oracle_passed=True,
        private_holdout_passed=True,
        required_evidence_missing=False,
        new_blocking_findings=[],
    )
    if updates.get("new_blocking_findings"):
        updates = {**updates, "new_blocking_findings": [Finding(tool="memcheck", category="new")]}
    observation = VerificationObservation(**{**base, **updates})
    assert decide_verdict(observation).value == want
