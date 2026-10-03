"""Verdicts distinguish required evidence gaps from valid negative evidence."""

from gpu_agent.execution.models import SanitizerTool
from gpu_agent.verification.models import CheckRequirement, VerificationObservation


def _observation(**updates):
    base = VerificationObservation(
        build_ok=True,
        runtime_ok=True,
        original_finding_present=False,
        public_oracle_passed=True,
        private_holdout_passed=True,
        check_requirements=[
            CheckRequirement(
                tool=SanitizerTool.MEMCHECK,
                required=True,
                support="SUPPORTED",
                reason_code="TARGET_TOOL",
            )
        ],
        check_outcomes={SanitizerTool.MEMCHECK: "CLEAN"},
    )
    return base.model_copy(update=updates)


def test_required_unsupported_prevents_success():
    from gpu_agent.verification.policy import decide_verdict

    required = CheckRequirement(
        tool="synccheck",
        required=True,
        support="UNSUPPORTED",
        reason_code="TARGET_TOOL_UNSUPPORTED",
    )
    observation = _observation(
        check_requirements=[required], check_outcomes={SanitizerTool.SYNCCHECK: "UNSUPPORTED"}
    )
    assert decide_verdict(observation).value == "INCONCLUSIVE"


def test_optional_error_is_a_limitation_not_a_success_blocker():
    from gpu_agent.verification.policy import decide_verdict

    optional = CheckRequirement(
        tool="racecheck",
        required=False,
        support="SUPPORTED",
        reason_code="NOT_SELECTED",
    )
    observation = _observation(
        check_requirements=[*_observation().check_requirements, optional],
        check_outcomes={"memcheck": "CLEAN", "racecheck": "TOOL_ERROR"},
    )
    assert decide_verdict(observation).value == "VERIFIED_FIXED"


def test_strict_additional_timeout_blocks_success():
    from gpu_agent.verification.policy import decide_verdict

    required = CheckRequirement(
        tool="racecheck", required=True, support="SUPPORTED", reason_code="STRICT_MODE"
    )
    observation = _observation(
        check_requirements=[*_observation().check_requirements, required],
        check_outcomes={"memcheck": "CLEAN", "racecheck": "TOOL_ERROR"},
    )
    assert decide_verdict(observation).value == "INCONCLUSIVE"


def test_persisting_target_still_returns_not_fixed():
    from gpu_agent.verification.policy import decide_verdict

    observation = _observation(
        original_finding_present=True,
        check_outcomes={SanitizerTool.MEMCHECK: "FINDING"},
    )
    assert decide_verdict(observation).value == "NOT_FIXED"
