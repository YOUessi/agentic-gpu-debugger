"""Check necessity is independent from detected runtime support."""

import pytest


class ControlledCapabilityReport(dict):
    """Test-only wrapper; production policy consumes probed support data."""

    def mark_unsupported(self, tool: str) -> None:
        self[tool] = "UNSUPPORTED"


@pytest.fixture
def capability_report():
    return ControlledCapabilityReport(
        memcheck="SUPPORTED",
        racecheck="SUPPORTED",
        initcheck="SUPPORTED",
        synccheck="SUPPORTED",
    )


def test_required_tool_cannot_be_skipped(capability_report):
    from gpu_agent.verification.policy import plan_checks

    capability_report.mark_unsupported("synccheck")
    checks = plan_checks("synccheck", "standard", capability_report)
    target = next(check for check in checks if check.tool == "synccheck")
    assert target.required is True
    assert target.support == "UNSUPPORTED"
    assert target.reason_code == "TARGET_TOOL_UNSUPPORTED"


@pytest.mark.parametrize("target", ["racecheck", "initcheck", "synccheck"])
def test_non_memory_tools_require_memcheck_precheck(target, capability_report):
    from gpu_agent.verification.policy import plan_checks

    checks = plan_checks(target, "standard", capability_report)
    required = [check.tool.value for check in checks if check.required]
    assert required == ["memcheck", target]
    assert checks[0].reason_code == "MEMORY_SAFETY_PRECHECK"


def test_strict_requires_every_applicable_tool(capability_report):
    from gpu_agent.verification.policy import plan_checks

    checks = plan_checks("racecheck", "strict", capability_report)
    assert [check.tool.value for check in checks if check.required] == [
        "memcheck",
        "racecheck",
        "initcheck",
        "synccheck",
    ]
    assert all(check.support == "SUPPORTED" for check in checks)


def test_unprobed_required_capability_fails_closed(capability_report):
    from gpu_agent.verification.policy import plan_checks

    del capability_report["initcheck"]
    check = next(
        check
        for check in plan_checks("initcheck", "standard", capability_report)
        if check.tool == "initcheck"
    )
    assert check.required
    assert check.support == "UNSUPPORTED"
    assert check.reason_code == "CAPABILITY_NOT_PROBED"


@pytest.mark.parametrize("target", ["nope", "MEMCHECK"])
def test_invalid_check_plan_input_is_rejected(target, capability_report):
    from gpu_agent.verification.policy import plan_checks

    with pytest.raises(ValueError):
        plan_checks(target, "standard", capability_report)
