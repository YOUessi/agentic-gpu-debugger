"""SYNTHETIC unit logs only: never benchmark or live acceptance evidence."""

import pytest

from gpu_agent.execution.process import ProcessCapture

CLEAN = b"========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"
OOB = (
    b"========= COMPUTE-SANITIZER\n"
    b"========= Invalid __global__ write of size 4 bytes\n"
    b"=========     at vector_add(float const *, float const *, float *, unsigned long)"
    b" in /input/kernel.cu:9\n"
    b"=========     by thread (1,0,0) in block (1,0,0)\n"
    b"========= ERROR SUMMARY: 1 error\n"
)
OOB_WITH_PRINT_LIMIT = OOB + (
    b"========= ERROR SUMMARY: 31 errors were not printed. "
    b"Use --print-limit option to adjust the number of printed errors\n"
)


@pytest.mark.parametrize(
    "log,exit_code,flags,want,completed",
    [
        (b"", 0, {}, "TOOL_ERROR", False),
        (CLEAN, 0, {}, "CLEAN", True),
        (OOB, 0, {}, "FINDING", True),
        (OOB, 86, {}, "FINDING", True),
        (OOB_WITH_PRINT_LIMIT, 86, {}, "FINDING", True),
        (OOB, 1, {}, "FINDING", True),
        (OOB, -11, {}, "FINDING", False),
        (CLEAN, 1, {}, "TOOL_ERROR", False),
        (CLEAN, -11, {}, "TOOL_ERROR", False),
        (CLEAN, 0, {"truncated": True}, "TOOL_ERROR", False),
        (OOB, 86, {"truncated": True}, "FINDING", False),
        (CLEAN, 0, {"timed_out": True}, "TOOL_ERROR", False),
        (CLEAN, 0, {"cancelled": True}, "TOOL_ERROR", False),
        (CLEAN, 0, {"tool_error": "CRASH"}, "TOOL_ERROR", False),
        (b"========= ERROR SUMMARY: 2 errors\n", 0, {}, "FINDING", True),
        (b"========= Invalid __global__ read of size 4 bytes\n", 0, {}, "FINDING", False),
    ],
)
def test_memcheck_requires_complete_normal_observation(log, exit_code, flags, want, completed):
    from gpu_agent.evidence.sanitizer import parse_sanitizer
    from gpu_agent.execution.models import SanitizerTool

    fields = {"exit_code": exit_code, "stdout": b"", "stderr": log, "timed_out": False}
    fields.update(flags)
    result = parse_sanitizer(SanitizerTool.MEMCHECK, ProcessCapture(**fields))
    assert result.check_outcome == want
    assert result.completed is completed
    assert result.parser_version


def test_source_line_and_kernel_are_observed_not_inferred():
    from gpu_agent.evidence.sanitizer import parse_sanitizer
    from gpu_agent.execution.models import SanitizerTool

    result = parse_sanitizer(SanitizerTool.MEMCHECK, ProcessCapture(0, b"", OOB, False))
    finding = result.findings[0]
    assert finding.tool == "memcheck"
    assert finding.category == "Invalid __global__ write"
    assert finding.kernel.startswith("vector_add(")
    assert finding.source_location.path == "/input/kernel.cu"
    assert finding.source_location.line == 9
    assert finding.raw_ref is None  # Parser alone has no persisted evidence.


def test_program_stdout_cannot_forge_sanitizer_summary():
    from gpu_agent.evidence.sanitizer import parse_sanitizer
    from gpu_agent.execution.models import SanitizerTool

    result = parse_sanitizer(SanitizerTool.MEMCHECK, ProcessCapture(0, CLEAN, b"", False))
    assert result.check_outcome == "TOOL_ERROR"


def test_other_tools_explicitly_unsupported():
    from gpu_agent.evidence.sanitizer import parse_sanitizer
    from gpu_agent.execution.models import SanitizerTool

    for tool in [SanitizerTool.RACECHECK, SanitizerTool.INITCHECK, SanitizerTool.SYNCCHECK]:
        result = parse_sanitizer(tool, ProcessCapture(0, b"", CLEAN, False))
        assert result.check_outcome == "UNSUPPORTED"
