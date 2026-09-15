"""SYNTHETIC unit logs only: never benchmark or live acceptance evidence."""

import pytest

from gpu_agent.execution.process import ProcessCapture

CLEAN = b"========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"
# Minimal excerpts captured from the locked CUDA 12.8.93 / Compute Sanitizer
# 2025.1.0.0 image on 2026-09-15. Full raw logs remain run artifacts, not fixtures.
RACE_CLEAN = (
    b"========= COMPUTE-SANITIZER\n"
    b"========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
)
RACE = (
    b"========= COMPUTE-SANITIZER\n"
    b"========= Error: Race reported between Write access at vector_add(float const *, "
    b"float const *, float *, unsigned long)+0x130 in /input/kernel.cu:9\n"
    b"=========     and Write access at vector_add(float const *, float const *, float *, "
    b"unsigned long)+0x130 in /input/kernel.cu:9 [34 hazards]\n"
    b"========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)\n"
)
INIT = (
    b"========= COMPUTE-SANITIZER\n"
    b"========= Uninitialized __global__ memory read of size 4 bytes\n"
    b"=========     at vector_add(float const *, float const *, float *, unsigned long)"
    b"+0x130 in /input/kernel.cu:8\n"
    b"=========     by thread (0,0,0) in block (0,0,0)\n"
    b"========= ERROR SUMMARY: 257 errors\n"
    b"========= ERROR SUMMARY: 157 errors were not printed. "
    b"Use --print-limit option to adjust the number of printed errors\n"
)
SYNC = (
    b"========= COMPUTE-SANITIZER\n"
    b"========= Barrier error detected. Invalid arguments.\n"
    b"=========     at __syncwarp(unsigned int)+0xe0 in sm_30_intrinsics.hpp:110\n"
    b"=========         Device Frame: broken_sync(float *, unsigned long)+0xb0 "
    b"in /input/kernel.cu:12\n"
    b"========= Target application returned an error\n"
    b"========= ERROR SUMMARY: 17 errors\n"
)
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


@pytest.mark.parametrize(
    "tool,log,exit_code,category,source_line",
    [
        ("racecheck", RACE, 86, "Race reported between Write access and Write access", 9),
        ("initcheck", INIT, 86, "Uninitialized __global__ memory read", 8),
        ("synccheck", SYNC, 86, "Barrier error detected. Invalid arguments.", 12),
    ],
)
def test_four_tool_parser_uses_real_version_formats(tool, log, exit_code, category, source_line):
    from gpu_agent.evidence.sanitizer import parse_sanitizer
    from gpu_agent.execution.models import SanitizerTool

    result = parse_sanitizer(SanitizerTool(tool), ProcessCapture(exit_code, b"", log, False))
    assert result.completed
    assert result.check_outcome == "FINDING"
    assert result.findings[0].category == category
    assert result.findings[0].source_location.path == "/input/kernel.cu"
    assert result.findings[0].source_location.line == source_line


@pytest.mark.parametrize(
    "tool,log",
    [("memcheck", CLEAN), ("racecheck", RACE_CLEAN), ("initcheck", CLEAN), ("synccheck", CLEAN)],
)
def test_each_tool_requires_its_own_complete_clean_summary(tool, log):
    from gpu_agent.evidence.sanitizer import parse_sanitizer
    from gpu_agent.execution.models import SanitizerTool

    result = parse_sanitizer(SanitizerTool(tool), ProcessCapture(0, b"", log, False))
    assert result.completed
    assert result.check_outcome == "CLEAN"


def test_racecheck_rejects_generic_error_summary_as_completion():
    from gpu_agent.evidence.sanitizer import parse_sanitizer
    from gpu_agent.execution.models import SanitizerTool

    result = parse_sanitizer(SanitizerTool.RACECHECK, ProcessCapture(0, b"", CLEAN, False))
    assert not result.completed
    assert result.check_outcome == "TOOL_ERROR"
