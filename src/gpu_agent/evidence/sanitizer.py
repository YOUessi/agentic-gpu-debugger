"""Conservative parser for Compute Sanitizer's dedicated memcheck log."""

import re

from gpu_agent.execution.models import (
    CheckOutcome,
    Finding,
    SanitizerResult,
    SanitizerTool,
    SourceLocation,
)
from gpu_agent.execution.process import ProcessCapture


def parse_sanitizer(tool: SanitizerTool, capture: ProcessCapture) -> SanitizerResult:
    if tool != SanitizerTool.MEMCHECK:
        return SanitizerResult()
    # stdout is program output, never a sanitizer evidence channel.
    log = capture.stderr.decode("utf-8", errors="replace")
    findings: list[Finding] = []
    for block in re.split(
        r"(?=^========= (?:Invalid |Misaligned |Uninitialized |Leaked ))", log, flags=re.MULTILINE
    )[1:]:
        heading = block.splitlines()[0].removeprefix("========= ")
        category = re.split(r" of size | of \d+ bytes", heading)[0]
        location = re.search(r"^=========\s+at (.*?) in (.*?):(\d+)\s*$", block, re.MULTILINE)
        # Older versions print `at 0x... in /path/file.cu:line:kernel(...)`.
        old_location = re.search(r"\bin (.*?\.cu):(\d+)(?::(.*))?", block)
        source = None
        kernel = None
        if location:
            kernel = location[1]
            source = SourceLocation(path=location[2], line=int(location[3]), function=kernel)
        elif old_location:
            kernel = old_location[3] or None
            source = SourceLocation(
                path=old_location[1], line=int(old_location[2]), function=kernel
            )
        findings.append(
            Finding(tool=tool, category=category, kernel=kernel, source_location=source)
        )
    summaries = re.findall(r"^========= ERROR SUMMARY: (\d+) errors?\s*$", log, re.MULTILINE)
    errors = max(map(int, summaries), default=0)
    if errors and not findings:
        findings.append(Finding(tool=tool, category="MEMCHECK_ERROR_SUMMARY"))
    complete_summary = bool(
        re.search(r"^========= ERROR SUMMARY: \d+ errors?\s*\Z", log, re.MULTILINE)
    )
    normal = not (capture.timed_out or capture.cancelled or capture.truncated or capture.tool_error)
    # --error-exitcode applies only when the target succeeds. A reported CUDA
    # error can make the harness exit 1 even though memcheck completed normally.
    exited = capture.exit_code is not None and 0 <= capture.exit_code < 128
    completed = normal and complete_summary and (exited if findings else capture.exit_code == 0)
    outcome: CheckOutcome = "FINDING" if findings else "CLEAN" if completed else "TOOL_ERROR"
    return SanitizerResult(
        status="COMPLETED" if completed else "TOOL_ERROR",
        findings=findings,
        completed=completed,
        check_outcome=outcome,
    )
