"""Conservative parsers for dedicated Compute Sanitizer tool logs."""

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
    # stdout is program output, never a sanitizer evidence channel.
    log = capture.stderr.decode("utf-8", errors="replace")
    if tool == SanitizerTool.RACECHECK:
        findings = _parse_racecheck(log, tool)
        summary = re.search(
            r"^========= RACECHECK SUMMARY: (\d+) hazards? displayed "
            r"\((\d+) errors?, (\d+) warnings?\)\s*$",
            log,
            re.MULTILINE,
        )
        observed_count = sum(map(int, summary.groups()[1:])) if summary else 0
        if observed_count and not findings:
            findings.append(Finding(tool=tool, category="RACECHECK_HAZARD_SUMMARY"))
        complete_summary = bool(summary and log.rstrip().endswith(summary[0].rstrip()))
    else:
        findings = _parse_error_findings(log, tool)
        summaries = re.findall(r"^========= ERROR SUMMARY: (\d+) errors?\s*$", log, re.MULTILINE)
        errors = max(map(int, summaries), default=0)
        if errors and not findings:
            findings.append(Finding(tool=tool, category=f"{tool.value.upper()}_ERROR_SUMMARY"))
        complete_summary = bool(
            re.search(
                r"^========= ERROR SUMMARY: \d+ errors?\n"
                r"(?:========= ERROR SUMMARY: \d+ errors were not printed\. "
                r"Use --print-limit option to adjust the number of printed errors\n)?\Z",
                log,
                re.MULTILINE,
            )
        )
    normal = not (capture.timed_out or capture.cancelled or capture.truncated or capture.tool_error)
    # --error-exitcode applies only when the target succeeds. A reported CUDA
    # error can make the harness exit nonzero even though the tool completed.
    exited = capture.exit_code is not None and 0 <= capture.exit_code < 128
    completed = normal and complete_summary and (exited if findings else capture.exit_code == 0)
    outcome: CheckOutcome = "FINDING" if findings else "CLEAN" if completed else "TOOL_ERROR"
    return SanitizerResult(
        status="COMPLETED" if completed else "TOOL_ERROR",
        findings=findings,
        completed=completed,
        parser_version="compute-sanitizer-2",
        check_outcome=outcome,
    )


def _source(block: str) -> tuple[str | None, SourceLocation | None]:
    location = re.search(r"^=========\s+at (.*?) in (.*?):(\d+)\s*$", block, re.MULTILINE)
    locations = re.findall(r"\bin (.*?\.cu):(\d+)(?::([^\n]*))?", block)
    if location and location[2].endswith(".cu"):
        kernel = location[1]
        return kernel, SourceLocation(path=location[2], line=int(location[3]), function=kernel)
    if locations:
        path, line, observed_kernel = locations[-1]
        kernel = observed_kernel or None
        # Device Frame puts the function before `in`, rather than after the line.
        if kernel is None:
            frame = re.search(r"Device Frame:\s*(.*?)\+0x[0-9a-f]+\s+in\s+", block)
            kernel = frame[1] if frame else None
        return kernel, SourceLocation(path=path, line=int(line), function=kernel)
    return None, None


def _parse_error_findings(log: str, tool: SanitizerTool) -> list[Finding]:
    findings: list[Finding] = []
    prefixes = {
        SanitizerTool.MEMCHECK: r"(?:Invalid |Misaligned |Uninitialized |Leaked )",
        SanitizerTool.INITCHECK: r"(?:Uninitialized |Unused )",
        SanitizerTool.SYNCCHECK: (
            r"(?:Barrier error detected\.|Warpgroup MMA sequence error detected)"
        ),
    }
    prefix = prefixes.get(tool)
    if prefix is None:
        return findings
    for block in re.split(rf"(?=^========= {prefix})", log, flags=re.MULTILINE)[1:]:
        heading = block.splitlines()[0].removeprefix("========= ")
        category = re.split(r" of size | of \d+ bytes", heading)[0]
        kernel, source = _source(block)
        findings.append(
            Finding(tool=tool, category=category, kernel=kernel, source_location=source)
        )
    return findings


def _parse_racecheck(log: str, tool: SanitizerTool) -> list[Finding]:
    findings: list[Finding] = []
    for block in re.split(
        r"(?=^========= (?:Error|Warning): Race reported between )", log, flags=re.MULTILINE
    )[1:]:
        heading = block.splitlines()[0]
        first = re.search(r"between ([A-Za-z]+ access)", heading)
        second = re.search(r"^=========\s+and ([A-Za-z]+ access)", block, re.MULTILINE)
        category = "Race reported"
        if first and second:
            category += f" between {first[1]} and {second[1]}"
        kernel, source = _source(block)
        findings.append(
            Finding(tool=tool, category=category, kernel=kernel, source_location=source)
        )
    return findings
