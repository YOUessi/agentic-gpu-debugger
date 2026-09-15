"""Pure verdict priority: infrastructure gaps, original defect, regressions, proof."""

import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Literal

from gpu_agent.execution.models import Finding, SanitizerResult, SanitizerTool
from gpu_agent.verification.models import (
    CheckRequirement,
    VerificationObservation,
    VerificationVerdict,
)

CheckSupport = Literal["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE"]


def plan_checks(
    target: SanitizerTool | str,
    mode: Literal["standard", "strict"],
    capability_report: Mapping[SanitizerTool | str, CheckSupport],
) -> list[CheckRequirement]:
    """Plan mandatory checks without conflating necessity and availability."""
    try:
        target_tool = SanitizerTool(target)
    except ValueError as error:
        raise ValueError("unknown sanitizer target") from error
    if mode not in {"standard", "strict"}:
        raise ValueError("unknown verification mode")
    probed: dict[SanitizerTool, CheckSupport] = {}
    for name, reported_support in capability_report.items():
        try:
            tool = SanitizerTool(name)
        except ValueError as error:
            raise ValueError("unknown sanitizer capability") from error
        if reported_support not in {"SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE"}:
            raise ValueError("unknown sanitizer support state")
        probed[tool] = reported_support

    required_tools = {SanitizerTool.MEMCHECK, target_tool}
    if mode == "strict":
        required_tools = set(SanitizerTool)
    checks: list[CheckRequirement] = []
    for tool in SanitizerTool:
        required = tool in required_tools
        if not required:
            checks.append(
                CheckRequirement(
                    tool=tool,
                    required=False,
                    support="NOT_APPLICABLE",
                    reason_code="NOT_SELECTED",
                )
            )
            continue
        support = probed.get(tool)
        if support is None:
            support, reason = "UNSUPPORTED", "CAPABILITY_NOT_PROBED"
        elif support != "SUPPORTED":
            reason = (
                "TARGET_TOOL_UNSUPPORTED" if tool == target_tool else "REQUIRED_TOOL_UNSUPPORTED"
            )
        elif tool == SanitizerTool.MEMCHECK and target_tool != SanitizerTool.MEMCHECK:
            reason = "MEMORY_SAFETY_PRECHECK"
        elif tool == target_tool:
            reason = "TARGET_TOOL"
        else:
            reason = "STRICT_MODE"
        checks.append(
            CheckRequirement(tool=tool, required=True, support=support, reason_code=reason)
        )
    return checks


def decide_verdict(observation: VerificationObservation) -> VerificationVerdict:
    o = observation
    if o.required_evidence_missing:
        return VerificationVerdict.INCONCLUSIVE
    if o.build_ok is False or o.original_finding_present is True:
        return VerificationVerdict.NOT_FIXED
    if o.original_finding_present is None:
        return VerificationVerdict.INCONCLUSIVE
    if o.new_blocking_findings or False in (
        o.runtime_ok,
        o.public_oracle_passed,
        o.private_holdout_passed,
    ):
        return VerificationVerdict.REGRESSION_DETECTED
    if (
        o.build_ok is True
        and o.runtime_ok is True
        and o.original_finding_present is False
        and o.public_oracle_passed is True
        and o.private_holdout_passed is True
    ):
        return VerificationVerdict.VERIFIED_FIXED
    return VerificationVerdict.INCONCLUSIVE


def finding_signature(finding: Finding) -> tuple[str, str, str, str] | None:
    access = re.search(r"\b(read|write|access|allocation)\b", finding.category, re.IGNORECASE)
    if not finding.kernel or not access:
        return None
    return (
        finding.tool.value,
        finding.category,
        re.sub(r"\s+", "", finding.kernel),
        access[0].lower(),
    )


def original_presence(
    original: list[Finding],
    result: SanitizerResult,
    line_map: dict[int, int | None],
    *,
    same_input: bool,
) -> bool | None:
    """Signature identity survives line drift; incomplete same-target evidence stays unknown.

    Exact mapped locations corroborate signatures; deleted/changed lines never establish absence.
    Conservatively count a same-signature finding at another line as retained, not repaired.
    """
    if not same_input or not original:
        return None
    uncertain = False
    for old in original:
        signature = finding_signature(old)
        location = old.source_location
        if (
            signature is None
            or location is None
            or PurePosixPath(location.path).name != "kernel.cu"
        ):
            uncertain = True
            continue
        mapped = line_map.get(location.line) if location.line else None
        for current in result.findings:
            current_signature = finding_signature(current)
            if current_signature is None:
                uncertain = True
            elif signature == current_signature:
                current_location = current.source_location
                if current_location and PurePosixPath(current_location.path).name != "kernel.cu":
                    uncertain = True
                    continue
                if current_location and mapped == current_location.line:
                    return True
                # A changed line or altered debug line table is not proof of repair.
                return True
    if uncertain or not result.completed or result.check_outcome not in {"CLEAN", "FINDING"}:
        return None
    tool = result.tool_result
    if tool is None or tool.tool_error or tool.timed_out or tool.cancelled or tool.truncated:
        return None
    payload = tool.typed_payload
    if (
        not payload.completed
        or payload.check_outcome != result.check_outcome
        or payload.binary_ref is None
        or payload.stdin_ref is None
        or any(f.tool.value != payload.tool for f in original)
    ):
        return None
    return False
