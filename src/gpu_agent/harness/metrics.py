"""Comparable PUBLIC repository-agent trajectory metrics, with no private oracle.

All tool actions, refusals and timings are counted from immutable public
RunStore artifacts, not from a model's own narrative. A public GPU pass never
becomes VERIFIED_FIXED without an independent evaluator.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.harness.journal import read_verified_journal, recovery_disposition
from gpu_agent.harness.models import RepoHarnessOutcome
from gpu_agent.store import RunStore, read_regular


class PublicToolStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    turn: int = Field(ge=1, le=40)
    tool: str = Field(min_length=1, max_length=64)
    status: Literal["OK", "FAILED", "REJECTED"]
    rejection_code: str | None = Field(default=None, max_length=80)


class RepositoryTrajectoryMetrics(BaseModel):
    """Only comparable operational metrics; NOT an independent accuracy score."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["gpu-repository-trajectory-metrics-v1"] = (
        "gpu-repository-trajectory-metrics-v1"
    )
    base_sha: str = Field(pattern=r"^[a-f0-9]{40}$")
    public_case_id: str | None = Field(default=None, pattern=r"^h00[1-3]$")
    context_mode: Literal["legacy", "evidence_first"] = "evidence_first"
    public_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    status: Literal["DIFF_STAGED_UNVERIFIED", "INCONCLUSIVE"]
    turns: int = Field(ge=1, le=40)
    model_physical_calls: int = Field(ge=0)
    model_tokens: int | None = Field(default=None, ge=0)
    inspected_actions: int = Field(ge=0)
    first_accepted_edit_turn: int | None = Field(default=None, ge=1)
    first_public_check_request_turn: int | None = Field(default=None, ge=1)
    first_public_pass_turn: int | None = Field(default=None, ge=1)
    first_accepted_finish_turn: int | None = Field(default=None, ge=1)
    rejected_actions: int = Field(ge=0)
    redundant_inspection_rejections: int = Field(ge=0)
    public_gpu_checks_attempted: int = Field(ge=0, le=3)
    public_checks_passed_final: bool = False
    independent_hidden_verification: Literal[False] = False
    independently_verified: Literal[False] = False
    tool_trajectory: tuple[PublicToolStep, ...]


_TOOL_NAME = re.compile(r"^repo-harness/v1/tool-([0-9]{3})\.json$")
_REDUNDANT = {"REDUNDANT_READ", "REDUNDANT_LIST", "REDUNDANT_DIFF"}
_READ_TOOLS = {"repo_list_files", "repo_read_file", "repo_git_diff"}


def summarize_repository_trajectory(output: Path) -> RepositoryTrajectoryMetrics:
    """Read a terminal session without executing code, CUDA or paid-model calls."""
    result = RepoHarnessOutcome.model_validate_json(read_regular(output / "result.json", 32_000))
    if result.status == "PAUSED_RESUMABLE":
        raise ValueError("paused sessions have no terminal trajectory metrics")
    journal = read_verified_journal(output / "session-events.jsonl")
    if recovery_disposition(journal) != "FINISHED":
        raise ValueError("trajectory is not a finished, verified session journal")
    started = journal[0]
    if started["kind"] != "session_started":
        raise ValueError("missing session creation evidence")
    payload = started.get("payload", {})
    if (
        payload.get("public_run_id") != result.public_run_id
        or payload.get("base_git_commit") != result.base_sha
    ):
        raise ValueError("trajectory session identity mismatch")
    public_case = payload.get("public_case_id")
    context_mode = payload.get("context_mode", "evidence_first")
    if context_mode not in {"legacy", "evidence_first"}:
        raise ValueError("unsupported public context mode")
    if result.context_mode != context_mode:
        raise ValueError("session context mode changed between start and finish")
    if public_case is not None and public_case not in {"h001", "h002", "h003"}:
        raise ValueError("unexpected public CUDA case in session")
    if journal[-1]["payload"].get("status") != result.status:
        raise ValueError("terminal journal and result status disagree")

    store = RunStore(output / "public")
    parent = store.load(result.public_run_id)
    if parent.kind != "gpu_repository_agent":
        raise ValueError("public run is not a repository coding session")
    expected_status = "COMPLETED" if result.status == "DIFF_STAGED_UNVERIFIED" else "FAILED"
    if parent.status.value != expected_status:
        raise ValueError("public RunStore did not finish in the expected state")

    steps: dict[int, PublicToolStep] = {}
    for ref in parent.artifact_refs:
        match = _TOOL_NAME.fullmatch(ref.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index in steps:
            raise ValueError("duplicate public tool event")
        body: object = json.loads(store.read(ref))
        if not isinstance(body, dict):
            raise ValueError("tool event is not an object")
        tool = body.get("tool")
        status = body.get("status")
        code = body.get("rejection_code")
        if (
            not isinstance(tool, str)
            or not isinstance(status, str)
            or status not in {"OK", "FAILED", "REJECTED"}
            or (code is not None and not isinstance(code, str))
        ):
            raise ValueError("invalid public tool event schema")
        steps[index] = PublicToolStep.model_validate(
            {"turn": index, "tool": tool, "status": status, "rejection_code": code}
        )
    if sorted(steps) != list(range(1, result.tool_calls + 1)):
        raise ValueError("tool events are missing or not contiguous")
    if result.tool_calls != result.turns:
        raise ValueError("tool counter differs from completed model turns")
    if sum(event["kind"] == "tool_completed" for event in journal) != result.turns:
        raise ValueError("journal completion count differs from tool trace")

    sequence = tuple(steps[index] for index in sorted(steps))
    first_edit = next(
        (
            step.turn
            for step in sequence
            if step.tool in {"repo_replace_snippet", "repo_replace_many"}
            and step.status == "OK"
        ),
        None,
    )
    first_check = next(
        (step.turn for step in sequence if step.tool == "repo_run_public_checks"),
        None,
    )
    first_public_pass = next(
        (
            step.turn
            for step in sequence
            if step.tool == "repo_run_public_checks" and step.status == "OK"
        ),
        None,
    )
    first_finish = next(
        (step.turn for step in sequence if step.tool == "finish" and step.status == "OK"),
        None,
    )
    if (first_finish is not None) != (result.status == "DIFF_STAGED_UNVERIFIED"):
        raise ValueError("completion state contradicts tool evidence")
    completed_gpu_calls = sum(
        step.tool == "repo_run_public_checks" and step.status in {"OK", "FAILED"}
        for step in sequence
    )
    if completed_gpu_calls > result.public_checks_attempted:
        raise ValueError("GPU evidence exceeds the recorded public check budget")
    if result.public_checks_passed and first_public_pass is None:
        raise ValueError("public success has no successful public GPU observation")

    return RepositoryTrajectoryMetrics(
        base_sha=result.base_sha,
        public_case_id=public_case,
        context_mode=context_mode,
        public_run_id=result.public_run_id,
        status=result.status,
        turns=result.turns,
        model_physical_calls=result.model_calls,
        model_tokens=result.model_tokens,
        inspected_actions=sum(step.tool in _READ_TOOLS for step in sequence),
        first_accepted_edit_turn=first_edit,
        first_public_check_request_turn=first_check,
        first_public_pass_turn=first_public_pass,
        first_accepted_finish_turn=first_finish,
        rejected_actions=sum(step.status == "REJECTED" for step in sequence),
        redundant_inspection_rejections=sum(
            step.status == "REJECTED" and step.rejection_code in _REDUNDANT for step in sequence
        ),
        public_gpu_checks_attempted=result.public_checks_attempted,
        public_checks_passed_final=result.public_checks_passed,
        tool_trajectory=sequence,
    )
