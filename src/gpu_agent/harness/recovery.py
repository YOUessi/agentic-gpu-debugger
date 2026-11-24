"""Crash-safe, fail-closed checkpoints for repository-only coding sessions.

The checkpoint is private to the operator, tied to the fsynced journal head,
the pinned Git tree and a public RunStore run ID. It is NOT authorization to
retry an uncertain paid model call or GPU tool. Only fully completed tool
boundaries may be resumed.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.harness.journal import read_verified_journal, recovery_disposition
from gpu_agent.harness.models import (
    HarnessDecision,
    SyntaxState,
    TaskPlan,
    TaskProgress,
    TaskReflection,
    ToolObservation,
)
from gpu_agent.harness.repo_audit import read_repository_tool_events
from gpu_agent.store import RunStore, read_regular, reject_symlinks, sync_directory

MAX_CHECKPOINT_BYTES = 256 * 1024


class RepositoryCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["gpu-repository-checkpoint-v1"] = "gpu-repository-checkpoint-v1"
    base_sha: str = Field(pattern=r"^[a-f0-9]{40}$")
    goal_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    public_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    journal_head_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    diff_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_counter: int = Field(ge=4, le=160)
    max_steps: int = Field(ge=1, le=38)
    max_model_calls: int = Field(ge=4, le=40)
    max_output_tokens: int = Field(default=4096, ge=256, le=16384)
    max_edited_files: int = Field(ge=1, le=20)
    public_case_id: str | None = Field(default=None, pattern=r"^h00[1-3]$")
    context_mode: Literal["legacy", "evidence_first"] = "evidence_first"
    allow_batch_edits: bool = True
    hierarchical_mode: bool = False
    syntax_mode: bool = False
    compile_gate: bool = False
    allow_revert: bool = False
    max_compiles: int = Field(default=6, ge=1, le=8)
    public_diagnostics: bool = False
    ast_target_binding: bool = False
    max_public_checks: int = Field(default=2, ge=1, le=3)
    public_checks_attempted: int = Field(default=0, ge=0, le=3)
    public_probes_attempted: int = Field(default=0, ge=0, le=4)
    public_probe_signatures: tuple[str, ...] = ()
    public_probe_run_ids: tuple[str, ...] = ()
    attempted_candidate_hashes: tuple[str, ...] = ()
    observed_query_fingerprints: tuple[str, ...] = ()
    syntax_state: SyntaxState | None = None
    rejected_syntax_edits: tuple[str, ...] = ()
    last_public_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    last_checked_diff_sha: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    last_checked_candidate_sha: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    last_public_passed: bool = False
    accepted_hashes: dict[str, str]
    state: dict[str, Any]


def _validated_state(value: dict[str, Any], max_steps: int) -> dict[str, Any]:
    """Check exact LangGraph tool boundary, not just JSON well-formedness."""
    expected = {"history", "decision", "observation", "turns", "stop", "reason"}
    extras = {"task_plan", "task_progress", "task_reflection"}
    if set(value) not in (expected, expected | extras):
        raise ValueError("checkpoint state has unexpected or missing fields")
    if set(value) == expected | extras:
        TaskPlan.model_validate(value["task_plan"])
        TaskProgress.model_validate(value["task_progress"])
        if value["task_reflection"] is not None:
            TaskReflection.model_validate(value["task_reflection"])
    turns = value["turns"]
    if type(turns) is not int or not 1 <= turns <= max_steps:
        raise ValueError("invalid checkpoint turn count")
    history = value["history"]
    if not isinstance(history, list) or len(history) != turns:
        raise ValueError("checkpoint history does not match completed turns")
    for entry in history:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"tool", "status", "content"}
            or any(not isinstance(v, str) for v in entry.values())
            or len(entry["content"]) > 3900
        ):
            raise ValueError("checkpoint contains invalid tool history")
    decision = HarnessDecision.model_validate(value["decision"])
    observation = ToolObservation.model_validate(value["observation"])
    if (
        history[-1]
        != {
            "tool": observation.tool,
            "status": observation.status,
            "content": observation.content[:3900],
        }
        or decision.action.tool != observation.tool
    ):
        raise ValueError("checkpoint action and observation disagree")
    if type(value["stop"]) is not bool or not isinstance(value["reason"], str):
        raise ValueError("invalid checkpoint stop metadata")
    reason = value["reason"]
    allowed = {
        "CONTINUE",
        "STEP_BUDGET_EXHAUSTED",
        "WORKSPACE_DIFF_STAGED_UNVERIFIED",
        "SESSION_PAUSED",
    }
    if reason not in allowed:
        raise ValueError("unknown checkpoint stop reason")
    if (reason == "CONTINUE") == value["stop"]:
        raise ValueError("checkpoint stop flag disagrees with reason")
    return value


def write_repository_checkpoint(path: Path, checkpoint: RepositoryCheckpoint) -> None:
    """Atomic 0600 replacement and directory fsync, never public RunStore data."""
    reject_symlinks(path)
    _validated_state(checkpoint.state, checkpoint.max_steps)
    if checkpoint.hierarchical_mode != ("task_plan" in checkpoint.state):
        raise ValueError("checkpoint planning mode and state shape disagree")
    if checkpoint.event_counter != 4 * checkpoint.state["turns"]:
        raise ValueError("checkpoint trace count disagrees with completed turns")
    if (
        checkpoint.public_probes_attempted != len(checkpoint.public_probe_signatures)
        or checkpoint.public_probes_attempted != len(checkpoint.public_probe_run_ids)
        or len(set(checkpoint.public_probe_signatures)) != checkpoint.public_probes_attempted
        or len(set(checkpoint.public_probe_run_ids)) != checkpoint.public_probes_attempted
    ):
        raise ValueError("checkpoint public functional probe ledger invalid")
    if len(checkpoint.accepted_hashes) > checkpoint.max_edited_files:
        raise ValueError("checkpoint file limit exceeded")
    if (
        checkpoint.public_checks_attempted > checkpoint.max_public_checks
        or len(set(checkpoint.attempted_candidate_hashes)) != checkpoint.public_checks_attempted
        or (checkpoint.public_case_id is None and checkpoint.public_checks_attempted)
    ):
        raise ValueError("checkpoint public GPU check budget invalid")
    if (
        len(checkpoint.observed_query_fingerprints) > checkpoint.state["turns"]
        or len(set(checkpoint.observed_query_fingerprints))
        != len(checkpoint.observed_query_fingerprints)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)
            for value in checkpoint.observed_query_fingerprints
        )
    ):
        raise ValueError("checkpoint inspection keys are invalid")
    for name, digest in checkpoint.accepted_hashes.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid checkpoint edited-file manifest")
    raw = checkpoint.model_dump_json().encode("utf-8") + b"\n"
    if len(raw) > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint exceeds bounded private storage")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp = tempfile.mkstemp(prefix=".session-checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        sync_directory(path.parent)
    finally:
        Path(temp).unlink(missing_ok=True)


def read_repository_checkpoint(path: Path) -> RepositoryCheckpoint:
    if path.is_symlink():
        raise ValueError("checkpoint cannot be a symlink")
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("checkpoint must be a private regular file")
    checkpoint = RepositoryCheckpoint.model_validate_json(read_regular(path, MAX_CHECKPOINT_BYTES))
    _validated_state(checkpoint.state, checkpoint.max_steps)
    if checkpoint.hierarchical_mode != ("task_plan" in checkpoint.state):
        raise ValueError("checkpoint planning mode and state shape disagree")
    if checkpoint.event_counter != 4 * checkpoint.state["turns"]:
        raise ValueError("checkpoint action trace is inconsistent")
    if (
        checkpoint.public_probes_attempted != len(checkpoint.public_probe_signatures)
        or checkpoint.public_probes_attempted != len(checkpoint.public_probe_run_ids)
        or len(set(checkpoint.public_probe_signatures)) != checkpoint.public_probes_attempted
        or len(set(checkpoint.public_probe_run_ids)) != checkpoint.public_probes_attempted
    ):
        raise ValueError("checkpoint public functional probe ledger invalid")
    if len(checkpoint.accepted_hashes) > checkpoint.max_edited_files:
        raise ValueError("checkpoint changed-file budget exceeded")
    if (
        checkpoint.public_checks_attempted > checkpoint.max_public_checks
        or len(set(checkpoint.attempted_candidate_hashes)) != checkpoint.public_checks_attempted
        or (checkpoint.public_case_id is None and checkpoint.public_checks_attempted)
    ):
        raise ValueError("checkpoint public GPU check evidence invalid")
    if (
        len(checkpoint.observed_query_fingerprints) > checkpoint.state["turns"]
        or len(set(checkpoint.observed_query_fingerprints))
        != len(checkpoint.observed_query_fingerprints)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)
            for value in checkpoint.observed_query_fingerprints
        )
    ):
        raise ValueError("checkpoint inspection keys are invalid")
    return checkpoint


def validated_quiescent_repository_checkpoint(
    output: Path,
    *,
    goal: str,
    base_sha: str,
    max_steps: int,
    max_model_calls: int,
    max_output_tokens: int = 4096,
    max_edited_files: int,
    public_case_id: str | None = None,
    max_public_checks: int = 2,
    context_mode: Literal["legacy", "evidence_first"] = "evidence_first",
    allow_batch_edits: bool = True,
    hierarchical_mode: bool = False,
    syntax_mode: bool = False,
    compile_gate: bool = False,
    allow_revert: bool = False,
    max_compiles: int = 6,
    public_diagnostics: bool = False,
    ast_target_binding: bool = False,
) -> RepositoryCheckpoint:
    """Reject uncertain side effects, terminal sessions or mismatched settings."""
    records = read_verified_journal(output / "session-events.jsonl")
    if (
        recovery_disposition(records) != "QUIESCENT_INSPECTION_ONLY"
        or records[-1]["kind"] != "tool_completed"
    ):
        raise ValueError("session is not at a completed tool boundary")
    checkpoint = read_repository_checkpoint(output / "session-checkpoint.json")
    if (
        checkpoint.journal_head_sha256 != records[-1]["sha256"]
        or checkpoint.base_sha != base_sha
        or checkpoint.goal_sha256 != hashlib.sha256(goal.encode()).hexdigest()
        or checkpoint.max_steps != max_steps
        or checkpoint.max_model_calls != max_model_calls
        or checkpoint.max_output_tokens != max_output_tokens
        or checkpoint.max_edited_files != max_edited_files
        or checkpoint.public_case_id != public_case_id
        or checkpoint.max_public_checks != max_public_checks
        or checkpoint.context_mode != context_mode
        or checkpoint.allow_batch_edits != allow_batch_edits
        or checkpoint.hierarchical_mode != hierarchical_mode
        or checkpoint.syntax_mode != syntax_mode
        or checkpoint.compile_gate != compile_gate
        or checkpoint.allow_revert != allow_revert
        or checkpoint.max_compiles != max_compiles
        or checkpoint.public_diagnostics != public_diagnostics
        or checkpoint.ast_target_binding != ast_target_binding
    ):
        raise ValueError("checkpoint identity, journal head or budgets differ")
    first = records[0]
    if (
        first["kind"] != "session_started"
        or first["payload"].get("base_git_commit") != base_sha
        or first["payload"].get("goal_sha256") != checkpoint.goal_sha256
        or first["payload"].get("public_run_id") != checkpoint.public_run_id
        or first["payload"].get("public_case_id") != checkpoint.public_case_id
        or first["payload"].get("max_public_checks", 2) != checkpoint.max_public_checks
        or first["payload"].get("context_mode", "evidence_first") != checkpoint.context_mode
        or first["payload"].get("allow_batch_edits", True) != checkpoint.allow_batch_edits
        or first["payload"].get("hierarchical_mode", False) != checkpoint.hierarchical_mode
        or first["payload"].get("syntax_mode", False) != checkpoint.syntax_mode
        or first["payload"].get("compile_gate", False) != checkpoint.compile_gate
        or first["payload"].get("allow_revert", False) != checkpoint.allow_revert
        or first["payload"].get("max_compiles", 6) != checkpoint.max_compiles
        or first["payload"].get("public_diagnostics", False) != checkpoint.public_diagnostics
        or first["payload"].get("ast_target_binding", False) != checkpoint.ast_target_binding
        or first["payload"].get("max_steps", max_steps) != max_steps
        or first["payload"].get("max_model_calls", max_model_calls) != max_model_calls
        or first["payload"].get("max_output_tokens", 4096) != checkpoint.max_output_tokens
    ):
        raise ValueError("checkpoint does not match session creation evidence")
    completed = sum(event["kind"] == "tool_completed" for event in records)
    if completed != checkpoint.state["turns"]:
        raise ValueError("tool completion ledger does not match checkpoint state")
    if checkpoint.event_counter != 4 * completed:
        raise ValueError("public event counter disagrees with journal")
    if syntax_mode and public_case_id is not None:
        events = read_repository_tool_events(
            RunStore(output / "public"),
            checkpoint.public_run_id,
            expected_count=completed,
            require_action=True,
        )
        completed_records = [event for event in records if event["kind"] == "tool_completed"]
        for entry, event, journal_record in zip(
            checkpoint.state["history"], events, completed_records, strict=True
        ):
            if (
                entry["tool"] != event["tool"]
                or entry["status"] != event["status"]
                or journal_record["payload"]["tool"] != event["tool"]
                or journal_record["payload"]["status"] != event["status"]
            ):
                raise ValueError("checkpoint history differs from public tool ledger")
            observation = event.get("observation")
            if observation is not None:
                if entry["content"] != observation[:3900]:
                    raise ValueError("checkpoint observation differs from public tool ledger")
            elif len(entry["content"]) < 3900 and (
                hashlib.sha256(entry["content"].encode()).hexdigest() != event["observation_sha256"]
            ):
                raise ValueError("checkpoint observation hash differs from public tool ledger")
        if checkpoint.state["decision"]["action"] != events[-1]["action"]:
            raise ValueError("checkpoint last action differs from public tool ledger")
    return checkpoint
