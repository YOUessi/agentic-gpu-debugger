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
from gpu_agent.harness.models import HarnessDecision, ToolObservation
from gpu_agent.store import read_regular, reject_symlinks, sync_directory

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
    max_edited_files: int = Field(ge=1, le=20)
    public_case_id: str | None = Field(default=None, pattern=r"^h00[1-3]$")
    max_public_checks: int = Field(default=2, ge=1, le=3)
    public_checks_attempted: int = Field(default=0, ge=0, le=3)
    attempted_candidate_hashes: tuple[str, ...] = ()
    last_public_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    last_checked_diff_sha: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    last_checked_candidate_sha: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    last_public_passed: bool = False
    accepted_hashes: dict[str, str]
    state: dict[str, Any]


def _validated_state(value: dict[str, Any], max_steps: int) -> dict[str, Any]:
    """Check exact LangGraph tool boundary, not just JSON well-formedness."""
    expected = {"history", "decision", "observation", "turns", "stop", "reason"}
    if set(value) != expected:
        raise ValueError("checkpoint state has unexpected or missing fields")
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
    if checkpoint.event_counter != 4 * checkpoint.state["turns"]:
        raise ValueError("checkpoint trace count disagrees with completed turns")
    if len(checkpoint.accepted_hashes) > checkpoint.max_edited_files:
        raise ValueError("checkpoint file limit exceeded")
    if (
        checkpoint.public_checks_attempted > checkpoint.max_public_checks
        or len(set(checkpoint.attempted_candidate_hashes)) != checkpoint.public_checks_attempted
        or (checkpoint.public_case_id is None and checkpoint.public_checks_attempted)
    ):
        raise ValueError("checkpoint public GPU check budget invalid")
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
    if checkpoint.event_counter != 4 * checkpoint.state["turns"]:
        raise ValueError("checkpoint action trace is inconsistent")
    if len(checkpoint.accepted_hashes) > checkpoint.max_edited_files:
        raise ValueError("checkpoint changed-file budget exceeded")
    if (
        checkpoint.public_checks_attempted > checkpoint.max_public_checks
        or len(set(checkpoint.attempted_candidate_hashes)) != checkpoint.public_checks_attempted
        or (checkpoint.public_case_id is None and checkpoint.public_checks_attempted)
    ):
        raise ValueError("checkpoint public GPU check evidence invalid")
    return checkpoint


def validated_quiescent_repository_checkpoint(
    output: Path,
    *,
    goal: str,
    base_sha: str,
    max_steps: int,
    max_model_calls: int,
    max_edited_files: int,
    public_case_id: str | None = None,
    max_public_checks: int = 2,
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
        or checkpoint.max_edited_files != max_edited_files
        or checkpoint.public_case_id != public_case_id
        or checkpoint.max_public_checks != max_public_checks
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
    ):
        raise ValueError("checkpoint does not match session creation evidence")
    completed = sum(event["kind"] == "tool_completed" for event in records)
    if completed != checkpoint.state["turns"]:
        raise ValueError("tool completion ledger does not match checkpoint state")
    if checkpoint.event_counter != 4 * completed:
        raise ValueError("public event counter disagrees with journal")
    return checkpoint
