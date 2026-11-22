"""Durable per-turn checkpoints, never automatic side-effect replay.

LangGraph owns scheduling; this SQLite projection persists the bounded Agent
state *after* the tool_completed write-ahead journal event is durable.
A checkpoint is inspection/reconciliation evidence, not permission to re-run
a paid model call or a Docker job after a crash.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gpu_agent.harness.journal import read_verified_journal, recovery_disposition

SHA = re.compile(r"^[a-f0-9]{64}$")
SOURCE_ID = re.compile(r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
SCHEMA_VERSION = 1
MAX_STATE_BYTES = 128_000


def _canonical(data: Mapping[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


class TurnCheckpointStore:
    """SQLite checkpoint writer; one process exclusively owns the session."""

    def __init__(self, path: Path, *, base_sha: str, run_id: str) -> None:
        if not SOURCE_ID.fullmatch(base_sha):
            raise ValueError("checkpoint requires a pinned Git SHA or source digest")
        if not re.fullmatch(r"^[a-f0-9]{32}$", run_id):
            raise ValueError("checkpoint requires a bounded public RunStore ID")
        if path.exists() or path.is_symlink():
            raise ValueError("checkpoint database must not overwrite an existing session")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(fd)
        self.path = path
        self.run_id = run_id
        self.base_sha = base_sha
        self._db = sqlite3.connect(str(path), isolation_level=None, timeout=5)
        self._db.execute("PRAGMA journal_mode=DELETE")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS turn_checkpoints(
               turn INTEGER PRIMARY KEY,
               source_sha TEXT NOT NULL,
               run_id TEXT NOT NULL,
               journal_sha TEXT NOT NULL,
               diff_sha TEXT NOT NULL,
               state_blob BLOB NOT NULL,
               state_sha TEXT NOT NULL
            )"""
        )
        self._closed = False

    def save(
        self,
        state: Mapping[str, Any],
        *,
        journal_sha256: str,
        worktree_diff_sha256: str,
    ) -> str:
        if self._closed:
            raise ValueError("checkpoint store closed")
        if not SHA.fullmatch(journal_sha256) or not SHA.fullmatch(worktree_diff_sha256):
            raise ValueError("checkpoint provenance hash invalid")
        turn = state.get("turns")
        if type(turn) is not int or not 1 <= turn <= 40:
            raise ValueError("turn number out of bounds")
        if "history" not in state or len(state["history"]) > 40:
            raise ValueError("bounded checkpoint history required")
        if state.get("decision") is None or state.get("observation") is None:
            raise ValueError("checkpoint may only be written after tool completion")
        blob = _canonical(state)
        if len(blob) > MAX_STATE_BYTES:
            raise ValueError("checkpoint state exceeds storage limit")
        digest = hashlib.sha256(blob).hexdigest()
        self._db.execute("BEGIN IMMEDIATE")
        try:
            last = self._db.execute(
                "SELECT COALESCE(MAX(turn), 0) FROM turn_checkpoints"
            ).fetchone()
            assert last is not None
            if turn != last[0] + 1:
                raise ValueError("checkpoint turn sequence broken")
            self._db.execute(
                """INSERT INTO turn_checkpoints
                (turn, source_sha, run_id, journal_sha, diff_sha, state_blob, state_sha)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    turn,
                    self.base_sha,
                    self.run_id,
                    journal_sha256,
                    worktree_diff_sha256,
                    blob,
                    digest,
                ),
            )
            self._db.execute("COMMIT")
        except Exception:
            self._db.execute("ROLLBACK")
            raise
        return digest

    def close(self) -> None:
        if not self._closed:
            self._db.close()
            self._closed = True

    def __enter__(self) -> TurnCheckpointStore:
        return self

    def __exit__(self, typ: object, val: object, tb: object) -> None:
        self.close()


def read_latest_checkpoint(
    database: Path,
    *,
    journal: Path,
    current_diff_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate journal and checkpoint hashes; never resume tool execution here."""
    if database.is_symlink() or not database.is_file():
        raise ValueError("missing or unsafe checkpoint database")
    if database.stat().st_mode & 0o077:
        raise ValueError("checkpoint database permissions too broad")
    events = read_verified_journal(journal)
    # SQLite URI with mode=ro prevents accidental mutation by the inspector.
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        row = db.execute(
            """SELECT turn, source_sha, run_id, journal_sha, diff_sha,
                      state_blob, state_sha
               FROM turn_checkpoints ORDER BY turn DESC LIMIT 1"""
        ).fetchone()
    if row is None:
        raise ValueError("no completed tool checkpoint recorded")
    turn, source, run_id, journal_sha, diff_sha, blob, digest = row
    if (
        not SOURCE_ID.fullmatch(source)
        or not re.fullmatch(r"^[a-f0-9]{32}$", run_id)
        or not SHA.fullmatch(journal_sha)
        or not SHA.fullmatch(diff_sha)
        or hashlib.sha256(blob).hexdigest() != digest
    ):
        raise ValueError("checkpoint integrity mismatch")
    state = json.loads(blob)
    if state.get("turns") != turn or not state.get("observation"):
        raise ValueError("invalid checkpoint controller state")
    matched = [
        event
        for event in events
        if event["sha256"] == journal_sha
        and event["kind"] == "tool_completed"
        and event.get("payload", {}).get("turn") == turn
    ]
    if len(matched) != 1:
        raise ValueError("checkpoint does not bind to a completed tool event")
    disposition = recovery_disposition(events)
    # Only a quiescent, exact journal head could be considered for manual
    # rehydration. Even then a caller MUST verify the live worktree diff first.
    exact_head = events[-1]["sha256"] == journal_sha
    if current_diff_sha256 is not None and (
        not SHA.fullmatch(current_diff_sha256) or current_diff_sha256 != diff_sha
    ):
        raise ValueError("workspace drift or invalid worktree diff hash")
    # Neither a completed journal nor a checkpoint alone authorizes resume.
    # Manual rehydration requires current workspace provenance separately.
    can_consider_resume = (
        disposition == "QUIESCENT_INSPECTION_ONLY"
        and exact_head
        and current_diff_sha256 is not None
    )
    return {
        "turn": turn,
        "source_sha": source,
        "public_run_id": run_id,
        "state_sha256": digest,
        "worktree_diff_sha256": diff_sha,
        "journal_sha256": journal_sha,
        "recovery_disposition": disposition,
        "manual_rehydration_candidate": can_consider_resume,
        "automatic_replay_permitted": False,
        "state": state,
    }
