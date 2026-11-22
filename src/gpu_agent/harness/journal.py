"""Durable append-only session event log for a self-hosted LangGraph harness.

Write-ahead records distinguish a model request/tool execution that *started*
from one which demonstrably finished. On interruption, the operator may inspect
an uncertain call, but no external CUDA/paid-model side effect is auto-replayed.
This is a journal, not yet a transparent cross-process LangGraph checkpointer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

ZERO_HASH = "0" * 64
EventKind = Literal[
    "session_started",
    "model_started",
    "model_selected",
    "tool_started",
    "tool_completed",
    "session_finished",
    "session_failed",
]


def _encoded(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )


class SessionJournal:
    """Exclusive, 0600, hash-chained JSONL; one writer per new session."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink():
            raise ValueError("session journal cannot be a symlink")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.path, flags, 0o600)
        self._file = os.fdopen(fd, "ab", buffering=0)
        self._prev_hash = ZERO_HASH
        self._sequence = 0
        self._closed = False

    def append(self, kind: EventKind, payload: dict[str, object]) -> str:
        if self._closed:
            raise ValueError("session journal is closed")
        if kind not in {
            "session_started",
            "model_started",
            "model_selected",
            "tool_started",
            "tool_completed",
            "session_finished",
            "session_failed",
        }:
            raise ValueError("unrecognized session event kind")
        if any(key in payload for key in ("api_key", "evaluator_secret", "hidden_input")):
            raise ValueError("private data forbidden in the session event log")
        body: dict[str, object] = {
            "sequence": self._sequence + 1,
            "previous_sha256": self._prev_hash,
            "kind": kind,
            "payload": payload,
        }
        serialized = _encoded(body)
        if len(serialized) > 12_000:
            raise ValueError("session event exceeds bounded public audit size")
        digest = hashlib.sha256(serialized).hexdigest()
        event = {**body, "sha256": digest}
        self._file.write(_encoded(event) + b"\n")
        os.fsync(self._file.fileno())
        self._sequence += 1
        self._prev_hash = digest
        return digest

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> SessionJournal:
        return self

    def __exit__(self, _t: object, _v: object, _tb: object) -> None:
        self.close()


def read_verified_journal(path: Path) -> list[dict[str, Any]]:
    """Reject truncation/corruption and all invalid hash/sequence transitions."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("invalid session journal")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("incomplete or empty session journal")
    if len(raw) > 1_000_000:
        raise ValueError("session journal exceeds audit size")
    prev = ZERO_HASH
    records: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), 1):
        try:
            value: dict[str, Any] = json.loads(line)
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError("invalid session event JSON") from None
        digest = value.pop("sha256", None)
        if (
            value.get("previous_sha256") != prev
            or value.get("sequence") != index
            or not isinstance(digest, str)
            or hashlib.sha256(_encoded(value)).hexdigest() != digest
        ):
            raise ValueError("session event chain mismatch")
        records.append({**value, "sha256": digest})
        prev = digest
    return records


def recovery_disposition(records: list[dict[str, Any]]) -> str:
    """Safety classification, *not* automatic side-effect replay permission."""
    if not records:
        return "INVALID"
    pending: set[tuple[str, int]] = set()
    for item in records:
        kind = item["kind"]
        payload = item.get("payload", {})
        turn = payload.get("turn")
        if not isinstance(turn, int):
            continue
        if kind == "model_started":
            pending.add(("model", turn))
        elif kind == "model_selected":
            pending.discard(("model", turn))
        elif kind == "tool_started":
            pending.add(("tool", turn))
        elif kind == "tool_completed":
            pending.discard(("tool", turn))
    if pending:
        return "UNCERTAIN_EXTERNAL_EFFECT_RECONCILE_MANUALLY"
    if records[-1]["kind"] == "session_finished":
        return "FINISHED"
    if records[-1]["kind"] == "session_failed":
        return "FAILED_REVIEW_REQUIRED"
    return "QUIESCENT_INSPECTION_ONLY"
