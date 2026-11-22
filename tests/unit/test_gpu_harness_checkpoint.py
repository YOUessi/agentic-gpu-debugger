"""Crash-safe checkpoint records: never replay ambiguous paid API/GPU calls."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import pytest

from gpu_agent.harness.checkpoints import TurnCheckpointStore, read_latest_checkpoint
from gpu_agent.harness.journal import SessionJournal

RUN = "a" * 32
SOURCE = "b" * 40
WORKSPACE_HASH = hashlib.sha256(b"clean workspace").hexdigest()


def _state(turn: int) -> dict[str, object]:
    return {
        "turns": turn,
        "history": [
            {
                "tool": "repo_read_file",
                "status": "OK",
                "content": "public CUDA source excerpt",
            }
        ],
        "decision": {
            "action": {"tool": "repo_read_file", "path": "src/kernel.cu"},
            "reason": "Read source",
        },
        "observation": {"tool": "repo_read_file", "status": "OK", "content": "public source"},
        "reason": "CONTINUE",
        "stop": False,
    }


def _first_turn(journal: SessionJournal) -> str:
    journal.append("session_started", {"public_run_id": RUN})
    journal.append("model_started", {"turn": 1})
    journal.append("model_selected", {"turn": 1, "tool": "repo_read_file"})
    journal.append("tool_started", {"turn": 1, "tool": "repo_read_file"})
    journal.append("tool_completed", {"turn": 1, "tool": "repo_read_file"})
    return journal.head_sha256


def test_sqlite_checkpoint_is_bound_to_completed_tool_and_journal(tmp_path: Path) -> None:
    jpath = tmp_path / "session-events.jsonl"
    dbpath = tmp_path / "turn-checkpoints.sqlite3"
    with SessionJournal(jpath) as journal:
        digest = _first_turn(journal)
        with TurnCheckpointStore(dbpath, base_sha=SOURCE, run_id=RUN) as db:
            stored = db.save(
                _state(1),
                journal_sha256=digest,
                worktree_diff_sha256=WORKSPACE_HASH,
            )
            assert len(stored) == 64
            with pytest.raises(ValueError, match="sequence"):
                db.save(_state(1), journal_sha256=digest, worktree_diff_sha256=WORKSPACE_HASH)
            without_workspace = read_latest_checkpoint(dbpath, journal=jpath)
            assert without_workspace["manual_rehydration_candidate"] is False
            snapshot = read_latest_checkpoint(
                dbpath, journal=jpath, current_diff_sha256=WORKSPACE_HASH
            )
            assert snapshot["manual_rehydration_candidate"] is True
            assert snapshot["automatic_replay_permitted"] is False
            assert snapshot["turn"] == 1
            assert snapshot["state"]["decision"]["action"]["path"] == "src/kernel.cu"
            assert snapshot["journal_sha256"] == digest
            assert snapshot["worktree_diff_sha256"] == WORKSPACE_HASH
            assert snapshot["state_sha256"] == stored

        journal.append("session_finished", {"status": "DIFF_STAGED_UNVERIFIED"})
    final = read_latest_checkpoint(dbpath, journal=jpath)
    assert final["recovery_disposition"] == "FINISHED"
    assert final["manual_rehydration_candidate"] is False
    assert dbpath.stat().st_mode & 0o077 == 0


def test_pending_external_operation_is_never_automatically_replayed(
    tmp_path: Path,
) -> None:
    jpath = tmp_path / "journal.jsonl"
    dbpath = tmp_path / "checkpoint.sqlite3"
    with SessionJournal(jpath) as journal:
        last = _first_turn(journal)
        with TurnCheckpointStore(dbpath, base_sha=SOURCE, run_id=RUN) as db:
            db.save(_state(1), journal_sha256=last, worktree_diff_sha256=WORKSPACE_HASH)
        journal.append("model_started", {"turn": 2})
    record = read_latest_checkpoint(dbpath, journal=jpath)
    assert "UNCERTAIN" in record["recovery_disposition"]
    assert record["manual_rehydration_candidate"] is False
    assert record["automatic_replay_permitted"] is False


def test_checkpoint_tampering_and_wrong_journal_are_rejected(tmp_path: Path) -> None:
    journal_path = tmp_path / "events.jsonl"
    dbpath = tmp_path / "checkpoint.sqlite3"
    with SessionJournal(journal_path) as journal:
        digest = _first_turn(journal)
        with TurnCheckpointStore(dbpath, base_sha=SOURCE, run_id=RUN) as db:
            db.save(_state(1), journal_sha256=digest, worktree_diff_sha256=WORKSPACE_HASH)

    with sqlite3.connect(dbpath) as db:
        db.execute(
            "UPDATE turn_checkpoints SET state_blob = ? WHERE turn=1",
            (b"tampered",),
        )
        db.commit()
    with pytest.raises(ValueError, match="integrity mismatch"):
        read_latest_checkpoint(dbpath, journal=journal_path)

    os.chmod(dbpath, 0o644)
    with pytest.raises(ValueError, match="permissions"):
        read_latest_checkpoint(dbpath, journal=journal_path)


def test_state_validation_and_reopen_prevention(tmp_path: Path) -> None:
    dbpath = tmp_path / "checkpoint.sqlite3"
    with TurnCheckpointStore(dbpath, base_sha=SOURCE, run_id=RUN) as db:
        with pytest.raises(ValueError, match="tool completion"):
            db.save(
                {"turns": 1, "history": []},
                journal_sha256=WORKSPACE_HASH,
                worktree_diff_sha256=WORKSPACE_HASH,
            )
        with pytest.raises(ValueError, match="provenance"):
            db.save(_state(1), journal_sha256="not-a-hash", worktree_diff_sha256=WORKSPACE_HASH)
    with pytest.raises(ValueError, match="overwrite"):
        TurnCheckpointStore(dbpath, base_sha=SOURCE, run_id=RUN)


def test_langgraph_checkpoint_callback_sees_full_state_after_tool_result() -> None:
    from gpu_agent.harness.models import (
        FinishTask,
        HarnessDecision,
        SessionSnapshot,
        ToolObservation,
    )
    from gpu_agent.harness.runner import run_tool_agent

    class Model:
        def choose_harness_v1(self, snapshot: SessionSnapshot) -> HarnessDecision:
            return HarnessDecision(
                action=FinishTask(summary="Public checks passed on the current kernel"),
                reason="Operator tool reported success",
            )

    class Host:
        def snapshot(self, goal: str, history: list[dict[str, str]], steps: int) -> SessionSnapshot:
            return SessionSnapshot(
                goal=goal,
                case_id="h001",
                source_hash="e" * 64,
                functional_requirement="Fix functional correctness of CUDA",
                allowed_tools=("finish",),
                remaining_steps=steps,
                history=tuple(history),
            )

        def invoke(self, action: FinishTask) -> ToolObservation:
            return ToolObservation(tool=action.tool, status="OK", content="Public checks done")

    snapshots: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    result = run_tool_agent(
        Model(),
        Host(),
        goal="Fix the CUDA bug",
        max_steps=2,
        on_event=events.append,
        on_checkpoint=lambda state: snapshots.append(dict(state)),
    )
    assert result["reason"] == "PUBLIC_CANDIDATE_ACCEPTED"
    assert len(snapshots) == 1
    assert snapshots[0]["turns"] == 1
    assert snapshots[0]["decision"]["action"]["tool"] == "finish"
    assert snapshots[0]["observation"]["status"] == "OK"
    assert events[-1]["stage"] == "tool"
