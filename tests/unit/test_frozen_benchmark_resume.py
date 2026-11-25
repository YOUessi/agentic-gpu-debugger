"""Fault injection of the actual resume reducer and durable hash-chain."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_frozen_benchmark_resume as resume  # noqa: E402


@pytest.fixture
def manifest():
    path = SCRIPTS.parent / "docs/benchmark/2026-11-25-formal-benchmark-manifest.json"
    return json.loads(path.read_text())


def new_ledger(tmp_path, manifest):
    return resume.Ledger(tmp_path / "events", manifest)


def test_startup_failure_before_reservation(tmp_path, manifest):
    ledger = new_ledger(tmp_path, manifest)
    ledger.append("INFRA_EVENT", details={"event": "server failed before first trial"})
    r = new_ledger(tmp_path, manifest)
    assert all(x["state"] == "PENDING" for x in r.states.values())
    tid = manifest["trial_order"][0]["id"]
    assert (
        resume.recovery_action(
            r.states[tid],
            record=False,
            agent_done=False,
            strict_done=False,
            any_agent_evidence=False,
        )
        == "RUN_NEW"
    )


@pytest.mark.parametrize(
    "boundary,phase,agent,strict,record,evidence,expected",
    [
        ("before_agent_launch", "RESERVED", False, False, False, False, "INFRA_ABORT"),
        ("model_call", "AGENT_STARTED", False, False, False, True, "UNKNOWN"),
        ("after_tool_completion", "AGENT_STARTED", False, False, False, True, "UNKNOWN"),
        (
            "after_agent_report",
            "AGENT_RECORDED",
            True,
            False,
            False,
            True,
            "FINALIZE_AGENT_WITHOUT_RESAMPLING",
        ),
        ("strict_in_flight", "STRICT_STARTED", True, False, False, True, "UNKNOWN"),
        ("after_strict_projection", "STRICT_STARTED", True, True, False, True, "ADOPT_STRICT"),
        ("after_strict_record", "STRICT_RECORDED", True, True, False, True, "ADOPT_STRICT"),
        ("after_result_write", "RESULT_WRITTEN", True, True, True, True, "ADOPT_RECORD"),
    ],
)
def test_fault_injection_no_resample(
    tmp_path, manifest, boundary, phase, agent, strict, record, evidence, expected
):
    ledger = new_ledger(tmp_path, manifest)
    tid = manifest["trial_order"][0]["id"]
    calls = {"fake_model": 0, "fake_gpu": 0}
    ledger.append("RESERVED", tid, state="RUNNING", phase="RESERVED", attempt="a" * 32)
    if phase != "RESERVED":
        ledger.phase(tid, "AGENT_STARTED")
        calls["fake_model"] += 1
    if boundary == "after_tool_completion" or agent:
        calls["fake_gpu"] += 1
    if resume.PHASES.index(phase) > resume.PHASES.index("AGENT_STARTED"):
        ledger.phase(tid, phase)
    # Simulated interruption: construct a fresh reader from disk.
    fresh = new_ledger(tmp_path, manifest)
    decision = resume.recovery_action(
        fresh.states[tid],
        record=record,
        agent_done=agent,
        strict_done=strict,
        any_agent_evidence=evidence,
    )
    assert decision == expected and decision != "RUN_NEW"
    old_calls = dict(calls)
    # Recovery never invokes the fake model again. For the completed-report
    # branch it may finish a never-started independent check, not inference.
    if decision == "FINALIZE_AGENT_WITHOUT_RESAMPLING":
        calls["fake_gpu"] += 1
    assert calls["fake_model"] == old_calls["fake_model"]
    if decision in ("ADOPT_RECORD", "ADOPT_STRICT"):
        assert calls["fake_gpu"] == old_calls["fake_gpu"]


def test_immutable_terminal_and_duplicate_reservation(tmp_path, manifest):
    ledger = new_ledger(tmp_path, manifest)
    tid = manifest["trial_order"][0]["id"]
    ledger.append(
        "LEGACY_ADOPTED",
        tid,
        state="COMPLETED_SUCCESS",
        phase="RESULT_WRITTEN",
        attempt="b" * 64,
        details={"seal": {"record_sha256": "c" * 64}},
    )
    fresh = new_ledger(tmp_path, manifest)
    assert fresh.states[tid]["state"] == "COMPLETED_SUCCESS"
    with pytest.raises(AssertionError):
        fresh.append("RESERVED", tid, state="RUNNING", phase="RESERVED", attempt="a" * 32)
    assert (
        resume.recovery_action(
            fresh.states[tid],
            record=True,
            agent_done=True,
            strict_done=True,
            any_agent_evidence=True,
        )
        == "VERIFY_TERMINAL"
    )


def test_pending_untracked_artifact_is_not_new_trial(tmp_path, manifest):
    ledger = new_ledger(tmp_path, manifest)
    tid = manifest["trial_order"][0]["id"]
    assert (
        resume.recovery_action(
            ledger.states[tid],
            record=False,
            agent_done=False,
            strict_done=False,
            any_agent_evidence=True,
        )
        == "UNKNOWN"
    )


def test_hash_chain_and_no_overwrite(tmp_path, manifest):
    ledger = new_ledger(tmp_path, manifest)
    ledger.append("INFRA_EVENT", details={"first": True})
    p = ledger.root / "event-00001.json"
    with pytest.raises(FileExistsError):
        resume.atomic_json(p, {"tampered": True})
    r = json.loads(p.read_text())
    r["details"]["first"] = False
    p.write_text(json.dumps(r))
    with pytest.raises(AssertionError):
        new_ledger(tmp_path, manifest)


def test_progress_atomic_replace(tmp_path):
    p = tmp_path / "progress.json"
    resume.atomic_json(p, {"count": 6}, replace=True)
    resume.atomic_json(p, {"count": 7}, replace=True)
    assert json.loads(p.read_text()) == {"count": 7}
    assert not list(tmp_path.glob(".pending-*"))


class InjectedCrash(BaseException):
    pass


@pytest.mark.parametrize(
    "boundary,expected,model_calls,strict_calls",
    [
        ("model_call", "OUTCOME_UNKNOWN", 1, 0),
        ("tool_completed", "OUTCOME_UNKNOWN", 1, 0),
        ("agent_report", "COMPLETED_SUCCESS", 1, 1),
        ("strict_started", "OUTCOME_UNKNOWN", 1, 1),
        ("strict_projection", "COMPLETED_SUCCESS", 1, 1),
        ("result_written", "COMPLETED_SUCCESS", 1, 1),
    ],
)
def test_real_orchestration_boundaries(
    tmp_path, manifest, monkeypatch, boundary, expected, model_calls, strict_calls
):
    output = tmp_path / "output"
    (output / "trials").mkdir(parents=True)
    args = SimpleNamespace(output=output, repository=tmp_path, evidence_root=tmp_path)
    tid = manifest["trial_order"][0]["id"]
    root = output / "scheduler-v2/events"
    ledger = resume.Ledger(root, manifest)
    counts = {"model": 0, "tool": 0, "strict": 0}
    sha = "d" * 64
    audit = {
        "public_passed_candidate_sha256": sha,
        "final_candidate_sha256": sha,
        "live_edits": [{"turn": 2, "sha256": sha}],
    }
    agent_seal = {"verified_fake_agent": True}
    projection = {"candidate_sha256": sha, "verdict": "VERIFIED_FIXED", "case_id": "h001"}
    strict = {
        "status": "VERIFIED_FIXED",
        "projection": projection,
        "strict_public_run_id": "a" * 32,
        "projection_sha256": "b" * 64,
        "wall_seconds": 1.0,
        "mode": "fake",
        "exit_code": 0,
        "timed_out": False,
    }
    folder = output / "trials" / tid
    crashed = False

    def fake_invoke(*unused):
        counts["model"] += 1
        session = folder / "session"
        session.mkdir()
        resume.atomic_json(session / "fake-invocation.json", {"admitted": True})
        if boundary == "model_call":
            raise InjectedCrash
        counts["tool"] += 1
        resume.atomic_json(session / "fake-checkpoint.json", {"tool_completed": True})
        if boundary == "tool_completed":
            raise InjectedCrash
        resume.atomic_json(session / "controlled-recovery.json", {"completed": True})
        if boundary == "agent_report":
            raise InjectedCrash
        return {"exit_code": 0, "timed_out": False, "wall_seconds": 1.0}

    def fake_verify_agent(*unused):
        assert resume.read(folder / "session/controlled-recovery.json")["completed"]
        return audit, agent_seal, "PUBLIC_PASS"

    def fake_strict(*unused):
        counts["strict"] += 1
        if boundary == "strict_started" and not crashed:
            raise InjectedCrash
        resume.atomic_json(folder / "strict-projection.json", projection)
        if boundary == "strict_projection" and not crashed:
            raise InjectedCrash
        return strict

    def fake_projection(*unused):
        assert resume.read(folder / "strict-projection.json") == projection
        return {**strict, "public_inventory": {"inventory_sha256": "f" * 64}}

    def fake_record(*unused):
        r = resume.read(folder / "trial-record.json")
        assert r["success"] and r["audit"] == audit
        assert r["strict"]["projection"] == projection
        return (
            r,
            {"record_sha256": resume.frozen.file_digest(folder / "trial-record.json")},
            "COMPLETED_SUCCESS",
        )

    monkeypatch.setattr(resume.frozen, "invoke", fake_invoke)
    monkeypatch.setattr(resume, "verify_agent", fake_verify_agent)
    monkeypatch.setattr(resume.frozen, "strict_check", fake_strict)
    monkeypatch.setattr(resume, "verify_strict_projection", fake_projection)
    monkeypatch.setattr(resume, "verify_record", fake_record)
    monkeypatch.setattr(resume.frozen, "session_environment", lambda *_: {})
    monkeypatch.setattr(resume, "trial_argv", lambda *_: ["fake-frozen-cli"])
    original_phase = ledger.phase

    def crash_phase(tid_arg, phase, details=None):
        if boundary == "result_written" and phase == "RESULT_WRITTEN":
            raise InjectedCrash
        return original_phase(tid_arg, phase, details)

    monkeypatch.setattr(ledger, "phase", crash_phase)
    with pytest.raises(InjectedCrash):
        resume.run_new(args, ledger, tid, {})
    crashed = True
    # The same scheduler recovery function used in production must reconcile
    # the durable files. Reopening the ledger simulates a new process.
    recovered = resume.Ledger(root, manifest)
    assert resume.recover_running(args, recovered, tid) == expected
    assert counts["model"] == model_calls and counts["strict"] == strict_calls
    assert recovered.states[tid]["state"] == expected
    assert sum(e["kind"] == "RESERVED" for e in recovered.records) == 1
    if expected == "COMPLETED_SUCCESS":
        assert resume.read(folder / "trial-record.json")["success"]
    assert all(x["state"] == "PENDING" for key, x in recovered.states.items() if key != tid)


def test_watchdog_never_resamples_and_next_trial_is_pending(tmp_path, manifest, monkeypatch):
    output = tmp_path / "output"
    (output / "trials").mkdir(parents=True)
    args = SimpleNamespace(output=output, repository=tmp_path, evidence_root=tmp_path)
    ledger = resume.Ledger(output / "scheduler-v2/events", manifest)
    tid, nxt = list(ledger.trials)[:2]
    calls = {"model": 0, "strict": 0, "cleanup": 0}

    def fake_invoke(*unused):
        calls["model"] += 1
        return {"exit_code": -15, "timed_out": True, "wall_seconds": 600.008}

    def forbidden(*unused, **kwargs):
        calls["strict"] += 1
        raise AssertionError("timed out trial must never be evaluated again")

    def cleanup(*unused):
        calls["cleanup"] += 1
        return {"verified_fake_cleanup": True}

    monkeypatch.setattr(resume, "trial_argv", lambda *_: ["fake-frozen-cli"])
    monkeypatch.setattr(resume.frozen, "invoke", fake_invoke)
    monkeypatch.setattr(resume, "finish_agent", forbidden)
    monkeypatch.setattr(resume, "idle_proof", cleanup)
    assert resume.run_new(args, ledger, tid, {}) == "OUTCOME_UNKNOWN"
    folder = output / "trials" / tid
    record = resume.read(folder / "timeout-record.json")
    assert record["strict_fix_within_budget"] is False
    assert record["candidate_correctness"] == "UNKNOWN"
    assert record["retry_same_replicate"] is False
    assert record["elapsed_seconds"] == 600.008
    assert resume.reconcile_watchdogs(args, ledger)
    fresh = resume.Ledger(ledger.root, manifest)
    assert fresh.states[nxt]["state"] == "PENDING"
    assert resume.reconcile_watchdogs(args, fresh)
    with pytest.raises(AssertionError):
        resume.run_new(args, fresh, tid, {})
    assert calls == {"model": 1, "strict": 0, "cleanup": 2}
    metrics = resume.budget_metrics(args, fresh, [tid, nxt])
    assert metrics["started"] == metrics["timeouts"] == 1
    assert metrics["primary_success_rate"] == 0
    assert metrics["secondary_completed_success_rate"] is None
    assert metrics["agent_wall_observed_seconds"] == 600.008
    assert metrics["total_tokens"] is None


def test_historical_watchdog_adoption_preserves_terminal_event(tmp_path, manifest, monkeypatch):
    args = SimpleNamespace(output=tmp_path)
    ledger = new_ledger(tmp_path, manifest)
    tid = next(iter(ledger.trials))
    folder = tmp_path / "trials" / tid
    folder.mkdir(parents=True)
    ledger.append("RESERVED", tid, state="RUNNING", phase="RESERVED", attempt="a" * 32)
    ledger.phase(tid, "AGENT_STARTED")
    resume.atomic_json(
        folder / "agent-execution.json",
        {"exit_code": -15, "timed_out": True, "wall_seconds": 600.008},
    )
    ledger.phase(tid, "AGENT_RECORDED")
    terminal = ledger.terminal(tid, "OUTCOME_UNKNOWN", {"watchdog": True, "no_resample": True})
    old = [p.read_bytes() for p in sorted(ledger.root.glob("event-*.json"))]
    monkeypatch.setattr(resume, "idle_proof", lambda *_: {"verified_fake_cleanup": True})
    assert resume.reconcile_watchdogs(args, ledger)
    assert ledger.states[tid]["details"] == terminal["details"]
    assert [p.read_bytes() for p in sorted(ledger.root.glob("event-*.json"))[: len(old)]] == old
    assert (folder / "timeout-record.json").is_file()


def test_unknown_without_watchdog_still_blocks(tmp_path, manifest):
    args = SimpleNamespace(output=tmp_path)
    ledger = new_ledger(tmp_path, manifest)
    tid = next(iter(ledger.trials))
    ledger.append("RESERVED", tid, state="RUNNING", phase="RESERVED", attempt="a" * 32)
    ledger.phase(tid, "AGENT_STARTED")
    ledger.terminal(tid, "OUTCOME_UNKNOWN", {"no_resample": True})
    assert not resume.reconcile_watchdogs(args, ledger)


def test_timeout_recovery_after_execution_write_never_finalizes(tmp_path, manifest, monkeypatch):
    args = SimpleNamespace(output=tmp_path)
    ledger = new_ledger(tmp_path, manifest)
    tid = next(iter(ledger.trials))
    folder = tmp_path / "trials" / tid
    folder.mkdir(parents=True)
    ledger.append("RESERVED", tid, state="RUNNING", phase="RESERVED", attempt="a" * 32)
    ledger.phase(tid, "AGENT_STARTED")
    resume.atomic_json(
        folder / "agent-execution.json",
        {"exit_code": -15, "timed_out": True, "wall_seconds": 600.008},
    )
    monkeypatch.setattr(resume, "finish_agent", lambda *_: pytest.fail("no additional action"))
    assert resume.recover_running(args, ledger, tid) == "OUTCOME_UNKNOWN"
    assert ledger.states[tid]["details"]["watchdog_stage"] == "agent"
