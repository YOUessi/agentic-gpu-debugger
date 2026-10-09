"""Restart recovery preserves immutable evidence and terminates controller state."""


def test_interrupted_run_is_discovered_and_failed_without_losing_artifacts(tmp_path):
    from gpu_agent.store import RunStore

    root = tmp_path / "runs"
    original = RunStore(root)
    run = original.create_run("recovery")
    original.transition(run.id, "RUNNING", "DIAGNOSING")
    evidence = original.put(run.id, "evidence/partial.log", b"retained", "public")

    restarted = RunStore(root)
    assert [item.id for item in restarted.recoverable_runs()] == [run.id]
    failed = restarted.fail_interrupted(run.id)
    assert failed.status == "FAILED"
    assert restarted.read(evidence) == b"retained"
    recovery = next(ref for ref in failed.artifact_refs if ref.name == "recovery/interruption.json")
    assert b"CONTROLLER_RESTARTED" in restarted.read(recovery)
    assert restarted.recoverable_runs() == []


def test_terminal_or_invalid_recovery_is_rejected(tmp_path):
    import pytest

    from gpu_agent.store import RunStore

    store = RunStore(tmp_path / "runs")
    run = store.create_run("recovery")
    store.transition(run.id, "RUNNING", "FINALIZING")
    store.transition(run.id, "COMPLETED", None)
    with pytest.raises(ValueError, match="not recoverable"):
        store.fail_interrupted(run.id)
    with pytest.raises(ValueError, match="invalid recovery reason"):
        store.fail_interrupted(run.id, 'BAD"JSON')
