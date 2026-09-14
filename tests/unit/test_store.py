import pytest


def test_terminal_state_preserves_last_successful_phase(store):
    run = store.create_run("diagnosis")
    store.transition(run.id, "RUNNING", "PREPARING")
    store.transition(run.id, "RUNNING", "COMPILING")
    done = store.transition(run.id, "FAILED", None)
    assert done.current_phase is None
    assert done.last_completed_phase == "PREPARING"
    assert store.load(run.id).status == "FAILED"
    assert [event.status for event in done.events] == ["QUEUED", "RUNNING", "RUNNING", "FAILED"]
    with pytest.raises(ValueError):
        store.transition(run.id, "RUNNING", "COMPILING")


@pytest.mark.parametrize("status,phase", [("RUNNING", None), ("COMPLETED", "COMPILING")])
def test_invalid_state_phase_pair_rejected(store, status, phase):
    run = store.create_run("diagnosis")
    with pytest.raises(ValueError):
        store.transition(run.id, status, phase)
    assert store.load(run.id).status == "QUEUED"


def test_artifact_round_trip_is_registered_immutable_and_hash_checked(store):
    run = store.create_run("diagnosis")
    ref = store.put(run.id, "logs/build.stdout", b"compiler evidence\xff", "public")
    assert store.read(ref) == b"compiler evidence\xff"
    assert ref in store.load(run.id).artifact_refs
    changed = ref.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError):
        store.read(changed)
    path = store.root / ref.relative_path
    path.chmod(0o600)
    path.write_bytes(b"altered")
    with pytest.raises(ValueError):
        store.read(ref)


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "a/../../escape", "a\\b"])
def test_path_traversal_is_rejected(store, name):
    run = store.create_run("diagnosis")
    with pytest.raises(ValueError):
        store.put(run.id, name, b"no", "public")
    with pytest.raises(ValueError):
        store.load(name)


def test_symlink_artifact_cannot_read_outside_store(store, tmp_path):
    run = store.create_run("diagnosis")
    ref = store.put(run.id, "data", b"public", "public")
    outside = tmp_path / "private"
    outside.write_bytes(b"private")
    path = store.root / ref.relative_path
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError):
        store.read(ref)


def test_symlink_run_directory_is_rejected(store, tmp_path):
    run = store.create_run("diagnosis")
    path = store.root / run.id
    moved = tmp_path / "moved"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError):
        store.load(run.id)


def test_failed_manifest_replace_preserves_previous_record(store, monkeypatch):
    import gpu_agent.store as module

    run = store.create_run("diagnosis")
    before = store.load(run.id)

    def fail_replace(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        store.put(run.id, "lost", b"not registered", "public")
    assert store.load(run.id) == before
    assert not list((store.root / run.id / "artifacts").iterdir())


def test_private_artifacts_require_separate_store_root(store):
    run = store.create_run("diagnosis")
    with pytest.raises(ValueError):
        store.put(run.id, "private", b"secret", "evaluator")


def test_independent_store_instances_do_not_lose_artifact_updates(store):
    from concurrent.futures import ThreadPoolExecutor

    from gpu_agent.store import RunStore

    run = store.create_run("diagnosis")
    other = RunStore(store.root)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(s.put, run.id, str(i), bytes([i]), "public")
            for i, s in enumerate([store, other])
        ]
        refs = [future.result() for future in futures]
    assert set(ref.id for ref in store.load(run.id).artifact_refs) == {ref.id for ref in refs}


def test_special_lock_file_is_rejected_without_writing_outside(store):
    import os

    run = store.create_run("diagnosis")
    os.mkfifo(store.root / run.id / ".lock")
    with pytest.raises(ValueError):
        store.put(run.id, "data", b"data", "public")


def test_new_run_directory_entry_is_durable_before_return(store, monkeypatch):
    import os
    from pathlib import Path

    import gpu_agent.store as module

    original = os.fsync
    synced = []

    def record_fsync(fd):
        synced.append(Path(os.readlink(f"/proc/self/fd/{fd}")))
        original(fd)

    monkeypatch.setattr(module.os, "fsync", record_fsync)
    run = store.create_run("diagnosis")
    assert store.root in synced
    assert synced.index(store.root / run.id) < synced.index(store.root)
