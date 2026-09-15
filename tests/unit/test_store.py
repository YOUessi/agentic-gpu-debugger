import pytest


def _binding(commit="1" * 40, purpose="evaluation", toolchain_hash="2" * 64):
    from gpu_agent.contracts import RepositorySnapshot, RunBinding

    return RunBinding(
        repository=RepositorySnapshot(commit=commit, tracked_tree_hash="3" * 64, clean=True),
        purpose=purpose,
        toolchain_lock_hash=toolchain_hash,
        prompt_version="diagnosis-v1",
        model_config_hash="4" * 64,
    )


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


def test_release_binding_is_persisted_at_creation_and_cannot_be_rebound(store):
    from pydantic import ValidationError

    binding = _binding()
    run = store.create_run("diagnosis", binding=binding)
    assert store.load(run.id).binding == binding
    with pytest.raises(ValidationError):
        run.binding = _binding(commit="5" * 40)  # type: ignore[misc]


def test_bound_child_inherits_exact_parent_binding(store):
    binding = _binding()
    parent = store.create_run("diagnosis", binding=binding)
    child = store.create_run("candidate", parent.id)
    assert child.binding == binding
    assert store.load(child.id).binding == store.load(parent.id).binding


def test_child_cannot_replace_or_add_a_parent_binding(store):
    parent = store.create_run("diagnosis", binding=_binding())
    with pytest.raises(ValueError, match="binding"):
        store.create_run("candidate", parent.id, binding=_binding(commit="5" * 40))

    old_parent = store.create_run("diagnosis")
    with pytest.raises(ValueError, match="binding"):
        store.create_run("candidate", old_parent.id, binding=_binding())


def test_child_inherits_external_origin_and_cannot_replace_or_add_it(store):
    from gpu_agent.contracts import ExternalRunOrigin

    origin = ExternalRunOrigin(run_id="6" * 32, visibility="evaluator")
    parent = store.create_run("diagnosis", external_origin=origin)
    child = store.create_run("candidate", parent.id)
    assert child.external_origin == origin

    replacement = ExternalRunOrigin(run_id="7" * 32, visibility="evaluator")
    with pytest.raises(ValueError, match="origin"):
        store.create_run("candidate", parent.id, external_origin=replacement)

    local_parent = store.create_run("diagnosis")
    with pytest.raises(ValueError, match="origin"):
        store.create_run("candidate", local_parent.id, external_origin=origin)


def test_legacy_unbound_manifest_loads_but_is_not_release_bound(store):
    run = store.create_run("diagnosis")
    assert store.load(run.id).binding is None


def test_cross_store_child_origin_is_immutable_and_cannot_mix_bindings(store, tmp_path):
    from pydantic import ValidationError

    from gpu_agent.contracts import ExternalRunOrigin
    from gpu_agent.store import RunStore

    origin = store.create_run("diagnosis", binding=_binding())
    evaluator = RunStore(tmp_path / "evaluator", visibility="evaluator")
    relation = ExternalRunOrigin(run_id=origin.id, visibility="public")
    audit = evaluator.create_run(
        "verification_audit", binding=origin.binding, external_origin=relation
    )
    assert evaluator.load(audit.id).external_origin == relation
    with pytest.raises(ValidationError):
        audit.external_origin = None  # type: ignore[misc]
    with pytest.raises(ValueError, match="external origin"):
        evaluator.create_run(
            "verification_input",
            parent_run_id=audit.id,
            binding=_binding(commit="5" * 40),
            external_origin=relation,
        )
    with pytest.raises(ValueError, match="visibility"):
        evaluator.create_run(
            "verification_input",
            binding=origin.binding,
            external_origin=ExternalRunOrigin(run_id=origin.id, visibility="evaluator"),
        )
