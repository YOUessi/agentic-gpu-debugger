import pytest


def _binding():
    from gpu_agent.contracts import RepositorySnapshot, RunBinding

    return RunBinding(
        repository=RepositorySnapshot(commit="1" * 40, tracked_tree_hash="2" * 64, clean=True),
        purpose="evaluation",
        toolchain_lock_hash="3" * 64,
        prompt_version="diagnosis-v1",
        model_config_hash="4" * 64,
    )


def test_configured_service_rejects_caller_constructed_binding(tmp_path, monkeypatch):
    from gpu_agent.service import ApplicationService

    monkeypatch.setenv("GPU_AGENT_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("GPU_AGENT_EVALUATOR_ROOT", str(tmp_path / "evaluator"))
    try:
        ApplicationService.configured(binding=_binding())  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("configured accepted a caller-constructed release binding")


def test_release_service_factory_captures_repository_and_lock_internally(tmp_path, monkeypatch):
    from gpu_agent.contracts import RepositorySnapshot
    from gpu_agent.environment import load_toolchain_lock
    from gpu_agent.execution.isolated import LOCK_PATH
    from gpu_agent.service import ApplicationService

    monkeypatch.setenv("GPU_AGENT_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("GPU_AGENT_EVALUATOR_ROOT", str(tmp_path / "evaluator"))
    captured = RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True)
    calls = []

    def capture(repo, *, expected_commit=None):
        calls.append((repo, expected_commit))
        return captured

    monkeypatch.setattr("gpu_agent.service.capture_repository_snapshot", capture)
    locked = load_toolchain_lock(LOCK_PATH)
    lock_calls = []

    def load_lock(path):
        lock_calls.append(path)
        return locked

    monkeypatch.setattr("gpu_agent.service.load_toolchain_lock", load_lock)
    service = ApplicationService.for_release(
        tmp_path,
        purpose="evaluation",
        expected_commit="a" * 40,
        prompt_version="diagnosis-v1",
        model_config_hash="c" * 64,
    )
    assert calls == [(tmp_path, "a" * 40), (tmp_path, "a" * 40)]
    assert lock_calls == [tmp_path / "containers/toolchain.lock.json"]
    assert service.binding.repository == captured
    assert service.binding.toolchain_lock_hash == locked.lock_hash


def test_release_service_rejects_repository_change_while_loading_lock(tmp_path, monkeypatch):
    from gpu_agent.contracts import RepositorySnapshot
    from gpu_agent.environment import load_toolchain_lock
    from gpu_agent.execution.isolated import LOCK_PATH
    from gpu_agent.service import ApplicationService

    before = RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True)
    after = before.model_copy(update={"tracked_tree_hash": "c" * 64})
    snapshots = iter([before, after])
    monkeypatch.setattr(
        "gpu_agent.service.capture_repository_snapshot",
        lambda *_args, **_kwargs: next(snapshots),
    )
    monkeypatch.setattr(
        "gpu_agent.service.load_toolchain_lock", lambda _path: load_toolchain_lock(LOCK_PATH)
    )
    monkeypatch.setattr("gpu_agent.service.read_regular", lambda *_args: b"registry")
    with pytest.raises(ValueError, match="changed"):
        ApplicationService.for_release(tmp_path, purpose="corpus_validation")


def test_corpus_release_binding_includes_bounded_registry_hash(tmp_path, monkeypatch):
    import hashlib

    from gpu_agent.contracts import RepositorySnapshot
    from gpu_agent.environment import load_toolchain_lock
    from gpu_agent.execution.isolated import LOCK_PATH
    from gpu_agent.service import ApplicationService

    snapshot = RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True)
    registry = b'{"schema_version":1,"cases":[]}'
    monkeypatch.setenv("GPU_AGENT_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("GPU_AGENT_EVALUATOR_ROOT", str(tmp_path / "evaluator"))
    monkeypatch.setattr(
        "gpu_agent.service.capture_repository_snapshot", lambda *_args, **_kwargs: snapshot
    )
    monkeypatch.setattr(
        "gpu_agent.service.load_toolchain_lock", lambda _path: load_toolchain_lock(LOCK_PATH)
    )
    monkeypatch.setattr("gpu_agent.service.read_regular", lambda *_args: registry)
    service = ApplicationService.for_release(tmp_path, purpose="corpus_validation")
    assert service.binding is not None
    assert service.binding.case_registry_hash == hashlib.sha256(registry).hexdigest()


def test_diagnosis_and_registered_children_inherit_service_binding(oob_service):
    from gpu_agent.environment import load_toolchain_lock
    from gpu_agent.execution.isolated import LOCK_PATH
    from gpu_agent.service import ApplicationService

    original, provider, source = oob_service
    binding = _binding().model_copy(
        update={"toolchain_lock_hash": load_toolchain_lock(LOCK_PATH).lock_hash}
    )
    service = ApplicationService(
        original.store,
        original.evaluator_root,
        provider=provider,
        backend_factory=original._backend_factory,
        knowledge=original.knowledge,
        knowledge_version=original.knowledge_version,
        _binding=binding,
    )
    run = service.diagnose(source)
    assert run.binding == binding
    candidate_id = service.candidates(run.id)[0]
    assert service.store.load(candidate_id).binding == binding
