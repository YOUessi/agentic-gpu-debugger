import pytest


def _configure_corpus_family(tmp_path, monkeypatch):
    from gpu_agent.benchmark.ledger import CorpusFamily

    family = CorpusFamily.provision(
        tmp_path.parent / f"{tmp_path.name}-controller",
        public_store=tmp_path / "runs",
        evaluator_store=tmp_path / "evaluator" / "runs",
        repository=tmp_path,
    )
    monkeypatch.setenv("GPU_AGENT_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("GPU_AGENT_EVALUATOR_ROOT", str(tmp_path / "evaluator"))
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    return family


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
    fingerprints = []
    monkeypatch.setattr(
        "gpu_agent.service.runtime_code_fingerprint",
        lambda repo: fingerprints.append(repo) or "d" * 64,
    )
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
    # Evaluation binds the executing code, not only the git snapshot.
    assert fingerprints == [tmp_path] and service.binding.runtime_code_hash == "d" * 64


def test_release_evaluation_service_selects_exact_family_visibility(tmp_path, monkeypatch):
    from gpu_agent.contracts import RepositorySnapshot
    from gpu_agent.environment import load_toolchain_lock
    from gpu_agent.execution.isolated import LOCK_PATH
    from gpu_agent.service import ApplicationService

    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    evaluator_root = tmp_path / "evaluator-root"
    evaluator_root.mkdir(mode=0o700)
    from gpu_agent.benchmark.ledger import CorpusFamily

    family = CorpusFamily.provision(
        tmp_path / "controller",
        public_store=tmp_path / "public",
        evaluator_store=evaluator_root / "runs",
        repository=repository,
    )
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    monkeypatch.setenv("GPU_AGENT_RUN_ROOT", str(tmp_path / "public"))
    monkeypatch.setenv("GPU_AGENT_EVALUATOR_ROOT", str(evaluator_root))
    snapshot = RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True)
    monkeypatch.setattr("gpu_agent.service.capture_repository_snapshot", lambda *_a, **_k: snapshot)
    monkeypatch.setattr("gpu_agent.service.runtime_code_fingerprint", lambda _repo: "d" * 64)
    monkeypatch.setattr(
        "gpu_agent.service.load_toolchain_lock", lambda _path: load_toolchain_lock(LOCK_PATH)
    )

    public = ApplicationService.for_release(
        repository,
        purpose="evaluation",
        expected_commit=snapshot.commit,
        prompt_version="diagnosis-v1",
        model_config_hash="c" * 64,
        require_corpus_family=True,
        workflow_visibility="public",
    )
    evaluator = ApplicationService.for_release(
        repository,
        purpose="evaluation",
        expected_commit=snapshot.commit,
        prompt_version="diagnosis-v1",
        model_config_hash="c" * 64,
        require_corpus_family=True,
        workflow_visibility="evaluator",
    )

    assert public.store.identity == family.corpus_store("public").identity
    assert public.evaluator_store.identity == family.corpus_store("evaluator").identity
    assert evaluator.store.identity == evaluator.evaluator_store.identity
    assert evaluator.store.identity == family.corpus_store("evaluator").identity


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
    _configure_corpus_family(tmp_path, monkeypatch)
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
    family = _configure_corpus_family(tmp_path, monkeypatch)
    service = ApplicationService.for_release(tmp_path, purpose="corpus_validation")
    assert service.binding is not None
    assert service.binding.case_registry_hash == hashlib.sha256(registry).hexdigest()
    assert service.binding.corpus_ledger_namespace_hash == family.namespace_hash


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
        original.evaluator_store,
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


def test_evaluator_diagnosis_keeps_all_artifacts_in_evaluator_store(
    evaluator_oob_service, tmp_path
):
    from gpu_agent.contracts import RunStatus

    service, _, source, public = evaluator_oob_service
    run = service.diagnose(source, mode="E")
    assert run.status == RunStatus.COMPLETED
    assert service.store.visibility == "evaluator"
    for path in service.store.root.rglob("manifest.json"):
        manifest = service.store.load(path.parent.name)
        assert all(ref.visibility == "evaluator" for ref in manifest.artifact_refs)
    assert not any(
        b"PRIVATE-SOURCE-CANARY" in path.read_bytes()
        for path in public.root.rglob("*")
        if path.is_file()
    )


def test_evaluator_candidate_and_verification_remain_parent_scoped(evaluator_oob_service):
    from gpu_agent.verification.models import VerificationVerdict

    service, _, source, _ = evaluator_oob_service
    diagnosis = service.diagnose(source, mode="E")
    candidate = service.candidates(diagnosis.id)
    assert len(candidate) == 1
    result = service.verify(diagnosis.id, candidate[0])
    assert isinstance(result.verdict, VerificationVerdict)
    assert all(
        ref.visibility == "evaluator"
        for run in (service.store.load(diagnosis.id), service.store.load(candidate[0]))
        for ref in run.artifact_refs
    )


def test_runtime_code_fingerprint_rejects_code_imported_from_another_checkout(tmp_path):
    from gpu_agent.provenance import runtime_code_fingerprint

    with pytest.raises(ValueError, match="not the captured repository"):
        runtime_code_fingerprint(tmp_path)


def test_runtime_code_fingerprint_binds_this_checkout_and_detects_drift(monkeypatch):
    from pathlib import Path

    import gpu_agent
    from gpu_agent.provenance import runtime_code_fingerprint

    repository = Path(gpu_agent.__file__).resolve().parents[2]
    first = runtime_code_fingerprint(repository)
    assert first == runtime_code_fingerprint(repository)
    original = Path.read_bytes

    def drifted(path):
        content = original(path)
        return content + b"# patched" if path.name == "policy.py" else content

    monkeypatch.setattr("gpu_agent.provenance.read_regular", lambda path, _limit: drifted(path))
    assert runtime_code_fingerprint(repository) != first


def test_evaluation_unit_refuses_code_that_drifted_after_binding(oob_service, tmp_path):
    from gpu_agent.contracts import RepositorySnapshot, RunBinding

    service = oob_service[0]
    service._binding = RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="evaluation",
        runtime_code_hash="0" * 64,
    )
    service._repository_root = __import__("pathlib").Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="executing code changed"):
        service._attest_runtime_code()
