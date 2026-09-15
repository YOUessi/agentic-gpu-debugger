def _binding():
    from gpu_agent.contracts import RepositorySnapshot, RunBinding

    return RunBinding(
        repository=RepositorySnapshot(commit="1" * 40, tracked_tree_hash="2" * 64, clean=True),
        purpose="evaluation",
        toolchain_lock_hash="3" * 64,
        prompt_version="diagnosis-v1",
        model_config_hash="4" * 64,
    )


def test_configured_service_accepts_optional_pre_execution_binding(tmp_path, monkeypatch):
    from gpu_agent.service import ApplicationService

    monkeypatch.setenv("GPU_AGENT_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("GPU_AGENT_EVALUATOR_ROOT", str(tmp_path / "evaluator"))
    binding = _binding()
    service = ApplicationService.configured(binding=binding)
    assert service.binding == binding


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
        binding=binding,
    )
    run = service.diagnose(source)
    assert run.binding == binding
    candidate_id = service.candidates(run.id)[0]
    assert service.store.load(candidate_id).binding == binding
