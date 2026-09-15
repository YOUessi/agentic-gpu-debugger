def test_evaluation_is_repeated_randomized_serial_and_native(native_evaluation_executor):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None

    class ObservedExecutor:
        calls = []

        def execute(self, case, template, mode, repeat):
            return executor.execute(case, template, mode, repeat)

        def execute_scheduled(self, item, attempt):
            self.calls.append((item.case_id, item.mode, item.repeat))
            return executor.execute_scheduled(item, attempt)

    observed = ObservedExecutor()
    runner = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        observed.execute,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
    )
    result = runner.run("D", "development", 3)
    assert len(result.records) == 3 and result.stopped_reason is None
    assert observed.calls == [
        ("case_0100", "D", 1),
        ("case_0100", "D", 0),
        ("case_0100", "D", 2),
    ]


def test_missing_cost_cap_stops_before_external_execution(tmp_path):
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.contracts import RepositorySnapshot, RunBinding
    from gpu_agent.store import RunStore

    runner = EvaluationRunner(
        RunStore(tmp_path / "runs"),
        {"case_0001": "index"},
        lambda *args: (_ for _ in ()).throw(AssertionError()),
        commit="c" * 40,
        prompt_version="v2",
        toolchain_hash="d" * 64,
        model_config_hash="e" * 64,
        binding=RunBinding(
            repository=RepositorySnapshot(commit="c" * 40, tracked_tree_hash="f" * 64, clean=True),
            purpose="evaluation",
            toolchain_lock_hash="d" * 64,
            prompt_version="v2",
            model_config_hash="e" * 64,
        ),
        max_cost_usd=None,
        max_unit_cost_usd=None,
    )
    result = runner.run("E", "holdout", 3)
    assert result.records == [] and result.stopped_reason == "COST_CAP_REQUIRED"
