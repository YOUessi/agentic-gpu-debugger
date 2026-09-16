from schedule_authority_support import schedule_client_for_test


def test_evaluation_is_repeated_randomized_serial_and_native(native_evaluation_executor):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None

    class ObservedExecutor:
        calls = []

        def validate_scheduled_record(self, record, item, attempt):
            return executor.validate_scheduled_record(record, item, attempt)

        def execute_scheduled(self, run_id, ordinal):
            record = executor.execute_scheduled(run_id, ordinal)
            self.calls.append((record.case_id, record.mode, record.repeat))
            return record

    observed = ObservedExecutor()
    native_execute = executor.execute_scheduled
    observed.execute_scheduled = lambda run_id, ordinal: (
        observed.calls.append(
            (
                (record := native_execute(run_id, ordinal)).case_id,
                record.mode,
                record.repeat,
            )
        )
        or record
    )
    executor.execute_scheduled = observed.execute_scheduled
    executor._execute = lambda *args: (_ for _ in ()).throw(
        AssertionError("instance internal monkeypatch must not run")
    )
    executor._schedule_verifier.verify = lambda *args: (_ for _ in ()).throw(
        AssertionError("instance schedule verifier monkeypatch must not run")
    )
    executor.validate_scheduled_record = lambda *args: (_ for _ in ()).throw(
        AssertionError("instance validator monkeypatch must not run")
    )
    runner = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
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
    assert observed.calls == []
    assert "evaluation/schedule-receipt.json" in {
        ref.name for ref in executor.service.store.load(result.run_id).artifact_refs
    }
    assert [(record.case_id, record.mode, record.repeat) for record in result.records] == [
        ("case_0100", "D", 1),
        ("case_0100", "D", 0),
        ("case_0100", "D", 2),
    ]


def test_missing_cost_cap_stops_before_external_execution(native_evaluation_executor):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None

    runner = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=None,
        max_unit_cost_usd=None,
    )
    result = runner.run("E", "development", 3)
    assert result.records == [] and result.stopped_reason == "COST_CAP_REQUIRED"
