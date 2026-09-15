def test_evaluation_is_repeated_randomized_serial_and_cost_capped(tmp_path):
    from gpu_agent.benchmark.evaluation import EvaluationRecord, EvaluationRunner
    from gpu_agent.store import RunStore

    calls = []

    def execute(case, template, mode, repeat):
        calls.append((case, mode, repeat))
        return EvaluationRecord(
            record_id=f"{case}-{mode}-{repeat}",
            case_id=case,
            template_id=template,
            mode=mode,
            repeat=repeat,
            input_hash="a" * 64,
            evidence_hash="b" * 64,
            executed_checks={"memcheck": "CLEAN"},
            status="COMPLETED",
            diagnosis={},
            verdict="VERIFIED_FIXED",
            latency_ms=1,
            cost_usd=0.01,
        )

    runner = EvaluationRunner(
        RunStore(tmp_path / "runs"),
        {"case_0001": "index", "case_0002": "race"},
        execute,
        commit="c" * 40,
        prompt_version="v2",
        toolchain_hash="d" * 64,
        model_config_hash="e" * 64,
        max_cost_usd=1.0,
        max_unit_cost_usd=0.05,
    )
    result = runner.run("all", "development", 3)
    assert len(result.records) == 30 and result.stopped_reason is None
    assert calls == [
        ("case_0001", "B", 1),
        ("case_0002", "D", 0),
        ("case_0002", "C", 0),
        ("case_0002", "E", 2),
        ("case_0001", "E", 2),
        ("case_0002", "C", 2),
        ("case_0002", "A", 1),
        ("case_0002", "E", 1),
        ("case_0001", "D", 1),
        ("case_0002", "D", 2),
        ("case_0002", "A", 2),
        ("case_0001", "C", 0),
        ("case_0001", "B", 0),
        ("case_0002", "B", 1),
        ("case_0001", "B", 2),
        ("case_0002", "A", 0),
        ("case_0001", "E", 1),
        ("case_0002", "C", 1),
        ("case_0001", "E", 0),
        ("case_0002", "E", 0),
        ("case_0001", "C", 1),
        ("case_0001", "A", 0),
        ("case_0001", "A", 1),
        ("case_0002", "B", 2),
        ("case_0002", "B", 0),
        ("case_0001", "D", 2),
        ("case_0001", "D", 0),
        ("case_0001", "C", 2),
        ("case_0002", "D", 1),
        ("case_0001", "A", 2),
    ]


def test_missing_cost_cap_stops_before_external_execution(tmp_path):
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.store import RunStore

    runner = EvaluationRunner(
        RunStore(tmp_path / "runs"),
        {"case_0001": "index"},
        lambda *args: (_ for _ in ()).throw(AssertionError()),
        commit="c" * 40,
        prompt_version="v2",
        toolchain_hash="d" * 64,
        model_config_hash="e" * 64,
        max_cost_usd=None,
        max_unit_cost_usd=None,
    )
    result = runner.run("E", "holdout", 3)
    assert result.records == [] and result.stopped_reason == "COST_CAP_REQUIRED"
