def test_evaluation_is_repeated_randomized_serial_and_cost_capped():
    from gpu_agent.benchmark.evaluation import EvaluationRecord, EvaluationRunner

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
        {"case_0001": "index", "case_0002": "race"}, execute, max_cost_usd=1.0
    )
    result = runner.run("all", "development", 3)
    assert len(result.records) == 30 and result.stopped_reason is None
    assert len(calls) == len(set(calls))


def test_missing_cost_cap_stops_before_external_execution():
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    runner = EvaluationRunner(
        {"case_0001": "index"},
        lambda *args: (_ for _ in ()).throw(AssertionError()),
        max_cost_usd=None,
    )
    result = runner.run("E", "holdout", 3)
    assert result.records == [] and result.stopped_reason == "COST_CAP_REQUIRED"
