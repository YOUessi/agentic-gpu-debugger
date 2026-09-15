def test_blind_view_excludes_mode_model_usage_and_trace():
    from gpu_agent.benchmark.evaluation import EvaluationRecord

    record = EvaluationRecord(
        record_id="blind-1",
        case_id="case_0001",
        template_id="index",
        mode="E",
        repeat=0,
        input_hash="a" * 64,
        evidence_hash="b" * 64,
        executed_checks={"memcheck": "CLEAN"},
        status="COMPLETED",
        diagnosis={"root_cause": "guard missing"},
        usage={"tokens": 12},
        latency_ms=5,
        cost_usd=0.01,
    )
    blind = record.blind()
    assert set(blind) == {"blind_id", "diagnosis", "evidence_hash"}
    assert "E" not in str(blind) and "tokens" not in str(blind)
