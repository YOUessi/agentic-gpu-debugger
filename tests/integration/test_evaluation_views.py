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


def test_public_evaluation_artifacts_exclude_hidden_truth_fields(tmp_path):
    from gpu_agent.benchmark.evaluation import (
        EvaluationManifest,
        EvaluationRecord,
        EvaluationRunner,
        PublicEvaluationRecord,
    )
    from gpu_agent.benchmark.metrics import Score
    from gpu_agent.store import RunStore

    store = RunStore(tmp_path / "runs")

    def execute(case, template, mode, repeat):
        return EvaluationRecord(
            record_id=f"{case}-{repeat}",
            case_id=case,
            template_id=template,
            mode=mode,
            repeat=repeat,
            input_hash="a" * 64,
            evidence_hash="b" * 64,
            executed_checks={"memcheck": "CLEAN"},
            status="COMPLETED",
            diagnosis={},
            latency_ms=1,
            cost_usd=0.01,
            should_be_inconclusive=True,
            score=Score(
                family_correct=True,
                root_cause_correct=True,
                location_correct=True,
                inconclusive_correct=True,
            ),
        )

    result = EvaluationRunner(
        store,
        {"case_0001": "index"},
        execute,
        commit="c" * 40,
        prompt_version="v2",
        toolchain_hash="d" * 64,
        model_config_hash="e" * 64,
        max_cost_usd=1.0,
        max_unit_cost_usd=0.1,
    ).run("E", "development", 3)

    artifacts = {ref.name: store.read(ref) for ref in store.load(result.run_id).artifact_refs}
    public_bytes = artifacts["evaluation/records/0.json"] + artifacts["evaluation/manifest.json"]
    assert b"should_be_inconclusive" not in public_bytes and b'"score"' not in public_bytes
    record = PublicEvaluationRecord.model_validate_json(artifacts["evaluation/records/0.json"])
    manifest = EvaluationManifest.model_validate_json(artifacts["evaluation/manifest.json"])
    assert not hasattr(record, "should_be_inconclusive") and not hasattr(record, "score")
    assert not hasattr(manifest.records[0], "should_be_inconclusive")
    assert not hasattr(result.records[0], "should_be_inconclusive")
