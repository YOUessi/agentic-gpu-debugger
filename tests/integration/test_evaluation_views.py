def _lineage():
    from gpu_agent.benchmark.evaluation import EvaluationLineage

    return EvaluationLineage(
        diagnosis_run_id="a" * 32,
        diagnosis_hash="b" * 64,
        evidence_hash="c" * 64,
        provider_invocation_hashes=[],
    )


def test_blind_view_excludes_mode_model_usage_and_trace():
    from gpu_agent.benchmark.evaluation import EvaluationRecord

    record = EvaluationRecord(
        record_id="blind-1",
        lineage=_lineage(),
        case_id="case_0001",
        template_id="index",
        mode="E",
        repeat=0,
        input_hash="a" * 64,
        evidence_hash="b" * 64,
        executed_checks={"memcheck": "CLEAN"},
        status="COMPLETED",
        diagnosis={"diagnostic_outcome": "DIAGNOSED", "root_cause": "guard missing"},
        usage={"tokens": 12},
        latency_ms=5,
        cost_usd=0.01,
    )
    blind = record.blind()
    assert set(blind) == {"blind_id", "diagnosis", "evidence_hash"}
    assert "tokens" not in str(blind)
    assert blind["diagnosis"]["root_cause"] == "guard missing"


def test_public_evaluation_artifacts_exclude_hidden_truth_fields(native_evaluation_executor):
    from gpu_agent.benchmark.evaluation import (
        EvaluationManifest,
        EvaluationRunner,
        PublicEvaluationRecord,
    )
    from gpu_agent.benchmark.metrics import Score

    executor = native_evaluation_executor
    store = executor.service.store
    binding = executor.service.binding
    assert binding is not None

    class PrivateScoringProjection:
        def execute(self, case, template, mode, repeat):
            return executor.execute(case, template, mode, repeat)

        def execute_scheduled(self, item, attempt):
            record = executor.execute_scheduled(item, attempt)
            return record.model_copy(
                update={
                    "executed_checks": {
                        **record.executed_checks,
                        "verification/private_oracle": "PRIVATE_CHECK_CANARY",
                    },
                    "should_be_inconclusive": True,
                    "private_holdout_passed": False,
                    "evaluator_labels": {
                        "citation_relevance": {"PRIVATE_LABEL_CANARY": True},
                        "claim_support": {"PRIVATE_SUPPORT_CANARY": False},
                    },
                    "score": Score(
                        family_correct=True,
                        root_cause_correct=True,
                        location_correct=True,
                        inconclusive_correct=True,
                    ),
                }
            )

    projection = PrivateScoringProjection()

    result = EvaluationRunner(
        store,
        {"case_0100": "vector-add"},
        projection.execute,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
    ).run("D", "development", 3)

    artifacts = {ref.name: store.read(ref) for ref in store.load(result.run_id).artifact_refs}
    public_bytes = artifacts["evaluation/records/0.json"] + artifacts["evaluation/manifest.json"]
    assert b"should_be_inconclusive" not in public_bytes and b'"score"' not in public_bytes
    record = PublicEvaluationRecord.model_validate_json(artifacts["evaluation/records/0.json"])
    manifest = EvaluationManifest.model_validate_json(artifacts["evaluation/manifest.json"])
    assert not hasattr(record, "should_be_inconclusive") and not hasattr(record, "score")
    assert not hasattr(manifest.records[0], "should_be_inconclusive")
    assert not hasattr(result.records[0], "should_be_inconclusive")
    for content in artifacts.values():
        assert b"PRIVATE_LABEL_CANARY" not in content
        assert b"PRIVATE_SUPPORT_CANARY" not in content
        assert b"private_holdout_passed" not in content
        assert b"evaluator_labels" not in content
        assert b"PRIVATE_CHECK_CANARY" not in content
        assert b"private_oracle" not in content
    assert record.executed_checks == {"memcheck": "FINDING"}


def test_blind_rejects_untyped_diagnosis_metadata():
    from gpu_agent.benchmark.evaluation import EvaluationRecord

    record = EvaluationRecord(
        record_id="x",
        lineage=_lineage(),
        case_id="c",
        template_id="t",
        mode="A",
        repeat=0,
        input_hash="a" * 64,
        evidence_hash="b" * 64,
        executed_checks={},
        status="COMPLETED",
        latency_ms=1,
        diagnosis={"root_cause": "guard missing", "mode": "NESTED_MODE_CANARY"},
    )
    assert "NESTED_MODE_CANARY" not in str(record.blind())


def test_blind_view_does_not_reveal_record_mapping_or_private_observations():
    from gpu_agent.benchmark.evaluation import EvaluationRecord

    record = EvaluationRecord(
        record_id="MODE_E_MODEL_CANARY",
        lineage=_lineage(),
        case_id="case_1",
        template_id="t",
        mode="E",
        repeat=0,
        input_hash="a" * 64,
        evidence_hash="b" * 64,
        executed_checks={},
        status="COMPLETED",
        diagnosis={"root_cause": "guard missing"},
        latency_ms=1,
        private_holdout_passed=True,
        verdict="VERIFIED_FIXED",
        evaluator_labels={"evidence_relevance": {"PRIVATE_EVIDENCE_CANARY": True}},
    )
    assert set(record.blind()) == {"blind_id", "diagnosis", "evidence_hash"}
    assert record.blind()["blind_id"] != record.record_id
    for canary in ("MODE_E_MODEL_CANARY", "PRIVATE_EVIDENCE_CANARY", "VERIFIED_FIXED"):
        assert canary not in str(record.blind())
    assert "private_holdout_passed" not in record.public().model_dump()
