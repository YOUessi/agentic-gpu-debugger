from types import SimpleNamespace

import pytest


def test_empty_data_is_not_perfect_accuracy():
    from gpu_agent.benchmark.metrics import _aggregate_records as aggregate

    summary = aggregate([])
    assert summary.root_cause_accuracy.value is None
    assert summary.root_cause_accuracy.n == 0
    assert summary.cost_total_usd is None
    assert summary.template_count == summary.record_count == summary.case_count == 0
    for name in (
        "evidence_precision",
        "citation_precision",
        "unsupported_claim_rate",
        "retrieval_hit_at_k",
        "patch_compile_rate",
        "public_oracle_pass_rate",
        "private_holdout_pass_rate",
        "regression_detection_rate",
        "tool_calls_per_case",
        "sanitizer_calls_per_case",
        "llm_calls_per_case",
        "tokens_per_case",
    ):
        assert getattr(summary, name).model_dump() == {"value": None, "n": 0, "numerator": 0}


def test_timeout_stays_in_end_to_end_denominator_but_not_diagnosis_denominator():
    from gpu_agent.benchmark.metrics import Score
    from gpu_agent.benchmark.metrics import _aggregate_records as aggregate

    good = SimpleNamespace(
        record_id="good",
        case_id="case_1",
        template_id="t",
        mode="A",
        diagnosis={},
        usage={},
        evaluator_labels=None,
        patch_compile_passed=None,
        oracle_passed=None,
        private_holdout_passed=None,
        status="COMPLETED",
        verdict="VERIFIED_FIXED",
        regression_detected=False,
        failure_reason=None,
        latency_ms=10,
        cost_usd=0.1,
        should_be_inconclusive=False,
        score=Score(
            family_correct=True,
            root_cause_correct=True,
            location_correct=True,
            inconclusive_correct=True,
        ),
    )
    timeout = SimpleNamespace(
        record_id="timeout",
        case_id="case_2",
        template_id="t",
        mode="A",
        diagnosis={},
        usage={},
        evaluator_labels=None,
        patch_compile_passed=None,
        oracle_passed=None,
        private_holdout_passed=None,
        status="TIMEOUT",
        verdict=None,
        regression_detected=False,
        failure_reason="TIMEOUT",
        latency_ms=100,
        cost_usd=0.1,
        should_be_inconclusive=None,
        score=None,
    )
    summary = aggregate([good, timeout])
    assert summary.end_to_end_success.value == 0.5
    assert summary.root_cause_accuracy.value == 1.0
    assert summary.root_cause_accuracy.n == 1
    assert summary.failure_ids == ["timeout"]


def test_score_uses_registered_family_labels_and_line_range():
    from gpu_agent.benchmark.metrics import HiddenTruth, Rubric
    from gpu_agent.benchmark.metrics import _score_record as score

    record = SimpleNamespace(
        diagnosis={
            "diagnostic_outcome": "DIAGNOSED",
            "failure_family": "race",
            "root_cause": "Missing barrier creates a shared memory hazard",
            "source_locations": [{"path": "kernel.cu", "line": 12}],
        }
    )
    result = score(
        record,
        HiddenTruth(
            failure_family="race",
            root_cause_labels=["barrier", "shared memory"],
            source_path="kernel.cu",
            line_start=10,
            line_end=14,
        ),
        Rubric(root_cause_required_labels=2),
    )
    assert result.family_correct and result.root_cause_correct and result.location_correct


@pytest.mark.parametrize("verdict", [None, "NOT_FIXED", "REGRESSION_DETECTED", "INCONCLUSIVE"])
def test_completed_workflow_without_verified_repair_is_a_benchmark_failure(verdict):
    from gpu_agent.benchmark.evaluation import EvaluationRecord
    from gpu_agent.benchmark.metrics import _aggregate_records as aggregate

    record = EvaluationRecord(
        record_id="unsuccessful",
        lineage={
            "diagnosis_run_id": "a" * 32,
            "diagnosis_hash": "c" * 64,
            "evidence_hash": "b" * 64,
            "provider_invocation_hashes": [],
        },
        case_id="case_0100",
        template_id="vector-add",
        mode="D",
        repeat=0,
        input_hash="a" * 64,
        evidence_hash="b" * 64,
        executed_checks={},
        status="COMPLETED",
        diagnosis={},
        verdict=verdict,
        latency_ms=1,
    )
    summary = aggregate([record])
    assert summary.end_to_end_success.n == 1
    assert summary.end_to_end_success.numerator == 0
    assert summary.failure_ids == ["unsuccessful"]


def evaluation_record(**updates):
    from gpu_agent.benchmark.evaluation import EvaluationRecord

    return EvaluationRecord.model_validate(
        {
            "record_id": "record",
            "lineage": {
                "diagnosis_run_id": "a" * 32,
                "diagnosis_hash": "c" * 64,
                "evidence_hash": "b" * 64,
                "provider_invocation_hashes": [],
            },
            "case_id": "case_1",
            "template_id": "template_1",
            "mode": "A",
            "repeat": 0,
            "input_hash": "a" * 64,
            "evidence_hash": "b" * 64,
            "executed_checks": {},
            "status": "COMPLETED",
            "diagnosis": {},
            "latency_ms": 10,
            **updates,
        }
    )


def test_relevance_is_not_inferred_from_citation_existence_or_text():
    from gpu_agent.benchmark.metrics import _aggregate_records as aggregate

    diagnosis = {
        "diagnostic_outcome": "DIAGNOSED",
        "root_cause": "doc-good",
        "observed_facts": [{"text": "observed", "citation_ids": ["artifact-bad"]}],
        "documentation_evidence": [{"text": "claim", "citation_ids": ["doc-bad"]}],
    }
    record = evaluation_record(diagnosis=diagnosis)
    summary = aggregate([record])
    for metric in (
        summary.evidence_precision,
        summary.citation_precision,
        summary.unsupported_claim_rate,
        summary.retrieval_hit_at_k,
    ):
        assert metric.model_dump() == {"value": None, "n": 0, "numerator": 0}


def test_explicit_labels_score_only_typed_claims_and_ranked_retrieval():
    from gpu_agent.benchmark.metrics import _aggregate_records as aggregate

    record = evaluation_record(
        diagnosis={
            "diagnostic_outcome": "DIAGNOSED",
            "root_cause": "doc-good",
            "observed_facts": [{"text": "fact", "citation_ids": ["a", "b", "a"]}],
            "documentation_evidence": [{"text": "claim", "citation_ids": ["doc-bad"]}],
        },
        evaluator_labels={
            "evidence_relevance": {"a": True, "b": False},
            "citation_relevance": {"doc-bad": False, "doc-good": True},
            "claim_support": {
                "observed_facts/0": True,
                "documentation_evidence/0": False,
                "model_inferences/99": False,
            },
            "retrievals": [
                {
                    "ranked_ids": ["doc-bad", "doc-good"],
                    "relevance": {"doc-bad": False, "doc-good": True},
                }
            ],
        },
    )
    summary = aggregate([record], retrieval_k=1)
    assert summary.evidence_precision.model_dump() == {"value": 0.5, "n": 2, "numerator": 1}
    assert summary.citation_precision.model_dump() == {"value": 0.0, "n": 1, "numerator": 0}
    assert summary.unsupported_claim_rate.value == 0.5
    assert summary.unsupported_claim_rate.n == 2
    assert summary.retrieval_hit_at_k.value == 0
    assert aggregate([record], retrieval_k=2).retrieval_hit_at_k.value == 1


def test_missing_partial_retrieval_labels_and_malformed_claims_are_not_scored():
    from gpu_agent.benchmark.metrics import _aggregate_records as aggregate

    record = evaluation_record(
        diagnosis={"diagnostic_outcome": "DIAGNOSED", "observed_facts": "a"},
        evaluator_labels={
            "evidence_relevance": {"a": True},
            "retrievals": [{"ranked_ids": ["a", "unlabeled"], "relevance": {"a": False}}],
        },
    )
    summary = aggregate([record], retrieval_k=2)
    assert summary.evidence_precision.n == summary.retrieval_hit_at_k.n == 0
    with pytest.raises(ValueError):
        aggregate([record], retrieval_k=0)


def test_repair_rates_distinguish_compile_public_private_and_regression_truth():
    from gpu_agent.benchmark.metrics import _aggregate_records as aggregate

    records = [
        evaluation_record(
            patch_compile_passed=True,
            oracle_passed=True,
            private_holdout_passed=False,
            verdict="REGRESSION_DETECTED",
            regression_detected=True,
            evaluator_labels={"regression_present": True},
        ),
        evaluation_record(
            patch_compile_passed=False,
            oracle_passed=None,
            evaluator_labels={"regression_present": True},
        ),
        evaluation_record(status="TIMEOUT"),
    ]
    summary = aggregate(records)
    assert summary.patch_compile_rate.value == 0.5 and summary.patch_compile_rate.n == 2
    assert summary.public_oracle_pass_rate.value == 1 and summary.public_oracle_pass_rate.n == 1
    assert summary.private_holdout_pass_rate.value == 0
    assert summary.private_holdout_pass_rate.n == 1
    assert summary.regression_detection_rate.value == 0.5
    assert summary.regression_detection_rate.n == 2
    assert summary.verified_rate.n == summary.end_to_end_success.n == 3


def test_repeats_group_by_mode_case_template_without_inflating_unique_counts():
    from gpu_agent.benchmark.metrics import _aggregate_grouped_records as aggregate_grouped

    records = [
        evaluation_record(record_id=str(i), repeat=i, mode=mode, case_id=case)
        for i, (mode, case) in enumerate(
            [("A", "case_1"), ("A", "case_1"), ("B", "case_1"), ("A", "case_2")]
        )
    ]
    grouped = aggregate_grouped(records)
    assert grouped.overall.record_count == 4
    assert grouped.overall.case_count == 2 and grouped.overall.template_count == 1
    assert grouped.by_mode["A"].record_count == 3
    assert grouped.by_case["case_1"].record_count == 3
    assert grouped.by_template["template_1"].case_count == 2
    assert grouped.by_mode_case["A"]["case_1"].record_count == 2


def test_public_metric_entry_points_reject_unvalidated_records():
    import gpu_agent.benchmark.metrics as metrics
    from gpu_agent.benchmark.metrics import (
        HiddenTruth,
        Rubric,
        aggregate,
        aggregate_grouped,
        score,
    )

    record = evaluation_record()
    assert not hasattr(metrics, "_EVALUATOR_AUTHORITY")
    with pytest.raises(ValueError, match="persisted evaluator"):
        aggregate([record])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="persisted evaluator"):
        aggregate([SimpleNamespace(record=record)])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="persisted evaluator"):
        aggregate_grouped([record])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="persisted evaluator"):
        score(  # type: ignore[arg-type]
            record,
            HiddenTruth(
                failure_family="memory",
                root_cause_labels=["bounds"],
                source_path="kernel.cu",
                line_start=1,
                line_end=2,
            ),
            Rubric(),
        )


def test_efficiency_counts_failed_units_and_unknown_cost_is_not_zero():
    from gpu_agent.benchmark.metrics import _aggregate_records as aggregate

    records = [
        evaluation_record(
            cost_usd=0.2,
            usage={
                "tool_calls": 6,
                "total_sanitizer_calls": 2,
                "physical_calls": 3,
                "total_tokens": 90,
            },
        ),
        evaluation_record(
            status="TIMEOUT",
            failure_reason="BUDGET_EXHAUSTED",
            latency_ms=30,
            usage={
                "tool_calls": 2,
                "total_sanitizer_calls": 1,
                "physical_calls": 1,
                "total_tokens": 10,
            },
        ),
    ]
    summary = aggregate(records)
    for metric, expected in [
        (summary.tool_calls_per_case, 4),
        (summary.sanitizer_calls_per_case, 1.5),
        (summary.llm_calls_per_case, 2),
        (summary.tokens_per_case, 50),
    ]:
        assert metric.value == expected and metric.n == 2
    assert summary.latency_mean_ms == summary.latency_median_ms == 20
    assert summary.latency_samples_ms == [10, 30]
    assert summary.budget_exhaustion_rate.value == 0.5
    assert summary.cost_total_usd is None
    assert summary.cost_known_partial_usd == 0.2 and summary.cost_known_record_count == 1
    assert aggregate([evaluation_record()]).tool_calls_per_case.value is None
    assert aggregate([evaluation_record()]).tool_calls_per_case.n == 1


def test_inconclusive_uses_diagnosis_outcome_not_repair_workflow_status():
    from gpu_agent.benchmark.metrics import _aggregate_records as aggregate

    records = [
        evaluation_record(
            status="INCONCLUSIVE",
            should_be_inconclusive=False,
            diagnosis={"diagnostic_outcome": "DIAGNOSED"},
        ),
        evaluation_record(
            status="FAILED",
            should_be_inconclusive=True,
            diagnosis={"diagnostic_outcome": "INCONCLUSIVE"},
        ),
    ]
    summary = aggregate(records)
    assert summary.inconclusive_precision.value == summary.inconclusive_recall.value == 1
    assert summary.inconclusive_precision.n == summary.inconclusive_recall.n == 1


def test_known_zero_cost_and_usage_are_measured_zero_not_missing():
    from gpu_agent.benchmark.metrics import _aggregate_records as aggregate

    summary = aggregate(
        [
            evaluation_record(
                cost_usd=0,
                usage={
                    "tool_calls": 0,
                    "total_sanitizer_calls": 0,
                    "physical_calls": 0,
                    "total_tokens": 0,
                },
            )
        ]
    )
    assert summary.cost_total_usd == 0
    assert summary.cost_known_record_count == 1
    assert summary.llm_calls_per_case.model_dump() == {"value": 0.0, "n": 1, "numerator": 0}


def test_budget_exhaustion_counts_the_production_agent_reason_code():
    from gpu_agent.benchmark.metrics import _aggregate_records as aggregate

    record = evaluation_record(
        status="FAILED",
        failure_reason="AGENT_BUDGET_EXHAUSTED",
        diagnosis={
            "diagnostic_outcome": "INCONCLUSIVE",
            "limitations": ["AGENT_BUDGET_EXHAUSTED"],
        },
    )
    assert aggregate([record]).budget_exhaustion_rate.model_dump() == {
        "value": 1.0,
        "n": 1,
        "numerator": 1,
    }


def test_claim_support_includes_root_recommendation_and_inferences_without_llm_scoring():
    from gpu_agent.benchmark.metrics import _aggregate_records as aggregate

    record = evaluation_record(
        diagnosis={
            "diagnostic_outcome": "DIAGNOSED",
            "root_cause": "guard missing",
            "recommended_change": "bound the index",
            "model_inferences": ["causes overflow"],
        },
        evaluator_labels={
            "claim_support": {
                "root_cause": False,
                "recommended_change": True,
                "model_inferences/0": False,
                "documentation_evidence/0": False,
            }
        },
    )
    assert aggregate([record]).unsupported_claim_rate.model_dump() == {
        "value": 2 / 3,
        "n": 3,
        "numerator": 2,
    }
