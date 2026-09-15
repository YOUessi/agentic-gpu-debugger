from types import SimpleNamespace


def test_empty_data_is_not_perfect_accuracy():
    from gpu_agent.benchmark.metrics import aggregate

    summary = aggregate([])
    assert summary.root_cause_accuracy.value is None
    assert summary.root_cause_accuracy.n == 0
    assert summary.cost_total_usd is None


def test_timeout_stays_in_end_to_end_denominator_but_not_diagnosis_denominator():
    from gpu_agent.benchmark.metrics import Score, aggregate

    good = SimpleNamespace(
        record_id="good",
        case_id="case_1",
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
    from gpu_agent.benchmark.metrics import HiddenTruth, Rubric, score

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
