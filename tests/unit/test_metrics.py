"""Trusted metrics are reachable only through persisted evaluator bindings."""

from types import SimpleNamespace

import pytest


def test_raw_metric_helpers_are_not_importable():
    import gpu_agent.benchmark.metrics as metrics

    for name in (
        "_aggregate_records",
        "_aggregate_grouped_records",
        "_score_record",
        "_diagnosis",
        "_inconclusive",
        "_quality_labels",
        "_usage_metric",
    ):
        assert not hasattr(metrics, name)


def test_public_metric_entry_points_reject_raw_records():
    from gpu_agent.benchmark.evaluation import EvaluationRecord
    from gpu_agent.benchmark.metrics import (
        HiddenTruth,
        Rubric,
        aggregate,
        aggregate_grouped,
        score,
    )

    record = EvaluationRecord(
        record_id="raw",
        lineage={
            "diagnosis_run_id": "a" * 32,
            "diagnosis_hash": "b" * 64,
            "evidence_hash": "c" * 64,
            "provider_invocation_hashes": [],
        },
        case_id="case_0100",
        template_id="vector-add",
        mode="A",
        repeat=0,
        input_hash="d" * 64,
        evidence_hash="c" * 64,
        executed_checks={},
        status="INCONCLUSIVE",
        diagnosis={"diagnostic_outcome": "INCONCLUSIVE", "limitations": ["RAW"]},
        latency_ms=0,
    )
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
