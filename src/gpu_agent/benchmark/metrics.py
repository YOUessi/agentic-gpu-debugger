"""Transparent benchmark scoring with explicit denominators and N/A values."""

from __future__ import annotations

from statistics import median
from typing import TYPE_CHECKING, cast

from pydantic import Field

from gpu_agent.execution.models import ExecutionModel

if TYPE_CHECKING:
    from gpu_agent.benchmark.evaluation import EvaluationRecord


class HiddenTruth(ExecutionModel):
    failure_family: str
    root_cause_labels: list[str]
    source_path: str
    line_start: int
    line_end: int
    should_be_inconclusive: bool = False


class Rubric(ExecutionModel):
    root_cause_required_labels: int = Field(default=1, ge=1)


class Score(ExecutionModel):
    family_correct: bool | None
    root_cause_correct: bool | None
    location_correct: bool | None
    inconclusive_correct: bool | None


class Metric(ExecutionModel):
    value: float | None
    n: int
    numerator: int = 0


class MetricSummary(ExecutionModel):
    case_count: int
    record_count: int
    end_to_end_success: Metric
    family_accuracy: Metric
    root_cause_accuracy: Metric
    source_location_accuracy: Metric
    verified_rate: Metric
    regression_detection_rate: Metric
    inconclusive_precision: Metric
    inconclusive_recall: Metric
    budget_exhaustion_rate: Metric
    latency_median_ms: float | None
    cost_total_usd: float | None
    failure_ids: list[str]


def _metric(values: list[bool]) -> Metric:
    return Metric(
        value=sum(values) / len(values) if values else None, n=len(values), numerator=sum(values)
    )


def score(record: EvaluationRecord, hidden_truth: HiddenTruth, rubric: Rubric) -> Score:
    diagnosis = record.diagnosis
    inconclusive = diagnosis.get("diagnostic_outcome") != "DIAGNOSED"
    family = (
        None if inconclusive else diagnosis.get("failure_family") == hidden_truth.failure_family
    )
    text = str(diagnosis.get("root_cause", "")).lower()
    matched = sum(label.lower() in text for label in hidden_truth.root_cause_labels)
    root = None if inconclusive else matched >= rubric.root_cause_required_labels
    locations = cast(list[dict[str, object]], diagnosis.get("source_locations", []))

    def valid_location(item: dict[str, object]) -> bool:
        line = item.get("line")
        return (
            item.get("path") == hidden_truth.source_path
            and type(line) is int
            and hidden_truth.line_start <= line <= hidden_truth.line_end
        )

    location = None if inconclusive else any(valid_location(item) for item in locations)
    return Score(
        family_correct=family,
        root_cause_correct=root,
        location_correct=location,
        inconclusive_correct=inconclusive == hidden_truth.should_be_inconclusive,
    )


def aggregate(records: list[EvaluationRecord]) -> MetricSummary:
    scored = [record.score for record in records if record.score is not None]
    family = [item.family_correct for item in scored if item.family_correct is not None]
    root = [item.root_cause_correct for item in scored if item.root_cause_correct is not None]
    locations = [item.location_correct for item in scored if item.location_correct is not None]
    actual_inc = [record for record in records if record.status == "INCONCLUSIVE"]
    truth_inc = [record for record in records if record.should_be_inconclusive is not None]
    precision_values = [
        bool(record.should_be_inconclusive)
        for record in actual_inc
        if record.should_be_inconclusive is not None
    ]
    recall_values = [
        record.status == "INCONCLUSIVE" for record in truth_inc if record.should_be_inconclusive
    ]
    known_costs = [record.cost_usd for record in records if record.cost_usd is not None]
    all_cost_known = len(known_costs) == len(records) and bool(records)
    return MetricSummary(
        case_count=len({record.case_id for record in records}),
        record_count=len(records),
        end_to_end_success=_metric([record.status == "COMPLETED" for record in records]),
        family_accuracy=_metric(family),
        root_cause_accuracy=_metric(root),
        source_location_accuracy=_metric(locations),
        verified_rate=_metric([record.verdict == "VERIFIED_FIXED" for record in records]),
        regression_detection_rate=_metric([record.regression_detected for record in records]),
        inconclusive_precision=_metric(precision_values),
        inconclusive_recall=_metric(recall_values),
        budget_exhaustion_rate=_metric(
            [record.failure_reason == "BUDGET_EXHAUSTED" for record in records]
        ),
        latency_median_ms=median(record.latency_ms for record in records) if records else None,
        cost_total_usd=sum(known_costs) if all_cost_known else None,
        failure_ids=[record.record_id for record in records if record.status != "COMPLETED"],
    )
