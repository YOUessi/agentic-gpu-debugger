"""Transparent benchmark scoring with explicit denominators and N/A values."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import TYPE_CHECKING, cast

from pydantic import Field, ValidationError

from gpu_agent.agent.models import DiagnosisResult
from gpu_agent.execution.models import ExecutionModel

if TYPE_CHECKING:
    from gpu_agent.benchmark.evaluation import EvaluationRecord


class ValidatedEvaluationRecord:
    """Opaque evaluator-loader output accepted by public metric entry points."""

    __slots__ = ("_record",)

    def __init__(self, record: EvaluationRecord, authority: object) -> None:
        if authority is not _EVALUATOR_AUTHORITY:
            raise ValueError("validated records are created only by the evaluator loader")
        self._record = record

    @property
    def record(self) -> EvaluationRecord:
        return self._record


_EVALUATOR_AUTHORITY = object()


class HiddenTruth(ExecutionModel):
    failure_family: str
    root_cause_labels: list[str]
    source_path: str
    line_start: int
    line_end: int
    should_be_inconclusive: bool = False


class Rubric(ExecutionModel):
    root_cause_required_labels: int = Field(default=1, ge=1)


class RetrievalLabels(ExecutionModel):
    """Evaluator-provided ranking and relevance; absent IDs are unjudged."""

    ranked_ids: list[str]
    relevance: dict[str, bool] = Field(default_factory=dict)


class EvaluationLabels(ExecutionModel):
    """Private human labels, never model self-assessments.

    Evidence/citation relevance keys are IDs in typed diagnosis claims. Evidence
    precision covers observed_facts/tool_findings; citation precision covers
    documentation_evidence. Repeated IDs count once per record and category.
    Support keys are claim paths, e.g. ``observed_facts/0`` or ``root_cause``;
    only nonempty claims actually present in the parsed diagnosis are counted.
    """

    evidence_relevance: dict[str, bool] = Field(default_factory=dict)
    citation_relevance: dict[str, bool] = Field(default_factory=dict)
    claim_support: dict[str, bool] = Field(default_factory=dict)
    retrievals: list[RetrievalLabels] = Field(default_factory=list)
    regression_present: bool | None = None


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
    template_count: int
    record_count: int
    end_to_end_success: Metric
    family_accuracy: Metric
    root_cause_accuracy: Metric
    source_location_accuracy: Metric
    evidence_precision: Metric
    citation_precision: Metric
    unsupported_claim_rate: Metric
    retrieval_hit_at_k: Metric
    retrieval_k: int
    patch_compile_rate: Metric
    public_oracle_pass_rate: Metric
    private_holdout_pass_rate: Metric
    verified_rate: Metric
    regression_detection_rate: Metric
    inconclusive_precision: Metric
    inconclusive_recall: Metric
    budget_exhaustion_rate: Metric
    tool_calls_per_case: Metric
    sanitizer_calls_per_case: Metric
    llm_calls_per_case: Metric
    tokens_per_case: Metric
    diagnostic_tool_calls_per_case: Metric
    diagnostic_sanitizer_calls_per_case: Metric
    latency_mean_ms: float | None
    latency_median_ms: float | None
    latency_samples_ms: list[float]
    cost_total_usd: float | None
    cost_known_partial_usd: float | None
    cost_known_record_count: int
    failure_ids: list[str]


def _metric(values: list[bool]) -> Metric:
    return Metric(
        value=sum(values) / len(values) if values else None, n=len(values), numerator=sum(values)
    )


def _score_record(record: EvaluationRecord, hidden_truth: HiddenTruth, rubric: Rubric) -> Score:
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


def score(record: ValidatedEvaluationRecord, hidden_truth: HiddenTruth, rubric: Rubric) -> Score:
    if not isinstance(record, ValidatedEvaluationRecord):
        raise ValueError("metrics require an evaluator-validated record")
    return _score_record(record.record, hidden_truth, rubric)


def _diagnosis(record: EvaluationRecord) -> DiagnosisResult | None:
    try:
        return DiagnosisResult.model_validate(record.diagnosis)
    except ValidationError:
        return None


def _inconclusive(record: EvaluationRecord) -> bool:
    diagnosis = _diagnosis(record)
    return diagnosis is not None and diagnosis.diagnostic_outcome != "DIAGNOSED"


def _quality_labels(
    records: list[EvaluationRecord], retrieval_k: int
) -> tuple[list[bool], list[bool], list[bool], list[bool]]:
    evidence: list[bool] = []
    citations: list[bool] = []
    unsupported: list[bool] = []
    retrieval: list[bool] = []
    for record in records:
        labels = record.evaluator_labels
        if labels is None:
            continue
        for ranked in labels.retrievals:
            top = ranked.ranked_ids[:retrieval_k]
            # Require a judgment for every returned top-k item; empty judged
            # retrievals are misses. An unlabeled result never becomes relevant.
            if all(item in ranked.relevance for item in top):
                retrieval.append(any(ranked.relevance[item] for item in top))
        diagnosis = _diagnosis(record)
        if diagnosis is None:
            continue
        evidence_ids = {
            citation
            for claim in diagnosis.observed_facts + diagnosis.tool_findings
            for citation in claim.citation_ids
        }
        citation_ids = {
            citation
            for claim in diagnosis.documentation_evidence
            for citation in claim.citation_ids
        }
        evidence.extend(
            labels.evidence_relevance[item]
            for item in sorted(evidence_ids)
            if item in labels.evidence_relevance
        )
        citations.extend(
            labels.citation_relevance[item]
            for item in sorted(citation_ids)
            if item in labels.citation_relevance
        )
        paths = {
            f"{name}/{index}"
            for name, claims in (
                ("observed_facts", diagnosis.observed_facts),
                ("tool_findings", diagnosis.tool_findings),
                ("documentation_evidence", diagnosis.documentation_evidence),
                ("model_inferences", diagnosis.model_inferences),
            )
            for index, claim in enumerate(claims)
            if claim
        }
        paths.update(
            name for name in ("root_cause", "recommended_change") if getattr(diagnosis, name)
        )
        unsupported.extend(
            not labels.claim_support[path] for path in sorted(paths) if path in labels.claim_support
        )
    return evidence, citations, unsupported, retrieval


def _usage_metric(records: list[EvaluationRecord], key: str) -> Metric:
    values = [record.usage.get(key) for record in records]
    # All attempted units stay in the denominator, even when measurement is
    # unavailable. Unknown counts cannot silently be interpreted as zero.
    if not values or any(value is None for value in values):
        return Metric(value=None, n=len(records), numerator=0)
    total = sum(value for value in values if value is not None)
    return Metric(value=total / len(records), n=len(records), numerator=total)


def _aggregate_records(records: list[EvaluationRecord], *, retrieval_k: int = 5) -> MetricSummary:
    """Micro-aggregate repeated attempts; counts expose the unique experimental units.

    Label-based rates use judged items only. Repair component rates use measured
    observations only. E2E/verified/efficiency/budget/latency retain all attempts.
    Regression detection is recall among evaluator-labeled regression cases.
    """
    if retrieval_k < 1:
        raise ValueError("retrieval_k must be positive")
    successful = [
        record.status == "COMPLETED" and record.verdict == "VERIFIED_FIXED" for record in records
    ]
    scored = [record.score for record in records if record.score is not None]
    family = [item.family_correct for item in scored if item.family_correct is not None]
    root = [item.root_cause_correct for item in scored if item.root_cause_correct is not None]
    locations = [item.location_correct for item in scored if item.location_correct is not None]
    actual_inc = [record for record in records if _inconclusive(record)]
    truth_inc = [record for record in records if record.should_be_inconclusive is not None]
    precision_values = [
        bool(record.should_be_inconclusive)
        for record in actual_inc
        if record.should_be_inconclusive is not None
    ]
    recall_values = [_inconclusive(record) for record in truth_inc if record.should_be_inconclusive]
    known_costs = [record.cost_usd for record in records if record.cost_usd is not None]
    all_cost_known = len(known_costs) == len(records) and bool(records)
    evidence, citations, unsupported, retrieval = _quality_labels(records, retrieval_k)
    return MetricSummary(
        case_count=len({record.case_id for record in records}),
        template_count=len({record.template_id for record in records}),
        record_count=len(records),
        end_to_end_success=_metric(successful),
        family_accuracy=_metric(family),
        root_cause_accuracy=_metric(root),
        source_location_accuracy=_metric(locations),
        evidence_precision=_metric(evidence),
        citation_precision=_metric(citations),
        unsupported_claim_rate=_metric(unsupported),
        retrieval_hit_at_k=_metric(retrieval),
        retrieval_k=retrieval_k,
        patch_compile_rate=_metric(
            [
                record.patch_compile_passed
                for record in records
                if record.patch_compile_passed is not None
            ]
        ),
        public_oracle_pass_rate=_metric(
            [record.oracle_passed for record in records if record.oracle_passed is not None]
        ),
        private_holdout_pass_rate=_metric(
            [
                record.private_holdout_passed
                for record in records
                if record.private_holdout_passed is not None
            ]
        ),
        verified_rate=_metric([record.verdict == "VERIFIED_FIXED" for record in records]),
        regression_detection_rate=_metric(
            [
                record.regression_detected
                for record in records
                if record.evaluator_labels is not None
                and record.evaluator_labels.regression_present
            ]
        ),
        inconclusive_precision=_metric(precision_values),
        inconclusive_recall=_metric(recall_values),
        budget_exhaustion_rate=_metric(
            [
                record.failure_reason in {"AGENT_BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED"}
                for record in records
            ]
        ),
        tool_calls_per_case=_usage_metric(records, "tool_calls"),
        sanitizer_calls_per_case=_usage_metric(records, "total_sanitizer_calls"),
        llm_calls_per_case=_usage_metric(records, "physical_calls"),
        tokens_per_case=_usage_metric(records, "total_tokens"),
        diagnostic_tool_calls_per_case=_usage_metric(records, "diagnostic_tool_calls"),
        diagnostic_sanitizer_calls_per_case=_usage_metric(records, "sanitizer_calls"),
        latency_mean_ms=mean(record.latency_ms for record in records) if records else None,
        latency_median_ms=median(record.latency_ms for record in records) if records else None,
        latency_samples_ms=[record.latency_ms for record in records],
        cost_total_usd=sum(known_costs) if all_cost_known else None,
        cost_known_partial_usd=sum(known_costs) if known_costs else None,
        cost_known_record_count=len(known_costs),
        failure_ids=[
            record.record_id
            for record, success in zip(records, successful, strict=True)
            if not success
        ],
    )


def aggregate(records: list[ValidatedEvaluationRecord], *, retrieval_k: int = 5) -> MetricSummary:
    if any(not isinstance(record, ValidatedEvaluationRecord) for record in records):
        raise ValueError("metrics require evaluator-validated records")
    return _aggregate_records([record.record for record in records], retrieval_k=retrieval_k)


class GroupedMetricSummary(ExecutionModel):
    """Descriptive attempt summaries, not independent samples or confidence intervals."""

    overall: MetricSummary
    by_mode: dict[str, MetricSummary]
    by_case: dict[str, MetricSummary]
    by_template: dict[str, MetricSummary]
    by_mode_case: dict[str, dict[str, MetricSummary]]
    by_mode_template: dict[str, dict[str, MetricSummary]]


def aggregate_grouped(
    records: list[ValidatedEvaluationRecord], *, retrieval_k: int = 5
) -> GroupedMetricSummary:
    if any(not isinstance(record, ValidatedEvaluationRecord) for record in records):
        raise ValueError("metrics require evaluator-validated records")
    return _aggregate_grouped_records(
        [record.record for record in records], retrieval_k=retrieval_k
    )


def _aggregate_grouped_records(
    records: list[EvaluationRecord], *, retrieval_k: int = 5
) -> GroupedMetricSummary:
    """Internal aggregation used only after native evaluator validation."""
    raw = records

    def grouped(items: list[EvaluationRecord], field: str) -> dict[str, MetricSummary]:
        groups: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for record in items:
            groups[str(getattr(record, field))].append(record)
        return {
            key: _aggregate_records(group, retrieval_k=retrieval_k)
            for key, group in sorted(groups.items())
        }

    modes = sorted({record.mode for record in raw})
    return GroupedMetricSummary(
        overall=_aggregate_records(raw, retrieval_k=retrieval_k),
        by_mode=grouped(raw, "mode"),
        by_case=grouped(raw, "case_id"),
        by_template=grouped(raw, "template_id"),
        by_mode_case={
            mode: grouped([r for r in raw if r.mode == mode], "case_id") for mode in modes
        },
        by_mode_template={
            mode: grouped([r for r in raw if r.mode == mode], "template_id") for mode in modes
        },
    )
