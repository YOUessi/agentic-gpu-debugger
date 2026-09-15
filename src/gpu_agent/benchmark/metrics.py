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
    from gpu_agent.benchmark.holdout import EvaluatorRecordBinding
    from gpu_agent.contracts import RunBinding
    from gpu_agent.store import RunStore


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


def _load_records(
    records: list[EvaluatorRecordBinding],
    public_store: RunStore | None,
    evaluator_store: RunStore | None,
    run_binding: RunBinding | None,
) -> list[EvaluationRecord]:
    from gpu_agent.benchmark.holdout import EvaluatorRecordBinding, HoldoutController

    if (
        public_store is None
        or evaluator_store is None
        or run_binding is None
        or any(not isinstance(record, EvaluatorRecordBinding) for record in records)
    ):
        raise ValueError("metrics require persisted evaluator record bindings")
    controller = HoldoutController(public_store, evaluator_store, binding=run_binding)
    return [controller._load_metric_record(record) for record in records]


def score(
    record: EvaluatorRecordBinding,
    hidden_truth: HiddenTruth,
    rubric: Rubric,
    *,
    public_store: RunStore | None = None,
    evaluator_store: RunStore | None = None,
    run_binding: RunBinding | None = None,
) -> Score:
    loaded = _load_records([record], public_store, evaluator_store, run_binding)[0]
    diagnosis = loaded.diagnosis
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

    return Score(
        family_correct=family,
        root_cause_correct=root,
        location_correct=None if inconclusive else any(valid_location(item) for item in locations),
        inconclusive_correct=inconclusive == hidden_truth.should_be_inconclusive,
    )


def aggregate(
    records: list[EvaluatorRecordBinding],
    *,
    public_store: RunStore | None = None,
    evaluator_store: RunStore | None = None,
    run_binding: RunBinding | None = None,
    retrieval_k: int = 5,
) -> MetricSummary:
    loaded = _load_records(records, public_store, evaluator_store, run_binding)
    if retrieval_k < 1:
        raise ValueError("retrieval_k must be positive")

    def diagnosis(record: EvaluationRecord) -> DiagnosisResult | None:
        try:
            return DiagnosisResult.model_validate(record.diagnosis)
        except ValidationError:
            return None

    def inconclusive(record: EvaluationRecord) -> bool:
        parsed = diagnosis(record)
        return parsed is not None and parsed.diagnostic_outcome != "DIAGNOSED"

    def usage_metric(key: str) -> Metric:
        values = [record.usage.get(key) for record in loaded]
        if not values or any(value is None for value in values):
            return Metric(value=None, n=len(loaded), numerator=0)
        total = sum(value for value in values if value is not None)
        return Metric(value=total / len(loaded), n=len(loaded), numerator=total)

    evidence: list[bool] = []
    citations: list[bool] = []
    unsupported: list[bool] = []
    retrieval: list[bool] = []
    for record in loaded:
        labels = record.evaluator_labels
        if labels is None:
            continue
        for ranked in labels.retrievals:
            top = ranked.ranked_ids[:retrieval_k]
            if all(item in ranked.relevance for item in top):
                retrieval.append(any(ranked.relevance[item] for item in top))
        parsed = diagnosis(record)
        if parsed is None:
            continue
        evidence_ids = {
            citation
            for claim in parsed.observed_facts + parsed.tool_findings
            for citation in claim.citation_ids
        }
        citation_ids = {
            citation for claim in parsed.documentation_evidence for citation in claim.citation_ids
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
                ("observed_facts", parsed.observed_facts),
                ("tool_findings", parsed.tool_findings),
                ("documentation_evidence", parsed.documentation_evidence),
                ("model_inferences", parsed.model_inferences),
            )
            for index, claim in enumerate(claims)
            if claim
        }
        paths.update(name for name in ("root_cause", "recommended_change") if getattr(parsed, name))
        unsupported.extend(
            not labels.claim_support[path] for path in sorted(paths) if path in labels.claim_support
        )

    successful = [
        record.status == "COMPLETED" and record.verdict == "VERIFIED_FIXED" for record in loaded
    ]
    scored = [record.score for record in loaded if record.score is not None]
    actual_inc = [record for record in loaded if inconclusive(record)]
    truth_inc = [record for record in loaded if record.should_be_inconclusive is not None]
    known_costs = [record.cost_usd for record in loaded if record.cost_usd is not None]
    return MetricSummary(
        case_count=len({record.case_id for record in loaded}),
        template_count=len({record.template_id for record in loaded}),
        record_count=len(loaded),
        end_to_end_success=_metric(successful),
        family_accuracy=_metric(
            [item.family_correct for item in scored if item.family_correct is not None]
        ),
        root_cause_accuracy=_metric(
            [item.root_cause_correct for item in scored if item.root_cause_correct is not None]
        ),
        source_location_accuracy=_metric(
            [item.location_correct for item in scored if item.location_correct is not None]
        ),
        evidence_precision=_metric(evidence),
        citation_precision=_metric(citations),
        unsupported_claim_rate=_metric(unsupported),
        retrieval_hit_at_k=_metric(retrieval),
        retrieval_k=retrieval_k,
        patch_compile_rate=_metric(
            [r.patch_compile_passed for r in loaded if r.patch_compile_passed is not None]
        ),
        public_oracle_pass_rate=_metric(
            [r.oracle_passed for r in loaded if r.oracle_passed is not None]
        ),
        private_holdout_pass_rate=_metric(
            [r.private_holdout_passed for r in loaded if r.private_holdout_passed is not None]
        ),
        verified_rate=_metric([r.verdict == "VERIFIED_FIXED" for r in loaded]),
        regression_detection_rate=_metric(
            [
                r.regression_detected
                for r in loaded
                if r.evaluator_labels is not None and r.evaluator_labels.regression_present
            ]
        ),
        inconclusive_precision=_metric(
            [
                bool(r.should_be_inconclusive)
                for r in actual_inc
                if r.should_be_inconclusive is not None
            ]
        ),
        inconclusive_recall=_metric(
            [inconclusive(r) for r in truth_inc if r.should_be_inconclusive]
        ),
        budget_exhaustion_rate=_metric(
            [r.failure_reason in {"AGENT_BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED"} for r in loaded]
        ),
        tool_calls_per_case=usage_metric("tool_calls"),
        sanitizer_calls_per_case=usage_metric("total_sanitizer_calls"),
        llm_calls_per_case=usage_metric("physical_calls"),
        tokens_per_case=usage_metric("total_tokens"),
        diagnostic_tool_calls_per_case=usage_metric("diagnostic_tool_calls"),
        diagnostic_sanitizer_calls_per_case=usage_metric("sanitizer_calls"),
        latency_mean_ms=mean(r.latency_ms for r in loaded) if loaded else None,
        latency_median_ms=median(r.latency_ms for r in loaded) if loaded else None,
        latency_samples_ms=[r.latency_ms for r in loaded],
        cost_total_usd=(
            sum(known_costs) if len(known_costs) == len(loaded) and bool(loaded) else None
        ),
        cost_known_partial_usd=sum(known_costs) if known_costs else None,
        cost_known_record_count=len(known_costs),
        failure_ids=[
            record.record_id
            for record, success in zip(loaded, successful, strict=True)
            if not success
        ],
    )


class GroupedMetricSummary(ExecutionModel):
    """Descriptive attempt summaries, not independent samples or confidence intervals."""

    overall: MetricSummary
    by_mode: dict[str, MetricSummary]
    by_case: dict[str, MetricSummary]
    by_template: dict[str, MetricSummary]
    by_mode_case: dict[str, dict[str, MetricSummary]]
    by_mode_template: dict[str, dict[str, MetricSummary]]


def aggregate_grouped(
    records: list[EvaluatorRecordBinding],
    *,
    public_store: RunStore | None = None,
    evaluator_store: RunStore | None = None,
    run_binding: RunBinding | None = None,
    retrieval_k: int = 5,
) -> GroupedMetricSummary:
    loaded = _load_records(records, public_store, evaluator_store, run_binding)

    def grouped(field: str, mode: str | None = None) -> dict[str, MetricSummary]:
        groups: dict[str, list[EvaluatorRecordBinding]] = defaultdict(list)
        for persisted, record in zip(records, loaded, strict=True):
            if mode is None or record.mode == mode:
                groups[str(getattr(record, field))].append(persisted)
        return {
            key: aggregate(
                group,
                public_store=public_store,
                evaluator_store=evaluator_store,
                run_binding=run_binding,
                retrieval_k=retrieval_k,
            )
            for key, group in sorted(groups.items())
        }

    modes = sorted({record.mode for record in loaded})
    return GroupedMetricSummary(
        overall=aggregate(
            records,
            public_store=public_store,
            evaluator_store=evaluator_store,
            run_binding=run_binding,
            retrieval_k=retrieval_k,
        ),
        by_mode=grouped("mode"),
        by_case=grouped("case_id"),
        by_template=grouped("template_id"),
        by_mode_case={mode: grouped("case_id", mode) for mode in modes},
        by_mode_template={mode: grouped("template_id", mode) for mode in modes},
    )
