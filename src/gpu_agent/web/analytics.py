"""Read-only public batch/evaluation projections for engineering analytics."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean
from threading import RLock
from typing import Any

from gpu_agent.contracts import ArtifactRef, RunManifest
from gpu_agent.store import RunStore
from gpu_agent.web.models import (
    AnalyticsOverview,
    BatchCard,
    BatchCaseRow,
    BatchDetail,
    EvaluationCard,
    EvaluationDetail,
    EvaluationModeSummary,
    EvaluationRecordRow,
)

_RUN_ID = re.compile(r"^[a-f0-9]{32}$")


class AnalyticsCatalog:
    """Project public operational artifacts without evaluator/private access."""

    def __init__(self, store: RunStore) -> None:
        if store.visibility != "public":
            raise ValueError("analytics catalog only supports a public RunStore")
        self.store = store
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = RLock()

    @staticmethod
    def _require_public_run(run: RunManifest) -> None:
        if any(ref.visibility != "public" for ref in run.artifact_refs):
            raise ValueError("analytics RunStore contains non-public artifacts")

    def _manifests(self) -> list[RunManifest]:
        result: list[RunManifest] = []
        for path in sorted(self.store.root.iterdir()):
            if not path.is_dir() or not _RUN_ID.fullmatch(path.name):
                continue
            run = self.store.load(path.name)
            self._require_public_run(run)
            result.append(run)
        return result

    @staticmethod
    def _ref(run: RunManifest, name: str) -> ArtifactRef | None:
        refs = [ref for ref in run.artifact_refs if ref.name == name]
        if len(refs) > 1:
            raise ValueError("analytics artifact name is ambiguous")
        return refs[0] if refs else None

    def _json_ref(self, ref: ArtifactRef) -> dict[str, Any]:
        key = (ref.run_id, ref.sha256)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        payload = json.loads(self.store.read(ref))
        if not isinstance(payload, dict):
            raise ValueError("analytics artifact is not a JSON object")
        with self._lock:
            if len(self._cache) >= 64:
                self._cache.clear()
            self._cache[key] = payload
        return payload

    def _evaluation_payload(self, run: RunManifest) -> dict[str, Any]:
        self._require_public_run(run)
        if run.kind != "evaluation":
            raise ValueError("not an evaluation run")
        manifest_ref = self._ref(run, "evaluation/manifest.json")
        if manifest_ref is not None:
            return self._json_ref(manifest_ref)

        schedule_ref = self._ref(run, "evaluation/schedule.json")
        schedule = self._json_ref(schedule_ref) if schedule_ref else {}
        records: list[dict[str, Any]] = []
        for ref in sorted(
            (
                ref
                for ref in run.artifact_refs
                if re.fullmatch(r"evaluation/records/[0-9]+\.json", ref.name)
            ),
            key=lambda item: int(item.name.rsplit("/", 1)[1].split(".", 1)[0]),
        ):
            records.append(self._json_ref(ref))
        return {
            "run_id": run.id,
            "expected_units": len(schedule.get("items", []))
            if isinstance(schedule.get("items"), list)
            else None,
            "executed_units": len(records),
            "modes": schedule.get("modes", []),
            "split": schedule.get("split"),
            "repeats": schedule.get("repeats"),
            "corpus_cutoff": schedule.get("corpus_cutoff"),
            "records": records,
        }

    def _batch_payload(self, run: RunManifest) -> dict[str, Any]:
        self._require_public_run(run)
        if run.kind != "seed_batch":
            raise ValueError("not a seed batch run")
        ref = self._ref(run, "batch/summary.json")
        if ref is None:
            progress = sorted(
                (item for item in run.artifact_refs if item.name.startswith("batch/progress/")),
                key=lambda item: item.name,
            )
            if not progress:
                raise ValueError("batch has no summary or progress artifact")
            ref = progress[-1]
        return self._json_ref(ref)
    @staticmethod
    def _datetime(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else None

    @staticmethod
    def _number(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _integer(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    @staticmethod
    def _record_diagnosis(record: dict[str, Any]) -> dict[str, Any]:
        value = record.get("diagnosis")
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _record_usage(record: dict[str, Any]) -> dict[str, Any]:
        value = record.get("usage")
        return value if isinstance(value, dict) else {}

    def _evaluation_records(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        records = payload.get("records", [])
        if not isinstance(records, list):
            return []
        return [item for item in records if isinstance(item, dict)]

    def _mode_summary(self, mode: str, records: list[dict[str, Any]]) -> EvaluationModeSummary:
        diagnoses = [self._record_diagnosis(record) for record in records]
        latencies = [
            value
            for record in records
            if (value := self._number(record.get("latency_ms"))) is not None
        ]
        calls = [
            value
            for record in records
            if (value := self._number(self._record_usage(record).get("physical_calls"))) is not None
        ]
        tokens = [
            value
            for record in records
            if (value := self._number(self._record_usage(record).get("total_tokens"))) is not None
        ]
        costs = [
            value
            for record in records
            if (value := self._number(record.get("cost_usd"))) is not None
        ]
        verified = sum(record.get("verdict") == "VERIFIED_FIXED" for record in records)
        return EvaluationModeSummary(
            mode=mode,
            record_count=len(records),
            diagnosed=sum(item.get("diagnostic_outcome") == "DIAGNOSED" for item in diagnoses),
            verified_fixed=verified,
            verified_rate=verified / len(records) if records else None,
            latency_mean_ms=mean(latencies) if latencies else None,
            llm_calls_mean=mean(calls) if calls else None,
            tokens_mean=mean(tokens) if tokens else None,
            known_cost_usd=sum(costs) if costs else None,
        )

    def _evaluation_card(self, run: RunManifest, payload: dict[str, Any]) -> EvaluationCard:
        records = self._evaluation_records(payload)
        overall = self._mode_summary("ALL", records)
        modes = payload.get("modes")
        if not isinstance(modes, list):
            modes = sorted(
                {str(record.get("mode")) for record in records if record.get("mode") is not None}
            )
        return EvaluationCard(
            run_id=run.id,
            status=run.status.value,
            split=str(payload.get("split")) if payload.get("split") is not None else None,
            corpus_cutoff=self._integer(payload.get("corpus_cutoff")),
            expected_units=self._integer(payload.get("expected_units")),
            executed_units=self._integer(payload.get("executed_units")) or len(records),
            modes=[str(item) for item in modes],
            repeats=self._integer(payload.get("repeats")),
            verified_fixed=overall.verified_fixed,
            verified_rate=overall.verified_rate,
            diagnosed=overall.diagnosed,
            latency_mean_ms=overall.latency_mean_ms,
            llm_calls_mean=overall.llm_calls_mean,
            tokens_mean=overall.tokens_mean,
            known_cost_usd=overall.known_cost_usd,
        )

    def _batch_card(self, run: RunManifest, payload: dict[str, Any]) -> BatchCard:
        cases = payload.get("cases", [])
        items = (
            [item for item in cases if isinstance(item, dict)]
            if isinstance(cases, list)
            else []
        )
        counts = Counter(str(item.get("status", "NOT_RUN")) for item in items)
        tools = Counter(str(item.get("target_tool", "unknown")) for item in items)
        started = payload.get("started_at")
        finished = payload.get("finished_at")
        return BatchCard(
            run_id=run.id,
            status=run.status.value,
            started_at=self._datetime(started),
            finished_at=self._datetime(finished),
            case_count=len(items),
            registered=counts["REGISTERED"],
            validated=counts["VALIDATED"],
            failed=counts["FAILED"],
            running=counts["RUNNING"],
            not_run=counts["NOT_RUN"],
            target_tools=dict(tools),
        )

    def overview(self) -> AnalyticsOverview:
        batches: list[BatchCard] = []
        evaluations: list[EvaluationCard] = []
        projection_errors: list[str] = []
        for run in self._manifests():
            try:
                if run.kind == "seed_batch":
                    batches.append(self._batch_card(run, self._batch_payload(run)))
                elif run.kind == "evaluation":
                    evaluations.append(self._evaluation_card(run, self._evaluation_payload(run)))
            except (OSError, ValueError, json.JSONDecodeError):
                if run.kind in {"seed_batch", "evaluation"}:
                    projection_errors.append(f"{run.id}:ANALYTICS_PROJECTION_INVALID")

        def last_event(run_id: str) -> float:
            run = self.store.load(run_id)
            return run.events[-1].at.timestamp() if run.events else 0.0

        batches.sort(key=lambda item: last_event(item.run_id), reverse=True)
        evaluations.sort(key=lambda item: last_event(item.run_id), reverse=True)
        return AnalyticsOverview(
            store_root=str(self.store.root),
            batch_count=len(batches),
            evaluation_count=len(evaluations),
            projection_errors=projection_errors,
            batches=batches,
            evaluations=evaluations,
        )

    def evaluation_detail(
        self,
        run_id: str,
        *,
        page: int,
        page_size: int,
        mode: str | None = None,
        verdict: str | None = None,
        query: str | None = None,
    ) -> EvaluationDetail:
        run = self.store.load(run_id)
        payload = self._evaluation_payload(run)
        all_records = self._evaluation_records(payload)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        families: Counter[str] = Counter()
        for record in all_records:
            record_mode = str(record.get("mode", ""))
            grouped[record_mode].append(record)
            diagnosis = self._record_diagnosis(record)
            family = diagnosis.get("failure_family")
            if isinstance(family, str) and family:
                families[family] += 1

        filtered = all_records
        if mode:
            filtered = [record for record in filtered if record.get("mode") == mode]
        if verdict:
            filtered = [record for record in filtered if record.get("verdict") == verdict]
        if query:
            needle = query.lower()
            filtered = [
                record
                for record in filtered
                if needle in str(record.get("case_id", "")).lower()
                or needle in str(record.get("template_id", "")).lower()
                or needle
                in str(self._record_diagnosis(record).get("failure_family", "")).lower()
            ]

        total = len(filtered)
        start = (page - 1) * page_size
        page_records = filtered[start : start + page_size]
        rows: list[EvaluationRecordRow] = []
        ordinal_lookup = {id(record): ordinal for ordinal, record in enumerate(all_records)}
        for record in page_records:
            diagnosis = self._record_diagnosis(record)
            usage = self._record_usage(record)
            rows.append(
                EvaluationRecordRow(
                    ordinal=ordinal_lookup[id(record)],
                    record_id=str(record.get("record_id", "")),
                    case_id=str(record.get("case_id", "")),
                    template_id=str(record.get("template_id", "")),
                    mode=str(record.get("mode", "")),
                    repeat=int(record.get("repeat", 0)),
                    status=str(record.get("status", "")),
                    diagnosis_outcome=(
                        str(diagnosis.get("diagnostic_outcome"))
                        if diagnosis.get("diagnostic_outcome") is not None
                        else None
                    ),
                    failure_family=(
                        str(diagnosis.get("failure_family"))
                        if diagnosis.get("failure_family") is not None
                        else None
                    ),
                    verdict=(
                        str(record.get("verdict")) if record.get("verdict") is not None else None
                    ),
                    oracle_passed=(
                        record.get("oracle_passed")
                        if isinstance(record.get("oracle_passed"), bool)
                        else None
                    ),
                    latency_ms=self._number(record.get("latency_ms")),
                    physical_calls=self._integer(usage.get("physical_calls")),
                    sanitizer_calls=self._integer(usage.get("sanitizer_calls")),
                    total_tokens=self._integer(usage.get("total_tokens")),
                    cost_usd=self._number(record.get("cost_usd")),
                    failure_reason=(
                        str(record.get("failure_reason"))
                        if record.get("failure_reason") is not None
                        else None
                    ),
                )
            )
        return EvaluationDetail(
            summary=self._evaluation_card(run, payload),
            mode_metrics=[
                self._mode_summary(mode_name, grouped[mode_name])
                for mode_name in sorted(grouped)
            ],
            failure_families=dict(families),
            records=rows,
            total=total,
            page=page,
            page_size=page_size,
        )

    def batch_detail(self, run_id: str) -> BatchDetail:
        run = self.store.load(run_id)
        payload = self._batch_payload(run)
        raw_cases = payload.get("cases", [])
        cases = (
            [item for item in raw_cases if isinstance(item, dict)]
            if isinstance(raw_cases, list)
            else []
        )
        rows: list[BatchCaseRow] = []
        for item in cases:
            clean = item.get("clean")
            mutant = item.get("mutant")
            clean = clean if isinstance(clean, dict) else {}
            mutant = mutant if isinstance(mutant, dict) else {}
            target_detections = mutant.get("target_detections", [])
            rows.append(
                BatchCaseRow(
                    case_id=str(item.get("case_id", "")),
                    target_tool=str(item.get("target_tool", "")),
                    repetitions=int(item.get("repetitions", 0)),
                    status=str(item.get("status", "NOT_RUN")),
                    clean_run_id=(
                        str(clean.get("run_id")) if clean.get("run_id") is not None else None
                    ),
                    clean_runtime_status=(
                        str(clean.get("runtime_status"))
                        if clean.get("runtime_status") is not None
                        else None
                    ),
                    clean_oracle_passed=(
                        clean.get("oracle_passed")
                        if isinstance(clean.get("oracle_passed"), bool)
                        else None
                    ),
                    clean_sanitizer_outcomes=[
                        str(value)
                        for value in clean.get("sanitizer_outcomes", [])
                        if isinstance(value, str)
                    ],
                    mutant_run_id=(
                        str(mutant.get("run_id")) if mutant.get("run_id") is not None else None
                    ),
                    mutant_runtime_status=(
                        str(mutant.get("runtime_status"))
                        if mutant.get("runtime_status") is not None
                        else None
                    ),
                    mutant_oracle_passed=(
                        mutant.get("oracle_passed")
                        if isinstance(mutant.get("oracle_passed"), bool)
                        else None
                    ),
                    mutant_sanitizer_outcomes=[
                        str(value)
                        for value in mutant.get("sanitizer_outcomes", [])
                        if isinstance(value, str)
                    ],
                    target_detections=[
                        bool(value)
                        for value in target_detections
                        if isinstance(value, bool)
                    ],
                    reason_code=(
                        str(item.get("reason_code"))
                        if item.get("reason_code") is not None
                        else None
                    ),
                )
            )
        return BatchDetail(
            summary=self._batch_card(run, payload),
            register_requested=bool(payload.get("register_requested", False)),
            stopped_reason=(
                str(payload.get("stopped_reason"))
                if payload.get("stopped_reason") is not None
                else None
            ),
            cases=rows,
        )
