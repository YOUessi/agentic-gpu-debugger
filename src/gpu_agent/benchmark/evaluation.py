"""Serial, cost-capped evaluation records and durable batch schedules."""

import hashlib
import json
import random
import re
from collections.abc import Callable
from typing import Literal

from pydantic import Field

from gpu_agent.benchmark.metrics import Score
from gpu_agent.contracts import ArtifactRef, CurrentPhase, RunStatus
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.store import RunStore

EvaluationMode = Literal["A", "B", "C", "D", "E"]
EvaluationSelection = EvaluationMode | Literal["all"]
EvaluationSplit = Literal["development", "holdout"]


class EvaluationRecord(ExecutionModel):
    record_id: str
    case_id: str
    template_id: str
    mode: EvaluationMode
    repeat: int = Field(ge=0)
    input_hash: str
    evidence_hash: str
    executed_checks: dict[str, str]
    status: Literal["COMPLETED", "FAILED", "TIMEOUT", "INCONCLUSIVE"]
    diagnosis: dict[str, object]
    patch_hash: str | None = None
    oracle_passed: bool | None = None
    verdict: str | None = None
    regression_detected: bool = False
    usage: dict[str, int | None] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    failure_reason: str | None = None
    should_be_inconclusive: bool | None = None
    score: Score | None = None

    def blind(self) -> dict[str, object]:
        return {
            "blind_id": self.record_id,
            "diagnosis": self.diagnosis,
            "evidence_hash": self.evidence_hash,
        }


class EvaluationBindings(ExecutionModel):
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    prompt_version: str = Field(min_length=1)
    toolchain_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_unit_cost_usd: float | None = Field(default=None, ge=0)


class EvaluationScheduleItem(ExecutionModel):
    ordinal: int = Field(ge=0)
    case_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    mode: EvaluationMode
    repeat: int = Field(ge=0)


class EvaluationSchedule(ExecutionModel):
    schema_version: Literal[1] = 1
    selection: EvaluationSelection
    modes: list[EvaluationMode]
    split: EvaluationSplit
    repeats: int = Field(ge=3)
    random_seed: int
    bindings: EvaluationBindings
    items: list[EvaluationScheduleItem]


class EvaluationManifest(ExecutionModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    prompt_version: str = Field(min_length=1)
    toolchain_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    schedule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_units: int = Field(ge=0)
    executed_units: int = Field(ge=0)
    modes: list[EvaluationMode]
    split: EvaluationSplit
    repeats: int = Field(ge=3)
    random_seed: int
    records: list[EvaluationRecord]
    stopped_reason: str | None = None


class EvaluationRunner:
    """Own one immutable, resumable evaluation batch in a ``RunStore``."""

    def __init__(
        self,
        store: RunStore,
        case_ids: dict[str, str],
        execute: Callable[[str, str, EvaluationMode, int], EvaluationRecord],
        *,
        commit: str,
        prompt_version: str,
        toolchain_hash: str,
        model_config_hash: str,
        max_cost_usd: float | None,
        max_unit_cost_usd: float | None,
        random_seed: int = 20260915,
    ) -> None:
        self.store = store
        self.case_ids = dict(case_ids)
        self.execute = execute
        self.bindings = EvaluationBindings(
            commit=commit,
            prompt_version=prompt_version,
            toolchain_hash=toolchain_hash,
            model_config_hash=model_config_hash,
            max_cost_usd=max_cost_usd,
            max_unit_cost_usd=max_unit_cost_usd,
        )
        self.random_seed = random_seed

    def run(
        self, mode: EvaluationSelection, split: EvaluationSplit, repeats: int
    ) -> EvaluationManifest:
        schedule = self._schedule(mode, split, repeats)
        run = self.store.create_run("evaluation")
        self.store.transition(run.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)
        self._put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode())
        return self._execute(run.id, schedule, [])

    def resume(
        self, run_id: str, mode: EvaluationSelection, split: EvaluationSplit, repeats: int
    ) -> EvaluationManifest:
        run = self.store.load(run_id)
        if run.kind != "evaluation" or run.status != RunStatus.RUNNING:
            raise ValueError("only a running evaluation run may be resumed")
        expected = self._schedule(mode, split, repeats)
        persisted = EvaluationSchedule.model_validate_json(
            self.store.read(self._one_artifact(run_id, "evaluation/schedule.json"))
        )
        if self._schedule_hash(persisted) != self._schedule_hash(expected) or persisted != expected:
            raise ValueError("evaluation schedule or bindings do not match")
        return self._execute(run_id, persisted, self._records(run_id, persisted))

    def _schedule(
        self, mode: EvaluationSelection, split: EvaluationSplit, repeats: int
    ) -> EvaluationSchedule:
        if repeats < 3:
            raise ValueError("evaluation requires at least three repeats")
        modes: list[EvaluationMode] = ["A", "B", "C", "D", "E"] if mode == "all" else [mode]
        units = [
            (case_id, template_id, item_mode, repeat)
            for repeat in range(repeats)
            for case_id, template_id in self.case_ids.items()
            for item_mode in modes
        ]
        random.Random(self.random_seed).shuffle(units)
        return EvaluationSchedule(
            selection=mode,
            modes=modes,
            split=split,
            repeats=repeats,
            random_seed=self.random_seed,
            bindings=self.bindings,
            items=[
                EvaluationScheduleItem(
                    ordinal=ordinal,
                    case_id=case_id,
                    template_id=template_id,
                    mode=item_mode,
                    repeat=repeat,
                )
                for ordinal, (case_id, template_id, item_mode, repeat) in enumerate(units)
            ],
        )

    @staticmethod
    def _schedule_hash(schedule: EvaluationSchedule) -> str:
        content = json.dumps(
            schedule.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(content).hexdigest()

    def _execute(
        self,
        run_id: str,
        schedule: EvaluationSchedule,
        records: list[EvaluationRecord],
    ) -> EvaluationManifest:
        if (
            schedule.bindings.max_cost_usd is None
            or schedule.bindings.max_unit_cost_usd is None
        ):
            return self._terminal(
                run_id, schedule, records, "COST_CAP_REQUIRED", RunStatus.COMPLETED
            )
        spent = 0.0
        for record in records:
            if record.cost_usd is None:
                return self._terminal(
                    run_id, schedule, records, "COST_UNKNOWN", RunStatus.COMPLETED
                )
            if record.cost_usd > schedule.bindings.max_unit_cost_usd:
                return self._terminal(
                    run_id, schedule, records, "UNIT_COST_CEILING_EXCEEDED", RunStatus.FAILED
                )
            spent += record.cost_usd

        for item in schedule.items[len(records) :]:
            if spent + schedule.bindings.max_unit_cost_usd > schedule.bindings.max_cost_usd:
                return self._terminal(
                    run_id,
                    schedule,
                    records,
                    "COST_CAP_RESERVATION_REQUIRED",
                    RunStatus.COMPLETED,
                )
            try:
                record = self.execute(item.case_id, item.template_id, item.mode, item.repeat)
                self._validate_record(record, item)
            except Exception:
                return self._terminal(
                    run_id, schedule, records, "EXECUTION_ERROR", RunStatus.FAILED
                )
            self._put(
                run_id,
                f"evaluation/records/{item.ordinal}.json",
                record.model_dump_json().encode(),
            )
            records.append(record)
            if record.cost_usd is None:
                return self._terminal(
                    run_id, schedule, records, "COST_UNKNOWN", RunStatus.COMPLETED
                )
            if record.cost_usd > schedule.bindings.max_unit_cost_usd:
                return self._terminal(
                    run_id, schedule, records, "UNIT_COST_CEILING_EXCEEDED", RunStatus.FAILED
                )
            spent += record.cost_usd
        return self._terminal(run_id, schedule, records, None, RunStatus.COMPLETED)

    def _terminal(
        self,
        run_id: str,
        schedule: EvaluationSchedule,
        records: list[EvaluationRecord],
        stopped_reason: str | None,
        status: RunStatus,
    ) -> EvaluationManifest:
        manifest = EvaluationManifest(
            run_id=run_id,
            commit=schedule.bindings.commit,
            prompt_version=schedule.bindings.prompt_version,
            toolchain_hash=schedule.bindings.toolchain_hash,
            model_config_hash=schedule.bindings.model_config_hash,
            schedule_hash=self._schedule_hash(schedule),
            expected_units=len(schedule.items),
            executed_units=len(records),
            modes=schedule.modes,
            split=schedule.split,
            repeats=schedule.repeats,
            random_seed=schedule.random_seed,
            records=records,
            stopped_reason=stopped_reason,
        )
        self._put(run_id, "evaluation/manifest.json", manifest.model_dump_json().encode())
        self.store.transition(run_id, status, None)
        return manifest

    def _records(self, run_id: str, schedule: EvaluationSchedule) -> list[EvaluationRecord]:
        records: dict[int, EvaluationRecord] = {}
        for ref in self.store.load(run_id).artifact_refs:
            if not ref.name.startswith("evaluation/records/"):
                continue
            match = re.fullmatch(r"evaluation/records/([0-9]+)\.json", ref.name)
            if match is None:
                raise ValueError("invalid evaluation record artifact name")
            ordinal = int(match.group(1))
            if ref.name != f"evaluation/records/{ordinal}.json" or ordinal in records:
                raise ValueError("duplicate evaluation record ordinal")
            if ordinal >= len(schedule.items):
                raise ValueError("evaluation record ordinal is out of range")
            record = EvaluationRecord.model_validate_json(self.store.read(ref))
            self._validate_record(record, schedule.items[ordinal])
            records[ordinal] = record
        if set(records) != set(range(len(records))):
            raise ValueError("evaluation record ordinals have a gap")
        return [records[ordinal] for ordinal in range(len(records))]

    @staticmethod
    def _validate_record(record: EvaluationRecord, item: EvaluationScheduleItem) -> None:
        if (record.case_id, record.template_id, record.mode, record.repeat) != (
            item.case_id,
            item.template_id,
            item.mode,
            item.repeat,
        ):
            raise ValueError("evaluation record does not match scheduled unit")

    def _one_artifact(self, run_id: str, name: str) -> ArtifactRef:
        refs = [ref for ref in self.store.load(run_id).artifact_refs if ref.name == name]
        if len(refs) != 1:
            raise ValueError(f"expected exactly one {name} artifact")
        return refs[0]

    def _put(self, run_id: str, name: str, content: bytes) -> None:
        self.store.put(run_id, name, content, self.store.visibility)
