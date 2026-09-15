"""Serial, cost-capped evaluation records and durable public batch schedules."""

import hashlib
import json
import random
import re
from collections.abc import Callable
from typing import Literal

from pydantic import Field, ValidationError

from gpu_agent.agent.models import DiagnosisResult
from gpu_agent.benchmark.metrics import EvaluationLabels, Score
from gpu_agent.contracts import ArtifactRef, CurrentPhase, RunBinding, RunStatus
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.store import RunStore

EvaluationMode = Literal["A", "B", "C", "D", "E"]
EvaluationSelection = EvaluationMode | Literal["all"]
EvaluationSplit = Literal["development", "holdout"]
StoppedReason = Literal[
    "COST_CAP_REQUIRED",
    "COST_UNKNOWN",
    "COST_CAP_RESERVATION_REQUIRED",
    "UNIT_COST_CEILING_EXCEEDED",
    "EXECUTION_ERROR",
    "AMBIGUOUS_STARTED_ATTEMPT",
    "RECORD_PERSISTENCE_ERROR",
]


class EvaluationLineage(ExecutionModel):
    diagnosis_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    diagnosis_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    provider_invocation_hashes: list[str]
    candidate_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    verification_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    public_verification_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class EvaluationProviderPolicy(ExecutionModel):
    """Controller-owned provider policy committed before a scheduled E-mode call."""

    schema_version: Literal[1] = 1
    provider: str = Field(min_length=1)
    configured_model: str = Field(min_length=1)
    allowed_response_models: list[str] = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    store_false_required: Literal[True] = True

    def content(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content()).hexdigest()


class PublicEvaluationRecord(ExecutionModel):
    """The public projection of an evaluation result; it contains no hidden truth."""

    record_id: str
    lineage: EvaluationLineage
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

    def blind(self) -> dict[str, object]:
        # Export only the typed diagnosis contract. Malformed legacy payloads
        # must not smuggle model/config/truth fields through this dictionary.
        try:
            diagnosis = DiagnosisResult.model_validate(self.diagnosis).model_dump(mode="json")
        except ValidationError:
            diagnosis = {}
        return {
            "blind_id": hashlib.sha256(f"blind-v1:{self.record_id}".encode()).hexdigest(),
            "diagnosis": diagnosis,
            "evidence_hash": self.evidence_hash,
        }


class EvaluationRecord(PublicEvaluationRecord):
    """Evaluator-private result that may carry truth-derived scoring fields."""

    should_be_inconclusive: bool | None = None
    score: Score | None = None
    evaluator_labels: EvaluationLabels | None = None
    patch_compile_passed: bool | None = None
    private_holdout_passed: bool | None = None

    def public(self) -> PublicEvaluationRecord:
        public_checks = {
            "build",
            "runtime",
            "public_oracle",
            "memcheck",
            "racecheck",
            "initcheck",
            "synccheck",
        }
        fields = {name: getattr(self, name) for name in PublicEvaluationRecord.model_fields}
        fields["executed_checks"] = {
            key: value
            for key, value in self.executed_checks.items()
            if key in public_checks
            or (
                key.startswith("verification/")
                and key.removeprefix("verification/") in public_checks
            )
        }
        return PublicEvaluationRecord.model_validate(fields)


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


class EvaluationAttempt(ExecutionModel):
    """A durable pre-execution reservation for exactly one scheduled ordinal."""

    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    ordinal: int = Field(ge=0)
    schedule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    idempotency_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    reserved_cost_usd: float = Field(ge=0)


class EvaluationUnitBinding(ExecutionModel):
    """Frozen schedule identity persisted before diagnosis execution starts."""

    schema_version: Literal[1] = 1
    evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    ordinal: int = Field(ge=0)
    schedule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    idempotency_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    reserved_cost_usd: float = Field(ge=0)
    case_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    mode: EvaluationMode
    repeat: int = Field(ge=0)


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
    records: list[PublicEvaluationRecord]
    stopped_reason: StoppedReason | None = None


PersistedOrReturnedRecord = PublicEvaluationRecord | EvaluationRecord


class EvaluationRunner:
    """Own one immutable, resumable evaluation batch in the public ``RunStore``."""

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
        binding: RunBinding,
        max_cost_usd: float | None,
        max_unit_cost_usd: float | None,
        random_seed: int = 20260915,
    ) -> None:
        if store.visibility != "public":
            raise ValueError("evaluation requires a public RunStore")
        if (
            binding.purpose != "evaluation"
            or binding.repository.commit != commit
            or binding.prompt_version != prompt_version
            or binding.toolchain_lock_hash != toolchain_hash
            or binding.model_config_hash != model_config_hash
        ):
            raise ValueError("evaluation schedule differs from its immutable run binding")
        self.store = store
        self.binding = binding
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
        run = self.store.create_run("evaluation", binding=self.binding)
        self.store.transition(run.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)
        self._put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode())
        return self._execute(run.id, schedule, [], {})

    def resume(
        self, run_id: str, mode: EvaluationSelection, split: EvaluationSplit, repeats: int
    ) -> EvaluationManifest:
        run = self.store.load(run_id)
        if run.kind != "evaluation" or run.status != RunStatus.RUNNING:
            raise ValueError("only a running evaluation run may be resumed")
        if run.binding != self.binding:
            raise ValueError("evaluation run binding does not match controller")
        expected = self._schedule(mode, split, repeats)
        persisted = EvaluationSchedule.model_validate_json(
            self.store.read(self._one_artifact(run_id, "evaluation/schedule.json"))
        )
        if self._schedule_hash(persisted) != self._schedule_hash(expected) or persisted != expected:
            raise ValueError("evaluation schedule or bindings do not match")
        attempts = self._attempts(run_id, persisted)
        records = self._records(run_id, persisted, attempts)
        return self._execute(run_id, persisted, records, attempts)

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

    def _attempt(
        self, run_id: str, schedule: EvaluationSchedule, item: EvaluationScheduleItem
    ) -> EvaluationAttempt:
        if schedule.bindings.max_unit_cost_usd is None:
            raise ValueError("evaluation attempt requires a unit cost reservation")
        schedule_hash = self._schedule_hash(schedule)
        key = hashlib.sha256(f"{run_id}:{schedule_hash}:{item.ordinal}".encode()).hexdigest()
        return EvaluationAttempt(
            run_id=run_id,
            ordinal=item.ordinal,
            schedule_hash=schedule_hash,
            idempotency_key=key,
            reserved_cost_usd=schedule.bindings.max_unit_cost_usd,
        )

    def _execute(
        self,
        run_id: str,
        schedule: EvaluationSchedule,
        records: list[PersistedOrReturnedRecord],
        attempts: dict[int, EvaluationAttempt],
    ) -> EvaluationManifest:
        if schedule.bindings.max_cost_usd is None or schedule.bindings.max_unit_cost_usd is None:
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

        incomplete_attempts = [
            attempt for ordinal, attempt in attempts.items() if ordinal >= len(records)
        ]
        if incomplete_attempts:
            spent += sum(attempt.reserved_cost_usd for attempt in incomplete_attempts)
            return self._terminal(
                run_id, schedule, records, "AMBIGUOUS_STARTED_ATTEMPT", RunStatus.FAILED
            )

        for item in schedule.items[len(records) :]:
            if spent + schedule.bindings.max_unit_cost_usd > schedule.bindings.max_cost_usd:
                return self._terminal(
                    run_id,
                    schedule,
                    records,
                    "COST_CAP_RESERVATION_REQUIRED",
                    RunStatus.COMPLETED,
                )
            attempt = self._attempt(run_id, schedule, item)
            self._put(
                run_id,
                f"evaluation/attempts/{item.ordinal}.json",
                attempt.model_dump_json().encode(),
            )
            attempts[item.ordinal] = attempt
            try:
                owner = getattr(self.execute, "__self__", None)
                if owner is None or not hasattr(owner, "execute_scheduled"):
                    raise ValueError("evaluation requires a schedule-bound native executor")
                record = owner.execute_scheduled(item, attempt)
                self._validate_record(record, item, attempt)
            except Exception:
                return self._terminal(
                    run_id, schedule, records, "EXECUTION_ERROR", RunStatus.FAILED
                )
            try:
                self._put(
                    run_id,
                    f"evaluation/records/{item.ordinal}.json",
                    record.public().model_dump_json().encode(),
                )
            except Exception:
                return self._terminal(
                    run_id, schedule, records, "RECORD_PERSISTENCE_ERROR", RunStatus.FAILED
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
        records: list[PersistedOrReturnedRecord],
        stopped_reason: StoppedReason | None,
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
            records=[self._public(record) for record in records],
            stopped_reason=stopped_reason,
        )
        self._put(run_id, "evaluation/manifest.json", manifest.model_dump_json().encode())
        self.store.transition(run_id, status, None)
        return manifest

    @staticmethod
    def _public(record: PersistedOrReturnedRecord) -> PublicEvaluationRecord:
        return record.public() if isinstance(record, EvaluationRecord) else record

    def _records(
        self,
        run_id: str,
        schedule: EvaluationSchedule,
        attempts: dict[int, EvaluationAttempt],
    ) -> list[PublicEvaluationRecord]:
        records: dict[int, PublicEvaluationRecord] = {}
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
            if ordinal not in attempts:
                raise ValueError("completed evaluation record has no durable attempt")
            record = PublicEvaluationRecord.model_validate_json(self.store.read(ref))
            self._validate_record(record, schedule.items[ordinal], attempts[ordinal])
            records[ordinal] = record
        if set(records) != set(range(len(records))):
            raise ValueError("evaluation record ordinals have a gap")
        return [records[ordinal] for ordinal in range(len(records))]

    def _attempts(self, run_id: str, schedule: EvaluationSchedule) -> dict[int, EvaluationAttempt]:
        attempts: dict[int, EvaluationAttempt] = {}
        for ref in self.store.load(run_id).artifact_refs:
            if not ref.name.startswith("evaluation/attempts/"):
                continue
            match = re.fullmatch(r"evaluation/attempts/([0-9]+)\.json", ref.name)
            if match is None:
                raise ValueError("invalid evaluation attempt artifact name")
            ordinal = int(match.group(1))
            if ref.name != f"evaluation/attempts/{ordinal}.json" or ordinal in attempts:
                raise ValueError("duplicate evaluation attempt ordinal")
            if ordinal >= len(schedule.items):
                raise ValueError("evaluation attempt ordinal is out of range")
            attempt = EvaluationAttempt.model_validate_json(self.store.read(ref))
            expected = self._attempt(run_id, schedule, schedule.items[ordinal])
            if attempt != expected:
                raise ValueError("evaluation attempt does not match scheduled unit")
            attempts[ordinal] = attempt
        return attempts

    def _validate_record(
        self,
        record: PersistedOrReturnedRecord,
        item: EvaluationScheduleItem,
        attempt: EvaluationAttempt,
    ) -> None:
        if (record.case_id, record.template_id, record.mode, record.repeat) != (
            item.case_id,
            item.template_id,
            item.mode,
            item.repeat,
        ):
            raise ValueError("evaluation record does not match scheduled unit")
        from gpu_agent.benchmark.executor import validate_evaluation_record

        validate_evaluation_record(self.store, record, item, attempt, self.binding)

    def _one_artifact(self, run_id: str, name: str) -> ArtifactRef:
        refs = [ref for ref in self.store.load(run_id).artifact_refs if ref.name == name]
        if len(refs) != 1:
            raise ValueError(f"expected exactly one {name} artifact")
        return refs[0]

    def _put(self, run_id: str, name: str, content: bytes) -> None:
        self.store.put(run_id, name, content, self.store.visibility)
