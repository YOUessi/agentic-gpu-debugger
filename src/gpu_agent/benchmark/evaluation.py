"""Serial, cost-capped evaluation records and durable public batch schedules."""

import fcntl
import hashlib
import json
import os
import random
import re
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from pydantic import Field, ValidationError

from gpu_agent.agent.models import DiagnosisResult
from gpu_agent.benchmark.metrics import EvaluationLabels, Score
from gpu_agent.contracts import ArtifactRef, RunBinding, RunStatus
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.store import RunStore, reject_symlinks

if TYPE_CHECKING:
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.benchmark.holdout import HoldoutBatch, HoldoutController
    from gpu_agent.benchmark.schedule_authority import ScheduleCommitClient

_CLAIM_GUARD = threading.Lock()
_CLAIM_LOCKS: dict[str, threading.Lock] = {}

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
    corpus_cutoff: int = Field(ge=1)
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
    endpoint_host: str = Field(min_length=1)
    configured_model: str = Field(min_length=1)
    allowed_response_models: list[str] = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    pricing_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    store_false_required: Literal[True] = True

    def content(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content()).hexdigest()


class PricingAttestation(ExecutionModel):
    """Controller-owned pricing evidence; only a private test producer exists today."""

    schema_version: Literal[1] = 1
    source: Literal["TEST_ONLY"]
    provider: str
    model: str
    repository_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)

    @classmethod
    def _for_test(
        cls, provider: str, model: str, commit: str, model_config_hash: str
    ) -> "PricingAttestation":
        return cls(
            source="TEST_ONLY",
            provider=provider,
            model=model,
            repository_commit=commit,
            model_config_hash=model_config_hash,
            input_usd_per_million=1,
            output_usd_per_million=1,
        )

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_million + output_tokens * self.output_usd_per_million
        ) / 1_000_000

    @property
    def rate_card_hash(self) -> str:
        content = json.dumps(
            {
                "source": self.source,
                "provider": self.provider,
                "model": self.model,
                "input_usd_per_million": self.input_usd_per_million,
                "output_usd_per_million": self.output_usd_per_million,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(content).hexdigest()


class PublicEvaluationRecord(ExecutionModel):
    """The public projection of an evaluation result; it contains no hidden truth."""

    record_id: str
    corpus_cutoff: int = Field(ge=1)
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


class HoldoutScheduleProof(ExecutionModel):
    public_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    aliases_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)


class EvaluationScheduleItem(ExecutionModel):
    ordinal: int = Field(ge=0)
    case_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    mode: EvaluationMode
    repeat: int = Field(ge=0)
    split: EvaluationSplit
    holdout_proof: HoldoutScheduleProof | None = None


class EvaluationSchedule(ExecutionModel):
    schema_version: Literal[1] = 1
    selection: EvaluationSelection
    modes: list[EvaluationMode]
    split: EvaluationSplit
    repeats: int = Field(ge=3)
    random_seed: int
    corpus_cutoff: int = Field(ge=1)
    bindings: EvaluationBindings
    items: list[EvaluationScheduleItem]
    holdout_proof: HoldoutScheduleProof | None = None


class EvaluationAttempt(ExecutionModel):
    """A durable pre-execution reservation for exactly one scheduled ordinal."""

    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    ordinal: int = Field(ge=0)
    schedule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)
    idempotency_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    reserved_cost_usd: float = Field(ge=0)


class EvaluationExecutionClaim(ExecutionModel):
    """Controller-created, durable proof that a scheduled unit began execution."""

    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    ordinal: int = Field(ge=0)
    schedule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)
    attempt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class EvaluationUnitBinding(ExecutionModel):
    """Frozen schedule identity persisted before diagnosis execution starts."""

    schema_version: Literal[1] = 1
    evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    ordinal: int = Field(ge=0)
    schedule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)
    idempotency_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    reserved_cost_usd: float = Field(ge=0)
    case_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    mode: EvaluationMode
    repeat: int = Field(ge=0)
    split: EvaluationSplit
    holdout_proof: HoldoutScheduleProof | None = None


class EvaluationManifest(ExecutionModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    prompt_version: str = Field(min_length=1)
    toolchain_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    schedule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)
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
        executor: "EvaluationExecutor",
        *,
        commit: str,
        prompt_version: str,
        toolchain_hash: str,
        model_config_hash: str,
        binding: RunBinding,
        max_cost_usd: float | None,
        max_unit_cost_usd: float | None,
        random_seed: int = 20260915,
        holdout_controller: "HoldoutController | None" = None,
        holdout_batch: "HoldoutBatch | None" = None,
        schedule_client: "ScheduleCommitClient | None" = None,
    ) -> None:
        if store.visibility != "public":
            raise ValueError("evaluation requires a public RunStore")
        from gpu_agent.benchmark.executor import EvaluationExecutor

        if type(executor) is not EvaluationExecutor:
            raise ValueError("evaluation requires the native executor implementation")
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
        self.executor = executor
        self.schedule_client = schedule_client
        self.bindings = EvaluationBindings(
            commit=commit,
            prompt_version=prompt_version,
            toolchain_hash=toolchain_hash,
            model_config_hash=model_config_hash,
            max_cost_usd=max_cost_usd,
            max_unit_cost_usd=max_unit_cost_usd,
        )
        self.random_seed = random_seed
        if (holdout_controller is None) != (holdout_batch is None):
            raise ValueError("holdout controller and batch must be configured together")
        self.holdout_controller, self.holdout_batch = holdout_controller, holdout_batch

    def run(
        self, mode: EvaluationSelection, split: EvaluationSplit, repeats: int
    ) -> EvaluationManifest:
        from gpu_agent.benchmark.schedule_authority import (
            activate_schedule,
            bind_reserved_schedule,
            reserve_evaluation_cutoff,
            seal_schedule,
        )

        if self.schedule_client is None:
            raise ValueError("external schedule authority is required")
        run = RunStore.create_run(self.store, "evaluation", binding=self.binding)
        modes: list[EvaluationMode] = ["A", "B", "C", "D", "E"] if mode == "all" else [mode]
        reservation = reserve_evaluation_cutoff(
            self.executor._corpus_family,
            self.store,
            run.id,
            self.binding,
            selection=mode,
            modes=modes,
            split=split,
            repeats=repeats,
            random_seed=self.random_seed,
            max_cost_usd=self.bindings.max_cost_usd,
            max_unit_cost_usd=self.bindings.max_unit_cost_usd,
        )
        schedule = EvaluationRunner._schedule(
            self, mode, split, repeats, corpus_cutoff=reservation.corpus_cutoff
        )
        bind_reserved_schedule(
            self.executor._corpus_family, self.store, run.id, schedule, self.binding
        )
        EvaluationRunner._put(
            self, run.id, "evaluation/schedule.json", schedule.model_dump_json().encode()
        )
        seal_schedule(
            self.executor._corpus_family,
            self.store,
            run.id,
            schedule,
            self.binding,
            self.schedule_client,
            self.executor._schedule_verifier,
        )
        activate_schedule(self.store, self.executor._schedule_verifier, run.id)
        return EvaluationRunner._execute(self, run.id, schedule, [], {})

    def resume(
        self, run_id: str, mode: EvaluationSelection, split: EvaluationSplit, repeats: int
    ) -> EvaluationManifest:
        with EvaluationRunner._claim(self, run_id):
            return EvaluationRunner._resume_claimed(self, run_id, mode, split, repeats)

    def _resume_claimed(
        self, run_id: str, mode: EvaluationSelection, split: EvaluationSplit, repeats: int
    ) -> EvaluationManifest:
        run = RunStore.load(self.store, run_id)
        if run.kind != "evaluation" or run.status not in {
            RunStatus.QUEUED,
            RunStatus.RUNNING,
        }:
            raise ValueError("only an activatable evaluation run may be resumed")
        if run.binding != self.binding:
            raise ValueError("evaluation run binding does not match controller")
        reservation = self.executor._corpus_family.ledger.evaluation_cutoff_reservation(run_id)
        schedule_refs = [ref for ref in run.artifact_refs if ref.name == "evaluation/schedule.json"]
        if len(schedule_refs) > 1:
            raise ValueError("evaluation schedule is ambiguous")
        if schedule_refs:
            persisted = EvaluationSchedule.model_validate_json(
                RunStore.read(self.store, schedule_refs[0])
            )
        elif run.status == RunStatus.QUEUED:
            persisted = EvaluationRunner._schedule(
                self, mode, split, repeats, corpus_cutoff=reservation.corpus_cutoff
            )
            from gpu_agent.benchmark.schedule_authority import bind_reserved_schedule

            bind_reserved_schedule(
                self.executor._corpus_family,
                self.store,
                run.id,
                persisted,
                self.binding,
            )
            EvaluationRunner._put(
                self,
                run_id,
                "evaluation/schedule.json",
                persisted.model_dump_json().encode(),
            )
            run = RunStore.load(self.store, run_id)
        else:
            raise ValueError("active evaluation schedule is unavailable")
        expected = EvaluationRunner._schedule(
            self, mode, split, repeats, corpus_cutoff=persisted.corpus_cutoff
        )
        if (
            EvaluationRunner._schedule_hash(persisted) != EvaluationRunner._schedule_hash(expected)
            or persisted != expected
        ):
            raise ValueError("evaluation schedule or bindings do not match")
        from gpu_agent.benchmark.schedule_authority import bind_reserved_schedule

        bind_reserved_schedule(
            self.executor._corpus_family, self.store, run_id, persisted, self.binding
        )
        from gpu_agent.benchmark.schedule_authority import (
            EvaluationScheduleVerifier,
            activate_schedule,
            seal_schedule,
        )

        if run.status == RunStatus.QUEUED:
            receipts = [
                ref for ref in run.artifact_refs if ref.name == "evaluation/schedule-receipt.json"
            ]
            if not receipts:
                seal_schedule(
                    self.executor._corpus_family,
                    self.store,
                    run_id,
                    persisted,
                    self.binding,
                    self.schedule_client,
                    self.executor._schedule_verifier,
                )
            elif len(receipts) == 1:
                EvaluationScheduleVerifier.verify(self.executor._schedule_verifier, run_id)
            else:
                raise ValueError("evaluation schedule receipt is ambiguous")
            activate_schedule(self.store, self.executor._schedule_verifier, run_id)
        EvaluationScheduleVerifier.verify(self.executor._schedule_verifier, run_id)
        attempts = EvaluationRunner._attempts(self, run_id, persisted)
        records = EvaluationRunner._records(self, run_id, persisted, attempts)
        return EvaluationRunner._execute(self, run_id, persisted, records, attempts)

    @contextmanager
    def _claim(self, run_id: str) -> Iterator[None]:
        """Serialize recovery across threads and processes for one immutable batch."""
        with _CLAIM_GUARD:
            local = _CLAIM_LOCKS.setdefault(str(self.store.root / run_id), threading.Lock())
        lock_path = self.store.root / f".evaluation-{run_id}.lock"
        reject_symlinks(lock_path)
        with local:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_mode & 0o077:
                    raise ValueError("evaluation claim lock is unsafe")
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)

    def _schedule(
        self,
        mode: EvaluationSelection,
        split: EvaluationSplit,
        repeats: int,
        *,
        corpus_cutoff: int | None = None,
    ) -> EvaluationSchedule:
        if repeats < 3:
            raise ValueError("evaluation requires at least three repeats")
        current_cutoff = len(self.executor._corpus_family.ledger.committed_through())
        cutoff = current_cutoff if corpus_cutoff is None else corpus_cutoff
        self.executor._corpus_family.ledger.committed_through(cutoff)
        if split == "holdout":
            if self.holdout_controller is None or self.holdout_batch is None:
                raise ValueError("holdout alias proof is required")
            holdout_proof = self.holdout_controller.validate_batch(self.holdout_batch)
            if self.holdout_batch.corpus_cutoff != cutoff or holdout_proof.corpus_cutoff != cutoff:
                raise ValueError("holdout alias cutoff differs from evaluation schedule")
            case_ids = {alias: alias for alias in self.holdout_batch.aliases}
        else:
            if self.holdout_controller is not None:
                raise ValueError("holdout alias proof cannot bind a development schedule")
            holdout_proof = None
            from gpu_agent.benchmark.executor import registered_cases

            cases = registered_cases(
                self.executor.corpus,
                self.binding,
                self.executor._corpus_family,
                cutoff=cutoff,
            )
            case_ids = {case.id: case.template_id for case in cases.values()}
            if not case_ids:
                raise ValueError("committed corpus transaction universe is empty")
        modes: list[EvaluationMode] = ["A", "B", "C", "D", "E"] if mode == "all" else [mode]
        units = [
            (case_id, template_id, item_mode, repeat)
            for repeat in range(repeats)
            for case_id, template_id in case_ids.items()
            for item_mode in modes
        ]
        random.Random(self.random_seed).shuffle(units)
        return EvaluationSchedule(
            selection=mode,
            modes=modes,
            split=split,
            repeats=repeats,
            random_seed=self.random_seed,
            corpus_cutoff=cutoff,
            bindings=self.bindings,
            items=[
                EvaluationScheduleItem(
                    ordinal=ordinal,
                    case_id=case_id,
                    template_id=template_id,
                    mode=item_mode,
                    repeat=repeat,
                    split=split,
                    holdout_proof=holdout_proof,
                )
                for ordinal, (case_id, template_id, item_mode, repeat) in enumerate(units)
            ],
            holdout_proof=holdout_proof,
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
        schedule_hash = EvaluationRunner._schedule_hash(schedule)
        key = hashlib.sha256(f"{run_id}:{schedule_hash}:{item.ordinal}".encode()).hexdigest()
        return EvaluationAttempt(
            run_id=run_id,
            ordinal=item.ordinal,
            schedule_hash=schedule_hash,
            corpus_cutoff=schedule.corpus_cutoff,
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
            return EvaluationRunner._terminal(
                self, run_id, schedule, records, "COST_CAP_REQUIRED", RunStatus.COMPLETED
            )

        spent = 0.0
        for record in records:
            if record.cost_usd is None:
                return EvaluationRunner._terminal(
                    self, run_id, schedule, records, "COST_UNKNOWN", RunStatus.COMPLETED
                )
            if record.cost_usd > schedule.bindings.max_unit_cost_usd:
                return EvaluationRunner._terminal(
                    self, run_id, schedule, records, "UNIT_COST_CEILING_EXCEEDED", RunStatus.FAILED
                )
            spent += record.cost_usd

        incomplete_attempts = [
            attempt for ordinal, attempt in attempts.items() if ordinal >= len(records)
        ]
        if incomplete_attempts:
            spent += sum(attempt.reserved_cost_usd for attempt in incomplete_attempts)
            return EvaluationRunner._terminal(
                self, run_id, schedule, records, "AMBIGUOUS_STARTED_ATTEMPT", RunStatus.FAILED
            )

        for item in schedule.items[len(records) :]:
            if spent + schedule.bindings.max_unit_cost_usd > schedule.bindings.max_cost_usd:
                return EvaluationRunner._terminal(
                    self,
                    run_id,
                    schedule,
                    records,
                    "COST_CAP_RESERVATION_REQUIRED",
                    RunStatus.COMPLETED,
                )
            attempt = EvaluationRunner._attempt(self, run_id, schedule, item)
            EvaluationRunner._put(
                self,
                run_id,
                f"evaluation/attempts/{item.ordinal}.json",
                attempt.model_dump_json().encode(),
            )
            attempts[item.ordinal] = attempt
            try:
                from gpu_agent.benchmark.executor import EvaluationExecutor

                record = EvaluationExecutor.execute_scheduled(self.executor, run_id, item.ordinal)
                EvaluationRunner._validate_record(self, record, item, attempt)
            except Exception:
                return EvaluationRunner._terminal(
                    self, run_id, schedule, records, "EXECUTION_ERROR", RunStatus.FAILED
                )
            try:
                EvaluationRunner._put(
                    self,
                    run_id,
                    f"evaluation/records/{item.ordinal}.json",
                    record.public().model_dump_json().encode(),
                )
            except Exception:
                return EvaluationRunner._terminal(
                    self, run_id, schedule, records, "RECORD_PERSISTENCE_ERROR", RunStatus.FAILED
                )
            attempts = EvaluationRunner._attempts(self, run_id, schedule)
            records = EvaluationRunner._records(self, run_id, schedule, attempts)
            if record.cost_usd is None:
                return EvaluationRunner._terminal(
                    self, run_id, schedule, records, "COST_UNKNOWN", RunStatus.COMPLETED
                )
            if record.cost_usd > schedule.bindings.max_unit_cost_usd:
                return EvaluationRunner._terminal(
                    self, run_id, schedule, records, "UNIT_COST_CEILING_EXCEEDED", RunStatus.FAILED
                )
            spent += record.cost_usd
        return EvaluationRunner._terminal(
            self, run_id, schedule, records, None, RunStatus.COMPLETED
        )

    def _terminal(
        self,
        run_id: str,
        schedule: EvaluationSchedule,
        records: list[PersistedOrReturnedRecord],
        stopped_reason: StoppedReason | None,
        status: RunStatus,
    ) -> EvaluationManifest:
        attempts = EvaluationRunner._attempts(self, run_id, schedule)
        persisted_records = EvaluationRunner._records(self, run_id, schedule, attempts)
        manifest = EvaluationManifest(
            run_id=run_id,
            commit=schedule.bindings.commit,
            prompt_version=schedule.bindings.prompt_version,
            toolchain_hash=schedule.bindings.toolchain_hash,
            model_config_hash=schedule.bindings.model_config_hash,
            schedule_hash=EvaluationRunner._schedule_hash(schedule),
            corpus_cutoff=schedule.corpus_cutoff,
            expected_units=len(schedule.items),
            executed_units=len(persisted_records),
            modes=schedule.modes,
            split=schedule.split,
            repeats=schedule.repeats,
            random_seed=schedule.random_seed,
            records=[EvaluationRunner._public(record) for record in persisted_records],
            stopped_reason=stopped_reason,
        )
        EvaluationRunner._put(
            self, run_id, "evaluation/manifest.json", manifest.model_dump_json().encode()
        )
        RunStore.transition(self.store, run_id, status, None)
        persisted = EvaluationManifest.model_validate_json(
            RunStore.read(
                self.store,
                EvaluationRunner._one_artifact(self, run_id, "evaluation/manifest.json"),
            )
        )
        if persisted != manifest:
            raise ValueError("persisted evaluation manifest differs from native records")
        return persisted

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
        for ref in RunStore.load(self.store, run_id).artifact_refs:
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
            record = PublicEvaluationRecord.model_validate_json(RunStore.read(self.store, ref))
            EvaluationRunner._validate_record(
                self, record, schedule.items[ordinal], attempts[ordinal]
            )
            records[ordinal] = record
        if set(records) != set(range(len(records))):
            raise ValueError("evaluation record ordinals have a gap")
        return [records[ordinal] for ordinal in range(len(records))]

    def _attempts(self, run_id: str, schedule: EvaluationSchedule) -> dict[int, EvaluationAttempt]:
        attempts: dict[int, EvaluationAttempt] = {}
        for ref in RunStore.load(self.store, run_id).artifact_refs:
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
            attempt = EvaluationAttempt.model_validate_json(RunStore.read(self.store, ref))
            expected = EvaluationRunner._attempt(self, run_id, schedule, schedule.items[ordinal])
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
        if (
            record.corpus_cutoff != attempt.corpus_cutoff
            or record.lineage.corpus_cutoff != attempt.corpus_cutoff
        ):
            raise ValueError("evaluation record corpus cutoff differs from scheduled unit")
        from gpu_agent.benchmark.executor import EvaluationExecutor

        EvaluationExecutor.validate_scheduled_record(self.executor, record, item, attempt)

    def _one_artifact(self, run_id: str, name: str) -> ArtifactRef:
        refs = [ref for ref in RunStore.load(self.store, run_id).artifact_refs if ref.name == name]
        if len(refs) != 1:
            raise ValueError(f"expected exactly one {name} artifact")
        return refs[0]

    def _put(self, run_id: str, name: str, content: bytes) -> None:
        RunStore.put(self.store, run_id, name, content, self.store.visibility)
