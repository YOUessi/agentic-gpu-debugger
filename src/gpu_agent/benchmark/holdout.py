"""Evaluator-owned holdout aliases and private score bindings."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

from pydantic import Field

from gpu_agent.benchmark.evaluation import (
    EvaluationAttempt,
    EvaluationExecutionClaim,
    EvaluationManifest,
    EvaluationRecord,
    EvaluationSchedule,
    EvaluationScheduleItem,
    EvaluationUnitBinding,
    HoldoutEvaluationLineage,
    HoldoutScheduleProof,
    NativeEvaluationLineage,
    PublicEvaluationRecord,
)
from gpu_agent.benchmark.metrics import EvaluationLabels, Score
from gpu_agent.contracts import (
    ArtifactRef,
    CurrentPhase,
    ExternalRunOrigin,
    RunBinding,
    RunManifest,
    RunStatus,
)
from gpu_agent.execution.models import ExecutionModel, SanitizerTool
from gpu_agent.store import (
    EvaluationRunLease,
    RunDirectorySetLease,
    RunStore,
    read_regular,
    reject_symlinks,
)

_SCORE_GUARD = threading.Lock()
_SCORE_LOCKS: dict[str, threading.Lock] = {}
_EXECUTION_GUARD = threading.Lock()
_EXECUTION_LOCKS: dict[str, threading.Lock] = {}

if TYPE_CHECKING:
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier
    from gpu_agent.service import ApplicationService


class HoldoutBatch(ExecutionModel):
    public_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    evaluator_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    aliases: list[str]
    public_alias_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)


class HoldoutExecutionBinding(ExecutionModel):
    schema_version: Literal[1] = 1
    public_evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    alias_mapping_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    ordinal: int = Field(ge=0)
    schedule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    attempt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)
    alias: str = Field(pattern=r"^[a-f0-9]{64}$")
    private_case_id: str
    private_template_id: str
    diagnosis_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    native_record_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    public_record_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class PreparedHoldoutExecution(ExecutionModel):
    execution_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    diagnosis_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    binding: HoldoutExecutionBinding
    evaluation_unit: EvaluationUnitBinding


class EvaluatorRecordBinding(ExecutionModel):
    evaluator_score_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    public_evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    public_record_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    public_record_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    private_case_id: str = Field(min_length=1)
    private_template_id: str = Field(min_length=1)
    private_score_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)


class PreparedHoldoutScore(ExecutionModel):
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    binding: EvaluatorRecordBinding
    private_score_content: bytes


@dataclass(frozen=True, slots=True)
class ResolvedHoldoutEvaluationRecord:
    """Private resolution of one blind public ordinal to its exact transaction."""

    public_record: PublicEvaluationRecord
    native_record: EvaluationRecord
    execution_binding: HoldoutExecutionBinding


@dataclass(frozen=True, slots=True)
class ValidatedHoldoutEvaluation:
    evaluation_run_id: str
    schedule: EvaluationSchedule
    schedule_hash: str
    records: tuple[PublicEvaluationRecord, ...]
    record_refs: tuple[ArtifactRef, ...]
    resolved_records: tuple[ResolvedHoldoutEvaluationRecord, ...] = ()


class _PrivateIdentity(ExecutionModel):
    alias: str = Field(pattern=r"^[a-f0-9]{64}$")
    private_case_id: str = Field(min_length=1)
    private_template_id: str = Field(min_length=1)


class _PrivateAliasMap(ExecutionModel):
    schema_version: int = 2
    corpus_cutoff: int = Field(ge=1)
    nonce_hex: str = Field(pattern=r"^[a-f0-9]{64}$")
    identities: list[_PrivateIdentity]


class _MetricLoadContext(NamedTuple):
    batch: HoldoutBatch
    evaluation: ValidatedHoldoutEvaluation
    identities: dict[str, _PrivateIdentity]


class _PrivateScore(ExecutionModel):
    schema_version: int = 2
    corpus_cutoff: int = Field(ge=1)
    public_record_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    labels: EvaluationLabels
    score: Score
    should_be_inconclusive: bool
    private_holdout_passed: bool

    def content(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()


class HoldoutController:
    """Keep private identities and judgments in an evaluator-only RunStore."""

    def __init__(
        self,
        public: RunStore,
        evaluator: RunStore,
        *,
        binding: RunBinding,
        _schedule_verifier: EvaluationScheduleVerifier | None = None,
        _schedule_family: CorpusFamily | None = None,
    ) -> None:
        if (
            public.visibility != "public"
            or evaluator.visibility != "evaluator"
            or binding.purpose != "evaluation"
        ):
            raise ValueError("holdout controller requires bound split stores")
        self.public, self.evaluator, self.binding = public, evaluator, binding
        from pathlib import Path

        from gpu_agent.benchmark.ledger import CorpusFamily

        if _schedule_family is None:
            family_root = os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT")
            if family_root is None:
                raise ValueError("trusted corpus family configuration is required")
            _schedule_family = CorpusFamily.open(Path(family_root))
        self._schedule_family = _schedule_family
        if _schedule_verifier is None:
            from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

            _schedule_verifier = EvaluationScheduleVerifier.for_family(
                self._schedule_family, public
            )
        self._schedule_verifier = _schedule_verifier

    def prepare(self) -> HoldoutBatch:
        from gpu_agent.benchmark.executor import registered_cases

        cutoff = len(self._schedule_family.ledger.committed_through())
        cases = registered_cases(self.evaluator, self.binding, self._schedule_family, cutoff=cutoff)
        private_identities = sorted((case.id, case.template_id) for case in cases.values())
        if not private_identities or len(set(private_identities)) != len(private_identities):
            raise ValueError("holdout preparation input is invalid")
        public_run = self.public.create_run("holdout_aliases", binding=self.binding)
        self.public.transition(public_run.id, RunStatus.RUNNING, "PREPARING")
        evaluator_run = self.evaluator.create_run(
            "holdout_alias_mapping",
            binding=self.binding,
            external_origin=ExternalRunOrigin(run_id=public_run.id, visibility="public"),
        )
        self.evaluator.transition(evaluator_run.id, RunStatus.RUNNING, "PREPARING")
        nonce = secrets.token_bytes(32)
        identities = [
            _PrivateIdentity(
                alias=hmac.new(
                    nonce,
                    json.dumps([case_id, template_id], separators=(",", ":")).encode(),
                    hashlib.sha256,
                ).hexdigest(),
                private_case_id=case_id,
                private_template_id=template_id,
            )
            for case_id, template_id in private_identities
        ]
        if len({item.alias for item in identities}) != len(identities):
            raise ValueError("holdout preparation input is invalid")
        public_content = json.dumps(
            {
                "schema_version": 2,
                "corpus_cutoff": cutoff,
                "aliases": [item.alias for item in identities],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        public_ref = self.public.put(
            public_run.id, "holdout/aliases.json", public_content, "public"
        )
        mapping = _PrivateAliasMap(
            corpus_cutoff=cutoff, nonce_hex=nonce.hex(), identities=identities
        )
        self.evaluator.put(
            evaluator_run.id,
            "holdout/private-alias-map.json",
            mapping.model_dump_json().encode(),
            "evaluator",
        )
        self.evaluator.transition(evaluator_run.id, RunStatus.RUNNING, "FINALIZING")
        self.evaluator.transition(evaluator_run.id, RunStatus.COMPLETED, None)
        self.public.transition(public_run.id, RunStatus.RUNNING, "FINALIZING")
        self.public.transition(public_run.id, RunStatus.COMPLETED, None)
        return HoldoutBatch(
            public_run_id=public_run.id,
            evaluator_run_id=evaluator_run.id,
            aliases=[item.alias for item in identities],
            public_alias_hash=public_ref.sha256,
            corpus_cutoff=cutoff,
        )

    def validate_batch(self, batch: HoldoutBatch) -> HoldoutScheduleProof:
        public_run = self.public.load(batch.public_run_id)
        refs = [ref for ref in public_run.artifact_refs if ref.name == "holdout/aliases.json"]
        if (
            public_run.kind != "holdout_aliases"
            or public_run.status != RunStatus.COMPLETED
            or public_run.binding != self.binding
            or len(refs) != 1
            or refs[0].sha256 != batch.public_alias_hash
            or not batch.aliases
            or len(set(batch.aliases)) != len(batch.aliases)
            or any(not re.fullmatch(r"[a-f0-9]{64}", alias) for alias in batch.aliases)
        ):
            raise ValueError("holdout binding is invalid")
        public_payload = json.loads(self.public.read(refs[0]))
        if public_payload != {
            "schema_version": 2,
            "corpus_cutoff": batch.corpus_cutoff,
            "aliases": batch.aliases,
        }:
            raise ValueError("holdout binding is invalid")
        mapping_run = self.evaluator.load(batch.evaluator_run_id)
        mapping_refs = [
            ref for ref in mapping_run.artifact_refs if ref.name == "holdout/private-alias-map.json"
        ]
        if (
            mapping_run.kind != "holdout_alias_mapping"
            or mapping_run.status != RunStatus.COMPLETED
            or mapping_run.binding != self.binding
            or mapping_run.external_origin
            != ExternalRunOrigin(run_id=batch.public_run_id, visibility="public")
            or len(mapping_refs) != 1
        ):
            raise ValueError("holdout binding is invalid")
        mapping = _PrivateAliasMap.model_validate_json(self.evaluator.read(mapping_refs[0]))
        if mapping.corpus_cutoff != batch.corpus_cutoff:
            raise ValueError("holdout binding cutoff is invalid")
        nonce = bytes.fromhex(mapping.nonce_hex)
        recomputed = [
            hmac.new(
                nonce,
                json.dumps(
                    [item.private_case_id, item.private_template_id], separators=(",", ":")
                ).encode(),
                hashlib.sha256,
            ).hexdigest()
            for item in mapping.identities
        ]
        from gpu_agent.benchmark.executor import registered_cases

        cases = registered_cases(
            self.evaluator,
            self.binding,
            self._schedule_family,
            cutoff=batch.corpus_cutoff,
        )
        expected_private = sorted((case.id, case.template_id) for case in cases.values())
        observed_private = [
            (item.private_case_id, item.private_template_id) for item in mapping.identities
        ]
        if (
            recomputed != batch.aliases
            or [item.alias for item in mapping.identities] != recomputed
            or observed_private != expected_private
        ):
            raise ValueError("holdout binding is invalid")
        return HoldoutScheduleProof(
            public_run_id=batch.public_run_id,
            aliases_hash=refs[0].sha256,
            corpus_cutoff=batch.corpus_cutoff,
        )

    @staticmethod
    def _content_hash(value: ExecutionModel) -> str:
        content = json.dumps(
            value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(content).hexdigest()

    def _prepared_execution(
        self,
        batch: HoldoutBatch,
        evaluation_run_id: str,
        item: EvaluationScheduleItem,
        attempt: EvaluationAttempt,
    ) -> PreparedHoldoutExecution:
        proof = self.validate_batch(batch)
        if (
            evaluation_run_id != attempt.run_id
            or item.ordinal != attempt.ordinal
            or item.split != "holdout"
            or item.case_id != item.template_id
            or item.case_id not in batch.aliases
            or item.holdout_proof != proof
            or item.holdout_proof is None
            or attempt.corpus_cutoff != batch.corpus_cutoff
        ):
            raise ValueError("holdout execution request is invalid")
        identity = self._identity(batch, item.case_id)
        attempt_hash = self._content_hash(attempt)
        execution_run_id = hashlib.sha256(
            (
                "holdout-execution-v1:"
                f"{batch.evaluator_run_id}:{evaluation_run_id}:{attempt.schedule_hash}:"
                f"{item.ordinal}:{attempt_hash}"
            ).encode()
        ).hexdigest()[:32]
        diagnosis_run_id = hashlib.sha256(
            f"holdout-diagnosis-v1:{execution_run_id}".encode()
        ).hexdigest()[:32]
        binding = HoldoutExecutionBinding(
            public_evaluation_run_id=evaluation_run_id,
            alias_mapping_run_id=batch.evaluator_run_id,
            ordinal=item.ordinal,
            schedule_hash=attempt.schedule_hash,
            attempt_hash=attempt_hash,
            corpus_cutoff=batch.corpus_cutoff,
            alias=item.case_id,
            private_case_id=identity.private_case_id,
            private_template_id=identity.private_template_id,
            diagnosis_run_id=diagnosis_run_id,
        )
        unit = EvaluationUnitBinding(
            evaluation_run_id=evaluation_run_id,
            ordinal=item.ordinal,
            schedule_hash=attempt.schedule_hash,
            corpus_cutoff=attempt.corpus_cutoff,
            idempotency_key=attempt.idempotency_key,
            reserved_cost_usd=attempt.reserved_cost_usd,
            case_id=identity.private_case_id,
            template_id=identity.private_template_id,
            mode=item.mode,
            repeat=item.repeat,
            split="holdout",
            holdout_proof=proof,
        )
        return PreparedHoldoutExecution(
            execution_run_id=execution_run_id,
            diagnosis_run_id=diagnosis_run_id,
            binding=binding,
            evaluation_unit=unit,
        )

    def _validate_execution_reservation(
        self,
        prepared: PreparedHoldoutExecution,
        lease: RunDirectorySetLease | None = None,
    ) -> tuple[RunManifest, RunManifest]:
        expected_origin = ExternalRunOrigin(
            run_id=prepared.binding.public_evaluation_run_id, visibility="public"
        )
        execution = (
            lease.load_optional(prepared.execution_run_id)
            if lease is not None
            else self.evaluator.load(prepared.execution_run_id)
        )
        diagnosis = (
            lease.load_optional(prepared.diagnosis_run_id)
            if lease is not None
            else self.evaluator.load(prepared.diagnosis_run_id)
        )
        if execution is None or diagnosis is None:
            raise ValueError("holdout execution transaction is incomplete")
        unit_refs = [ref for ref in diagnosis.artifact_refs if ref.name == "evaluation/unit.json"]
        if (
            execution.kind != "holdout_execution"
            or execution.parent_run_id is not None
            or execution.binding != self.binding
            or execution.external_origin != expected_origin
            or execution.status not in {RunStatus.RUNNING, RunStatus.COMPLETED}
            or diagnosis.kind != "diagnosis"
            or diagnosis.parent_run_id != execution.id
            or diagnosis.binding != self.binding
            or diagnosis.external_origin != expected_origin
            or diagnosis.status not in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED}
            or len(unit_refs) != 1
            or EvaluationUnitBinding.model_validate_json(
                lease.read(unit_refs[0]) if lease is not None else self.evaluator.read(unit_refs[0])
            )
            != prepared.evaluation_unit
        ):
            raise ValueError("holdout execution transaction is invalid")
        return execution, diagnosis

    def _run_path_state(self, run_id: str) -> Literal["absent", "directory"]:
        path = self.evaluator.root / run_id
        try:
            info = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            return "absent"
        except OSError as exc:
            raise ValueError("holdout execution path is unavailable") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("holdout execution path is unsafe")
        reject_symlinks(path)
        return "directory"

    def _validate_execution_parent(
        self, prepared: PreparedHoldoutExecution, execution: RunManifest
    ) -> None:
        expected_origin = ExternalRunOrigin(
            run_id=prepared.binding.public_evaluation_run_id, visibility="public"
        )
        if (
            execution.kind != "holdout_execution"
            or execution.parent_run_id is not None
            or execution.binding != self.binding
            or execution.external_origin != expected_origin
            or execution.status not in {RunStatus.RUNNING, RunStatus.COMPLETED}
        ):
            raise ValueError("holdout execution transaction is invalid")
        if execution.status == RunStatus.RUNNING and (
            execution.current_phase != "PREPARING" or execution.artifact_refs
        ):
            raise ValueError("holdout execution transaction is incomplete")

    def reserve_execution(
        self,
        batch: HoldoutBatch,
        *,
        evaluation_run_id: str,
        item: EvaluationScheduleItem,
        attempt: EvaluationAttempt,
    ) -> PreparedHoldoutExecution:
        prepared = self._prepared_execution(batch, evaluation_run_id, item, attempt)
        with self._execution_claim(prepared.execution_run_id):
            with self.evaluator.run_directory_set_lease(
                (prepared.execution_run_id, prepared.diagnosis_run_id)
            ) as lease:
                execution = lease.load_optional(prepared.execution_run_id)
                diagnosis = lease.load_optional(prepared.diagnosis_run_id)
                if execution is None and diagnosis is not None:
                    raise ValueError("holdout execution transaction is invalid")
                if execution is None:
                    lease.validate()
                    self.evaluator.create_run(
                        "holdout_execution",
                        binding=self.binding,
                        external_origin=ExternalRunOrigin(
                            run_id=evaluation_run_id, visibility="public"
                        ),
                        _run_id=prepared.execution_run_id,
                    )
                    execution = lease.adopt(prepared.execution_run_id)
                    self.evaluator.transition(
                        prepared.execution_run_id, RunStatus.RUNNING, "PREPARING"
                    )
                    execution = lease.load_optional(prepared.execution_run_id)
                    if execution is None:
                        raise ValueError("holdout execution transaction is incomplete")
                else:
                    self._validate_execution_parent(prepared, execution)
                    if execution.status == RunStatus.COMPLETED:
                        if diagnosis is None:
                            raise ValueError("holdout execution transaction is incomplete")
                        self._recover_prepared_execution(prepared, lease=lease)
                        return prepared
                if diagnosis is None:
                    lease.validate()
                    self.evaluator.create_run(
                        "diagnosis",
                        parent_run_id=prepared.execution_run_id,
                        _run_id=prepared.diagnosis_run_id,
                    )
                    lease.adopt(prepared.diagnosis_run_id)
                    self.evaluator.put_if_absent_exact(
                        prepared.diagnosis_run_id,
                        "evaluation/unit.json",
                        prepared.evaluation_unit.model_dump_json().encode(),
                        "evaluator",
                    )
                self._validate_execution_reservation(prepared, lease)
                lease.validate()
                return prepared

    def _validate_public_reserved_authority(
        self,
        lease: EvaluationRunLease,
        prepared: PreparedHoldoutExecution,
    ) -> None:
        from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

        parent = lease.load()
        if (
            parent.kind != "evaluation"
            or parent.status != RunStatus.RUNNING
            or parent.binding != self.binding
        ):
            raise ValueError("holdout public execution authority is invalid")
        EvaluationScheduleVerifier._verify_leased(self._schedule_verifier, lease)
        schedule_ref = self._one_named_ref(parent, "evaluation/schedule.json")
        schedule_content = lease.read(schedule_ref)
        schedule = EvaluationSchedule.model_validate_json(schedule_content)
        schedule_hash = hashlib.sha256(
            json.dumps(
                schedule.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        ordinal = prepared.binding.ordinal
        if ordinal >= len(schedule.items):
            raise ValueError("holdout scheduled unit is invalid")
        attempt_ref = self._one_named_ref(parent, f"evaluation/attempts/{ordinal}.json")
        attempt_content = lease.read(attempt_ref)
        attempt = EvaluationAttempt.model_validate_json(attempt_content)
        claim_ref = self._one_named_ref(parent, f"evaluation/claims/{ordinal}.json")
        claim = EvaluationExecutionClaim.model_validate_json(lease.read(claim_ref))
        attempt_ordinals: list[int] = []
        claim_ordinals: list[int] = []
        record_ordinals: list[int] = []
        for ref in parent.artifact_refs:
            for prefix, target in (
                ("evaluation/attempts/", attempt_ordinals),
                ("evaluation/claims/", claim_ordinals),
                ("evaluation/records/", record_ordinals),
            ):
                if ref.name.startswith(prefix):
                    match = re.fullmatch(re.escape(prefix) + r"([0-9]+)\.json", ref.name)
                    if match is None:
                        raise ValueError("holdout artifact namespace is invalid")
                    target.append(int(match.group(1)))
        expected_ordinals = list(range(ordinal + 1))
        if (
            schedule_hash != prepared.binding.schedule_hash
            or schedule.items[ordinal]
            != EvaluationScheduleItem(
                ordinal=ordinal,
                case_id=prepared.binding.alias,
                template_id=prepared.binding.alias,
                mode=prepared.evaluation_unit.mode,
                repeat=prepared.evaluation_unit.repeat,
                split="holdout",
                holdout_proof=prepared.evaluation_unit.holdout_proof,
            )
            or attempt
            != EvaluationAttempt(
                run_id=prepared.binding.public_evaluation_run_id,
                ordinal=ordinal,
                schedule_hash=prepared.binding.schedule_hash,
                corpus_cutoff=prepared.binding.corpus_cutoff,
                idempotency_key=prepared.evaluation_unit.idempotency_key,
                reserved_cost_usd=prepared.evaluation_unit.reserved_cost_usd,
            )
            or claim
            != EvaluationExecutionClaim(
                run_id=attempt.run_id,
                ordinal=ordinal,
                schedule_hash=schedule_hash,
                corpus_cutoff=attempt.corpus_cutoff,
                attempt_hash=hashlib.sha256(attempt_content).hexdigest(),
            )
            or sorted(attempt_ordinals) != expected_ordinals
            or sorted(claim_ordinals) != expected_ordinals
            or sorted(record_ordinals) != list(range(ordinal))
            or len(attempt_ordinals) != len(set(attempt_ordinals))
            or len(claim_ordinals) != len(set(claim_ordinals))
            or len(record_ordinals) != len(set(record_ordinals))
        ):
            raise ValueError("holdout scheduled authority is invalid")

    def authorize_and_start_reserved(
        self,
        batch: HoldoutBatch,
        prepared: PreparedHoldoutExecution,
        service: ApplicationService,
    ) -> RunManifest:
        """Validate complete authority and atomically start one exact reservation."""
        from gpu_agent.service import ApplicationService

        if (
            type(batch) is not HoldoutBatch
            or type(prepared) is not PreparedHoldoutExecution
            or type(service) is not ApplicationService
            or service.store.identity != self.evaluator.identity
            or service.evaluator_store.identity != self.evaluator.identity
            or service.binding != self.binding
        ):
            raise ValueError("reserved diagnosis service is invalid")
        expected = self._prepared_execution(
            batch,
            prepared.binding.public_evaluation_run_id,
            EvaluationScheduleItem(
                ordinal=prepared.binding.ordinal,
                case_id=prepared.binding.alias,
                template_id=prepared.binding.alias,
                mode=prepared.evaluation_unit.mode,
                repeat=prepared.evaluation_unit.repeat,
                split="holdout",
                holdout_proof=prepared.evaluation_unit.holdout_proof,
            ),
            EvaluationAttempt(
                run_id=prepared.binding.public_evaluation_run_id,
                ordinal=prepared.binding.ordinal,
                schedule_hash=prepared.binding.schedule_hash,
                corpus_cutoff=prepared.binding.corpus_cutoff,
                idempotency_key=prepared.evaluation_unit.idempotency_key,
                reserved_cost_usd=prepared.evaluation_unit.reserved_cost_usd,
            ),
        )
        if expected != prepared:
            raise ValueError("prepared holdout execution is invalid")
        with self._execution_claim(prepared.execution_run_id):
            with self.public.evaluation_run_lease(
                prepared.binding.public_evaluation_run_id
            ) as public_lease:
                self._validate_public_reserved_authority(public_lease, prepared)
                with self.evaluator.run_directory_set_lease(
                    (prepared.execution_run_id, prepared.diagnosis_run_id)
                ) as evaluator_lease:
                    execution, diagnosis = self._validate_execution_reservation(
                        prepared, evaluator_lease
                    )
                    if (
                        execution.status != RunStatus.RUNNING
                        or diagnosis.status != RunStatus.QUEUED
                    ):
                        raise ValueError("holdout diagnosis reservation is not startable")
                    started = evaluator_lease.start_queued(
                        prepared.diagnosis_run_id, CurrentPhase.PREPARING
                    )
                    if (
                        started is None
                        or started.status != RunStatus.RUNNING
                        or started.current_phase != "PREPARING"
                    ):
                        raise ValueError("holdout diagnosis authorization failed")
                    evaluator_lease.validate()
                public_lease.validate()
                return started

    def execute_reserved_diagnosis(
        self,
        batch: HoldoutBatch,
        prepared: PreparedHoldoutExecution,
        service: ApplicationService,
        source: Path,
        *,
        required_tools: tuple[SanitizerTool, ...],
        expected_source_hash: str,
        expected_input_hash: str | None = None,
    ) -> RunManifest:
        """Execute through the service's authority-validating reserved entry."""
        return service._diagnose_reserved(
            source,
            controller=self,
            batch=batch,
            prepared=prepared,
            mode=prepared.evaluation_unit.mode,
            required_tools=required_tools,
            expected_source_hash=expected_source_hash,
            expected_input_hash=expected_input_hash,
        )

    @staticmethod
    def _one_named_ref(run: RunManifest, name: str) -> ArtifactRef:
        refs = [ref for ref in run.artifact_refs if ref.name == name]
        if len(refs) != 1:
            raise ValueError("holdout native record is invalid")
        return refs[0]

    def _validate_native_execution(
        self,
        prepared: PreparedHoldoutExecution,
        native: EvaluationRecord,
        lease: RunDirectorySetLease | None = None,
    ) -> PublicEvaluationRecord:
        _, diagnosis = self._validate_execution_reservation(prepared, lease)
        if diagnosis.status != RunStatus.COMPLETED:
            raise ValueError("holdout native record is invalid")
        item = EvaluationScheduleItem(
            ordinal=prepared.binding.ordinal,
            case_id=prepared.binding.private_case_id,
            template_id=prepared.binding.private_template_id,
            mode=prepared.evaluation_unit.mode,
            repeat=prepared.evaluation_unit.repeat,
            split="holdout",
            holdout_proof=prepared.evaluation_unit.holdout_proof,
        )
        attempt = EvaluationAttempt(
            run_id=prepared.binding.public_evaluation_run_id,
            ordinal=prepared.binding.ordinal,
            schedule_hash=prepared.binding.schedule_hash,
            corpus_cutoff=prepared.binding.corpus_cutoff,
            idempotency_key=prepared.evaluation_unit.idempotency_key,
            reserved_cost_usd=prepared.evaluation_unit.reserved_cost_usd,
        )
        from gpu_agent.benchmark.executor import validate_evaluation_record

        return validate_evaluation_record(
            self.evaluator,
            native,
            item,
            attempt,
            self.binding,
            self.evaluator,
            self.evaluator,
            self._schedule_family,
            prepared.binding.private_case_id,
            diagnosis_parent_run_id=prepared.execution_run_id,
            expected_visibility="evaluator",
        )

    def _execution_commitment(
        self, batch: HoldoutBatch, prepared: PreparedHoldoutExecution, native_hash: str
    ) -> str:
        mapping_run = self.evaluator.load(batch.evaluator_run_id)
        mapping_ref = self._one_named_ref(mapping_run, "holdout/private-alias-map.json")
        mapping = _PrivateAliasMap.model_validate_json(self.evaluator.read(mapping_ref))
        return hmac.new(
            bytes.fromhex(mapping.nonce_hex),
            f"holdout-execution-commitment-v1:{prepared.execution_run_id}:{native_hash}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def _public_projection(
        self,
        batch: HoldoutBatch,
        prepared: PreparedHoldoutExecution,
        native: PublicEvaluationRecord,
        native_hash: str,
    ) -> PublicEvaluationRecord:
        lineage = native.lineage
        if not isinstance(lineage, NativeEvaluationLineage):
            raise ValueError("holdout native record is invalid")
        return PublicEvaluationRecord(
            record_id=hashlib.sha256(
                (
                    "holdout-public-record-v1:"
                    f"{prepared.binding.public_evaluation_run_id}:"
                    f"{prepared.binding.schedule_hash}:{prepared.binding.ordinal}"
                ).encode()
            ).hexdigest()[:32],
            corpus_cutoff=prepared.binding.corpus_cutoff,
            case_id=prepared.binding.alias,
            template_id=prepared.binding.alias,
            mode=prepared.evaluation_unit.mode,
            repeat=prepared.evaluation_unit.repeat,
            input_hash=native.input_hash,
            evidence_hash=native.evidence_hash,
            executed_checks=dict(native.executed_checks),
            status=native.status,
            diagnosis={},
            patch_hash=native.patch_hash,
            oracle_passed=native.oracle_passed,
            verdict=native.verdict,
            regression_detected=native.regression_detected,
            usage=dict(native.usage),
            latency_ms=native.latency_ms,
            cost_usd=native.cost_usd,
            failure_reason=None,
            lineage=HoldoutEvaluationLineage(
                corpus_cutoff=prepared.binding.corpus_cutoff,
                execution_commitment=self._execution_commitment(batch, prepared, native_hash),
                diagnosis_hash=lineage.diagnosis_hash,
                evidence_hash=lineage.evidence_hash,
                provider_invocation_hashes=lineage.provider_invocation_hashes,
                candidate_hash=native.patch_hash,
                verification_hash=lineage.public_verification_hash,
            ),
        )

    def _validate_prepared_execution(self, prepared: PreparedHoldoutExecution) -> HoldoutBatch:
        batch = self._batch_for_execution(prepared)
        unit = prepared.evaluation_unit
        item = EvaluationScheduleItem(
            ordinal=prepared.binding.ordinal,
            case_id=prepared.binding.alias,
            template_id=prepared.binding.alias,
            mode=unit.mode,
            repeat=unit.repeat,
            split="holdout",
            holdout_proof=unit.holdout_proof,
        )
        attempt = EvaluationAttempt(
            run_id=prepared.binding.public_evaluation_run_id,
            ordinal=prepared.binding.ordinal,
            schedule_hash=prepared.binding.schedule_hash,
            corpus_cutoff=prepared.binding.corpus_cutoff,
            idempotency_key=unit.idempotency_key,
            reserved_cost_usd=unit.reserved_cost_usd,
        )
        if self._prepared_execution(batch, attempt.run_id, item, attempt) != prepared:
            raise ValueError("prepared holdout execution is invalid")
        return batch

    def complete_execution(
        self, prepared: PreparedHoldoutExecution, native_record: EvaluationRecord
    ) -> PublicEvaluationRecord:
        batch = self._validate_prepared_execution(prepared)
        native_content = native_record.model_dump_json().encode()
        execution = self.evaluator.load(prepared.execution_run_id)
        if execution.status == RunStatus.COMPLETED:
            recovered = self._recover_prepared_execution(prepared)
            native_ref = self._one_named_ref(execution, "holdout/native-record.json")
            if self.evaluator.read(native_ref) != native_content:
                raise ValueError("completed holdout execution differs")
            return recovered
        validated_native = self._validate_native_execution(prepared, native_record)
        native_hash = hashlib.sha256(native_content).hexdigest()
        public = self._public_projection(batch, prepared, validated_native, native_hash)
        public_content = public.model_dump_json().encode()
        final_binding = prepared.binding.model_copy(
            update={
                "native_record_hash": native_hash,
                "public_record_hash": hashlib.sha256(public_content).hexdigest(),
            }
        )
        self.evaluator.put_if_absent_exact(
            execution.id, "holdout/native-record.json", native_content, "evaluator"
        )
        self.evaluator.put_if_absent_exact(
            execution.id, "holdout/public-record.json", public_content, "evaluator"
        )
        self.evaluator.put_if_absent_exact(
            execution.id,
            "holdout/execution-binding.json",
            final_binding.model_dump_json().encode(),
            "evaluator",
        )
        execution = self.evaluator.load(execution.id)
        if execution.status == RunStatus.RUNNING:
            self.evaluator.transition(execution.id, RunStatus.RUNNING, "FINALIZING")
            self.evaluator.transition(execution.id, RunStatus.COMPLETED, None)
        return self._recover_prepared_execution(prepared)

    def _batch_for_execution(self, prepared: PreparedHoldoutExecution) -> HoldoutBatch:
        if prepared.binding.alias_mapping_run_id == "":
            raise ValueError("holdout execution transaction is invalid")
        mapping = self.evaluator.load(prepared.binding.alias_mapping_run_id)
        if mapping.external_origin is None:
            raise ValueError("holdout execution transaction is invalid")
        public_alias = self.public.load(mapping.external_origin.run_id)
        alias_ref = self._one_named_ref(public_alias, "holdout/aliases.json")
        alias_payload = json.loads(self.public.read(alias_ref))
        return HoldoutBatch(
            public_run_id=public_alias.id,
            evaluator_run_id=mapping.id,
            aliases=alias_payload["aliases"],
            public_alias_hash=alias_ref.sha256,
            corpus_cutoff=alias_payload["corpus_cutoff"],
        )

    def _recover_prepared_execution(
        self,
        prepared: PreparedHoldoutExecution,
        lease: RunDirectorySetLease | None = None,
    ) -> PublicEvaluationRecord:
        return self._resolve_prepared_execution(prepared, lease).public_record

    def _resolve_prepared_execution(
        self,
        prepared: PreparedHoldoutExecution,
        lease: RunDirectorySetLease | None = None,
    ) -> ResolvedHoldoutEvaluationRecord:
        """Validate and load the exact terminal evaluator transaction."""
        batch = self._validate_prepared_execution(prepared)
        execution, _ = self._validate_execution_reservation(prepared, lease)
        expected_names = {
            "holdout/native-record.json",
            "holdout/public-record.json",
            "holdout/execution-binding.json",
        }
        if (
            execution.status != RunStatus.COMPLETED
            or {ref.name for ref in execution.artifact_refs} != expected_names
        ):
            raise ValueError("holdout execution transaction is incomplete")
        native_ref = self._one_named_ref(execution, "holdout/native-record.json")
        public_ref = self._one_named_ref(execution, "holdout/public-record.json")
        binding_ref = self._one_named_ref(execution, "holdout/execution-binding.json")
        native_content = (
            lease.read(native_ref) if lease is not None else self.evaluator.read(native_ref)
        )
        public_content = (
            lease.read(public_ref) if lease is not None else self.evaluator.read(public_ref)
        )
        binding_content = (
            lease.read(binding_ref) if lease is not None else self.evaluator.read(binding_ref)
        )
        native = EvaluationRecord.model_validate_json(native_content)
        public = PublicEvaluationRecord.model_validate_json(public_content)
        final_binding = HoldoutExecutionBinding.model_validate_json(binding_content)
        expected_binding = prepared.binding.model_copy(
            update={
                "native_record_hash": hashlib.sha256(native_content).hexdigest(),
                "public_record_hash": hashlib.sha256(public_content).hexdigest(),
            }
        )
        validated_native = self._validate_native_execution(prepared, native, lease)
        expected_public = self._public_projection(
            batch, prepared, validated_native, expected_binding.native_record_hash or ""
        )
        if (
            final_binding != expected_binding
            or native_ref.sha256 != expected_binding.native_record_hash
            or public_ref.sha256 != expected_binding.public_record_hash
            or native.model_dump_json().encode() != native_content
            or public.model_dump_json().encode() != public_content
            or expected_public.model_dump_json().encode() != public_content
        ):
            raise ValueError("holdout execution transaction is invalid")
        if lease is not None:
            lease.validate()
        return ResolvedHoldoutEvaluationRecord(
            public_record=public,
            native_record=native,
            execution_binding=final_binding,
        )

    def recover_execution(
        self,
        batch: HoldoutBatch,
        item: EvaluationScheduleItem,
        attempt: EvaluationAttempt,
    ) -> PublicEvaluationRecord | None:
        prepared = self._prepared_execution(batch, attempt.run_id, item, attempt)
        with self._execution_claim(prepared.execution_run_id):
            with self.evaluator.run_directory_set_lease(
                (prepared.execution_run_id, prepared.diagnosis_run_id)
            ) as lease:
                execution = lease.load_optional(prepared.execution_run_id)
                diagnosis = lease.load_optional(prepared.diagnosis_run_id)
                if execution is None and diagnosis is None:
                    return None
                if execution is None or diagnosis is None:
                    raise ValueError("holdout execution transaction is incomplete")
                return self._recover_prepared_execution(prepared, lease=lease)

    def prepare_score(
        self,
        batch: HoldoutBatch,
        alias: str,
        public_record_ref: ArtifactRef,
        *,
        labels: EvaluationLabels | None,
        score: Score,
        should_be_inconclusive: bool,
        private_holdout_passed: bool,
    ) -> PreparedHoldoutScore:
        if labels is None:
            raise ValueError("private evaluation labels are required")
        record = self._validated_public_record(public_record_ref, batch)
        return self._prepare_validated_score(
            batch,
            alias,
            public_record_ref,
            record,
            labels=labels,
            score=score,
            should_be_inconclusive=should_be_inconclusive,
            private_holdout_passed=private_holdout_passed,
        )

    def _prepare_validated_score(
        self,
        batch: HoldoutBatch,
        alias: str,
        public_record_ref: ArtifactRef,
        record: PublicEvaluationRecord,
        *,
        labels: EvaluationLabels,
        score: Score,
        should_be_inconclusive: bool,
        private_holdout_passed: bool,
        identity: _PrivateIdentity | None = None,
    ) -> PreparedHoldoutScore:
        identity = identity or self._identity(batch, alias)
        if record.case_id != alias or record.template_id != alias:
            raise ValueError("public evaluation record is invalid")
        private_score = _PrivateScore(
            corpus_cutoff=batch.corpus_cutoff,
            public_record_hash=public_record_ref.sha256,
            labels=labels,
            score=score,
            should_be_inconclusive=should_be_inconclusive,
            private_holdout_passed=private_holdout_passed,
        )
        score_run_id = self._score_run_id(batch, record.record_id)
        binding = EvaluatorRecordBinding(
            evaluator_score_run_id=score_run_id,
            public_evaluation_run_id=public_record_ref.run_id,
            public_record_id=record.record_id,
            public_record_hash=public_record_ref.sha256,
            private_case_id=identity.private_case_id,
            private_template_id=identity.private_template_id,
            private_score_hash=hashlib.sha256(private_score.content()).hexdigest(),
            corpus_cutoff=batch.corpus_cutoff,
        )
        return PreparedHoldoutScore(
            run_id=score_run_id, binding=binding, private_score_content=private_score.content()
        )

    def bind_score(
        self,
        batch: HoldoutBatch,
        alias: str,
        public_record_ref: ArtifactRef,
        *,
        labels: EvaluationLabels | None,
        score: Score,
        should_be_inconclusive: bool,
        private_holdout_passed: bool,
    ) -> EvaluatorRecordBinding:
        if labels is None:
            raise ValueError("private evaluation labels are required")
        record = self._validated_public_record(public_record_ref, batch)
        identity = self._identity(batch, alias)
        prepared = self._prepare_validated_score(
            batch,
            alias,
            public_record_ref,
            record,
            labels=labels,
            score=score,
            should_be_inconclusive=should_be_inconclusive,
            private_holdout_passed=private_holdout_passed,
            identity=identity,
        )
        return self._bind_prepared_score(
            batch,
            prepared,
            alias=alias,
            public_record_ref=public_record_ref,
            public_record=record,
            private_case_id=identity.private_case_id,
            private_template_id=identity.private_template_id,
        )

    def _bind_prepared_score(
        self,
        batch: HoldoutBatch,
        prepared: PreparedHoldoutScore,
        *,
        alias: str,
        public_record_ref: ArtifactRef,
        public_record: PublicEvaluationRecord,
        private_case_id: str,
        private_template_id: str,
        _reload: bool = True,
    ) -> EvaluatorRecordBinding:
        """Persist one score already derived from current native validation."""
        score_run_id, binding = prepared.run_id, prepared.binding
        try:
            private_score = _PrivateScore.model_validate_json(prepared.private_score_content)
        except ValueError:
            raise ValueError("prepared holdout score is invalid") from None
        canonical_score = private_score.content()
        expected_run_id = self._score_run_id(batch, public_record.record_id)
        expected_binding = EvaluatorRecordBinding(
            evaluator_score_run_id=expected_run_id,
            public_evaluation_run_id=public_record_ref.run_id,
            public_record_id=public_record.record_id,
            public_record_hash=public_record_ref.sha256,
            private_case_id=private_case_id,
            private_template_id=private_template_id,
            private_score_hash=hashlib.sha256(canonical_score).hexdigest(),
            corpus_cutoff=batch.corpus_cutoff,
        )
        if (
            public_record.case_id != alias
            or public_record.template_id != alias
            or score_run_id != expected_run_id
            or binding != expected_binding
            or canonical_score != prepared.private_score_content
            or private_score.public_record_hash != public_record_ref.sha256
            or private_score.corpus_cutoff != batch.corpus_cutoff
        ):
            raise ValueError("prepared holdout score is invalid")
        expected_origin = ExternalRunOrigin(run_id=batch.public_run_id, visibility="public")
        mapping = self.evaluator.load(batch.evaluator_run_id)
        if (
            mapping.kind != "holdout_alias_mapping"
            or mapping.parent_run_id is not None
            or mapping.status != RunStatus.COMPLETED
            or mapping.binding != self.binding
            or mapping.external_origin != expected_origin
        ):
            raise ValueError("holdout score transaction is invalid")

        def validate_run(run: RunManifest) -> None:
            if (
                run.kind != "holdout_score"
                or run.parent_run_id != batch.evaluator_run_id
                or run.binding != self.binding
                or run.external_origin != expected_origin
                or run.status not in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED}
                or any(ref.run_id != run.id for ref in run.artifact_refs)
            ):
                raise ValueError("holdout score transaction is invalid")

        if (self.evaluator.root / score_run_id).exists():
            validate_run(self.evaluator.load(score_run_id))
        with self._score_claim(score_run_id):
            try:
                run = self.evaluator.load(score_run_id)
            except ValueError:
                run = self.evaluator.create_run(
                    "holdout_score",
                    parent_run_id=batch.evaluator_run_id,
                    _run_id=score_run_id,
                )
            validate_run(run)
            if run.status == RunStatus.QUEUED:
                self.evaluator.transition(run.id, RunStatus.RUNNING, "FINALIZING")
            self.evaluator.put_if_absent_exact(
                run.id, "holdout/private-score.json", prepared.private_score_content, "evaluator"
            )
            self.evaluator.put_if_absent_exact(
                run.id,
                "holdout/record-binding.json",
                binding.model_dump_json().encode(),
                "evaluator",
            )
            run = self.evaluator.load(run.id)
            if run.status == RunStatus.RUNNING:
                self.evaluator.transition(run.id, RunStatus.COMPLETED, None)
            if _reload:
                self._load_score(binding)
            return binding

    def _load_metric_record(
        self,
        binding: EvaluatorRecordBinding,
        *,
        _context: _MetricLoadContext | None = None,
    ) -> EvaluationRecord:
        """Reload a scored record for the metric module's persisted-reference API."""
        private_score, public, native = self._load_score(binding, _context=_context)
        # Metrics are keyed by the immutable public record binding.  Native
        # evaluator evidence supplies the adjudicated diagnosis and verification
        # result, but its private diagnosis-run identity must not replace the
        # public record identity returned by this persisted-reference API.
        raw = public.model_dump(mode="json")
        build = native.executed_checks.get("verification/build")
        raw.update(
            diagnosis=native.diagnosis,
            score=private_score.score,
            evaluator_labels=private_score.labels,
            should_be_inconclusive=private_score.should_be_inconclusive,
            private_holdout_passed=private_score.private_holdout_passed,
            patch_compile_passed=(
                {"CLEAN": True, "FAILED": False}.get(build) if build is not None else None
            ),
        )
        return EvaluationRecord.model_validate(raw)

    def _metric_load_context(self, binding: EvaluatorRecordBinding) -> _MetricLoadContext:
        """Validate shared native evaluation state once for one metric invocation."""
        run = self.evaluator.load(binding.evaluator_score_run_id)
        if (
            run.kind != "holdout_score"
            or run.status != RunStatus.COMPLETED
            or run.binding != self.binding
            or run.parent_run_id is None
        ):
            raise ValueError("holdout score transaction is invalid")
        batch = self._score_batch(run)
        evaluation = self.validated_evaluation(batch, binding.public_evaluation_run_id)
        identities = {alias: self._identity(batch, alias) for alias in batch.aliases}
        return _MetricLoadContext(batch=batch, evaluation=evaluation, identities=identities)

    def resolve_private(self, batch: HoldoutBatch, alias: str) -> tuple[str, str]:
        """Resolve privately; callers must never persist the result publicly."""
        identity = self._identity(batch, alias)
        return identity.private_case_id, identity.private_template_id

    def _identity(self, batch: HoldoutBatch, alias: str) -> _PrivateIdentity:
        self.validate_batch(batch)
        run = self.evaluator.load(batch.evaluator_run_id)
        if (
            run.kind != "holdout_alias_mapping"
            or run.status != RunStatus.COMPLETED
            or run.binding != self.binding
            or run.external_origin
            != ExternalRunOrigin(run_id=batch.public_run_id, visibility="public")
        ):
            raise ValueError("holdout binding is invalid")
        refs = [ref for ref in run.artifact_refs if ref.name == "holdout/private-alias-map.json"]
        if len(refs) != 1:
            raise ValueError("holdout binding is invalid")
        mapping = _PrivateAliasMap.model_validate_json(self.evaluator.read(refs[0]))
        matches = [item for item in mapping.identities if item.alias == alias]
        if len(matches) != 1 or alias not in batch.aliases:
            raise ValueError("holdout binding is invalid")
        return matches[0]

    def _validated_public_record(
        self, ref: ArtifactRef, batch: HoldoutBatch | None = None
    ) -> PublicEvaluationRecord:
        try:
            record, schedule, ordinal = self._resolve_public_record(ref, batch)
            return record
        except (ValueError, KeyError, IndexError):
            raise ValueError("public evaluation record is invalid") from None

    def _resolve_public_record(
        self, ref: ArtifactRef, batch: HoldoutBatch | None = None
    ) -> tuple[PublicEvaluationRecord, EvaluationSchedule, int]:
        match = re.fullmatch(r"evaluation/records/(0|[1-9][0-9]*)\.json", ref.name)
        if ref.visibility != "public" or match is None or batch is None:
            raise ValueError("public evaluation record is invalid")
        evaluation = self.validated_evaluation(batch, ref.run_id)
        ordinal = int(match.group(1))
        if ordinal >= len(evaluation.records) or evaluation.record_refs[ordinal] != ref:
            raise ValueError("public evaluation record is invalid")
        return evaluation.records[ordinal], evaluation.schedule, ordinal

    def validated_evaluation(
        self, batch: HoldoutBatch, evaluation_run_id: str
    ) -> ValidatedHoldoutEvaluation:
        """Validate every native record once, without persisting any evaluator data."""
        proof = self.validate_batch(batch)
        run = self.public.load(evaluation_run_id)
        if (
            run.kind != "evaluation"
            or run.status != RunStatus.COMPLETED
            or run.binding != self.binding
            or any(ref.run_id != run.id for ref in run.artifact_refs)
        ):
            raise ValueError("public evaluation record is invalid")
        from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

        try:
            read_regular(self.public.root / run.id / ".lock", 1024)
        except (ValueError, OSError):
            raise ValueError("public evaluation record is invalid") from None
        EvaluationScheduleVerifier.verify(self._schedule_verifier, run.id)
        schedule_refs = [r for r in run.artifact_refs if r.name == "evaluation/schedule.json"]
        manifest_refs = [r for r in run.artifact_refs if r.name == "evaluation/manifest.json"]
        all_attempt_refs = [
            r for r in run.artifact_refs if r.name.startswith("evaluation/attempts/")
        ]
        all_record_refs = [r for r in run.artifact_refs if r.name.startswith("evaluation/records/")]
        if len(schedule_refs) != 1 or len(manifest_refs) != 1:
            raise ValueError("public evaluation record is invalid")
        schedule = EvaluationSchedule.model_validate_json(self.public.read(schedule_refs[0]))
        schedule_hash = hashlib.sha256(
            json.dumps(
                schedule.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        expected_bindings = {
            "commit": self.binding.repository.commit,
            "prompt_version": self.binding.prompt_version,
            "toolchain_hash": self.binding.toolchain_lock_hash,
            "model_config_hash": self.binding.model_config_hash,
        }
        if (
            not schedule.items
            or schedule.split != "holdout"
            or schedule.corpus_cutoff != batch.corpus_cutoff
            or schedule.holdout_proof != proof
            or [item.ordinal for item in schedule.items] != list(range(len(schedule.items)))
            or len(all_attempt_refs) != len(schedule.items)
            or len(all_record_refs) != len(schedule.items)
            or any(
                getattr(schedule.bindings, key) != value for key, value in expected_bindings.items()
            )
        ):
            raise ValueError("public evaluation record is invalid")
        attempts: dict[int, EvaluationAttempt] = {}
        for attempt_ref in all_attempt_refs:
            match = re.fullmatch(r"evaluation/attempts/(0|[1-9][0-9]*)\.json", attempt_ref.name)
            if match is None or int(match.group(1)) in attempts:
                raise ValueError("public evaluation record is invalid")
            attempt_ordinal = int(match.group(1))
            if attempt_ordinal >= len(schedule.items):
                raise ValueError("public evaluation record is invalid")
            observed_attempt = EvaluationAttempt.model_validate_json(self.public.read(attempt_ref))
            expected_attempt_key = hashlib.sha256(
                f"{run.id}:{schedule_hash}:{attempt_ordinal}".encode()
            ).hexdigest()
            if (
                observed_attempt.run_id != run.id
                or observed_attempt.ordinal != attempt_ordinal
                or observed_attempt.schedule_hash != schedule_hash
                or observed_attempt.corpus_cutoff != schedule.corpus_cutoff
                or observed_attempt.idempotency_key != expected_attempt_key
                or observed_attempt.reserved_cost_usd != schedule.bindings.max_unit_cost_usd
            ):
                raise ValueError("public evaluation record is invalid")
            attempts[attempt_ordinal] = observed_attempt
        records: dict[int, PublicEvaluationRecord] = {}
        record_refs: dict[int, ArtifactRef] = {}
        for record_ref in all_record_refs:
            match = re.fullmatch(r"evaluation/records/(0|[1-9][0-9]*)\.json", record_ref.name)
            if match is None or int(match.group(1)) in records:
                raise ValueError("public evaluation record is invalid")
            record_ordinal = int(match.group(1))
            if record_ordinal not in attempts:
                raise ValueError("public evaluation record is invalid")
            records[record_ordinal] = PublicEvaluationRecord.model_validate_json(
                self.public.read(record_ref)
            )
            record_refs[record_ordinal] = record_ref
        manifest = EvaluationManifest.model_validate_json(self.public.read(manifest_refs[0]))
        ordered_records = [records[index] for index in sorted(records)]
        if (
            sorted(records) != list(range(len(schedule.items)))
            or len({record.record_id for record in ordered_records}) != len(ordered_records)
            or any(not re.fullmatch(r"[a-f0-9]{32}", r.record_id) for r in ordered_records)
            or manifest.stopped_reason is not None
            or manifest.run_id != run.id
            or manifest.schedule_hash != schedule_hash
            or manifest.corpus_cutoff != schedule.corpus_cutoff
            or manifest.expected_units != len(schedule.items)
            or manifest.executed_units != len(records)
            or manifest.records != ordered_records
            or manifest.split != schedule.split
            or manifest.repeats != schedule.repeats
            or manifest.random_seed != schedule.random_seed
            or manifest.modes != schedule.modes
            or any(getattr(manifest, key) != value for key, value in expected_bindings.items())
        ):
            raise ValueError("public evaluation record is invalid")
        mapping_run = self.evaluator.load(batch.evaluator_run_id)
        mapping_ref = self._one_named_ref(mapping_run, "holdout/private-alias-map.json")
        mapping = _PrivateAliasMap.model_validate_json(self.evaluator.read(mapping_ref))
        identities = {identity.alias: identity for identity in mapping.identities}
        from gpu_agent.benchmark.executor import (
            _validate_evaluation_record_against_case,
            registered_cases,
        )

        trusted_cases = registered_cases(
            self.evaluator,
            self.binding,
            self._schedule_family,
            cutoff=batch.corpus_cutoff,
        )
        if (
            mapping_run.kind != "holdout_alias_mapping"
            or mapping_run.status != RunStatus.COMPLETED
            or mapping_run.binding != self.binding
            or mapping_run.external_origin
            != ExternalRunOrigin(run_id=batch.public_run_id, visibility="public")
            or mapping.corpus_cutoff != batch.corpus_cutoff
            or list(identities) != batch.aliases
            or len(identities) != len(mapping.identities)
            or any(
                identity.private_case_id not in trusted_cases
                or trusted_cases[identity.private_case_id].template_id
                != identity.private_template_id
                for identity in mapping.identities
            )
        ):
            raise ValueError("public evaluation record is invalid")

        def resolve_record(
            scheduled_item: EvaluationScheduleItem,
            attempt: EvaluationAttempt,
            observed_record: PublicEvaluationRecord,
        ) -> ResolvedHoldoutEvaluationRecord:
            identity = identities.get(scheduled_item.case_id)
            if (
                identity is None
                or attempt.run_id != evaluation_run_id
                or scheduled_item.ordinal != attempt.ordinal
                or scheduled_item.split != "holdout"
                or scheduled_item.case_id != scheduled_item.template_id
                or scheduled_item.holdout_proof != proof
                or attempt.corpus_cutoff != batch.corpus_cutoff
            ):
                raise ValueError("public evaluation record is invalid")
            attempt_hash = self._content_hash(attempt)
            execution_run_id = hashlib.sha256(
                (
                    "holdout-execution-v1:"
                    f"{batch.evaluator_run_id}:{evaluation_run_id}:{attempt.schedule_hash}:"
                    f"{scheduled_item.ordinal}:{attempt_hash}"
                ).encode()
            ).hexdigest()[:32]
            diagnosis_run_id = hashlib.sha256(
                f"holdout-diagnosis-v1:{execution_run_id}".encode()
            ).hexdigest()[:32]
            prepared = PreparedHoldoutExecution(
                execution_run_id=execution_run_id,
                diagnosis_run_id=diagnosis_run_id,
                binding=HoldoutExecutionBinding(
                    public_evaluation_run_id=evaluation_run_id,
                    alias_mapping_run_id=batch.evaluator_run_id,
                    ordinal=scheduled_item.ordinal,
                    schedule_hash=attempt.schedule_hash,
                    attempt_hash=attempt_hash,
                    corpus_cutoff=batch.corpus_cutoff,
                    alias=scheduled_item.case_id,
                    private_case_id=identity.private_case_id,
                    private_template_id=identity.private_template_id,
                    diagnosis_run_id=diagnosis_run_id,
                ),
                evaluation_unit=EvaluationUnitBinding(
                    evaluation_run_id=evaluation_run_id,
                    ordinal=scheduled_item.ordinal,
                    schedule_hash=attempt.schedule_hash,
                    corpus_cutoff=attempt.corpus_cutoff,
                    idempotency_key=attempt.idempotency_key,
                    reserved_cost_usd=attempt.reserved_cost_usd,
                    case_id=identity.private_case_id,
                    template_id=identity.private_template_id,
                    mode=scheduled_item.mode,
                    repeat=scheduled_item.repeat,
                    split="holdout",
                    holdout_proof=proof,
                ),
            )
            execution, diagnosis = self._validate_execution_reservation(prepared)
            expected_names = {
                "holdout/native-record.json",
                "holdout/public-record.json",
                "holdout/execution-binding.json",
            }
            if (
                execution.status != RunStatus.COMPLETED
                or diagnosis.status != RunStatus.COMPLETED
                or {ref.name for ref in execution.artifact_refs} != expected_names
            ):
                raise ValueError("holdout execution transaction is incomplete")
            native_ref = self._one_named_ref(execution, "holdout/native-record.json")
            public_ref = self._one_named_ref(execution, "holdout/public-record.json")
            binding_ref = self._one_named_ref(execution, "holdout/execution-binding.json")
            native_content = self.evaluator.read(native_ref)
            public_content = self.evaluator.read(public_ref)
            binding_content = self.evaluator.read(binding_ref)
            native = EvaluationRecord.model_validate_json(native_content)
            public = PublicEvaluationRecord.model_validate_json(public_content)
            final_binding = HoldoutExecutionBinding.model_validate_json(binding_content)
            expected_binding = prepared.binding.model_copy(
                update={
                    "native_record_hash": hashlib.sha256(native_content).hexdigest(),
                    "public_record_hash": hashlib.sha256(public_content).hexdigest(),
                }
            )
            trusted_case = trusted_cases.get(identity.private_case_id)
            if trusted_case is None:
                raise ValueError("public evaluation record is invalid")
            native_item = scheduled_item.model_copy(
                update={
                    "case_id": identity.private_case_id,
                    "template_id": identity.private_template_id,
                }
            )
            validated_native = _validate_evaluation_record_against_case(
                self.evaluator,
                native,
                native_item,
                attempt,
                self.binding,
                trusted_case,
                identity.private_case_id,
                evaluator=self.evaluator,
                diagnosis_parent_run_id=execution_run_id,
                expected_visibility="evaluator",
            )
            expected_public = self._public_projection(
                batch,
                prepared,
                validated_native,
                expected_binding.native_record_hash or "",
            )
            if (
                final_binding != expected_binding
                or native_ref.sha256 != expected_binding.native_record_hash
                or public_ref.sha256 != expected_binding.public_record_hash
                or native.model_dump_json().encode() != native_content
                or public.model_dump_json().encode() != public_content
                or expected_public.model_dump_json().encode() != public_content
                or public.model_dump_json() != observed_record.model_dump_json()
            ):
                raise ValueError("public evaluation record is invalid")
            return ResolvedHoldoutEvaluationRecord(
                public_record=public,
                native_record=native,
                execution_binding=final_binding,
            )

        resolved_records: dict[int, ResolvedHoldoutEvaluationRecord] = {}
        for record_ordinal, observed_record in records.items():
            scheduled_item = schedule.items[record_ordinal]
            if (
                scheduled_item.split != "holdout"
                or scheduled_item.holdout_proof != proof
                or scheduled_item.case_id != observed_record.case_id
                or scheduled_item.template_id != observed_record.template_id
                or scheduled_item.case_id != scheduled_item.template_id
            ):
                raise ValueError("public evaluation record is invalid")
            resolved_records[record_ordinal] = resolve_record(
                scheduled_item,
                attempts[record_ordinal],
                observed_record,
            )
        return ValidatedHoldoutEvaluation(
            evaluation_run_id=run.id,
            schedule=schedule,
            schedule_hash=schedule_hash,
            records=tuple(ordered_records),
            record_refs=tuple(record_refs[index] for index in range(len(records))),
            resolved_records=tuple(resolved_records[index] for index in range(len(records))),
        )

    def resolve_evaluation_record(
        self, batch: HoldoutBatch, evaluation_run_id: str, ordinal: int
    ) -> ResolvedHoldoutEvaluationRecord:
        """Resolve a public evaluation locator to exactly one validated native record."""
        evaluation = self.validated_evaluation(batch, evaluation_run_id)
        if ordinal < 0 or ordinal >= len(evaluation.resolved_records):
            raise ValueError("public evaluation ordinal is invalid")
        return evaluation.resolved_records[ordinal]

    def _score_run_id(self, batch: HoldoutBatch, record_id: str) -> str:
        run = self.evaluator.load(batch.evaluator_run_id)
        mapping = _PrivateAliasMap.model_validate_json(
            self.evaluator.read(
                next(
                    ref for ref in run.artifact_refs if ref.name == "holdout/private-alias-map.json"
                )
            )
        )
        return hmac.new(
            bytes.fromhex(mapping.nonce_hex),
            f"holdout-score-v1:{record_id}".encode(),
            hashlib.sha256,
        ).hexdigest()[:32]

    def _validate_execution_claim(
        self,
        root_fd: int,
        root_identity: os.stat_result,
        lock_fd: int,
        lock_name: str,
    ) -> None:
        root_descriptor = os.fstat(root_fd)
        root_path = os.stat(self.evaluator.root, follow_symlinks=False)
        lock_descriptor = os.fstat(lock_fd)
        lock_path = os.stat(lock_name, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(root_descriptor.st_mode)
            or not stat.S_ISDIR(root_path.st_mode)
            or root_descriptor.st_mode & 0o077
            or (root_descriptor.st_dev, root_descriptor.st_ino)
            != (root_identity.st_dev, root_identity.st_ino)
            or (root_descriptor.st_dev, root_descriptor.st_ino)
            != (root_path.st_dev, root_path.st_ino)
            or not stat.S_ISREG(lock_descriptor.st_mode)
            or not stat.S_ISREG(lock_path.st_mode)
            or stat.S_IMODE(lock_descriptor.st_mode) != 0o600
            or stat.S_IMODE(lock_path.st_mode) != 0o600
            or lock_descriptor.st_nlink != 1
            or lock_path.st_nlink != 1
            or (lock_descriptor.st_dev, lock_descriptor.st_ino)
            != (lock_path.st_dev, lock_path.st_ino)
        ):
            raise ValueError("holdout execution claim changed or became unsafe")

    @contextmanager
    def _execution_claim(self, execution_run_id: str) -> Iterator[None]:
        """Serialize one deterministic evaluator transaction across threads/processes."""
        with _EXECUTION_GUARD:
            local = _EXECUTION_LOCKS.setdefault(execution_run_id, threading.Lock())
        with local:
            try:
                root_fd = os.open(
                    self.evaluator.root,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                )
            except OSError as exc:
                raise ValueError("holdout execution claim root is unsafe") from exc
            fd = -1
            try:
                root_info = os.fstat(root_fd)
                lock_name = f".holdout-execution-{execution_run_id}.lock"
                fd = os.open(
                    lock_name,
                    os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=root_fd,
                )
                self._validate_execution_claim(root_fd, root_info, fd, lock_name)
                fcntl.flock(fd, fcntl.LOCK_EX)
                self._validate_execution_claim(root_fd, root_info, fd, lock_name)
                try:
                    yield
                finally:
                    self._validate_execution_claim(root_fd, root_info, fd, lock_name)
            except OSError as exc:
                raise ValueError("holdout execution claim is unavailable or unsafe") from exc
            finally:
                if fd >= 0:
                    os.close(fd)
                os.close(root_fd)

    @contextmanager
    def _score_claim(self, score_run_id: str) -> Iterator[None]:
        with _SCORE_GUARD:
            local = _SCORE_LOCKS.setdefault(score_run_id, threading.Lock())
        with local:
            path = self.evaluator.root / f".holdout-score-{score_run_id}.lock"
            reject_symlinks(path)
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError("holdout score claim must be a regular file")
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)

    def _score_batch(self, run: RunManifest) -> HoldoutBatch:
        if run.parent_run_id is None:
            raise ValueError("holdout score transaction is invalid")
        mapping_run = self.evaluator.load(run.parent_run_id)
        if (
            mapping_run.kind != "holdout_alias_mapping"
            or mapping_run.status != RunStatus.COMPLETED
            or mapping_run.binding != self.binding
            or mapping_run.external_origin is None
            or mapping_run.external_origin.visibility != "public"
        ):
            raise ValueError("holdout score transaction is invalid")
        public_alias_run = self.public.load(mapping_run.external_origin.run_id)
        mapping_refs = [
            ref for ref in mapping_run.artifact_refs if ref.name == "holdout/private-alias-map.json"
        ]
        if len(mapping_refs) != 1:
            raise ValueError("holdout score transaction is invalid")
        mapping = _PrivateAliasMap.model_validate_json(self.evaluator.read(mapping_refs[0]))
        alias_refs = [
            ref for ref in public_alias_run.artifact_refs if ref.name == "holdout/aliases.json"
        ]
        if len(alias_refs) != 1:
            raise ValueError("holdout score transaction is invalid")
        alias_payload = json.loads(self.public.read(alias_refs[0]))
        return HoldoutBatch(
            public_run_id=public_alias_run.id,
            evaluator_run_id=mapping_run.id,
            aliases=alias_payload.get("aliases", []),
            public_alias_hash=alias_refs[0].sha256,
            corpus_cutoff=mapping.corpus_cutoff,
        )

    def _load_score(
        self,
        binding: EvaluatorRecordBinding,
        *,
        _context: _MetricLoadContext | None = None,
    ) -> tuple[_PrivateScore, PublicEvaluationRecord, EvaluationRecord]:
        run = self.evaluator.load(binding.evaluator_score_run_id)
        score_refs = [r for r in run.artifact_refs if r.name == "holdout/private-score.json"]
        binding_refs = [r for r in run.artifact_refs if r.name == "holdout/record-binding.json"]
        if (
            run.kind != "holdout_score"
            or run.status != RunStatus.COMPLETED
            or run.binding != self.binding
            or run.parent_run_id is None
            or len(score_refs) != 1
            or len(binding_refs) != 1
            or EvaluatorRecordBinding.model_validate_json(self.evaluator.read(binding_refs[0]))
            != binding
        ):
            raise ValueError("holdout score transaction is invalid")
        private_score = _PrivateScore.model_validate_json(self.evaluator.read(score_refs[0]))
        if (
            score_refs[0].sha256 != binding.private_score_hash
            or private_score.public_record_hash != binding.public_record_hash
            or private_score.corpus_cutoff != binding.corpus_cutoff
        ):
            raise ValueError("holdout score transaction is invalid")
        evaluation_run = self.public.load(binding.public_evaluation_run_id)
        refs = [
            ref
            for ref in evaluation_run.artifact_refs
            if ref.sha256 == binding.public_record_hash
            and ref.name.startswith("evaluation/records/")
        ]
        if len(refs) != 1:
            raise ValueError("holdout score transaction is invalid")
        batch = self._score_batch(run)
        if _context is None:
            evaluation = self.validated_evaluation(batch, binding.public_evaluation_run_id)
            try:
                ordinal = evaluation.record_refs.index(refs[0])
                public = evaluation.records[ordinal]
                native = evaluation.resolved_records[ordinal].native_record
            except (ValueError, IndexError):
                raise ValueError("holdout score transaction is invalid") from None
            identity = self._identity(batch, public.case_id)
        else:
            if batch != _context.batch or binding.public_evaluation_run_id != (
                _context.evaluation.evaluation_run_id
            ):
                raise ValueError("holdout score transaction is invalid")
            try:
                ordinal = _context.evaluation.record_refs.index(refs[0])
                public = _context.evaluation.records[ordinal]
                native = _context.evaluation.resolved_records[ordinal].native_record
                identity = _context.identities[public.case_id]
            except (KeyError, ValueError, IndexError):
                raise ValueError("holdout score transaction is invalid") from None
        if (
            run.id != self._score_run_id(batch, public.record_id)
            or binding.evaluator_score_run_id != run.id
            or binding.public_evaluation_run_id != refs[0].run_id
            or public.record_id != binding.public_record_id
            or identity.private_case_id != binding.private_case_id
            or identity.private_template_id != binding.private_template_id
            or public.corpus_cutoff != binding.corpus_cutoff
            or public.lineage.corpus_cutoff != binding.corpus_cutoff
            or batch.corpus_cutoff != binding.corpus_cutoff
        ):
            raise ValueError("holdout score transaction is invalid")
        return private_score, public, native
