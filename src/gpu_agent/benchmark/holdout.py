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
from typing import TYPE_CHECKING, NamedTuple

from pydantic import Field

from gpu_agent.benchmark.evaluation import (
    EvaluationAttempt,
    EvaluationManifest,
    EvaluationRecord,
    EvaluationSchedule,
    HoldoutScheduleProof,
    PublicEvaluationRecord,
)
from gpu_agent.benchmark.metrics import EvaluationLabels, Score
from gpu_agent.contracts import ArtifactRef, ExternalRunOrigin, RunBinding, RunManifest, RunStatus
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.store import RunStore, read_regular, reject_symlinks

_SCORE_GUARD = threading.Lock()
_SCORE_LOCKS: dict[str, threading.Lock] = {}

if TYPE_CHECKING:
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier


class HoldoutBatch(ExecutionModel):
    public_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    evaluator_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    aliases: list[str]
    public_alias_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)


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


class ValidatedHoldoutEvaluation(ExecutionModel):
    evaluation_run_id: str
    schedule: EvaluationSchedule
    schedule_hash: str
    records: tuple[PublicEvaluationRecord, ...]
    record_refs: tuple[ArtifactRef, ...]


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

        family_root = os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT")
        if family_root is None:
            raise ValueError("trusted corpus family configuration is required")
        self._schedule_family = CorpusFamily.open(Path(family_root))
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
        prepared = self.prepare_score(
            batch,
            alias,
            public_record_ref,
            labels=labels,
            score=score,
            should_be_inconclusive=should_be_inconclusive,
            private_holdout_passed=private_holdout_passed,
        )
        return self._bind_prepared_score(batch, prepared)

    def _bind_prepared_score(
        self,
        batch: HoldoutBatch,
        prepared: PreparedHoldoutScore,
        *,
        _reload: bool = True,
    ) -> EvaluatorRecordBinding:
        """Persist one score already derived from current native validation."""
        score_run_id, binding = prepared.run_id, prepared.binding
        try:
            private_score = _PrivateScore.model_validate_json(prepared.private_score_content)
        except ValueError:
            raise ValueError("prepared holdout score is invalid") from None
        if (
            score_run_id != binding.evaluator_score_run_id
            or score_run_id != self._score_run_id(batch, binding.public_record_id)
            or binding.corpus_cutoff != batch.corpus_cutoff
            or private_score.content() != prepared.private_score_content
            or private_score.public_record_hash != binding.public_record_hash
            or private_score.corpus_cutoff != binding.corpus_cutoff
            or hashlib.sha256(prepared.private_score_content).hexdigest()
            != binding.private_score_hash
        ):
            raise ValueError("prepared holdout score is invalid")
        with self._score_claim(score_run_id):
            try:
                run = self.evaluator.load(score_run_id)
            except ValueError:
                run = self.evaluator.create_run(
                    "holdout_score",
                    parent_run_id=batch.evaluator_run_id,
                    _run_id=score_run_id,
                )
            if (
                run.kind != "holdout_score"
                or run.parent_run_id != batch.evaluator_run_id
                or run.binding != self.binding
                or run.status == RunStatus.FAILED
            ):
                raise ValueError("holdout score transaction is invalid")
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
        private_score, public = self._load_score(binding, _context=_context)
        raw = public.model_dump(mode="json")
        build = public.executed_checks.get("verification/build")
        raw.update(
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
        from gpu_agent.benchmark.executor import validate_evaluation_record

        identities = {alias: self._identity(batch, alias) for alias in batch.aliases}
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
            identity = identities.get(scheduled_item.case_id)
            if identity is None:
                raise ValueError("public evaluation record is invalid")
            registered_case_id = identity.private_case_id
            validate_evaluation_record(
                self.public,
                observed_record,
                scheduled_item,
                attempts[record_ordinal],
                self.binding,
                self.evaluator,
                self._schedule_family.corpus_store("evaluator"),
                self._schedule_family,
                registered_case_id,
            )
        return ValidatedHoldoutEvaluation(
            evaluation_run_id=run.id,
            schedule=schedule,
            schedule_hash=schedule_hash,
            records=tuple(ordered_records),
            record_refs=tuple(record_refs[index] for index in range(len(records))),
        )

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
    ) -> tuple[_PrivateScore, PublicEvaluationRecord]:
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
            public = self._validated_public_record(refs[0], batch)
            identity = self._identity(batch, public.case_id)
        else:
            if batch != _context.batch or binding.public_evaluation_run_id != (
                _context.evaluation.evaluation_run_id
            ):
                raise ValueError("holdout score transaction is invalid")
            try:
                ordinal = _context.evaluation.record_refs.index(refs[0])
                public = _context.evaluation.records[ordinal]
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
        return private_score, public
