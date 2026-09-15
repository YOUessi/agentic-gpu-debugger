"""Evaluator-owned holdout aliases and private score bindings."""

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

from pydantic import Field

from gpu_agent.benchmark.evaluation import (
    EvaluationAttempt,
    EvaluationSchedule,
    HoldoutScheduleProof,
    PublicEvaluationRecord,
)
from gpu_agent.benchmark.metrics import (
    _EVALUATOR_AUTHORITY,
    EvaluationLabels,
    Score,
    ValidatedEvaluationRecord,
)
from gpu_agent.contracts import ArtifactRef, ExternalRunOrigin, RunBinding, RunStatus
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.store import RunStore, reject_symlinks

_SCORE_GUARD = threading.Lock()
_SCORE_LOCKS: dict[str, threading.Lock] = {}


class HoldoutBatch(ExecutionModel):
    public_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    evaluator_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    aliases: list[str]
    public_alias_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class EvaluatorRecordBinding(ExecutionModel):
    evaluator_score_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    public_evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    public_record_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    public_record_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    private_case_id: str = Field(min_length=1)
    private_template_id: str = Field(min_length=1)
    private_score_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class _PrivateIdentity(ExecutionModel):
    alias: str = Field(pattern=r"^[a-f0-9]{64}$")
    private_case_id: str = Field(min_length=1)
    private_template_id: str = Field(min_length=1)


class _PrivateAliasMap(ExecutionModel):
    schema_version: int = 1
    nonce_hex: str = Field(pattern=r"^[a-f0-9]{64}$")
    identities: list[_PrivateIdentity]


class _PrivateScore(ExecutionModel):
    schema_version: int = 1
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

    def __init__(self, public: RunStore, evaluator: RunStore, *, binding: RunBinding) -> None:
        if (
            public.visibility != "public"
            or evaluator.visibility != "evaluator"
            or binding.purpose != "evaluation"
        ):
            raise ValueError("holdout controller requires bound split stores")
        self.public, self.evaluator, self.binding = public, evaluator, binding

    def prepare(self, private_identities: list[tuple[str, str]]) -> HoldoutBatch:
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
            {"schema_version": 1, "aliases": [item.alias for item in identities]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        public_ref = self.public.put(
            public_run.id, "holdout/aliases.json", public_content, "public"
        )
        mapping = _PrivateAliasMap(nonce_hex=nonce.hex(), identities=identities)
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
        if public_payload != {"schema_version": 1, "aliases": batch.aliases}:
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
        if recomputed != batch.aliases or [item.alias for item in mapping.identities] != recomputed:
            raise ValueError("holdout binding is invalid")
        return HoldoutScheduleProof(public_run_id=batch.public_run_id, aliases_hash=refs[0].sha256)

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
        identity = self._identity(batch, alias)
        record = self._validated_public_record(public_record_ref, batch)
        if record.case_id != alias or record.template_id != alias:
            raise ValueError("public evaluation record is invalid")
        private_score = _PrivateScore(
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
        )
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
                run.id, "holdout/private-score.json", private_score.content(), "evaluator"
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
            self._load_score(binding)
            return binding

    def validated_record(self, binding: EvaluatorRecordBinding) -> ValidatedEvaluationRecord:
        """Resolve evaluator-private score and native public lineage for metric consumers."""
        private_score, public = self._load_score(binding)
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
        from gpu_agent.benchmark.evaluation import EvaluationRecord

        return ValidatedEvaluationRecord(EvaluationRecord.model_validate(raw), _EVALUATOR_AUTHORITY)

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
            record, schedule, ordinal = self._resolve_public_record(ref)
            if batch is not None:
                proof = self.validate_batch(batch)
                item = schedule.items[ordinal]
                if (
                    schedule.split != "holdout"
                    or schedule.holdout_proof != proof
                    or item.split != "holdout"
                    or item.holdout_proof != proof
                    or item.case_id != record.case_id
                    or item.template_id != record.template_id
                ):
                    raise ValueError("public evaluation record is invalid")
            return record
        except (ValueError, KeyError, IndexError):
            raise ValueError("public evaluation record is invalid") from None

    def _resolve_public_record(
        self, ref: ArtifactRef
    ) -> tuple[PublicEvaluationRecord, EvaluationSchedule, int]:
        if ref.visibility != "public" or not re.fullmatch(
            r"evaluation/records/[0-9]+\.json", ref.name
        ):
            raise ValueError("public evaluation record is invalid")
        run = self.public.load(ref.run_id)
        if (
            run.kind != "evaluation"
            or run.status not in {RunStatus.COMPLETED, RunStatus.FAILED}
            or run.binding != self.binding
            or ref not in run.artifact_refs
        ):
            raise ValueError("public evaluation record is invalid")
        ordinal = int(ref.name.split("/")[-1].removesuffix(".json"))
        schedule_refs = [r for r in run.artifact_refs if r.name == "evaluation/schedule.json"]
        attempt_refs = [
            r for r in run.artifact_refs if r.name == f"evaluation/attempts/{ordinal}.json"
        ]
        if len(schedule_refs) != 1 or len(attempt_refs) != 1:
            raise ValueError("public evaluation record is invalid")
        schedule = EvaluationSchedule.model_validate_json(self.public.read(schedule_refs[0]))
        if ordinal >= len(schedule.items):
            raise ValueError("public evaluation record is invalid")
        attempt = EvaluationAttempt.model_validate_json(self.public.read(attempt_refs[0]))
        record = PublicEvaluationRecord.model_validate_json(self.public.read(ref))
        from gpu_agent.benchmark.executor import validate_evaluation_record

        validate_evaluation_record(
            self.public, record, schedule.items[ordinal], attempt, self.binding
        )
        return record, schedule, ordinal

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

    def _load_score(
        self, binding: EvaluatorRecordBinding
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
        alias_refs = [
            ref for ref in public_alias_run.artifact_refs if ref.name == "holdout/aliases.json"
        ]
        if len(alias_refs) != 1:
            raise ValueError("holdout score transaction is invalid")
        alias_payload = json.loads(self.public.read(alias_refs[0]))
        batch = HoldoutBatch(
            public_run_id=public_alias_run.id,
            evaluator_run_id=mapping_run.id,
            aliases=alias_payload.get("aliases", []),
            public_alias_hash=alias_refs[0].sha256,
        )
        public = self._validated_public_record(refs[0], batch)
        identity = self._identity(batch, public.case_id)
        if (
            public.record_id != binding.public_record_id
            or identity.private_case_id != binding.private_case_id
            or identity.private_template_id != binding.private_template_id
        ):
            raise ValueError("holdout score transaction is invalid")
        return private_score, public
