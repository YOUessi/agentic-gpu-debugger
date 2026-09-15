"""Evaluator-owned holdout aliases and private score bindings."""

import hashlib
import hmac
import json
import re
import secrets

from pydantic import Field

from gpu_agent.benchmark.evaluation import (
    EvaluationAttempt,
    EvaluationSchedule,
    PublicEvaluationRecord,
)
from gpu_agent.benchmark.metrics import EvaluationLabels, Score
from gpu_agent.contracts import ArtifactRef, ExternalRunOrigin, RunBinding, RunStatus
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.store import RunStore


class HoldoutBatch(ExecutionModel):
    public_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    evaluator_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    aliases: list[str]


class EvaluatorRecordBinding(ExecutionModel):
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
        self.public.put(public_run.id, "holdout/aliases.json", public_content, "public")
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
        )

    def bind_score(
        self,
        batch: HoldoutBatch,
        alias: str,
        public_record_ref: ArtifactRef,
        *,
        labels: EvaluationLabels | None,
        score: Score,
    ) -> EvaluatorRecordBinding:
        if labels is None:
            raise ValueError("private evaluation labels are required")
        identity = self._identity(batch, alias)
        record = self._validated_public_record(public_record_ref)
        if record.case_id != alias or record.template_id != alias:
            raise ValueError("public evaluation record is invalid")
        private_score = _PrivateScore(
            public_record_hash=public_record_ref.sha256, labels=labels, score=score
        )
        binding = EvaluatorRecordBinding(
            public_record_id=record.record_id,
            public_record_hash=public_record_ref.sha256,
            private_case_id=identity.private_case_id,
            private_template_id=identity.private_template_id,
            private_score_hash=hashlib.sha256(private_score.content()).hexdigest(),
        )
        for path in self.evaluator.root.iterdir():
            if not path.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", path.name):
                continue
            existing_run = self.evaluator.load(path.name)
            if existing_run.kind != "holdout_score":
                continue
            refs = [
                ref
                for ref in existing_run.artifact_refs
                if ref.name == "holdout/record-binding.json"
            ]
            if len(refs) != 1:
                raise ValueError("holdout score history is invalid")
            existing = EvaluatorRecordBinding.model_validate_json(self.evaluator.read(refs[0]))
            if existing.public_record_id == binding.public_record_id:
                raise ValueError("public evaluation record was already scored")
        run = self.evaluator.create_run("holdout_score", parent_run_id=batch.evaluator_run_id)
        self.evaluator.transition(run.id, RunStatus.RUNNING, "FINALIZING")
        self.evaluator.put(
            run.id, "holdout/private-score.json", private_score.content(), "evaluator"
        )
        self.evaluator.put(
            run.id,
            "holdout/record-binding.json",
            binding.model_dump_json().encode(),
            "evaluator",
        )
        self.evaluator.transition(run.id, RunStatus.COMPLETED, None)
        return binding

    def resolve_private(self, batch: HoldoutBatch, alias: str) -> tuple[str, str]:
        """Resolve privately; callers must never persist the result publicly."""
        identity = self._identity(batch, alias)
        return identity.private_case_id, identity.private_template_id

    def _identity(self, batch: HoldoutBatch, alias: str) -> _PrivateIdentity:
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

    def _validated_public_record(self, ref: ArtifactRef) -> PublicEvaluationRecord:
        try:
            return self._resolve_public_record(ref)
        except (ValueError, KeyError, IndexError):
            raise ValueError("public evaluation record is invalid") from None

    def _resolve_public_record(self, ref: ArtifactRef) -> PublicEvaluationRecord:
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
        return record
