"""Read-only evaluator-owned holdout judgment validation and score preparation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, ValidationError

from gpu_agent.benchmark.controller_artifacts import read_private_external
from gpu_agent.benchmark.evaluation import EvaluationSchedule
from gpu_agent.benchmark.holdout import (
    EvaluatorRecordBinding,
    HoldoutBatch,
    HoldoutController,
    PreparedHoldoutScore,
)
from gpu_agent.benchmark.metrics import EvaluationLabels, Score
from gpu_agent.contracts import ArtifactRef, RepositorySnapshot, RunBinding, RunStatus
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.provenance import capture_repository_snapshot
from gpu_agent.store import RunStore, read_regular

if TYPE_CHECKING:
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _tracked_rubric_hash(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "--error-unmatch", "--", "evaluation/rubric.md"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0 or result.stdout != b"evaluation/rubric.md\n":
        raise ValueError("holdout requires a tracked rubric")
    return _hash(read_regular(repository / "evaluation/rubric.md", 16 * 1024 * 1024))


def _require_existing_lock(store: RunStore, run_id: str) -> None:
    # evaluation_run_lease may create a lock for recovery. Preflight cannot.
    store.load(run_id)
    try:
        read_regular(store.root / run_id / ".lock", 1024)
    except (ValueError, OSError):
        raise ValueError("holdout preflight requires an existing run lock") from None


class HoldoutJudgment(ExecutionModel):
    blind_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    public_record_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    blind_payload_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    labels: EvaluationLabels
    score: Score
    should_be_inconclusive: bool
    private_holdout_passed: bool


class HoldoutLabelPackage(ExecutionModel):
    schema_version: Literal[1] = 1
    evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    private_binding_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    schedule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    aliases_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)
    expected_record_count: Literal[120] = 120
    record_set_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    rubric_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    judgments: tuple[HoldoutJudgment, ...] = Field(min_length=120, max_length=120)


class HoldoutScoringPlanItem(ExecutionModel):
    ordinal: int = Field(ge=0, lt=120)
    alias: str = Field(pattern=r"^[a-f0-9]{64}$")
    public_record_ref: ArtifactRef
    judgment: HoldoutJudgment
    prepared_score: PreparedHoldoutScore
    existing_binding: EvaluatorRecordBinding | None = None


class HoldoutScoringPlan(ExecutionModel):
    evaluation_run_id: str
    private_binding_run_id: str
    batch: HoldoutBatch
    package: HoldoutLabelPackage
    package_hash: str
    repository_snapshot: RepositorySnapshot
    items: tuple[HoldoutScoringPlanItem, ...] = Field(min_length=120, max_length=120)


class HoldoutScoringController:
    """Validate a complete evaluator package before a future persistence operation."""

    def __init__(
        self,
        public: RunStore,
        evaluator: RunStore,
        *,
        binding: RunBinding,
        _schedule_verifier: EvaluationScheduleVerifier | None = None,
    ) -> None:
        self.public, self.evaluator, self.binding = public, evaluator, binding
        self.holdout = HoldoutController(
            public, evaluator, binding=binding, _schedule_verifier=_schedule_verifier
        )

    def preflight(
        self,
        evaluation_run_id: str,
        private_binding_run_id: str,
        labels_path: Path,
        repository: Path,
    ) -> HoldoutScoringPlan:
        """Return all 120 prepared scores, with no store or session writes."""
        snapshot = capture_repository_snapshot(
            repository, expected_commit=self.binding.repository.commit
        )
        if snapshot != self.binding.repository:
            raise ValueError("holdout repository binding is invalid")
        family = self.holdout._schedule_family
        family.require_store(self.public)
        family.require_store(self.evaluator)
        family.reject_repository_overlap(repository)
        content = read_private_external(
            labels_path,
            repository=repository,
            forbidden_roots=(self.public.root, self.evaluator.root),
            limit=16 * 1024 * 1024,
        )
        try:
            package = HoldoutLabelPackage.model_validate_json(content)
        except ValidationError:
            raise ValueError("holdout label package is incomplete or invalid") from None
        if content != _canonical(package.model_dump(mode="json")):
            raise ValueError("holdout label package is noncanonical")
        mapping = self.evaluator.load(private_binding_run_id)
        _require_existing_lock(self.evaluator, mapping.id)
        if mapping.external_origin is None or mapping.external_origin.visibility != "public":
            raise ValueError("holdout binding is invalid")
        alias_run = self.public.load(mapping.external_origin.run_id)
        alias_refs = [ref for ref in alias_run.artifact_refs if ref.name == "holdout/aliases.json"]
        if len(alias_refs) != 1:
            raise ValueError("holdout binding is invalid")
        alias_payload = json.loads(self.public.read(alias_refs[0]))
        batch = HoldoutBatch(
            public_run_id=alias_run.id,
            evaluator_run_id=mapping.id,
            aliases=alias_payload.get("aliases", []),
            public_alias_hash=alias_refs[0].sha256,
            corpus_cutoff=alias_payload.get("corpus_cutoff", 0),
        )
        # Reject impossible package bindings before the expensive native scan.
        # The evaluation-wide loader below still authenticates the entire input.
        evaluation_run = self.public.load(evaluation_run_id)
        _require_existing_lock(self.public, evaluation_run.id)
        schedule_refs = [
            ref for ref in evaluation_run.artifact_refs if ref.name == "evaluation/schedule.json"
        ]
        if len(schedule_refs) != 1:
            raise ValueError("holdout evaluation schedule is invalid")
        schedule = EvaluationSchedule.model_validate_json(self.public.read(schedule_refs[0]))
        rubric_hash = _tracked_rubric_hash(repository)
        if (
            package.evaluation_run_id != evaluation_run_id
            or package.private_binding_run_id != private_binding_run_id
            or package.schedule_hash != _hash(_canonical(schedule.model_dump(mode="json")))
            or package.aliases_hash != batch.public_alias_hash
            or package.corpus_cutoff != batch.corpus_cutoff
            or package.rubric_hash != rubric_hash
        ):
            raise ValueError("holdout label package binding is invalid")
        evaluation = self.holdout.validated_evaluation(batch, evaluation_run_id)
        schedule = evaluation.schedule
        expected_product = set(product(batch.aliases, ("A", "B", "C", "D", "E"), range(3)))
        observed_product = {(item.case_id, item.mode, item.repeat) for item in schedule.items}
        if (
            schedule.selection != "all"
            or schedule.modes != ["A", "B", "C", "D", "E"]
            or schedule.repeats != 3
            or len(batch.aliases) != 8
            or len(schedule.items) != 120
            or len(evaluation.records) != 120
            or observed_product != expected_product
        ):
            raise ValueError("holdout evaluation requires the exact 8 x 5 x 3 Cartesian product")
        blind_hashes = [_hash(_canonical(record.blind())) for record in evaluation.records]
        record_set_hash = _hash(
            _canonical(
                [
                    (ordinal, ref.sha256, blind_hashes[ordinal])
                    for ordinal, ref in enumerate(evaluation.record_refs)
                ]
            )
        )
        if (
            package.evaluation_run_id != evaluation_run_id
            or package.private_binding_run_id != private_binding_run_id
            or package.schedule_hash != evaluation.schedule_hash
            or package.aliases_hash != batch.public_alias_hash
            or package.corpus_cutoff != batch.corpus_cutoff
            or package.record_set_hash != record_set_hash
            or package.rubric_hash != rubric_hash
        ):
            raise ValueError("holdout label package binding is invalid")
        judgments = {judgment.blind_id: judgment for judgment in package.judgments}
        blind_ids = [str(record.blind()["blind_id"]) for record in evaluation.records]
        if len(judgments) != 120 or set(judgments) != set(blind_ids):
            raise ValueError("holdout label package is incomplete")
        items: list[HoldoutScoringPlanItem] = []
        identities = {alias: self.holdout._identity(batch, alias) for alias in batch.aliases}
        for ordinal, (record, ref) in enumerate(
            zip(evaluation.records, evaluation.record_refs, strict=True)
        ):
            judgment = judgments[blind_ids[ordinal]]
            if (
                judgment.public_record_hash != ref.sha256
                or judgment.blind_payload_hash != blind_hashes[ordinal]
            ):
                raise ValueError("holdout label package record binding is invalid")
            prepared = self.holdout._prepare_validated_score(
                batch,
                record.case_id,
                ref,
                record,
                labels=judgment.labels,
                score=judgment.score,
                should_be_inconclusive=judgment.should_be_inconclusive,
                private_holdout_passed=judgment.private_holdout_passed,
                identity=identities[record.case_id],
            )
            items.append(
                HoldoutScoringPlanItem(
                    ordinal=ordinal,
                    alias=record.case_id,
                    public_record_ref=ref,
                    judgment=judgment,
                    prepared_score=prepared,
                )
            )
        expected = {item.prepared_score.run_id: item for item in items}
        existing: dict[str, EvaluatorRecordBinding] = {}
        try:
            with self.evaluator.evaluation_run_lease(mapping.id) as lease:
                children = lease.children()
                seen_records: set[str] = set()
                for child in children:
                    item = expected.get(child.id)
                    if (
                        item is None
                        or child.kind != "holdout_score"
                        or child.status != RunStatus.COMPLETED
                        or child.binding != self.binding
                        or child.parent_run_id != mapping.id
                        or child.external_origin != mapping.external_origin
                    ):
                        raise ValueError("conflict")
                    binding_refs = [
                        r for r in child.artifact_refs if r.name == "holdout/record-binding.json"
                    ]
                    score_refs = [
                        r for r in child.artifact_refs if r.name == "holdout/private-score.json"
                    ]
                    prepared = item.prepared_score
                    if (
                        len(child.artifact_refs) != 2
                        or len(binding_refs) != 1
                        or len(score_refs) != 1
                        or self.evaluator.read(binding_refs[0])
                        != prepared.binding.model_dump_json().encode()
                        or self.evaluator.read(score_refs[0]) != prepared.private_score_content
                        or prepared.binding.public_record_id in seen_records
                    ):
                        raise ValueError("conflict")
                    seen_records.add(prepared.binding.public_record_id)
                    existing[child.id] = prepared.binding
                # A deterministic ID already occupied outside the mapping also conflicts.
                for run_id in expected.keys() - existing.keys():
                    if (self.evaluator.root / run_id).exists():
                        raise ValueError("conflict")
        except (ValueError, OSError):
            raise ValueError("existing holdout score conflicts") from None
        if capture_repository_snapshot(repository, expected_commit=snapshot.commit) != snapshot:
            raise ValueError("holdout repository changed during preflight")
        return HoldoutScoringPlan(
            evaluation_run_id=evaluation_run_id,
            private_binding_run_id=private_binding_run_id,
            batch=batch,
            package=package,
            package_hash=_hash(content),
            repository_snapshot=snapshot,
            items=tuple(
                item.model_copy(
                    update={"existing_binding": existing.get(item.prepared_score.run_id)}
                )
                for item in items
            ),
        )
