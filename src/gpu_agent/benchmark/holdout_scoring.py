"""Evaluator-owned holdout judgment validation and durable scoring sessions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, ValidationError

from gpu_agent.benchmark.controller_artifacts import read_private_external
from gpu_agent.benchmark.evaluation import EvaluationSchedule, PublicEvaluationRecord
from gpu_agent.benchmark.holdout import (
    EvaluatorRecordBinding,
    HoldoutBatch,
    HoldoutController,
    PreparedHoldoutScore,
)
from gpu_agent.benchmark.metrics import EvaluationLabels, Score, aggregate_grouped
from gpu_agent.contracts import (
    ArtifactRef,
    CurrentPhase,
    ExternalRunOrigin,
    RepositorySnapshot,
    RunBinding,
    RunManifest,
    RunStatus,
)
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.provenance import capture_repository_snapshot
from gpu_agent.store import RunStore, read_regular, reject_symlinks

if TYPE_CHECKING:
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

_SESSION_GUARD = threading.Lock()
_SESSION_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_SESSION_ARTIFACT_NAMES = tuple(
    f"holdout-scoring/{name}.json" for name in ("input-binding", "bindings", "metrics", "result")
)
_SESSION_ARTIFACT_PREFIXES = {
    _SESSION_ARTIFACT_NAMES[:length] for length in range(len(_SESSION_ARTIFACT_NAMES) + 1)
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _tracked_rubric_hash(repository: Path, commit: str = "HEAD") -> str:
    limit = 16 * 1024 * 1024

    def blob_command(option: str) -> bytes:
        result = subprocess.run(
            [
                "/usr/bin/git",
                "--no-replace-objects",
                "-C",
                str(repository),
                "cat-file",
                option,
                f"{commit}:evaluation/rubric.md",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError("invalid rubric object")
        return result.stdout

    try:
        if blob_command("-t") != b"blob\n" or not 0 <= int(blob_command("-s")) <= limit:
            raise ValueError("invalid rubric object")
        committed = blob_command("blob")
        if (
            len(committed) > limit
            or read_regular(repository / "evaluation/rubric.md", limit) != committed
        ):
            raise ValueError("rubric content mismatch")
    except (ValueError, OSError, subprocess.SubprocessError):
        raise ValueError("holdout committed rubric is invalid") from None
    return _hash(committed)


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
    public_record: PublicEvaluationRecord
    private_case_id: str = Field(min_length=1)
    private_template_id: str = Field(min_length=1)
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


class HoldoutScoringResult(ExecutionModel):
    schema_version: Literal[1] = 1
    scoring_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    private_binding_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    package_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    scored_count: Literal[120] = 120
    expected_record_count: Literal[120] = 120
    input_binding_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    bindings_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    metrics_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class HoldoutScoringController:
    """Validate and score a complete package under one durable evaluator claim."""

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
        self.schedule_verifier = self.holdout._schedule_verifier

    @contextmanager
    def _session_claim(self, run_id: str) -> Iterator[None]:
        key = (str(self.evaluator.root), run_id)
        with _SESSION_GUARD:
            local = _SESSION_LOCKS.setdefault(key, threading.Lock())
        with local:
            path = self.evaluator.root / f".holdout-scoring-{run_id}.lock"
            try:
                reject_symlinks(path)
                # A terminal retry must never even create a replacement lock.
                exists = (self.evaluator.root / run_id).exists()
                completed = exists and self.evaluator.load(run_id).status == RunStatus.COMPLETED
                flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
                fd = os.open(path, flags if completed else flags | os.O_CREAT, 0o600)
            except (OSError, ValueError):
                raise ValueError("holdout scoring claim is unsafe") from None
            try:
                info = os.fstat(fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077
                    or info.st_nlink != 1
                ):
                    raise ValueError("holdout scoring claim is unsafe")
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)

    def _check_package_claim(self, run_id: str, labels_path: Path, repository: Path) -> None:
        """Give a losing package one stable conflict, including before native preflight."""
        content = read_private_external(
            labels_path,
            repository=repository,
            forbidden_roots=(self.public.root, self.evaluator.root),
            limit=16 * 1024 * 1024,
        )
        if not (self.evaluator.root / run_id).exists():
            return
        run = self.evaluator.load(run_id)
        refs = [r for r in run.artifact_refs if r.name == "holdout-scoring/input-binding.json"]
        if len(refs) > 1:
            raise ValueError("holdout scoring session conflicts")
        if refs:
            if json.loads(self.evaluator.read(refs[0])).get("package_hash") != _hash(content):
                raise ValueError("holdout scoring package conflicts")

    def _session(self, run_id: str, evaluation_run_id: str) -> RunManifest:
        origin = ExternalRunOrigin(run_id=evaluation_run_id, visibility="public")
        if not (self.evaluator.root / run_id).exists():
            return self.evaluator.create_run(
                "holdout_scoring",
                binding=self.binding,
                external_origin=origin,
                _run_id=run_id,
            )
        try:
            run = self.evaluator.load(run_id)
        except ValueError:
            raise ValueError("holdout scoring session conflicts") from None
        names = tuple(ref.name for ref in run.artifact_refs)
        events = tuple((event.status, event.phase) for event in run.events)
        queued_events = ((RunStatus.QUEUED, None),)
        preparing_events = (*queued_events, (RunStatus.RUNNING, CurrentPhase.PREPARING))
        finalizing_events = (*preparing_events, (RunStatus.RUNNING, CurrentPhase.FINALIZING))
        completed_events = (*finalizing_events, (RunStatus.COMPLETED, None))
        valid_state = (
            (
                run.status == RunStatus.QUEUED
                and run.current_phase is None
                and run.last_completed_phase is None
                and names == ()
                and events == queued_events
            )
            or (
                run.status == RunStatus.RUNNING
                and run.current_phase == CurrentPhase.PREPARING
                and run.last_completed_phase is None
                and names in _SESSION_ARTIFACT_PREFIXES
                and events == preparing_events
            )
            or (
                run.status == RunStatus.RUNNING
                and run.current_phase == CurrentPhase.FINALIZING
                and run.last_completed_phase == CurrentPhase.PREPARING
                and names == _SESSION_ARTIFACT_NAMES
                and events == finalizing_events
            )
            or (
                run.status == RunStatus.COMPLETED
                and run.current_phase is None
                and run.last_completed_phase == CurrentPhase.FINALIZING
                and names == _SESSION_ARTIFACT_NAMES
                and events == completed_events
            )
        )
        if (
            run.kind != "holdout_scoring"
            or run.parent_run_id is not None
            or run.binding != self.binding
            or run.external_origin != origin
            or any(
                ref.run_id != run_id or ref.visibility != "evaluator" for ref in run.artifact_refs
            )
            or not valid_state
        ):
            raise ValueError("holdout scoring session conflicts")
        return run

    def score(
        self,
        evaluation_run_id: str,
        private_binding_run_id: str,
        labels_path: Path,
        repository: Path,
    ) -> HoldoutScoringResult:
        if any(
            re.fullmatch(r"[a-f0-9]{32}", value) is None
            for value in (evaluation_run_id, private_binding_run_id)
        ):
            raise ValueError("holdout scoring roots are invalid")
        run_id = _hash(f"holdout-scoring-v1:{evaluation_run_id}:{private_binding_run_id}".encode())[
            :32
        ]
        self._check_package_claim(run_id, labels_path, repository)
        # Invalid initial evidence cannot leave even a claim file in either store.
        try:
            self.preflight(evaluation_run_id, private_binding_run_id, labels_path, repository)
        except ValueError:
            self._check_package_claim(run_id, labels_path, repository)
            raise
        with self._session_claim(run_id):
            self._check_package_claim(run_id, labels_path, repository)
            plan = self.preflight(
                evaluation_run_id, private_binding_run_id, labels_path, repository
            )
            run = self._session(run_id, evaluation_run_id)
            completed = run.status == RunStatus.COMPLETED
            if run.status == RunStatus.QUEUED:
                run = self.evaluator.transition(run_id, RunStatus.RUNNING, "PREPARING")
            input_content = _canonical(
                {
                    **plan.package.model_dump(mode="json", exclude={"judgments"}),
                    "package_hash": plan.package_hash,
                }
            )

            def persist(name: str, content: bytes) -> None:
                name = f"holdout-scoring/{name}.json"
                if completed:
                    refs = [r for r in run.artifact_refs if r.name == name]
                    if len(refs) != 1 or self.evaluator.read(refs[0]) != content:
                        raise ValueError("holdout scoring artifacts conflict")
                else:
                    self.evaluator.put_if_absent_exact(run_id, name, content, "evaluator")

            persist("input-binding", input_content)
            bindings: list[EvaluatorRecordBinding] = []
            for ordinal, item in enumerate(plan.items):
                if item.ordinal != ordinal:
                    raise ValueError("holdout scoring plan order is invalid")
                binding = item.existing_binding
                if binding is None:
                    if completed:
                        raise ValueError("holdout scoring completed session is incomplete")
                    binding = self.holdout._bind_prepared_score(
                        plan.batch,
                        item.prepared_score,
                        alias=item.alias,
                        public_record_ref=item.public_record_ref,
                        public_record=item.public_record,
                        private_case_id=item.private_case_id,
                        private_template_id=item.private_template_id,
                        _reload=False,
                    )
                bindings.append(binding)
            bindings_content = _canonical([b.model_dump(mode="json") for b in bindings])
            persist("bindings", bindings_content)
            metrics = aggregate_grouped(
                bindings,
                public_store=self.public,
                evaluator_store=self.evaluator,
                run_binding=self.binding,
                schedule_verifier=self.schedule_verifier,
            )
            metrics_content = _canonical(metrics.model_dump(mode="json"))
            persist("metrics", metrics_content)
            result = HoldoutScoringResult(
                scoring_run_id=run_id,
                evaluation_run_id=evaluation_run_id,
                private_binding_run_id=private_binding_run_id,
                package_hash=plan.package_hash,
                input_binding_hash=_hash(input_content),
                bindings_hash=_hash(bindings_content),
                metrics_hash=_hash(metrics_content),
            )
            persist("result", _canonical(result.model_dump(mode="json")))
            if completed:
                if len(run.artifact_refs) != 4:
                    raise ValueError("holdout scoring artifacts conflict")
            else:
                run = self.evaluator.load(run_id)
                if run.current_phase != CurrentPhase.FINALIZING:
                    self.evaluator.transition(run_id, RunStatus.RUNNING, "FINALIZING")
                self.evaluator.transition(run_id, RunStatus.COMPLETED, None)
            return result

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
        rubric_hash = _tracked_rubric_hash(repository, self.binding.repository.commit)
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
            or [item.ordinal for item in schedule.items] != list(range(len(schedule.items)))
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
            identity = identities[record.case_id]
            prepared = self.holdout._prepare_validated_score(
                batch,
                record.case_id,
                ref,
                record,
                labels=judgment.labels,
                score=judgment.score,
                should_be_inconclusive=judgment.should_be_inconclusive,
                private_holdout_passed=judgment.private_holdout_passed,
                identity=identity,
            )
            items.append(
                HoldoutScoringPlanItem(
                    ordinal=ordinal,
                    alias=record.case_id,
                    public_record_ref=ref,
                    public_record=record,
                    private_case_id=identity.private_case_id,
                    private_template_id=identity.private_template_id,
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
                        or any(ref.run_id != child.id for ref in child.artifact_refs)
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
