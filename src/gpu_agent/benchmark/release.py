"""Evidence-derived V2 release gate.

The release manifest is a set of claims. It is never an evidence source. Release
facts are reconstructed from an explicit selection, the corpus ledger and immutable
public/evaluator RunStore artifacts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, TypeAdapter, model_validator

from gpu_agent.benchmark.controller_artifacts import (
    validate_external_artifact_path,
    write_private_atomic_new,
)
from gpu_agent.benchmark.evaluation import (
    EvaluationAttempt,
    EvaluationManifest,
    EvaluationSchedule,
    PublicEvaluationRecord,
)
from gpu_agent.benchmark.holdout import (
    EvaluatorRecordBinding,
    _PrivateAliasMap,
    _PrivateScore,
)
from gpu_agent.benchmark.holdout_scoring import HoldoutScoringResult, _tracked_rubric_hash
from gpu_agent.benchmark.ledger import CorpusFamily
from gpu_agent.benchmark.models import CaseManifest
from gpu_agent.contracts import (
    ArtifactRef,
    CurrentPhase,
    ExternalRunOrigin,
    RepositorySnapshot,
    RunBinding,
    RunManifest,
    RunStatus,
)
from gpu_agent.execution.models import ExecutionModel, SanitizerTool
from gpu_agent.provenance import capture_repository_snapshot
from gpu_agent.store import RunStore


def validate_external_release_artifact_path(
    path: Path,
    repository: Path,
    *,
    forbidden_roots: Sequence[Path] = (),
) -> Path:
    """Reject release artifacts that could become part of the bound Git checkout."""
    try:
        return validate_external_artifact_path(
            path,
            repository=repository,
            forbidden_roots=forbidden_roots,
        )
    except ValueError as exc:
        raise ValueError("release artifact must use an external absolute path") from exc


def external_release_artifact_path(
    variable: str,
    repository: Path,
    *,
    forbidden_roots: Sequence[Path] = (),
) -> Path:
    """Resolve a required controller artifact without dirtying the bound repository."""
    configured = os.environ.get(variable)
    if not configured:
        raise ValueError(f"{variable} must name an external absolute path")
    try:
        return validate_external_release_artifact_path(
            Path(configured),
            repository,
            forbidden_roots=forbidden_roots,
        )
    except ValueError as exc:
        raise ValueError(f"{variable} must name an external absolute path") from exc


class TestCounts(ExecutionModel):
    expected: int = Field(ge=0)
    executed: int = Field(ge=0)
    skipped_required: int = Field(ge=0)
    failed: int = Field(ge=0)


class ReleaseManifest(ExecutionModel):
    """Declarative release claims; every field is checked against derived evidence."""

    schema_version: Literal[1] = 1
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    toolchain_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    test_counts: TestCounts
    public_case_count: int = Field(ge=0)
    private_case_count: int = Field(ge=0)
    evidence_run_ids: dict[str, list[str]]
    unresolved_items: list[str]

    @classmethod
    def from_evidence(cls, evidence: ReleaseEvidenceIndex) -> ReleaseManifest:
        """Create claims only after a complete evidence index has been derived."""
        if (
            evidence.reason_codes
            or evidence.toolchain_hash is None
            or evidence.corpus_hash is None
            or evidence.model_config_hash is None
        ):
            raise ValueError("complete release evidence is required")
        return cls(
            commit=evidence.repository.commit,
            toolchain_hash=evidence.toolchain_hash,
            corpus_hash=evidence.corpus_hash,
            model_config_hash=evidence.model_config_hash,
            test_counts=evidence.test_counts,
            public_case_count=evidence.public_case_count,
            private_case_count=evidence.private_case_count,
            evidence_run_ids=evidence.evidence_run_ids,
            unresolved_items=[],
        )


class ReleaseEvidenceSelection(ExecutionModel):
    """Frozen, explicit roots of the release evidence graph."""

    schema_version: Literal[1] = 1
    repository: RepositorySnapshot
    public_case_run_ids: list[str] = Field(min_length=1)
    private_case_run_ids: list[str] = Field(min_length=1)
    development_evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    holdout_evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    private_binding_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    acceptance_run_ids: dict[str, list[str]]
    release_test_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")

    @model_validator(mode="after")
    def unique_roots(self) -> ReleaseEvidenceSelection:
        for values in self.acceptance_run_ids.values():
            if len(values) != len(set(values)):
                raise ValueError("release evidence selection contains duplicate run IDs")
        isolation = self.acceptance_run_ids.get("isolation", [])
        if any(run_id != self.release_test_run_id for run_id in isolation):
            raise ValueError("isolation acceptance must reference the release test run")
        ids = [
            *self.public_case_run_ids,
            *self.private_case_run_ids,
            self.development_evaluation_run_id,
            self.holdout_evaluation_run_id,
            self.private_binding_run_id,
            self.release_test_run_id,
            *(
                run_id
                for category, values in self.acceptance_run_ids.items()
                if category != "isolation"
                for run_id in values
            ),
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("release evidence selection contains duplicate run IDs")
        return self


class ReleaseEvidenceRoots(ExecutionModel):
    """The four operator-selected roots from which release evidence is derived."""

    development_evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    holdout_evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    private_binding_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    release_test_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class _HoldoutScoringInputBinding(ExecutionModel):
    schema_version: Literal[1] = 1
    evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    private_binding_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    schedule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    aliases_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)
    expected_record_count: Literal[120] = 120
    record_set_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    rubric_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    package_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class ReleaseTestEvidence(ExecutionModel):
    """Bound result of the controller-owned release test invocation."""

    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    repository: RepositorySnapshot
    toolchain_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    prompt_version: str = Field(min_length=1, max_length=128)
    corpus_cutoff: int = Field(ge=1)
    test_counts: TestCounts
    collected_node_ids: list[str] = Field(min_length=1)
    collection_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    invocation_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    exit_code: Literal[0]


class ReleaseEvidenceIndex(ExecutionModel):
    """Facts derived from selected native evidence, never copied from a manifest."""

    schema_version: Literal[1] = 1
    repository: RepositorySnapshot
    toolchain_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    corpus_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    prompt_version: str | None = None
    corpus_cutoff: int | None = Field(default=None, ge=1)
    test_counts: TestCounts = Field(
        default_factory=lambda: TestCounts(expected=0, executed=0, skipped_required=0, failed=0)
    )
    public_case_count: int = Field(default=0, ge=0)
    private_case_count: int = Field(default=0, ge=0)
    private_template_count: int = Field(default=0, ge=0)
    private_operator_count: int = Field(default=0, ge=0)
    public_tool_case_counts: dict[str, int] = Field(default_factory=dict)
    development_units: int = Field(default=0, ge=0)
    holdout_units: int = Field(default=0, ge=0)
    evaluation_modes: list[str] = Field(default_factory=list)
    evaluation_repeats: int = Field(default=0, ge=0)
    evidence_run_ids: dict[str, list[str]] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)

    @classmethod
    def derive(
        cls,
        selection: ReleaseEvidenceSelection,
        public_store: RunStore,
        evaluator_store: RunStore,
        corpus_family: CorpusFamily,
        repository: Path,
        actual_repository: RepositorySnapshot,
    ) -> ReleaseEvidenceIndex:
        """Derive the complete index, returning a typed closed gate on any defect."""

        try:
            roots = ReleaseEvidenceRoots(
                development_evaluation_run_id=selection.development_evaluation_run_id,
                holdout_evaluation_run_id=selection.holdout_evaluation_run_id,
                private_binding_run_id=selection.private_binding_run_id,
                release_test_run_id=selection.release_test_run_id,
            )
            return (
                _ReleaseEvidenceResolver(
                    roots,
                    public_store,
                    evaluator_store,
                    corpus_family,
                    repository,
                    actual_repository,
                    expected_selection=selection,
                )
                .resolve()
                .evidence
            )
        except _ReleaseEvidenceError as exc:
            return cls(repository=actual_repository, reason_codes=[exc.code])
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return cls(
                repository=actual_repository,
                reason_codes=["EVIDENCE_DERIVATION_FAILED"],
            )


class ReleaseEvidenceResolution(ExecutionModel):
    """A canonical selection and the evidence derived from exactly that selection."""

    selection: ReleaseEvidenceSelection
    evidence: ReleaseEvidenceIndex


class FrozenReleaseSelection(ExecutionModel):
    """Safe publication summary for one canonical release selection."""

    output: Path
    selection: ReleaseEvidenceSelection
    selection_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)
    public_case_count: int = Field(ge=0)
    private_case_count: int = Field(ge=0)
    acceptance_run_count: int = Field(ge=0)


class ReleaseGateResult(ExecutionModel):
    passed: bool
    reason_codes: list[str]


class ReleaseGate:
    REQUIRED_ACCEPTANCE = {
        "four_tools",
        "isolation",
        "private_oracle",
        "live_llm",
    }
    REQUIRED_MODES = {"A", "B", "C", "D", "E"}

    def check(
        self,
        manifest: ReleaseManifest,
        evidence: ReleaseEvidenceIndex | None = None,
    ) -> ReleaseGateResult:
        reasons: list[str] = []
        counts = manifest.test_counts
        if counts.expected == 0:
            reasons.append("REQUIRED_TESTS_MISSING")
        if counts.executed != counts.expected:
            reasons.append("TEST_COUNT_INCOMPLETE")
        if counts.skipped_required:
            reasons.append("REQUIRED_TEST_SKIPPED")
        if counts.failed:
            reasons.append("TEST_FAILURE")
        if manifest.public_case_count < 16 or manifest.private_case_count < 8:
            reasons.append("CORPUS_COUNT_INSUFFICIENT")
        if manifest.unresolved_items:
            reasons.append("UNRESOLVED_ITEMS")
        if evidence is None:
            reasons.append("EVIDENCE_INDEX_REQUIRED")
            return ReleaseGateResult(passed=False, reason_codes=_deduplicate(reasons))

        reasons.extend(evidence.reason_codes)
        if evidence.repository.commit != manifest.commit:
            reasons.append("REPOSITORY_BINDING_MISMATCH")
        if (
            evidence.toolchain_hash != manifest.toolchain_hash
            or evidence.corpus_hash != manifest.corpus_hash
            or evidence.model_config_hash != manifest.model_config_hash
        ):
            reasons.append("CONFIGURATION_BINDING_MISMATCH")
        if evidence.test_counts != manifest.test_counts:
            reasons.append("TEST_EVIDENCE_MISMATCH")
        if (
            evidence.public_case_count != manifest.public_case_count
            or evidence.private_case_count != manifest.private_case_count
        ):
            reasons.append("CORPUS_CLAIM_MISMATCH")
        if evidence.public_case_count < 16 or evidence.private_case_count < 8:
            reasons.append("CORPUS_COUNT_INSUFFICIENT")
        if evidence.private_template_count < 8 or evidence.private_operator_count < 8:
            reasons.append("PRIVATE_DIVERSITY_INSUFFICIENT")
        if any(evidence.public_tool_case_counts.get(tool.value, 0) < 4 for tool in SanitizerTool):
            reasons.append("SANITIZER_FAMILY_COVERAGE_INSUFFICIENT")
        if (
            set(evidence.evaluation_modes) != self.REQUIRED_MODES
            or evidence.evaluation_repeats != 3
            or evidence.development_units != 16 * 5 * 3
            or evidence.holdout_units != 8 * 5 * 3
        ):
            reasons.append("FIVE_MODE_EVALUATION_INCOMPLETE")
        if manifest.evidence_run_ids != evidence.evidence_run_ids:
            reasons.append("EVIDENCE_SELECTION_MISMATCH")
        missing = self.REQUIRED_ACCEPTANCE - {
            key for key, run_ids in evidence.evidence_run_ids.items() if run_ids
        }
        if missing:
            reasons.append("LIVE_EVIDENCE_MISSING")
        return ReleaseGateResult(passed=not reasons, reason_codes=_deduplicate(reasons))


class _ReleaseEvidenceError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _ReleaseRootError(_ReleaseEvidenceError):
    """A malformed or foreign caller-selected root, not derived evidence."""


class _EvaluationEvidence:
    def __init__(
        self,
        run: RunManifest,
        binding: RunBinding,
        schedule: EvaluationSchedule,
        manifest: EvaluationManifest,
        records: list[PublicEvaluationRecord],
        record_hashes: dict[int, str],
    ) -> None:
        self.run = run
        self.binding = binding
        self.schedule = schedule
        self.manifest = manifest
        self.records = records
        self.record_hashes = record_hashes


class _ReleaseEvidenceResolver:
    def __init__(
        self,
        roots: ReleaseEvidenceRoots,
        public: RunStore,
        evaluator: RunStore,
        family: CorpusFamily,
        repository: Path,
        actual_repository: RepositorySnapshot,
        *,
        expected_selection: ReleaseEvidenceSelection | None = None,
    ) -> None:
        self.roots = roots
        self.public = public
        self.evaluator = evaluator
        self.family = family
        self.repository = repository
        self.actual_repository = actual_repository
        self.expected_selection = expected_selection

    def resolve(self) -> ReleaseEvidenceResolution:
        self._validate_roots()
        development = self._evaluation(self.roots.development_evaluation_run_id, "development")
        holdout = self._evaluation(self.roots.holdout_evaluation_run_id, "holdout")
        self._same_evaluation_binding(development, holdout)
        cutoff = development.schedule.corpus_cutoff
        public_cases, private_cases, public_run_ids, private_run_ids = self._corpus(
            development.binding, cutoff
        )
        self._validate_development_records(development, public_cases)
        private_templates, private_score_bindings, private_score_run_ids = (
            self._validate_holdout_records(holdout, private_cases, cutoff)
        )
        scoring_run_id = self._scoring_session(holdout, private_score_bindings)
        test_counts = self._release_tests(development.binding, cutoff)
        acceptance = self._acceptance(
            public_cases,
            development,
            holdout,
            private_score_run_ids,
        )
        corpus_hash = self._corpus_hash(public_cases, private_cases, cutoff)
        evidence_ids = {
            **acceptance,
            "five_mode_evaluation": [development.run.id, holdout.run.id],
            "public_corpus": public_run_ids,
            "private_corpus": private_run_ids,
            "private_scoring": [self.roots.private_binding_run_id, scoring_run_id],
            "release_tests": [self.roots.release_test_run_id],
        }
        tool_counts = Counter(case.target_tool.value for case in public_cases.values())
        selection = ReleaseEvidenceSelection(
            repository=self.actual_repository,
            public_case_run_ids=public_run_ids,
            private_case_run_ids=private_run_ids,
            development_evaluation_run_id=development.run.id,
            holdout_evaluation_run_id=holdout.run.id,
            private_binding_run_id=self.roots.private_binding_run_id,
            acceptance_run_ids=acceptance,
            release_test_run_id=self.roots.release_test_run_id,
        )
        evidence = ReleaseEvidenceIndex(
            repository=self.actual_repository,
            toolchain_hash=development.binding.toolchain_lock_hash,
            corpus_hash=corpus_hash,
            model_config_hash=development.binding.model_config_hash,
            prompt_version=development.binding.prompt_version,
            corpus_cutoff=cutoff,
            test_counts=test_counts,
            public_case_count=len(public_cases),
            private_case_count=len(private_cases),
            private_template_count=len(private_templates),
            private_operator_count=len({case.mutation_id for case in private_cases.values()}),
            public_tool_case_counts=dict(tool_counts),
            development_units=len(development.records),
            holdout_units=len(holdout.records),
            evaluation_modes=list(development.schedule.modes),
            evaluation_repeats=development.schedule.repeats,
            evidence_run_ids=evidence_ids,
        )
        return ReleaseEvidenceResolution(selection=selection, evidence=evidence)

    def _validate_roots(self) -> None:
        if (
            self.expected_selection is not None
            and self.expected_selection.repository != self.actual_repository
        ):
            raise _ReleaseRootError("ACTUAL_REPOSITORY_MISMATCH")
        try:
            self.family.require_store(self.public)
            self.family.require_store(self.evaluator)
        except ValueError as exc:
            raise _ReleaseRootError("STORE_FAMILY_MISMATCH") from exc
        if self.public.visibility != "public" or self.evaluator.visibility != "evaluator":
            raise _ReleaseRootError("STORE_FAMILY_MISMATCH")

    def _evaluation(
        self, run_id: str, split: Literal["development", "holdout"]
    ) -> _EvaluationEvidence:
        try:
            from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

            try:
                run = self.public.load(run_id)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise _ReleaseRootError("EVALUATION_EVIDENCE_INVALID") from exc
            if (
                run.kind != "evaluation"
                or run.parent_run_id is not None
                or run.external_origin is not None
                or run.status != RunStatus.COMPLETED
                or run.binding is None
                or run.binding.purpose != "evaluation"
                or run.binding.repository != self.actual_repository
                or run.binding.toolchain_lock_hash is None
                or run.binding.prompt_version is None
                or run.binding.model_config_hash is None
            ):
                raise _ReleaseRootError("EVALUATION_EVIDENCE_INVALID")
            refs = _owned_artifacts(
                run,
                expected_visibility="public",
                expected_names=None,
                expected_external_origin=None,
            )
            verifier = EvaluationScheduleVerifier.for_family(self.family, self.public)
            EvaluationScheduleVerifier.verify(verifier, run_id)
            schedule = EvaluationSchedule.model_validate_json(
                self.public.read(refs["evaluation/schedule.json"])
            )
            expected_names = {
                "evaluation/schedule.json",
                "evaluation/schedule-receipt.json",
                "evaluation/manifest.json",
                *(f"evaluation/attempts/{ordinal}.json" for ordinal in range(len(schedule.items))),
                *(f"evaluation/records/{ordinal}.json" for ordinal in range(len(schedule.items))),
            }
            refs = _owned_artifacts(
                run,
                expected_visibility="public",
                expected_names=expected_names,
                expected_external_origin=None,
            )
            manifest = EvaluationManifest.model_validate_json(
                self.public.read(refs["evaluation/manifest.json"])
            )
            schedule_hash = _model_hash(schedule)
            if (
                schedule.split != split
                or schedule.selection != "all"
                or set(schedule.modes) != ReleaseGate.REQUIRED_MODES
                or len(schedule.modes) != 5
                or schedule.repeats != 3
                or schedule.bindings.commit != run.binding.repository.commit
                or schedule.bindings.prompt_version != run.binding.prompt_version
                or schedule.bindings.toolchain_hash != run.binding.toolchain_lock_hash
                or schedule.bindings.model_config_hash != run.binding.model_config_hash
                or schedule.bindings.max_cost_usd is None
                or schedule.bindings.max_unit_cost_usd is None
                or [item.ordinal for item in schedule.items] != list(range(len(schedule.items)))
                or any(item.split != split for item in schedule.items)
                or (
                    split == "development"
                    and (
                        schedule.holdout_proof is not None
                        or any(item.holdout_proof is not None for item in schedule.items)
                    )
                )
                or (
                    split == "holdout"
                    and (
                        schedule.holdout_proof is None
                        or any(
                            item.holdout_proof != schedule.holdout_proof for item in schedule.items
                        )
                    )
                )
            ):
                raise ValueError("evaluation schedule is incomplete or unbound")
            attempt_refs = _ordinal_refs(run, "attempts")
            record_refs = _ordinal_refs(run, "records")
            expected_ordinals = set(range(len(schedule.items)))
            if set(attempt_refs) != expected_ordinals or set(record_refs) != expected_ordinals:
                raise ValueError("evaluation units are incomplete")
            records: list[PublicEvaluationRecord] = []
            record_hashes: dict[int, str] = {}
            for ordinal in range(len(schedule.items)):
                attempt = EvaluationAttempt.model_validate_json(
                    self.public.read(attempt_refs[ordinal])
                )
                item = schedule.items[ordinal]
                expected_key = hashlib.sha256(
                    f"{run.id}:{schedule_hash}:{ordinal}".encode()
                ).hexdigest()
                if (
                    attempt.run_id != run.id
                    or attempt.ordinal != ordinal
                    or attempt.schedule_hash != schedule_hash
                    or attempt.corpus_cutoff != schedule.corpus_cutoff
                    or attempt.idempotency_key != expected_key
                    or attempt.reserved_cost_usd != schedule.bindings.max_unit_cost_usd
                ):
                    raise ValueError("evaluation attempt differs from schedule")
                record = PublicEvaluationRecord.model_validate_json(
                    self.public.read(record_refs[ordinal])
                )
                if (
                    (record.case_id, record.template_id, record.mode, record.repeat)
                    != (item.case_id, item.template_id, item.mode, item.repeat)
                    or record.corpus_cutoff != schedule.corpus_cutoff
                    or record.lineage.corpus_cutoff != schedule.corpus_cutoff
                    or record.status != "COMPLETED"
                    or record.failure_reason is not None
                    or record.cost_usd is None
                ):
                    raise ValueError("evaluation record is failed, stopped, or unbound")
                records.append(record)
                record_hashes[ordinal] = record_refs[ordinal].sha256
            total_cost = sum(record.cost_usd or 0 for record in records)
            if (
                manifest.run_id != run.id
                or manifest.commit != schedule.bindings.commit
                or manifest.prompt_version != schedule.bindings.prompt_version
                or manifest.toolchain_hash != schedule.bindings.toolchain_hash
                or manifest.model_config_hash != schedule.bindings.model_config_hash
                or manifest.schedule_hash != schedule_hash
                or manifest.corpus_cutoff != schedule.corpus_cutoff
                or manifest.expected_units != len(schedule.items)
                or manifest.executed_units != len(records)
                or manifest.modes != schedule.modes
                or manifest.split != split
                or manifest.repeats != schedule.repeats
                or manifest.random_seed != schedule.random_seed
                or manifest.records != records
                or manifest.stopped_reason is not None
                or any(
                    (record.cost_usd or 0) > schedule.bindings.max_unit_cost_usd
                    for record in records
                )
                or total_cost > schedule.bindings.max_cost_usd
            ):
                raise ValueError("evaluation terminal manifest differs from records")
            return _EvaluationEvidence(run, run.binding, schedule, manifest, records, record_hashes)
        except _ReleaseEvidenceError:
            raise
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise _ReleaseRootError("EVALUATION_EVIDENCE_INVALID") from exc

    @staticmethod
    def _same_evaluation_binding(
        development: _EvaluationEvidence, holdout: _EvaluationEvidence
    ) -> None:
        left, right = development.binding, holdout.binding
        if (
            left.repository != right.repository
            or left.toolchain_lock_hash != right.toolchain_lock_hash
            or left.prompt_version != right.prompt_version
            or left.model_config_hash != right.model_config_hash
            or development.schedule.corpus_cutoff != holdout.schedule.corpus_cutoff
        ):
            raise _ReleaseRootError("EVALUATION_BINDING_MISMATCH")

    def _corpus(
        self, binding: RunBinding, cutoff: int
    ) -> tuple[dict[str, CaseManifest], dict[str, CaseManifest], list[str], list[str]]:
        try:
            from gpu_agent.benchmark.executor import registered_cases

            transactions = self.family.ledger.committed_through(cutoff)
            public_target = self.family.ledger.target_store_hash(self.public)
            private_target = self.family.ledger.target_store_hash(self.evaluator)
            authoritative_public = [
                item.run_id
                for item in transactions
                if item.visibility == "public" and item.target_store_hash == public_target
            ]
            authoritative_private = [
                item.run_id
                for item in transactions
                if item.visibility == "evaluator" and item.target_store_hash == private_target
            ]
            selected = self.expected_selection
            if selected is not None and (
                authoritative_public != selected.public_case_run_ids
                or authoritative_private != selected.private_case_run_ids
            ):
                raise _ReleaseEvidenceError("CORPUS_SELECTION_MISMATCH")
            public_cases = registered_cases(self.public, binding, self.family, cutoff=cutoff)
            private_cases = registered_cases(self.evaluator, binding, self.family, cutoff=cutoff)
            self._selected_manifests(self.public, authoritative_public, public_cases)
            self._selected_manifests(self.evaluator, authoritative_private, private_cases)
            if set(public_cases) & set(private_cases):
                raise ValueError("case identity reused across corpus splits")
            return public_cases, private_cases, authoritative_public, authoritative_private
        except _ReleaseEvidenceError:
            raise
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise _ReleaseEvidenceError("CORPUS_EVIDENCE_INVALID") from exc

    @staticmethod
    def _selected_manifests(
        store: RunStore, run_ids: list[str], cases: dict[str, CaseManifest]
    ) -> None:
        observed: dict[str, CaseManifest] = {}
        for run_id in run_ids:
            run = store.load(run_id)
            refs = _owned_artifacts(
                run,
                expected_visibility=store.visibility,
                expected_names={
                    "case-manifest.json",
                    "validation/ledger-transaction.json",
                },
                expected_external_origin=None,
            )
            case = CaseManifest.model_validate_json(store.read(refs["case-manifest.json"]))
            if case.id in observed or cases.get(case.id) != case:
                raise ValueError("selected case registration is duplicated or stale")
            observed[case.id] = case
        if observed != cases:
            raise ValueError("selected case registrations are incomplete")

    def _validate_development_records(
        self,
        evidence: _EvaluationEvidence,
        cases: dict[str, CaseManifest],
    ) -> None:
        try:
            from gpu_agent.benchmark.executor import validate_evaluation_record

            observed = {
                (item.case_id, item.template_id, item.mode, item.repeat)
                for item in evidence.schedule.items
            }
            expected = {
                (case.id, case.template_id, mode, repeat)
                for case in cases.values()
                for mode in ReleaseGate.REQUIRED_MODES
                for repeat in range(3)
            }
            if observed != expected or len(observed) != len(evidence.schedule.items):
                raise ValueError("development schedule is not the full Cartesian product")
            attempts = self._attempts(evidence)
            for item, record in zip(evidence.schedule.items, evidence.records, strict=True):
                validate_evaluation_record(
                    self.public,
                    record,
                    item,
                    attempts[item.ordinal],
                    evidence.binding,
                    self.evaluator,
                    self.public,
                    self.family,
                    item.case_id,
                )
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise _ReleaseEvidenceError("DEVELOPMENT_EVALUATION_INVALID") from exc

    def _validate_holdout_records(
        self,
        evidence: _EvaluationEvidence,
        cases: dict[str, CaseManifest],
        cutoff: int,
    ) -> tuple[set[str], list[EvaluatorRecordBinding], set[str]]:
        try:
            try:
                try:
                    mapping_run = self.evaluator.load(self.roots.private_binding_run_id)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise _ReleaseRootError("PRIVATE_SCORING_INCOMPLETE") from exc
                proof = evidence.schedule.holdout_proof
                if (
                    mapping_run.kind != "holdout_alias_mapping"
                    or mapping_run.parent_run_id is not None
                    or mapping_run.status != RunStatus.COMPLETED
                    or mapping_run.binding != evidence.binding
                    or proof is None
                    or mapping_run.external_origin
                    != ExternalRunOrigin(run_id=proof.public_run_id, visibility="public")
                ):
                    raise _ReleaseRootError("PRIVATE_SCORING_INCOMPLETE")
                mapping_refs = _owned_artifacts(
                    mapping_run,
                    expected_visibility="evaluator",
                    expected_names={"holdout/private-alias-map.json"},
                    expected_external_origin=mapping_run.external_origin,
                )
                mapping = _PrivateAliasMap.model_validate_json(
                    self.evaluator.read(mapping_refs["holdout/private-alias-map.json"])
                )
            except _ReleaseRootError:
                raise
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise _ReleaseRootError("PRIVATE_SCORING_INCOMPLETE") from exc
            alias_run = self.public.load(mapping_run.external_origin.run_id)
            alias_refs = _owned_artifacts(
                alias_run,
                expected_visibility="public",
                expected_names={"holdout/aliases.json"},
                expected_external_origin=None,
            )
            alias_ref = alias_refs["holdout/aliases.json"]
            alias_payload = json.loads(self.public.read(alias_ref))
            aliases = [identity.alias for identity in mapping.identities]
            nonce = bytes.fromhex(mapping.nonce_hex)
            recomputed = [
                hmac.new(
                    nonce,
                    json.dumps(
                        [identity.private_case_id, identity.private_template_id],
                        separators=(",", ":"),
                    ).encode(),
                    hashlib.sha256,
                ).hexdigest()
                for identity in mapping.identities
            ]
            if (
                alias_run.kind != "holdout_aliases"
                or alias_run.parent_run_id is not None
                or alias_run.status != RunStatus.COMPLETED
                or alias_run.binding != evidence.binding
                or mapping.corpus_cutoff != cutoff
                or aliases != recomputed
                or len(aliases) != len(set(aliases))
                or alias_payload
                != {"schema_version": 2, "corpus_cutoff": cutoff, "aliases": aliases}
                or proof is None
                or proof.public_run_id != alias_run.id
                or proof.aliases_hash != alias_ref.sha256
                or proof.corpus_cutoff != cutoff
            ):
                raise ValueError("private alias mapping is invalid")
            private_identities = {
                (identity.private_case_id, identity.private_template_id)
                for identity in mapping.identities
            }
            if private_identities != {
                (case.id, case.template_id) for case in cases.values()
            } or len(private_identities) != len(mapping.identities):
                raise ValueError("private aliases differ from selected corpus")
            alias_map = {
                identity.alias: (identity.private_case_id, identity.private_template_id)
                for identity in mapping.identities
            }
            expected = {
                (alias, alias, mode, repeat)
                for alias in aliases
                for mode in ReleaseGate.REQUIRED_MODES
                for repeat in range(3)
            }
            observed = {
                (item.case_id, item.template_id, item.mode, item.repeat)
                for item in evidence.schedule.items
            }
            if observed != expected or len(observed) != len(evidence.schedule.items):
                raise ValueError("holdout schedule is not the full Cartesian product")
            from gpu_agent.benchmark.executor import validate_evaluation_record

            attempts = self._attempts(evidence)
            for item, record in zip(evidence.schedule.items, evidence.records, strict=True):
                private_case_id, _ = alias_map[item.case_id]
                validate_evaluation_record(
                    self.public,
                    record,
                    item,
                    attempts[item.ordinal],
                    evidence.binding,
                    self.evaluator,
                    self.evaluator,
                    self.family,
                    private_case_id,
                )
            private_score_bindings, private_score_run_ids = self._private_scores(
                mapping_run, mapping, evidence, alias_map
            )
            return (
                {template for _, template in private_identities},
                private_score_bindings,
                private_score_run_ids,
            )
        except _ReleaseEvidenceError:
            raise
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise _ReleaseEvidenceError("PRIVATE_SCORING_INCOMPLETE") from exc

    def _private_scores(
        self,
        mapping_run: RunManifest,
        mapping: _PrivateAliasMap,
        evidence: _EvaluationEvidence,
        alias_map: dict[str, tuple[str, str]],
    ) -> tuple[list[EvaluatorRecordBinding], set[str]]:
        with self.evaluator.evaluation_run_lease(mapping_run.id) as lease:
            if lease.load() != mapping_run:
                raise ValueError("private mapping changed during score inventory")
            score_runs = lease.children()
        if any(run.kind != "holdout_score" for run in score_runs):
            raise ValueError("private mapping has an unexpected direct child")
        if len(score_runs) != len(evidence.records):
            raise ValueError("private score count differs from holdout records")
        by_record: dict[str, EvaluatorRecordBinding] = {}
        nonce = bytes.fromhex(mapping.nonce_hex)
        for run in score_runs:
            if (
                run.status != RunStatus.COMPLETED
                or run.binding != evidence.binding
                or run.parent_run_id != mapping_run.id
                or run.external_origin != mapping_run.external_origin
            ):
                raise ValueError("private score run is not terminal and bound")
            try:
                refs = _owned_artifacts(
                    run,
                    expected_visibility="evaluator",
                    expected_names={
                        "holdout/record-binding.json",
                        "holdout/private-score.json",
                    },
                    expected_external_origin=mapping_run.external_origin,
                )
            except ValueError:
                raise ValueError("private score artifact inventory is invalid") from None
            binding_ref = refs["holdout/record-binding.json"]
            score_ref = refs["holdout/private-score.json"]
            binding = EvaluatorRecordBinding.model_validate_json(self.evaluator.read(binding_ref))
            private_score = _PrivateScore.model_validate_json(self.evaluator.read(score_ref))
            matches = [
                (ordinal, record)
                for ordinal, record in enumerate(evidence.records)
                if record.record_id == binding.public_record_id
            ]
            if len(matches) != 1:
                raise ValueError("private score has no unique public record")
            ordinal, record = matches[0]
            private_case_id, private_template_id = alias_map[record.case_id]
            expected_run_id = hmac.new(
                nonce,
                f"holdout-score-v1:{record.record_id}".encode(),
                hashlib.sha256,
            ).hexdigest()[:32]
            if (
                record.record_id in by_record
                or run.id != expected_run_id
                or binding.evaluator_score_run_id != run.id
                or binding.public_evaluation_run_id != evidence.run.id
                or binding.public_record_hash != evidence.record_hashes[ordinal]
                or binding.private_case_id != private_case_id
                or binding.private_template_id != private_template_id
                or binding.corpus_cutoff != evidence.schedule.corpus_cutoff
                or binding.private_score_hash != score_ref.sha256
                or private_score.public_record_hash != binding.public_record_hash
                or private_score.corpus_cutoff != binding.corpus_cutoff
            ):
                raise ValueError("private score binding differs from native evidence")
            by_record[record.record_id] = binding
        if set(by_record) != {record.record_id for record in evidence.records}:
            raise ValueError("private score bindings are incomplete")
        ordered = [by_record[record.record_id] for record in evidence.records]
        return ordered, {run.id for run in score_runs}

    def _attempts(self, evidence: _EvaluationEvidence) -> dict[int, EvaluationAttempt]:
        refs = _ordinal_refs(evidence.run, "attempts")
        return {
            ordinal: EvaluationAttempt.model_validate_json(self.public.read(ref))
            for ordinal, ref in refs.items()
        }

    def _scoring_metrics(
        self,
        bindings: list[EvaluatorRecordBinding],
        binding: RunBinding,
    ) -> dict[str, object]:
        from gpu_agent.benchmark.metrics import aggregate_grouped
        from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

        metrics = aggregate_grouped(
            bindings,
            public_store=self.public,
            evaluator_store=self.evaluator,
            run_binding=binding,
            schedule_verifier=EvaluationScheduleVerifier.for_family(self.family, self.public),
        )
        return metrics.model_dump(mode="json")

    def _scoring_session(
        self,
        evidence: _EvaluationEvidence,
        bindings: list[EvaluatorRecordBinding],
    ) -> str:
        try:
            run_id = hashlib.sha256(
                (
                    f"holdout-scoring-v1:{evidence.run.id}:{self.roots.private_binding_run_id}"
                ).encode()
            ).hexdigest()[:32]
            run = self.evaluator.load(run_id)
            expected_events = (
                (RunStatus.QUEUED, None),
                (RunStatus.RUNNING, CurrentPhase.PREPARING),
                (RunStatus.RUNNING, CurrentPhase.FINALIZING),
                (RunStatus.COMPLETED, None),
            )
            names = {
                "holdout-scoring/input-binding.json",
                "holdout-scoring/bindings.json",
                "holdout-scoring/metrics.json",
                "holdout-scoring/result.json",
            }
            if (
                run.kind != "holdout_scoring"
                or run.parent_run_id is not None
                or run.status != RunStatus.COMPLETED
                or run.current_phase is not None
                or run.last_completed_phase != CurrentPhase.FINALIZING
                or run.binding != evidence.binding
                or run.external_origin is None
                or run.external_origin.visibility != "public"
                or run.external_origin.run_id != evidence.run.id
                or tuple((event.status, event.phase) for event in run.events) != expected_events
            ):
                raise ValueError("holdout scoring session is not exact and completed")
            refs = _owned_artifacts(
                run,
                expected_visibility="evaluator",
                expected_names=names,
                expected_external_origin=ExternalRunOrigin(
                    run_id=evidence.run.id,
                    visibility="public",
                ),
            )
            input_content = self.evaluator.read(refs["holdout-scoring/input-binding.json"])
            input_binding = _HoldoutScoringInputBinding.model_validate_json(input_content)
            record_set_hash = hashlib.sha256(
                _canonical_json(
                    [
                        (
                            ordinal,
                            evidence.record_hashes[ordinal],
                            hashlib.sha256(_canonical_json(record.blind())).hexdigest(),
                        )
                        for ordinal, record in enumerate(evidence.records)
                    ]
                )
            ).hexdigest()
            proof = evidence.schedule.holdout_proof
            rubric_hash = _tracked_rubric_hash(
                self.repository,
                evidence.binding.repository.commit,
            )
            if (
                input_content != _canonical_json(input_binding.model_dump(mode="json"))
                or input_binding.evaluation_run_id != evidence.run.id
                or input_binding.private_binding_run_id != self.roots.private_binding_run_id
                or input_binding.schedule_hash != _model_hash(evidence.schedule)
                or proof is None
                or input_binding.aliases_hash != proof.aliases_hash
                or input_binding.corpus_cutoff != evidence.schedule.corpus_cutoff
                or input_binding.record_set_hash != record_set_hash
                or input_binding.rubric_hash != rubric_hash
                or len(evidence.records) != 120
                or len(bindings) != 120
            ):
                raise ValueError("holdout scoring input binding differs")
            bindings_content = self.evaluator.read(refs["holdout-scoring/bindings.json"])
            stored_bindings = TypeAdapter(list[EvaluatorRecordBinding]).validate_json(
                bindings_content
            )
            if (
                bindings_content
                != _canonical_json([binding.model_dump(mode="json") for binding in stored_bindings])
                or stored_bindings != bindings
            ):
                raise ValueError("holdout scoring bindings differ")
            metrics_content = self.evaluator.read(refs["holdout-scoring/metrics.json"])
            metrics_payload = json.loads(metrics_content)
            expected_metrics = self._scoring_metrics(bindings, evidence.binding)
            if (
                metrics_content != _canonical_json(metrics_payload)
                or metrics_payload != expected_metrics
            ):
                raise ValueError("holdout scoring metrics differ")
            result_content = self.evaluator.read(refs["holdout-scoring/result.json"])
            result = HoldoutScoringResult.model_validate_json(result_content)
            if (
                result_content != _canonical_json(result.model_dump(mode="json"))
                or result.scoring_run_id != run_id
                or result.evaluation_run_id != evidence.run.id
                or result.private_binding_run_id != self.roots.private_binding_run_id
                or result.package_hash != input_binding.package_hash
                or result.input_binding_hash != hashlib.sha256(input_content).hexdigest()
                or result.bindings_hash != hashlib.sha256(bindings_content).hexdigest()
                or result.metrics_hash != hashlib.sha256(metrics_content).hexdigest()
                or result.scored_count != len(bindings)
                or result.expected_record_count != len(bindings)
            ):
                raise ValueError("holdout scoring result differs")
            return run_id
        except _ReleaseEvidenceError:
            raise
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise _ReleaseEvidenceError("PRIVATE_SCORING_INCOMPLETE") from exc

    def _release_tests(self, binding: RunBinding, cutoff: int) -> TestCounts:
        try:
            from gpu_agent.release_controller import verify_persisted_release_artifacts

            try:
                run = self.public.load(self.roots.release_test_run_id)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise _ReleaseRootError("RELEASE_TEST_EVIDENCE_INVALID") from exc
            if (
                run.kind != "release_test"
                or run.parent_run_id is not None
                or run.external_origin is not None
                or run.status != RunStatus.COMPLETED
                or not _release_test_binding(run, binding)
            ):
                raise _ReleaseRootError("RELEASE_TEST_EVIDENCE_INVALID")
            refs = _owned_artifacts(
                run,
                expected_visibility="public",
                expected_names={
                    "release/test-evidence.json",
                    "release/test-invocation.json",
                },
                expected_external_origin=None,
            )
            evidence_ref = refs["release/test-evidence.json"]
            invocation_ref = refs["release/test-invocation.json"]
            evidence = ReleaseTestEvidence.model_validate_json(self.public.read(evidence_ref))
            verify_persisted_release_artifacts(self.public, run)
            expected_hash = hashlib.sha256(
                json.dumps(sorted(evidence.collected_node_ids), separators=(",", ":")).encode()
            ).hexdigest()
            if (
                evidence.run_id != run.id
                or evidence.repository != self.actual_repository
                or evidence.toolchain_hash != binding.toolchain_lock_hash
                or evidence.model_config_hash != binding.model_config_hash
                or evidence.prompt_version != binding.prompt_version
                or evidence.corpus_cutoff != cutoff
                or evidence.collection_hash != expected_hash
                or evidence.invocation_hash != invocation_ref.sha256
                or evidence.test_counts.expected == 0
                or evidence.test_counts.executed != evidence.test_counts.expected
                or evidence.test_counts.skipped_required
                or evidence.test_counts.failed
                or len(evidence.collected_node_ids) != len(set(evidence.collected_node_ids))
            ):
                raise ValueError("release test evidence is incomplete")
            return evidence.test_counts
        except _ReleaseEvidenceError:
            raise
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise _ReleaseRootError("RELEASE_TEST_EVIDENCE_INVALID") from exc

    def _acceptance(
        self,
        public_cases: dict[str, CaseManifest],
        development: _EvaluationEvidence,
        holdout: _EvaluationEvidence,
        private_score_run_ids: set[str],
    ) -> dict[str, list[str]]:
        """Match each category to native runs already validated above."""

        try:
            mode_e_ids = [
                record.lineage.diagnosis_run_id
                for evaluation in (development, holdout)
                for item, record in zip(
                    evaluation.schedule.items,
                    evaluation.records,
                    strict=True,
                )
                if item.mode == "E"
            ]
            if len(mode_e_ids) != len(set(mode_e_ids)):
                raise ValueError("Mode-E diagnosis lineage is not unique")
            expected = {
                "isolation": {self.roots.release_test_run_id},
                "four_tools": {case.validation_run_ids[1] for case in public_cases.values()},
                "private_oracle": private_score_run_ids,
                "live_llm": set(mode_e_ids),
            }
            if any(not run_ids for run_ids in expected.values()):
                raise ValueError("required native acceptance evidence is empty")
            selected_evidence = self.expected_selection
            if selected_evidence is not None:
                if set(selected_evidence.acceptance_run_ids) != ReleaseGate.REQUIRED_ACCEPTANCE:
                    raise ValueError("acceptance selection has missing or extra categories")
                for category, expected_ids in expected.items():
                    selected = selected_evidence.acceptance_run_ids[category]
                    if selected != sorted(expected_ids):
                        raise ValueError("acceptance selection differs from native evidence")
            return {
                category: sorted(expected[category])
                for category in sorted(ReleaseGate.REQUIRED_ACCEPTANCE)
            }
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise _ReleaseEvidenceError("RELEASE_ACCEPTANCE_INVALID") from exc

    def _corpus_hash(
        self,
        public_cases: dict[str, CaseManifest],
        private_cases: dict[str, CaseManifest],
        cutoff: int,
    ) -> str:
        transactions = self.family.ledger.committed_through(cutoff)
        payload = {
            "schema_version": 1,
            "ledger_namespace_hash": self.family.namespace_hash,
            "corpus_cutoff": cutoff,
            "transactions": [item.model_dump(mode="json") for item in transactions],
            "public_cases": [
                public_cases[key].model_dump(mode="json") for key in sorted(public_cases)
            ],
            "private_cases": [
                private_cases[key].model_dump(mode="json") for key in sorted(private_cases)
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class ReleaseEvidenceFreezer:
    """Resolve four explicit roots into one canonical release selection."""

    capture_repository = staticmethod(capture_repository_snapshot)

    @staticmethod
    def expected_development_commit(roots: ReleaseEvidenceRoots, public_store: RunStore) -> str:
        """Read only the selected development root needed to bind the first snapshot."""
        try:
            run = public_store.load(roots.development_evaluation_run_id)
            if run.kind != "evaluation" or run.binding is None:
                raise ValueError("selected development root is invalid")
            return run.binding.repository.commit
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("release roots are invalid") from exc

    @staticmethod
    def resolve(
        roots: ReleaseEvidenceRoots,
        public_store: RunStore,
        evaluator_store: RunStore,
        corpus_family: CorpusFamily,
        repository: Path,
        actual_repository: RepositorySnapshot,
    ) -> ReleaseEvidenceResolution:
        return _ReleaseEvidenceResolver(
            roots,
            public_store,
            evaluator_store,
            corpus_family,
            repository,
            actual_repository,
        ).resolve()

    def freeze(
        self,
        roots: ReleaseEvidenceRoots,
        public_store: RunStore,
        evaluator_store: RunStore,
        corpus_family: CorpusFamily,
        repository: Path,
        output: Path,
    ) -> FrozenReleaseSelection:
        """Gate and atomically publish the selection derived from exactly four roots."""
        try:
            expected_commit = self.expected_development_commit(roots, public_store)
            actual = self.capture_repository(repository, expected_commit=expected_commit)
            resolution = self.resolve(
                roots,
                public_store,
                evaluator_store,
                corpus_family,
                repository,
                actual,
            )
        except _ReleaseRootError as exc:
            raise ValueError(f"release roots are invalid: {exc.code}") from exc
        except _ReleaseEvidenceError as exc:
            raise ValueError(f"release evidence is incomplete: {exc.code}") from exc
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ValueError("release roots are invalid") from exc

        evidence = resolution.evidence
        try:
            manifest = ReleaseManifest.from_evidence(evidence)
        except ValueError as exc:
            raise ValueError("release evidence is incomplete") from exc
        gate = ReleaseGate().check(manifest, evidence)
        if not gate.passed:
            raise ValueError("release evidence is incomplete: " + ",".join(gate.reason_codes))
        cutoff = evidence.corpus_cutoff
        if cutoff is None:
            raise ValueError("release evidence is incomplete")

        try:
            recaptured = self.capture_repository(
                repository,
                expected_commit=actual.commit,
            )
        except (OSError, ValueError) as exc:
            raise ValueError("release repository changed") from exc
        if recaptured != actual:
            raise ValueError("release repository changed")

        content = resolution.selection.model_dump_json(indent=2).encode() + b"\n"
        digest = write_private_atomic_new(
            output,
            content,
            repository=repository,
            forbidden_roots=(public_store.root, evaluator_store.root),
        )
        return FrozenReleaseSelection(
            output=output,
            selection=resolution.selection,
            selection_sha256=digest,
            corpus_cutoff=cutoff,
            public_case_count=evidence.public_case_count,
            private_case_count=evidence.private_case_count,
            acceptance_run_count=sum(
                len(run_ids) for run_ids in resolution.selection.acceptance_run_ids.values()
            ),
        )


def _owned_artifacts(
    run: RunManifest,
    *,
    expected_visibility: Literal["public", "evaluator"],
    expected_names: set[str] | None,
    expected_external_origin: ExternalRunOrigin | None,
) -> dict[str, ArtifactRef]:
    """Validate one containing run's complete owned artifact inventory."""
    refs: dict[str, ArtifactRef] = {}
    for ref in run.artifact_refs:
        if (
            ref.name in refs
            or ref.run_id != run.id
            or ref.visibility != expected_visibility
            or ref.relative_path != f"{run.id}/artifacts/{ref.id}"
        ):
            raise ValueError("run artifact inventory is not owned and canonical")
        refs[ref.name] = ref
    if run.external_origin != expected_external_origin or (
        expected_names is not None and set(refs) != expected_names
    ):
        raise ValueError("run artifact inventory is not exact")
    return refs


def _ordinal_refs(run: RunManifest, kind: Literal["attempts", "records"]) -> dict[int, ArtifactRef]:
    prefix = f"evaluation/{kind}/"
    refs: dict[int, ArtifactRef] = {}
    for ref in run.artifact_refs:
        if not ref.name.startswith(prefix):
            continue
        suffix = ref.name.removeprefix(prefix)
        if not suffix.endswith(".json") or not suffix.removesuffix(".json").isdigit():
            raise ValueError(f"invalid evaluation {kind} artifact name")
        ordinal = int(suffix.removesuffix(".json"))
        if ref.name != f"{prefix}{ordinal}.json" or ordinal in refs:
            raise ValueError(f"duplicate evaluation {kind} ordinal")
        refs[ordinal] = ref
    return refs


def _model_hash(model: ExecutionModel) -> str:
    return hashlib.sha256(_canonical_json(model.model_dump(mode="json"))).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _release_test_binding(run: RunManifest, evaluation: RunBinding) -> bool:
    binding = run.binding
    return bool(
        binding is not None
        and binding.purpose == "release_acceptance"
        and binding.repository == evaluation.repository
        and binding.toolchain_lock_hash == evaluation.toolchain_lock_hash
        and binding.corpus_ledger_namespace_hash == evaluation.corpus_ledger_namespace_hash
        and binding.case_registry_hash == evaluation.case_registry_hash
        and binding.model_config_hash == evaluation.model_config_hash
        and binding.prompt_version == evaluation.prompt_version
    )


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
