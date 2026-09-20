"""Evidence-derived V2 release gate.

The release manifest is a set of claims. It is never an evidence source. Release
facts are reconstructed from an explicit selection, the corpus ledger and immutable
public/evaluator RunStore artifacts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections import Counter
from typing import Literal

from pydantic import Field, model_validator

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
from gpu_agent.benchmark.ledger import CorpusFamily
from gpu_agent.benchmark.models import CaseManifest
from gpu_agent.contracts import (
    ArtifactRef,
    RepositorySnapshot,
    RunBinding,
    RunManifest,
    RunStatus,
)
from gpu_agent.execution.models import ExecutionModel, SanitizerTool
from gpu_agent.store import RunStore


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
        ids = [
            *self.public_case_run_ids,
            *self.private_case_run_ids,
            self.development_evaluation_run_id,
            self.holdout_evaluation_run_id,
            self.private_binding_run_id,
            self.release_test_run_id,
            *(run_id for values in self.acceptance_run_ids.values() for run_id in values),
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("release evidence selection contains duplicate run IDs")
        return self


class ReleaseTestEvidence(ExecutionModel):
    """Bound result of the controller-owned release test invocation."""

    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    repository: RepositorySnapshot
    toolchain_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)
    test_counts: TestCounts
    collected_node_ids: list[str] = Field(min_length=1)
    collection_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
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
        actual_repository: RepositorySnapshot,
    ) -> ReleaseEvidenceIndex:
        """Derive the complete index, returning a typed closed gate on any defect."""

        try:
            return _ReleaseEvidenceDeriver(
                selection,
                public_store,
                evaluator_store,
                corpus_family,
                actual_repository,
            ).derive()
        except _ReleaseEvidenceError as exc:
            return cls(repository=actual_repository, reason_codes=[exc.code])
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return cls(
                repository=actual_repository,
                reason_codes=["EVIDENCE_DERIVATION_FAILED"],
            )


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


class _ReleaseEvidenceDeriver:
    def __init__(
        self,
        selection: ReleaseEvidenceSelection,
        public: RunStore,
        evaluator: RunStore,
        family: CorpusFamily,
        actual_repository: RepositorySnapshot,
    ) -> None:
        self.selection = selection
        self.public = public
        self.evaluator = evaluator
        self.family = family
        self.actual_repository = actual_repository

    def derive(self) -> ReleaseEvidenceIndex:
        self._validate_roots()
        development = self._evaluation(self.selection.development_evaluation_run_id, "development")
        holdout = self._evaluation(self.selection.holdout_evaluation_run_id, "holdout")
        self._same_evaluation_binding(development, holdout)
        cutoff = development.schedule.corpus_cutoff
        public_cases, private_cases = self._corpus(development.binding, cutoff)
        self._validate_development_records(development, public_cases)
        private_templates, private_score_run_ids = self._validate_holdout_records(
            holdout, private_cases, cutoff
        )
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
            "public_corpus": list(self.selection.public_case_run_ids),
            "private_corpus": list(self.selection.private_case_run_ids),
            "private_scoring": [self.selection.private_binding_run_id],
            "release_tests": [self.selection.release_test_run_id],
        }
        tool_counts = Counter(case.target_tool.value for case in public_cases.values())
        return ReleaseEvidenceIndex(
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

    def _validate_roots(self) -> None:
        if self.selection.repository != self.actual_repository:
            raise _ReleaseEvidenceError("ACTUAL_REPOSITORY_MISMATCH")
        try:
            self.family.require_store(self.public)
            self.family.require_store(self.evaluator)
        except ValueError as exc:
            raise _ReleaseEvidenceError("STORE_FAMILY_MISMATCH") from exc
        if self.public.visibility != "public" or self.evaluator.visibility != "evaluator":
            raise _ReleaseEvidenceError("STORE_FAMILY_MISMATCH")

    def _evaluation(
        self, run_id: str, split: Literal["development", "holdout"]
    ) -> _EvaluationEvidence:
        try:
            from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

            verifier = EvaluationScheduleVerifier.for_family(self.family, self.public)
            EvaluationScheduleVerifier.verify(verifier, run_id)
            run = self.public.load(run_id)
            if (
                run.kind != "evaluation"
                or run.status != RunStatus.COMPLETED
                or run.binding is None
                or run.binding.purpose != "evaluation"
                or run.binding.repository != self.actual_repository
                or run.binding.toolchain_lock_hash is None
                or run.binding.prompt_version is None
                or run.binding.model_config_hash is None
            ):
                raise ValueError("evaluation run is not terminal and bound")
            schedule = EvaluationSchedule.model_validate_json(
                self.public.read(_one(run, "evaluation/schedule.json"))
            )
            manifest = EvaluationManifest.model_validate_json(
                self.public.read(_one(run, "evaluation/manifest.json"))
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
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise _ReleaseEvidenceError("EVALUATION_EVIDENCE_INVALID") from exc

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
            raise _ReleaseEvidenceError("EVALUATION_BINDING_MISMATCH")

    def _corpus(
        self, binding: RunBinding, cutoff: int
    ) -> tuple[dict[str, CaseManifest], dict[str, CaseManifest]]:
        try:
            from gpu_agent.benchmark.executor import registered_cases

            transactions = self.family.ledger.committed_through(cutoff)
            public_target = self.family.ledger.target_store_hash(self.public)
            private_target = self.family.ledger.target_store_hash(self.evaluator)
            authoritative_public = {
                item.run_id
                for item in transactions
                if item.visibility == "public" and item.target_store_hash == public_target
            }
            authoritative_private = {
                item.run_id
                for item in transactions
                if item.visibility == "evaluator" and item.target_store_hash == private_target
            }
            if authoritative_public != set(self.selection.public_case_run_ids) or (
                authoritative_private != set(self.selection.private_case_run_ids)
            ):
                raise _ReleaseEvidenceError("CORPUS_SELECTION_MISMATCH")
            public_cases = registered_cases(self.public, binding, self.family, cutoff=cutoff)
            private_cases = registered_cases(self.evaluator, binding, self.family, cutoff=cutoff)
            self._selected_manifests(self.public, self.selection.public_case_run_ids, public_cases)
            self._selected_manifests(
                self.evaluator, self.selection.private_case_run_ids, private_cases
            )
            if set(public_cases) & set(private_cases):
                raise ValueError("case identity reused across corpus splits")
            return public_cases, private_cases
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
            case = CaseManifest.model_validate_json(store.read(_one(run, "case-manifest.json")))
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
    ) -> tuple[set[str], set[str]]:
        try:
            mapping_run = self.evaluator.load(self.selection.private_binding_run_id)
            if (
                mapping_run.kind != "holdout_alias_mapping"
                or mapping_run.status != RunStatus.COMPLETED
                or mapping_run.binding != evidence.binding
                or mapping_run.external_origin is None
                or mapping_run.external_origin.visibility != "public"
            ):
                raise ValueError("private mapping run is not terminal and bound")
            mapping = _PrivateAliasMap.model_validate_json(
                self.evaluator.read(_one(mapping_run, "holdout/private-alias-map.json"))
            )
            alias_run = self.public.load(mapping_run.external_origin.run_id)
            alias_ref = _one(alias_run, "holdout/aliases.json")
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
            proof = evidence.schedule.holdout_proof
            if (
                alias_run.kind != "holdout_aliases"
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
            private_score_run_ids = self._private_scores(mapping_run, mapping, evidence, alias_map)
            return (
                {template for _, template in private_identities},
                private_score_run_ids,
            )
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise _ReleaseEvidenceError("PRIVATE_SCORING_INCOMPLETE") from exc

    def _private_scores(
        self,
        mapping_run: RunManifest,
        mapping: _PrivateAliasMap,
        evidence: _EvaluationEvidence,
        alias_map: dict[str, tuple[str, str]],
    ) -> set[str]:
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
            ):
                raise ValueError("private score run is not terminal and bound")
            binding_ref = _one(run, "holdout/record-binding.json")
            score_ref = _one(run, "holdout/private-score.json")
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
        return {run.id for run in score_runs}

    def _attempts(self, evidence: _EvaluationEvidence) -> dict[int, EvaluationAttempt]:
        refs = _ordinal_refs(evidence.run, "attempts")
        return {
            ordinal: EvaluationAttempt.model_validate_json(self.public.read(ref))
            for ordinal, ref in refs.items()
        }

    def _release_tests(self, binding: RunBinding, cutoff: int) -> TestCounts:
        try:
            run = self.public.load(self.selection.release_test_run_id)
            if (
                run.kind != "release_test"
                or run.status != RunStatus.COMPLETED
                or not _release_test_binding(run, binding)
            ):
                raise ValueError("release test run is not terminal and bound")
            evidence = ReleaseTestEvidence.model_validate_json(
                self.public.read(_one(run, "release/test-evidence.json"))
            )
            expected_hash = hashlib.sha256(
                json.dumps(sorted(evidence.collected_node_ids), separators=(",", ":")).encode()
            ).hexdigest()
            if (
                evidence.run_id != run.id
                or evidence.repository != self.actual_repository
                or evidence.toolchain_hash != binding.toolchain_lock_hash
                or evidence.model_config_hash != binding.model_config_hash
                or evidence.corpus_cutoff != cutoff
                or evidence.collection_hash != expected_hash
                or evidence.test_counts.expected == 0
                or evidence.test_counts.executed != evidence.test_counts.expected
                or evidence.test_counts.skipped_required
                or evidence.test_counts.failed
                or len(evidence.collected_node_ids) != len(set(evidence.collected_node_ids))
            ):
                raise ValueError("release test evidence is incomplete")
            return evidence.test_counts
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise _ReleaseEvidenceError("RELEASE_TEST_EVIDENCE_INVALID") from exc

    def _acceptance(
        self,
        public_cases: dict[str, CaseManifest],
        development: _EvaluationEvidence,
        holdout: _EvaluationEvidence,
        private_score_run_ids: set[str],
    ) -> dict[str, list[str]]:
        """Match each category to native runs already validated above."""

        try:
            if set(self.selection.acceptance_run_ids) != ReleaseGate.REQUIRED_ACCEPTANCE:
                raise ValueError("acceptance selection has missing or extra categories")
            expected = {
                "isolation": {case.validation_run_ids[0] for case in public_cases.values()},
                "four_tools": {case.validation_run_ids[1] for case in public_cases.values()},
                "private_oracle": private_score_run_ids,
                "live_llm": {
                    record.lineage.diagnosis_run_id
                    for evaluation in (development, holdout)
                    for item, record in zip(
                        evaluation.schedule.items,
                        evaluation.records,
                        strict=True,
                    )
                    if item.mode == "E"
                },
            }
            if any(not run_ids for run_ids in expected.values()):
                raise ValueError("required native acceptance evidence is empty")
            for category, expected_ids in expected.items():
                selected = self.selection.acceptance_run_ids[category]
                if len(selected) != len(set(selected)) or set(selected) != expected_ids:
                    raise ValueError("acceptance selection differs from native evidence")
            return {
                category: list(self.selection.acceptance_run_ids[category])
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


def _one(run: RunManifest, name: str) -> ArtifactRef:
    refs = [ref for ref in run.artifact_refs if ref.name == name]
    if len(refs) != 1:
        raise ValueError(f"expected exactly one {name} artifact")
    return refs[0]


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
    return hashlib.sha256(
        json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
    )


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
