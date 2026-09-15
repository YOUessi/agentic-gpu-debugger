"""Fail-closed release checks derived from immutable controller artifacts."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal, cast
from xml.etree import ElementTree

from pydantic import Field

from gpu_agent.agent.models import DiagnosisResult
from gpu_agent.agent.provider import Invocation
from gpu_agent.benchmark.evaluation import (
    EvaluationAttempt,
    EvaluationManifest,
    EvaluationSchedule,
    PublicEvaluationRecord,
)
from gpu_agent.benchmark.models import CaseManifest
from gpu_agent.contracts import ArtifactRef, RunManifest, RunStatus
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.store import RunStore, read_regular
from gpu_agent.verification.models import VerificationObservation

EvidenceCategory = Literal[
    "four_tools",
    "isolation",
    "private_oracle",
    "live_llm",
    "five_mode_evaluation",
    "corpus_validation",
]
ReleaseSurface = Literal[
    "isolation",
    "four_tools",
    "private_oracle",
    "live_llm",
    "five_modes",
    "corpus",
    "release_manifest",
]


class TestCounts(ExecutionModel):
    expected: int = Field(ge=0)
    executed: int = Field(ge=0)
    skipped_required: int = Field(ge=0)
    failed: int = Field(ge=0)


class ReleaseManifest(ExecutionModel):
    """Declarative public release claim; never a source of evidence."""

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


class ReleaseEvidenceReceipt(ExecutionModel):
    """Controller-owned binding from a completed run to hashed proof artifacts."""

    schema_version: Literal[1] = 1
    category: EvidenceCategory
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    source_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    toolchain_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_hashes: dict[str, str] = Field(min_length=1)


class CorpusValidationProof(ExecutionModel):
    clean_oracle_passed: Literal[True]
    required_checks_clean: Literal[True]
    target_confirmed: Literal[True]


class IsolationAcceptanceProof(ExecutionModel):
    backend: Literal["isolated_gpu"]
    network_disabled: Literal[True]
    read_only_root: Literal[True]
    bounded_resources: Literal[True]
    timeout_cleanup_verified: Literal[True]
    candidate_build_and_run_verified: Literal[True]


class FourToolsAcceptanceProof(ExecutionModel):
    clean_outcomes: dict[str, Literal["CLEAN"]]
    fault_outcomes: dict[str, Literal["FINDING"]]


class ReleaseTestReport(ExecutionModel):
    """Persisted controller result for the explicitly selected release surface."""

    schema_version: Literal[1] = 1
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    junit_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    test_counts: TestCounts
    surface_counts: dict[ReleaseSurface, int]
    selected_node_ids: list[str] = Field(min_length=1)


class HoldoutIdentity(ExecutionModel):
    """Evaluator-only link from an opaque public evaluation ID to private corpus identity."""

    evaluation_case_id: str = Field(min_length=1)
    evaluation_template_id: str = Field(min_length=1)
    private_case_id: str = Field(pattern=r"^case_[0-9]{4}$")
    private_template_id: str = Field(min_length=1)


class HoldoutIdentityMap(ExecutionModel):
    schema_version: Literal[1] = 1
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    toolchain_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    identities: list[HoldoutIdentity] = Field(min_length=1)


class ReleaseEvidenceIndex(ExecutionModel):
    """Private controller projection derived from public/evaluator RunStores.

    It deliberately contains only identities, counts, and hashes. Private case manifests
    and evaluator payloads are validated in place and are never copied into this model.
    """

    schema_version: Literal[1] = 1
    current_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    toolchain_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    corpus_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_config_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    test_counts: TestCounts = Field(
        default_factory=lambda: TestCounts(
            expected=0, executed=0, skipped_required=0, failed=0
        )
    )
    public_case_count: int = Field(ge=0)
    private_case_count: int = Field(ge=0)
    public_tool_counts: dict[str, int]
    evidence_run_ids: dict[str, list[str]]
    release_surfaces_complete: bool
    evaluation_complete: bool
    integrity_errors: list[str]

    @classmethod
    def derive(
        cls, public: RunStore, evaluator: RunStore, *, current_commit: str
    ) -> "ReleaseEvidenceIndex":
        if public.visibility != "public" or evaluator.visibility != "evaluator":
            raise ValueError("release evidence stores have the wrong visibility")
        if not re.fullmatch(r"[a-f0-9]{40}", current_commit):
            raise ValueError("current commit must be a full lowercase Git hash")
        builder = _EvidenceBuilder(public, evaluator, current_commit)
        return builder.build()


class ReleaseGateResult(ExecutionModel):
    passed: bool
    reason_codes: list[str]


class ReleaseGate:
    REQUIRED_EVIDENCE = {
        "four_tools",
        "isolation",
        "private_oracle",
        "live_llm",
        "five_mode_evaluation",
    }

    def check(
        self, manifest: ReleaseManifest, evidence: ReleaseEvidenceIndex
    ) -> ReleaseGateResult:
        reasons: list[str] = []
        counts = manifest.test_counts
        if counts.expected == 0 or counts.executed == 0:
            reasons.append("TEST_COUNT_ZERO")
        if counts.executed != counts.expected:
            reasons.append("TEST_COUNT_INCOMPLETE")
        if counts != evidence.test_counts:
            reasons.append("TEST_COUNT_EVIDENCE_MISMATCH")
        if counts.skipped_required:
            reasons.append("REQUIRED_TEST_SKIPPED")
        if counts.failed:
            reasons.append("TEST_FAILURE")
        if manifest.commit != evidence.current_commit:
            reasons.append("COMMIT_MISMATCH")
        if manifest.toolchain_hash != evidence.toolchain_hash:
            reasons.append("TOOLCHAIN_HASH_MISMATCH")
        if manifest.corpus_hash != evidence.corpus_hash:
            reasons.append("CORPUS_HASH_MISMATCH")
        if manifest.model_config_hash != evidence.model_config_hash:
            reasons.append("MODEL_CONFIG_HASH_MISMATCH")
        if (
            manifest.public_case_count != evidence.public_case_count
            or manifest.private_case_count != evidence.private_case_count
            or evidence.public_case_count < 16
            or evidence.private_case_count < 8
            or any(evidence.public_tool_counts.get(tool, 0) < 4 for tool in _TOOLS)
        ):
            reasons.append("CORPUS_COUNT_INSUFFICIENT")
        if evidence.integrity_errors:
            reasons.append("EVIDENCE_INTEGRITY_FAILURE")
        if not evidence.release_surfaces_complete:
            reasons.append("RELEASE_SURFACE_INCOMPLETE")
        if not evidence.evaluation_complete:
            reasons.append("EVALUATION_INCOMPLETE")
        claimed = {
            key: sorted(set(value))
            for key, value in manifest.evidence_run_ids.items()
            if key in self.REQUIRED_EVIDENCE
        }
        derived = {
            key: sorted(set(evidence.evidence_run_ids.get(key, [])))
            for key in self.REQUIRED_EVIDENCE
        }
        if set(claimed) != self.REQUIRED_EVIDENCE or any(not ids for ids in claimed.values()):
            reasons.append("LIVE_EVIDENCE_MISSING")
        if claimed != derived or set(manifest.evidence_run_ids) != self.REQUIRED_EVIDENCE:
            reasons.append("EVIDENCE_RUN_MISMATCH")
        if manifest.unresolved_items:
            reasons.append("UNRESOLVED_ITEMS")
        unique = list(dict.fromkeys(reasons))
        return ReleaseGateResult(passed=not unique, reason_codes=unique)


_TOOLS = ("memcheck", "racecheck", "initcheck", "synccheck")
_MODES = ("A", "B", "C", "D", "E")
_SURFACES: set[ReleaseSurface] = {
    "isolation",
    "four_tools",
    "private_oracle",
    "live_llm",
    "five_modes",
    "corpus",
    "release_manifest",
}


class _EvidenceBuilder:
    def __init__(self, public: RunStore, evaluator: RunStore, commit: str) -> None:
        self.public = public
        self.evaluator = evaluator
        self.commit = commit
        self.errors: set[str] = set()
        self.toolchains: set[str] = set()
        self.model_configs: set[str] = set()
        self.receipts: dict[tuple[str, str], ReleaseEvidenceReceipt] = {}
        self.evidence_ids: dict[str, list[str]] = {
            category: []
            for category in (
                "four_tools",
                "isolation",
                "private_oracle",
                "live_llm",
                "five_mode_evaluation",
            )
        }

    def build(self) -> ReleaseEvidenceIndex:
        public_runs = self._runs(self.public)
        evaluator_runs = self._runs(self.evaluator)
        self._load_receipts(self.public, public_runs)
        self._load_receipts(self.evaluator, evaluator_runs)
        public_cases = self._cases(self.public, public_runs, "public")
        private_cases = self._cases(self.evaluator, evaluator_runs, "private")
        self._split_integrity(public_cases, private_cases)
        holdout_maps = self._holdout_maps(evaluator_runs)
        evaluation_complete = self._evaluations(
            public_runs, public_cases, private_cases, holdout_maps
        )
        test_counts, surfaces_complete = self._test_report(public_runs)
        corpus_payload = [
            case.model_dump(mode="json")
            for case in sorted([*public_cases, *private_cases], key=lambda item: item.id)
        ]
        corpus_hash = hashlib.sha256(_canonical(corpus_payload)).hexdigest()
        tool_counts = Counter(case.target_tool.value for case in public_cases)
        return ReleaseEvidenceIndex(
            current_commit=self.commit,
            toolchain_hash=self._single(self.toolchains, "TOOLCHAIN_BINDING_AMBIGUOUS"),
            corpus_hash=corpus_hash,
            model_config_hash=self._single(
                self.model_configs, "MODEL_CONFIG_BINDING_AMBIGUOUS"
            ),
            test_counts=test_counts,
            public_case_count=len(public_cases),
            private_case_count=len(private_cases),
            public_tool_counts=dict(tool_counts),
            evidence_run_ids={key: sorted(value) for key, value in self.evidence_ids.items()},
            release_surfaces_complete=surfaces_complete,
            evaluation_complete=evaluation_complete,
            integrity_errors=sorted(self.errors),
        )

    def _single(self, values: set[str], reason: str) -> str | None:
        if len(values) != 1:
            if len(values) > 1:
                self.errors.add(reason)
            return None
        return next(iter(values))

    def _runs(self, store: RunStore) -> dict[str, RunManifest]:
        runs: dict[str, RunManifest] = {}
        try:
            paths = sorted(store.root.iterdir())
        except OSError:
            self.errors.add("STORE_UNAVAILABLE")
            return runs
        for path in paths:
            if path.is_symlink():
                self.errors.add("UNSAFE_STORE_ENTRY")
                continue
            if not re.fullmatch(r"[a-f0-9]{32}", path.name) or not path.is_dir():
                continue
            try:
                runs[path.name] = store.load(path.name)
            except (OSError, ValueError):
                self.errors.add("ARTIFACT_INTEGRITY_FAILURE")
        return runs

    @staticmethod
    def _ref(run: RunManifest, name: str) -> ArtifactRef:
        refs = [ref for ref in run.artifact_refs if ref.name == name]
        if len(refs) != 1:
            raise ValueError("artifact is missing or ambiguous")
        return refs[0]

    def _read(self, store: RunStore, run: RunManifest, name: str) -> bytes:
        try:
            return store.read(self._ref(run, name))
        except (OSError, ValueError):
            self.errors.add("ARTIFACT_INTEGRITY_FAILURE")
            raise

    def _load_receipts(self, store: RunStore, runs: dict[str, RunManifest]) -> None:
        for run in runs.values():
            if not any(ref.name == "release/evidence.json" for ref in run.artifact_refs):
                continue
            try:
                if run.status != RunStatus.COMPLETED:
                    raise ValueError("evidence run is not completed")
                receipt = ReleaseEvidenceReceipt.model_validate_json(
                    self._read(store, run, "release/evidence.json")
                )
                if run.id != receipt.run_id or run.kind != "release_evidence":
                    raise ValueError("receipt identity or kind mismatch")
                if receipt.commit != self.commit:
                    self.errors.add("COMMIT_BINDING_MISMATCH")
                    continue
                if receipt.category == "private_oracle":
                    if store.visibility != "evaluator":
                        raise ValueError("private evidence is in the public store")
                elif store.visibility != "public" and receipt.category != "corpus_validation":
                    raise ValueError("public evidence is in the evaluator store")
                source = runs.get(receipt.source_run_id)
                if source is None or source.status != RunStatus.COMPLETED:
                    raise ValueError("evidence source run is unavailable")
                for name, expected_hash in receipt.artifact_hashes.items():
                    if name == "release/evidence.json":
                        raise ValueError("receipt cannot attest itself")
                    ref = self._ref(source, name)
                    content = store.read(ref)
                    if (
                        ref.sha256 != expected_hash
                        or hashlib.sha256(content).hexdigest() != expected_hash
                    ):
                        raise ValueError("receipt artifact hash mismatch")
                self._validate_category(store, source, receipt)
                self.toolchains.add(receipt.toolchain_hash)
                self.receipts[(store.visibility, source.id)] = receipt
                if receipt.category in self.evidence_ids:
                    self.evidence_ids[receipt.category].append(run.id)
            except (OSError, ValueError):
                self.errors.add("ARTIFACT_INTEGRITY_FAILURE")

    @staticmethod
    def _validate_category(
        store: RunStore, source: RunManifest, receipt: ReleaseEvidenceReceipt
    ) -> None:
        def read(name: str) -> bytes:
            return store.read(_EvidenceBuilder._ref(source, name))

        names = set(receipt.artifact_hashes)
        if receipt.category == "corpus_validation":
            if source.kind != "benchmark_validation" or names != {"validation/result.json"}:
                raise ValueError("invalid corpus validation proof source")
            CorpusValidationProof.model_validate_json(
                read("validation/result.json")
            )
        elif receipt.category == "isolation":
            if source.kind != "isolation_acceptance" or names != {"acceptance/isolation.json"}:
                raise ValueError("invalid isolation proof source")
            IsolationAcceptanceProof.model_validate_json(
                read("acceptance/isolation.json")
            )
        elif receipt.category == "four_tools":
            if source.kind != "sanitizer_acceptance" or names != {
                "acceptance/four-tools.json"
            }:
                raise ValueError("invalid four-tool proof source")
            proof = FourToolsAcceptanceProof.model_validate_json(
                read("acceptance/four-tools.json")
            )
            if set(proof.clean_outcomes) != set(_TOOLS) or set(proof.fault_outcomes) != set(
                _TOOLS
            ):
                raise ValueError("four-tool acceptance is incomplete")
        elif receipt.category == "private_oracle":
            if source.kind != "verification_audit" or names != {"observation.json"}:
                raise ValueError("invalid private Oracle proof source")
            observation = VerificationObservation.model_validate_json(
                read("observation.json")
            )
            if (
                observation.private_holdout_passed is not True
                or observation.required_evidence_missing
            ):
                raise ValueError("private Oracle did not complete successfully")
        elif receipt.category == "live_llm":
            if source.kind != "diagnosis" or "diagnosis.json" not in names:
                raise ValueError("invalid live LLM proof source")
            diagnosis = DiagnosisResult.model_validate_json(
                read("diagnosis.json")
            )
            provider_names = names - {"diagnosis.json"}
            invocations = [
                Invocation.model_validate_json(read(name))
                for name in provider_names
                if name.startswith("provider/") and name.endswith("/COMPLETED.json")
            ]
            if (
                diagnosis.diagnostic_outcome != "DIAGNOSED"
                or not invocations
                or len(invocations) != len(provider_names)
                or any(
                    item.run_id != source.id
                    or item.state != "COMPLETED"
                    or not item.store_false_sent
                    or not item.response_id
                    for item in invocations
                )
            ):
                raise ValueError("live LLM proof is incomplete")
        else:
            raise ValueError("evaluation evidence is derived without receipts")

    def _cases(
        self, store: RunStore, runs: dict[str, RunManifest], split: Literal["public", "private"]
    ) -> list[CaseManifest]:
        cases: list[CaseManifest] = []
        seen: set[str] = set()
        for run in runs.values():
            if run.kind != "benchmark_case":
                continue
            try:
                if run.status != RunStatus.COMPLETED:
                    raise ValueError("case run is not completed")
                case = CaseManifest.model_validate_json(
                    self._read(store, run, "case-manifest.json")
                )
                if case.split != split or case.id in seen:
                    raise ValueError("case split or identity mismatch")
                if len(set(case.validation_run_ids)) != len(case.validation_run_ids):
                    raise ValueError("case validation IDs are duplicated")
                for validation_id in case.validation_run_ids:
                    validation = runs.get(validation_id)
                    receipt = self.receipts.get((store.visibility, validation_id))
                    if (
                        validation is None
                        or validation.status != RunStatus.COMPLETED
                        or receipt is None
                        or receipt.category != "corpus_validation"
                        or receipt.toolchain_hash != case.toolchain_hash
                    ):
                        raise ValueError("case validation evidence is unavailable")
                seen.add(case.id)
                self.toolchains.add(case.toolchain_hash)
                cases.append(case)
            except (OSError, ValueError):
                self.errors.add("CORPUS_EVIDENCE_INVALID")
        return cases

    def _split_integrity(
        self, public_cases: list[CaseManifest], private_cases: list[CaseManifest]
    ) -> None:
        public_ids = {case.id for case in public_cases}
        private_ids = {case.id for case in private_cases}
        public_templates = {case.template_id for case in public_cases}
        private_templates = {case.template_id for case in private_cases}
        public_mutations = {case.mutation_id for case in public_cases}
        private_mutations = {case.mutation_id for case in private_cases}
        if (
            public_ids & private_ids
            or public_templates & private_templates
            or public_mutations & private_mutations
        ):
            self.errors.add("CORPUS_SPLIT_OVERLAP")
        if len(private_templates) != len(private_cases) or len(private_mutations) != len(
            private_cases
        ):
            self.errors.add("PRIVATE_HOLDOUT_NOT_DISTINCT")

    def _evaluations(
        self,
        runs: dict[str, RunManifest],
        public_cases: list[CaseManifest],
        private_cases: list[CaseManifest],
        holdout_maps: dict[str, HoldoutIdentityMap],
    ) -> bool:
        expected = {
            "development": {case.id: case.template_id for case in public_cases},
            "holdout": {case.id: case.template_id for case in private_cases},
        }
        valid: dict[str, list[str]] = {"development": [], "holdout": []}
        prompt_versions: set[str] = set()
        for run in runs.values():
            if run.kind != "evaluation":
                continue
            try:
                manifest, schedule = self._validated_evaluation(
                    run, expected, holdout_maps.get(run.id)
                )
                valid[manifest.split].append(run.id)
                self.toolchains.add(manifest.toolchain_hash)
                self.model_configs.add(manifest.model_config_hash)
                prompt_versions.add(manifest.prompt_version)
            except (OSError, ValueError):
                self.errors.add("EVALUATION_EVIDENCE_INVALID")
        complete = (
            bool(expected["development"])
            and bool(expected["holdout"])
            and len(valid["development"]) == 1
            and len(valid["holdout"]) == 1
            and len(prompt_versions) == 1
        )
        if complete:
            self.evidence_ids["five_mode_evaluation"] = sorted(
                [*valid["development"], *valid["holdout"]]
            )
        return complete

    def _validated_evaluation(
        self,
        run: RunManifest,
        expected: dict[str, dict[str, str]],
        holdout_map: HoldoutIdentityMap | None,
    ) -> tuple[EvaluationManifest, EvaluationSchedule]:
        if run.status != RunStatus.COMPLETED:
            raise ValueError("evaluation run is not completed")
        schedule = EvaluationSchedule.model_validate_json(
            self._read(self.public, run, "evaluation/schedule.json")
        )
        manifest = EvaluationManifest.model_validate_json(
            self._read(self.public, run, "evaluation/manifest.json")
        )
        schedule_hash = hashlib.sha256(_canonical(schedule.model_dump(mode="json"))).hexdigest()
        if (
            manifest.run_id != run.id
            or manifest.commit != self.commit
            or schedule.bindings.commit != self.commit
            or manifest.schedule_hash != schedule_hash
            or manifest.commit != schedule.bindings.commit
            or manifest.prompt_version != schedule.bindings.prompt_version
            or manifest.toolchain_hash != schedule.bindings.toolchain_hash
            or manifest.model_config_hash != schedule.bindings.model_config_hash
            or manifest.split != schedule.split
            or manifest.repeats != schedule.repeats
            or manifest.random_seed != schedule.random_seed
            or manifest.modes != schedule.modes
            or manifest.stopped_reason is not None
            or manifest.expected_units == 0
            or manifest.expected_units != manifest.executed_units
            or manifest.expected_units != len(schedule.items)
            or manifest.executed_units != len(manifest.records)
            or schedule.selection != "all"
            or set(schedule.modes) != set(_MODES)
            or set(manifest.modes) != set(_MODES)
            or schedule.repeats < 3
        ):
            raise ValueError("evaluation bindings or counts disagree")
        expected_cases = expected[manifest.split]
        scheduled_cases = expected_cases
        if manifest.split == "holdout":
            if holdout_map is None:
                raise ValueError("holdout evaluation has no evaluator-only identity map")
            private_pairs = {(case_id, template) for case_id, template in expected_cases.items()}
            mapped_private = {
                (item.private_case_id, item.private_template_id)
                for item in holdout_map.identities
            }
            aliases = {
                item.evaluation_case_id: item.evaluation_template_id
                for item in holdout_map.identities
            }
            if (
                holdout_map.evaluation_run_id != run.id
                or holdout_map.commit != self.commit
                or holdout_map.toolchain_hash != manifest.toolchain_hash
                or len(aliases) != len(holdout_map.identities)
                or len(mapped_private) != len(holdout_map.identities)
                or mapped_private != private_pairs
                or any(
                    item.evaluation_case_id == item.private_case_id
                    or item.evaluation_template_id == item.private_template_id
                    or not re.fullmatch(
                        r"holdout-[a-f0-9]{16,64}", item.evaluation_case_id
                    )
                    or not re.fullmatch(
                        r"opaque-[a-f0-9]{16,64}", item.evaluation_template_id
                    )
                    for item in holdout_map.identities
                )
            ):
                raise ValueError("holdout identity map is incomplete or identifying")
            scheduled_cases = aliases
        wanted = {
            (case_id, template_id, mode, repeat)
            for case_id, template_id in scheduled_cases.items()
            for mode in _MODES
            for repeat in range(schedule.repeats)
        }
        actual = {
            (item.case_id, item.template_id, item.mode, item.repeat) for item in schedule.items
        }
        if (
            not expected_cases
            or actual != wanted
            or len(actual) != len(schedule.items)
            or [item.ordinal for item in schedule.items] != list(range(len(schedule.items)))
        ):
            raise ValueError("evaluation schedule does not cover the frozen corpus")
        persisted: list[PublicEvaluationRecord] = []
        expected_record_names = {
            f"evaluation/records/{item.ordinal}.json" for item in schedule.items
        }
        expected_attempt_names = {
            f"evaluation/attempts/{item.ordinal}.json" for item in schedule.items
        }
        actual_record_names = {
            ref.name
            for ref in run.artifact_refs
            if ref.name.startswith("evaluation/records/")
        }
        actual_attempt_names = {
            ref.name
            for ref in run.artifact_refs
            if ref.name.startswith("evaluation/attempts/")
        }
        if (
            actual_record_names != expected_record_names
            or actual_attempt_names != expected_attempt_names
        ):
            raise ValueError("evaluation unit artifacts are missing or unexpected")
        for item in schedule.items:
            record = PublicEvaluationRecord.model_validate_json(
                self._read(self.public, run, f"evaluation/records/{item.ordinal}.json")
            )
            attempt = EvaluationAttempt.model_validate_json(
                self._read(self.public, run, f"evaluation/attempts/{item.ordinal}.json")
            )
            if (
                (record.case_id, record.template_id, record.mode, record.repeat)
                != (item.case_id, item.template_id, item.mode, item.repeat)
                or attempt.run_id != run.id
                or attempt.ordinal != item.ordinal
                or attempt.schedule_hash != schedule_hash
            ):
                raise ValueError("persisted evaluation unit does not match its schedule")
            persisted.append(record)
        if persisted != manifest.records:
            raise ValueError("evaluation manifest does not match persisted records")
        return manifest, schedule

    def _holdout_maps(
        self, runs: dict[str, RunManifest]
    ) -> dict[str, HoldoutIdentityMap]:
        maps: dict[str, HoldoutIdentityMap] = {}
        for run in runs.values():
            if run.kind != "holdout_identity_map":
                continue
            try:
                if run.status != RunStatus.COMPLETED:
                    raise ValueError("holdout identity map is not completed")
                identity_map = HoldoutIdentityMap.model_validate_json(
                    self._read(
                        self.evaluator, run, "release/holdout-identity-map.json"
                    )
                )
                if (
                    identity_map.commit != self.commit
                    or identity_map.evaluation_run_id in maps
                ):
                    raise ValueError("holdout identity map binding is invalid")
                self.toolchains.add(identity_map.toolchain_hash)
                maps[identity_map.evaluation_run_id] = identity_map
            except (OSError, ValueError):
                self.errors.add("HOLDOUT_IDENTITY_MAP_INVALID")
        return maps

    def _test_report(self, runs: dict[str, RunManifest]) -> tuple[TestCounts, bool]:
        reports: list[ReleaseTestReport] = []
        for run in runs.values():
            if run.kind != "release_tests":
                continue
            try:
                if run.status != RunStatus.COMPLETED:
                    raise ValueError("test run is not completed")
                report = ReleaseTestReport.model_validate_json(
                    self._read(self.public, run, "release/test-report.json")
                )
                junit = self._read(self.public, run, "release/pytest.xml")
                if report != _parse_pytest_report(junit, self.commit):
                    raise ValueError("test report commit mismatch")
                reports.append(report)
            except (OSError, ValueError):
                self.errors.add("RELEASE_TEST_EVIDENCE_INVALID")
        if len(reports) != 1:
            if reports:
                self.errors.add("RELEASE_TEST_EVIDENCE_AMBIGUOUS")
            return TestCounts(expected=0, executed=0, skipped_required=0, failed=0), False
        report = reports[0]
        surfaces_complete = (
            set(report.surface_counts) == _SURFACES
            and all(report.surface_counts[surface] > 0 for surface in _SURFACES)
        )
        return report.test_counts, surfaces_complete


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


_SURFACE_FILES: dict[str, ReleaseSurface] = {
    "tests/integration/test_isolation.py": "isolation",
    "tests/gpu/test_failure_families.py": "four_tools",
    "tests/gpu/test_candidate_verification.py": "private_oracle",
    "tests/e2e/test_oob_flow.py": "live_llm",
    "tests/unit/test_evaluation_modes.py": "five_modes",
    "tests/gpu/test_mutation_validation.py": "corpus",
    "tests/e2e/test_release_acceptance.py": "release_manifest",
}


def _parse_pytest_report(content: bytes, current_commit: str) -> ReleaseTestReport:
    if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
        raise ValueError("XML declarations with entities are not accepted")
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        raise ValueError("invalid pytest JUnit report") from exc
    cases = list(root.iter("testcase"))
    if root.tag == "testsuite":
        expected = int(root.attrib.get("tests", len(cases)))
    else:
        suites = [item for item in root if item.tag == "testsuite"]
        expected = sum(int(item.attrib.get("tests", 0)) for item in suites)
    skipped = failed = 0
    surface_counts: Counter[ReleaseSurface] = Counter()
    node_ids: list[str] = []
    for case in cases:
        file_name = case.attrib.get("file", "").replace("\\", "/")
        if not file_name:
            module_name = case.attrib.get("classname", "")
            file_name = f"{module_name.replace('.', '/')}.py"
        name = case.attrib.get("name", "")
        node_ids.append(f"{file_name}::{name}")
        is_skipped = case.find("skipped") is not None
        is_failed = case.find("failure") is not None or case.find("error") is not None
        skipped += int(is_skipped)
        failed += int(is_failed)
        surface = next(
            (value for suffix, value in _SURFACE_FILES.items() if file_name.endswith(suffix)),
            None,
        )
        if surface is not None and not is_skipped and not is_failed:
            surface_counts[surface] += 1
    skipped = max(skipped, int(root.attrib.get("skipped", 0)))
    failed = max(
        failed,
        int(root.attrib.get("failures", 0)) + int(root.attrib.get("errors", 0)),
    )
    return ReleaseTestReport(
        commit=current_commit,
        junit_sha256=hashlib.sha256(content).hexdigest(),
        test_counts=TestCounts(
            expected=expected,
            executed=len(cases),
            skipped_required=skipped,
            failed=failed,
        ),
        surface_counts=dict(surface_counts),
        selected_node_ids=node_ids,
    )


def record_pytest_report(
    store: RunStore, junit_path: Path, *, current_commit: str
) -> ReleaseTestReport:
    """Persist a release-test report derived from bounded pytest JUnit output."""

    if store.visibility != "public":
        raise ValueError("release test evidence must use the public store")
    if not re.fullmatch(r"[a-f0-9]{40}", current_commit):
        raise ValueError("current commit must be a full lowercase Git hash")
    content = _normalize_pytest_xml(
        read_regular(junit_path.absolute(), 8 * 1024 * 1024)
    )
    report = _parse_pytest_report(content, current_commit)
    run = store.create_run("release_tests")
    store.put(run.id, "release/pytest.xml", content, "public")
    store.put(
        run.id,
        "release/test-report.json",
        report.model_dump_json().encode(),
        "public",
    )
    store.transition(run.id, RunStatus.RUNNING, "FINALIZING")
    store.transition(run.id, RunStatus.COMPLETED, None)
    return report


def _normalize_pytest_xml(content: bytes) -> bytes:
    """Retain only collection/outcome fields; discard host, logs and error text."""

    if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
        raise ValueError("XML declarations with entities are not accepted")
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        raise ValueError("invalid pytest JUnit report") from exc
    cases = list(root.iter("testcase"))
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = [item for item in root if item.tag == "testsuite"]
    totals = {
        field: sum(int(suite.attrib.get(field, 0)) for suite in suites)
        for field in ("tests", "errors", "failures", "skipped")
    }
    if totals["tests"] == 0:
        totals["tests"] = len(cases)
    normalized = ElementTree.Element(
        "testsuite", {field: str(value) for field, value in totals.items()}
    )
    for case in cases:
        attributes = {
            key: case.attrib[key]
            for key in ("classname", "file", "name")
            if key in case.attrib
        }
        clean = ElementTree.SubElement(normalized, "testcase", attributes)
        for outcome in ("skipped", "failure", "error"):
            if case.find(outcome) is not None:
                ElementTree.SubElement(clean, outcome)
    return cast(bytes, ElementTree.tostring(normalized, encoding="utf-8"))


def record_release_evidence(
    store: RunStore,
    source_run_id: str,
    category: EvidenceCategory,
    *,
    current_commit: str,
    toolchain_hash: str,
) -> ReleaseEvidenceReceipt:
    """Validate a category-specific source run before publishing its immutable receipt."""

    if category == "five_mode_evaluation":
        raise ValueError("five-mode evidence is derived directly from evaluation artifacts")
    if not re.fullmatch(r"[a-f0-9]{40}", current_commit) or not re.fullmatch(
        r"[a-f0-9]{64}", toolchain_hash
    ):
        raise ValueError("release bindings must use full lowercase hashes")
    if (
        (category == "private_oracle" and store.visibility != "evaluator")
        or (
            category in {"isolation", "four_tools", "live_llm"}
            and store.visibility != "public"
        )
    ):
        raise ValueError("release evidence has the wrong visibility")
    source = store.load(source_run_id)
    if source.status != RunStatus.COMPLETED:
        raise ValueError("release evidence source must be completed")
    fixed_names = {
        "corpus_validation": {"validation/result.json"},
        "isolation": {"acceptance/isolation.json"},
        "four_tools": {"acceptance/four-tools.json"},
        "private_oracle": {"observation.json"},
    }
    names = fixed_names.get(category, set())
    if category == "live_llm":
        names = {"diagnosis.json"} | {
            ref.name
            for ref in source.artifact_refs
            if ref.name.startswith("provider/") and ref.name.endswith("/COMPLETED.json")
        }
    refs = {name: _EvidenceBuilder._ref(source, name) for name in names}
    hashes = {name: hashlib.sha256(store.read(ref)).hexdigest() for name, ref in refs.items()}
    provisional = ReleaseEvidenceReceipt(
        category=category,
        run_id="0" * 32,
        source_run_id=source.id,
        commit=current_commit,
        toolchain_hash=toolchain_hash,
        artifact_hashes=hashes,
    )
    _EvidenceBuilder._validate_category(store, source, provisional)
    run = store.create_run("release_evidence")
    receipt = provisional.model_copy(update={"run_id": run.id})
    store.put(
        run.id,
        "release/evidence.json",
        receipt.model_dump_json().encode(),
        store.visibility,
    )
    store.transition(run.id, RunStatus.RUNNING, "FINALIZING")
    store.transition(run.id, RunStatus.COMPLETED, None)
    return receipt


def record_holdout_identity_map(
    public: RunStore, evaluator: RunStore, identity_map: HoldoutIdentityMap
) -> str:
    """Persist a private mapping only after it exactly covers a frozen opaque schedule."""

    if public.visibility != "public" or evaluator.visibility != "evaluator":
        raise ValueError("holdout map stores have the wrong visibility")
    evaluation = public.load(identity_map.evaluation_run_id)
    if evaluation.kind != "evaluation" or evaluation.status != RunStatus.COMPLETED:
        raise ValueError("holdout map requires a completed evaluation run")
    schedule = EvaluationSchedule.model_validate_json(
        public.read(_EvidenceBuilder._ref(evaluation, "evaluation/schedule.json"))
    )
    aliases = {
        (item.evaluation_case_id, item.evaluation_template_id)
        for item in identity_map.identities
    }
    scheduled = {(item.case_id, item.template_id) for item in schedule.items}
    if (
        schedule.split != "holdout"
        or schedule.bindings.commit != identity_map.commit
        or schedule.bindings.toolchain_hash != identity_map.toolchain_hash
        or len(aliases) != len(identity_map.identities)
        or aliases != scheduled
        or any(
            not re.fullmatch(r"holdout-[a-f0-9]{16,64}", case_id)
            or not re.fullmatch(r"opaque-[a-f0-9]{16,64}", template_id)
            for case_id, template_id in aliases
        )
    ):
        raise ValueError("holdout identity map does not match the opaque schedule")
    run = evaluator.create_run("holdout_identity_map")
    evaluator.put(
        run.id,
        "release/holdout-identity-map.json",
        identity_map.model_dump_json().encode(),
        "evaluator",
    )
    evaluator.transition(run.id, RunStatus.RUNNING, "FINALIZING")
    evaluator.transition(run.id, RunStatus.COMPLETED, None)
    return run.id
