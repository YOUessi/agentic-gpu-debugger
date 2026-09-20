import pytest
from pydantic import ValidationError


@pytest.fixture
def repository():
    from gpu_agent.contracts import RepositorySnapshot

    return RepositorySnapshot(commit="a" * 40, tracked_tree_hash="f" * 64, clean=True)


@pytest.fixture
def evidence(repository):
    from gpu_agent.benchmark.release import ReleaseEvidenceIndex, TestCounts

    run_ids = {
        "four_tools": ["1" * 32],
        "isolation": ["2" * 32],
        "private_oracle": ["3" * 32],
        "live_llm": ["4" * 32],
        "five_mode_evaluation": ["5" * 32, "6" * 32],
        "public_corpus": [f"{index:032x}" for index in range(16, 32)],
        "private_corpus": [f"{index:032x}" for index in range(32, 40)],
        "private_scoring": ["7" * 32],
        "release_tests": ["8" * 32],
    }
    return ReleaseEvidenceIndex(
        repository=repository,
        toolchain_hash="b" * 64,
        corpus_hash="c" * 64,
        model_config_hash="d" * 64,
        prompt_version="v2",
        corpus_cutoff=24,
        test_counts=TestCounts(expected=12, executed=12, skipped_required=0, failed=0),
        public_case_count=16,
        private_case_count=8,
        private_template_count=8,
        private_operator_count=8,
        public_tool_case_counts={
            "memcheck": 4,
            "racecheck": 4,
            "initcheck": 4,
            "synccheck": 4,
        },
        development_units=240,
        holdout_units=120,
        evaluation_modes=["A", "B", "C", "D", "E"],
        evaluation_repeats=3,
        evidence_run_ids=run_ids,
    )


@pytest.fixture
def manifest(evidence):
    from gpu_agent.benchmark.release import ReleaseManifest

    return ReleaseManifest(
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


def test_complete_derived_evidence_passes(manifest, evidence):
    from gpu_agent.benchmark.release import ReleaseGate

    assert ReleaseGate().check(manifest, evidence).passed


def test_manifest_cannot_pass_without_derived_index(manifest):
    from gpu_agent.benchmark.release import ReleaseGate

    result = ReleaseGate().check(manifest)
    assert not result.passed
    assert "EVIDENCE_INDEX_REQUIRED" in result.reason_codes


@pytest.mark.parametrize(
    "manifest_update,evidence_update,reason",
    [
        (
            {
                "test_counts": {
                    "expected": 12,
                    "executed": 11,
                    "skipped_required": 0,
                    "failed": 0,
                }
            },
            {},
            "TEST_COUNT_INCOMPLETE",
        ),
        (
            {
                "test_counts": {
                    "expected": 12,
                    "executed": 12,
                    "skipped_required": 1,
                    "failed": 0,
                }
            },
            {},
            "REQUIRED_TEST_SKIPPED",
        ),
        ({"commit": "9" * 40}, {}, "REPOSITORY_BINDING_MISMATCH"),
        ({"toolchain_hash": "9" * 64}, {}, "CONFIGURATION_BINDING_MISMATCH"),
        ({"corpus_hash": "9" * 64}, {}, "CONFIGURATION_BINDING_MISMATCH"),
        ({"model_config_hash": "9" * 64}, {}, "CONFIGURATION_BINDING_MISMATCH"),
        ({"public_case_count": 15}, {}, "CORPUS_COUNT_INSUFFICIENT"),
        ({}, {"private_template_count": 7}, "PRIVATE_DIVERSITY_INSUFFICIENT"),
        ({}, {"private_operator_count": 7}, "PRIVATE_DIVERSITY_INSUFFICIENT"),
        (
            {},
            {
                "public_tool_case_counts": {
                    "memcheck": 4,
                    "racecheck": 4,
                    "initcheck": 4,
                    "synccheck": 3,
                }
            },
            "SANITIZER_FAMILY_COVERAGE_INSUFFICIENT",
        ),
        ({}, {"development_units": 239}, "FIVE_MODE_EVALUATION_INCOMPLETE"),
        ({}, {"holdout_units": 119}, "FIVE_MODE_EVALUATION_INCOMPLETE"),
        ({}, {"evaluation_modes": ["A", "B", "C", "D"]}, "FIVE_MODE_EVALUATION_INCOMPLETE"),
        ({}, {"evaluation_repeats": 2}, "FIVE_MODE_EVALUATION_INCOMPLETE"),
        ({"evidence_run_ids": {}}, {}, "EVIDENCE_SELECTION_MISMATCH"),
        ({"unresolved_items": ["paid evaluation pending"]}, {}, "UNRESOLVED_ITEMS"),
        ({}, {"reason_codes": ["PRIVATE_SCORING_INCOMPLETE"]}, "PRIVATE_SCORING_INCOMPLETE"),
    ],
)
def test_claim_or_evidence_defect_fails(
    manifest, evidence, manifest_update, evidence_update, reason
):
    from gpu_agent.benchmark.release import (
        ReleaseEvidenceIndex,
        ReleaseGate,
        ReleaseManifest,
    )

    changed_manifest = ReleaseManifest.model_validate({**manifest.model_dump(), **manifest_update})
    changed_evidence = ReleaseEvidenceIndex.model_validate(
        {**evidence.model_dump(), **evidence_update}
    )
    result = ReleaseGate().check(changed_manifest, changed_evidence)
    assert not result.passed
    assert reason in result.reason_codes


def test_zero_tests_cannot_pass_even_when_claim_counts_match(manifest, evidence):
    from gpu_agent.benchmark.release import (
        ReleaseEvidenceIndex,
        ReleaseGate,
        ReleaseManifest,
        TestCounts,
    )

    zero = TestCounts(expected=0, executed=0, skipped_required=0, failed=0)
    manifest = ReleaseManifest.model_validate(
        {**manifest.model_dump(), "test_counts": zero.model_dump()}
    )
    evidence = ReleaseEvidenceIndex.model_validate(
        {**evidence.model_dump(), "test_counts": zero.model_dump()}
    )
    result = ReleaseGate().check(manifest, evidence)
    assert not result.passed
    assert "REQUIRED_TESTS_MISSING" in result.reason_codes


def test_selection_rejects_reused_run_ids(repository):
    from gpu_agent.benchmark.release import ReleaseEvidenceSelection

    with pytest.raises(ValidationError, match="duplicate run IDs"):
        ReleaseEvidenceSelection(
            repository=repository,
            public_case_run_ids=["1" * 32],
            private_case_run_ids=["2" * 32],
            development_evaluation_run_id="3" * 32,
            holdout_evaluation_run_id="4" * 32,
            private_binding_run_id="5" * 32,
            acceptance_run_ids={"four_tools": ["1" * 32]},
            release_test_run_id="6" * 32,
        )


def test_derivation_rejects_repository_drift(tmp_path, repository):
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.release import ReleaseEvidenceIndex
    from gpu_agent.contracts import RepositorySnapshot
    from gpu_agent.store import RunStore

    public = RunStore(tmp_path / "public")
    evaluator = RunStore(tmp_path / "evaluator", visibility="evaluator")
    family = CorpusFamily.provision(
        tmp_path / "controller",
        public_store=public.root,
        evaluator_store=evaluator.root,
        repository=tmp_path / "repository",
    )
    selection = _selection(repository)
    actual = RepositorySnapshot(
        commit="e" * 40, tracked_tree_hash=repository.tracked_tree_hash, clean=True
    )
    result = ReleaseEvidenceIndex.derive(selection, public, evaluator, family, actual)
    assert result.reason_codes == ["ACTUAL_REPOSITORY_MISMATCH"]


def test_derivation_rejects_nonexistent_selected_runs(tmp_path, repository):
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.release import ReleaseEvidenceIndex
    from gpu_agent.store import RunStore

    public = RunStore(tmp_path / "public")
    evaluator = RunStore(tmp_path / "evaluator", visibility="evaluator")
    family = CorpusFamily.provision(
        tmp_path / "controller",
        public_store=public.root,
        evaluator_store=evaluator.root,
        repository=tmp_path / "repository",
    )
    result = ReleaseEvidenceIndex.derive(
        _selection(repository), public, evaluator, family, repository
    )
    assert result.reason_codes == ["EVALUATION_EVIDENCE_INVALID"]


def test_manifest_can_only_be_created_from_complete_derived_evidence(repository):
    from gpu_agent.benchmark.release import (
        ReleaseEvidenceIndex,
        ReleaseManifest,
        TestCounts,
    )

    incomplete = ReleaseEvidenceIndex(repository=repository)
    with pytest.raises(ValueError, match="complete"):
        ReleaseManifest.from_evidence(incomplete)

    evidence = ReleaseEvidenceIndex(
        repository=repository,
        toolchain_hash="1" * 64,
        corpus_hash="2" * 64,
        model_config_hash="3" * 64,
        test_counts=TestCounts(expected=10, executed=10, skipped_required=0, failed=0),
        public_case_count=16,
        private_case_count=8,
        evidence_run_ids={"five_mode_evaluation": ["4" * 32]},
    )
    manifest = ReleaseManifest.from_evidence(evidence)
    assert manifest.commit == repository.commit
    assert manifest.public_case_count == 16
    assert manifest.unresolved_items == []


def _selection(repository):
    from gpu_agent.benchmark.release import ReleaseEvidenceSelection

    return ReleaseEvidenceSelection(
        repository=repository,
        public_case_run_ids=["1" * 32],
        private_case_run_ids=["2" * 32],
        development_evaluation_run_id="3" * 32,
        holdout_evaluation_run_id="4" * 32,
        private_binding_run_id="5" * 32,
        acceptance_run_ids={
            "four_tools": ["6" * 32],
            "isolation": ["7" * 32],
            "private_oracle": ["8" * 32],
            "live_llm": ["9" * 32],
        },
        release_test_run_id="a" * 32,
    )
