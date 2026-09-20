import hashlib
import json
import stat
from types import SimpleNamespace

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


@pytest.mark.parametrize("configured", [None, "relative/release-selection.json", "inside"])
def test_release_artifact_path_rejects_missing_relative_or_repository_local_configuration(
    tmp_path, monkeypatch, configured
):
    from gpu_agent.benchmark.release import external_release_artifact_path

    repository = tmp_path / "repository"
    repository.mkdir()
    variable = "GPU_AGENT_RELEASE_SELECTION"
    if configured is None:
        monkeypatch.delenv(variable, raising=False)
    elif configured == "inside":
        monkeypatch.setenv(variable, str(repository / "evaluation/release-selection.json"))
    else:
        monkeypatch.setenv(variable, configured)

    with pytest.raises(ValueError, match="external absolute path"):
        external_release_artifact_path(variable, repository)


def test_release_artifact_path_accepts_controller_owned_absolute_path(tmp_path, monkeypatch):
    from gpu_agent.benchmark.release import external_release_artifact_path

    repository = tmp_path / "repository"
    repository.mkdir()
    configured = tmp_path / "controller" / "release-selection.json"
    monkeypatch.setenv("GPU_AGENT_RELEASE_SELECTION", str(configured))

    assert external_release_artifact_path("GPU_AGENT_RELEASE_SELECTION", repository) == configured


def test_release_artifact_wrapper_rejects_forbidden_store(tmp_path):
    from gpu_agent.benchmark.release import validate_external_release_artifact_path

    store = tmp_path / "public"
    store.mkdir()

    with pytest.raises(ValueError, match="external absolute path"):
        validate_external_release_artifact_path(
            store / "selection.json",
            tmp_path / "repository",
            forbidden_roots=(store,),
        )


@pytest.mark.parametrize("command", ["derive-manifest", "check"])
def test_release_cli_rejects_repository_local_artifact_paths(tmp_path, command):
    from typer.testing import CliRunner

    from gpu_agent.cli import app

    repository = tmp_path / "repository"
    evaluation = repository / "evaluation"
    evaluation.mkdir(parents=True)
    selection = evaluation / "release-selection.json"
    manifest = evaluation / "release-manifest.json"
    selection.write_text("{}")
    manifest.write_text("{}")
    arguments = [
        "release",
        command,
        "--selection",
        str(selection),
        "--repository",
        str(repository),
    ]
    if command == "check":
        arguments.extend(["--manifest", str(manifest)])

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    assert "RELEASE_ARTIFACT_PATH_INVALID" in result.output


def test_release_cli_accepts_external_artifact_path_before_evidence_validation(tmp_path):
    from typer.testing import CliRunner

    from gpu_agent.cli import app

    repository = tmp_path / "repository"
    repository.mkdir()
    selection = tmp_path / "controller" / "release-selection.json"
    selection.parent.mkdir()
    selection.write_text("{}")

    result = CliRunner().invoke(
        app,
        [
            "release",
            "derive-manifest",
            "--selection",
            str(selection),
            "--repository",
            str(repository),
        ],
    )

    assert result.exit_code == 2
    assert "RELEASE_ARTIFACT_PATH_INVALID" not in result.output
    assert "RELEASE_EVIDENCE_INCOMPLETE" in result.output


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
            "isolation": ["a" * 32],
            "private_oracle": ["8" * 32],
            "live_llm": ["9" * 32],
        },
        release_test_run_id="a" * 32,
    )


def test_freezer_resolves_the_same_selection_checked_by_release_gate(repository, monkeypatch):
    from gpu_agent.benchmark.release import (
        ReleaseEvidenceFreezer,
        ReleaseEvidenceIndex,
        ReleaseEvidenceRoots,
        TestCounts,
        _ReleaseEvidenceResolver,
    )

    roots = ReleaseEvidenceRoots(
        development_evaluation_run_id="1" * 32,
        holdout_evaluation_run_id="2" * 32,
        private_binding_run_id="3" * 32,
        release_test_run_id="4" * 32,
    )
    binding = SimpleNamespace(
        toolchain_lock_hash="1" * 64,
        model_config_hash="2" * 64,
        prompt_version="v2",
    )

    def evaluation(run_id, split):
        diagnosis_id = "d" * 32 if split == "development" else "e" * 32
        return SimpleNamespace(
            run=SimpleNamespace(id=run_id),
            binding=binding,
            schedule=SimpleNamespace(
                corpus_cutoff=4,
                modes=["A", "B", "C", "D", "E"],
                repeats=3,
                items=[SimpleNamespace(mode="E")],
            ),
            records=[SimpleNamespace(lineage=SimpleNamespace(diagnosis_run_id=diagnosis_id))],
        )

    public_cases = {
        "case-b": SimpleNamespace(
            target_tool=SimpleNamespace(value="memcheck"),
            validation_run_ids=["0" * 32, "a" * 32],
        ),
        "case-a": SimpleNamespace(
            target_tool=SimpleNamespace(value="racecheck"),
            validation_run_ids=["0" * 32, "9" * 32],
        ),
    }
    private_cases = {
        "private-b": SimpleNamespace(mutation_id="operator-b"),
        "private-a": SimpleNamespace(mutation_id="operator-a"),
    }
    monkeypatch.setattr(_ReleaseEvidenceResolver, "_validate_roots", lambda self: None)
    monkeypatch.setattr(
        _ReleaseEvidenceResolver,
        "_evaluation",
        lambda self, run_id, split: evaluation(run_id, split),
    )
    monkeypatch.setattr(
        _ReleaseEvidenceResolver,
        "_same_evaluation_binding",
        lambda self, development, holdout: None,
    )
    monkeypatch.setattr(
        _ReleaseEvidenceResolver,
        "_corpus",
        lambda self, selected_binding, cutoff: (
            public_cases,
            private_cases,
            ["6" * 32, "5" * 32],
            ["8" * 32, "7" * 32],
        ),
    )
    monkeypatch.setattr(
        _ReleaseEvidenceResolver,
        "_validate_development_records",
        lambda self, development, cases: None,
    )
    monkeypatch.setattr(
        _ReleaseEvidenceResolver,
        "_validate_holdout_records",
        lambda self, holdout, cases, cutoff: (
            {"template-a", "template-b"},
            [],
            {"c" * 32, "b" * 32},
        ),
    )
    monkeypatch.setattr(
        _ReleaseEvidenceResolver,
        "_scoring_session",
        lambda self, holdout, bindings: "f" * 32,
        raising=False,
    )
    monkeypatch.setattr(
        _ReleaseEvidenceResolver,
        "_release_tests",
        lambda self, selected_binding, cutoff: TestCounts(
            expected=1, executed=1, skipped_required=0, failed=0
        ),
    )
    monkeypatch.setattr(
        _ReleaseEvidenceResolver,
        "_corpus_hash",
        lambda self, public, private, cutoff: "3" * 64,
    )

    public = evaluator = family = object()
    resolution = ReleaseEvidenceFreezer.resolve(roots, public, evaluator, family, repository)
    checked = ReleaseEvidenceIndex.derive(
        resolution.selection, public, evaluator, family, repository
    )

    assert resolution.selection.public_case_run_ids == ["6" * 32, "5" * 32]
    assert resolution.selection.private_case_run_ids == ["8" * 32, "7" * 32]
    assert resolution.selection.acceptance_run_ids == {
        "four_tools": ["9" * 32, "a" * 32],
        "isolation": ["4" * 32],
        "live_llm": ["d" * 32, "e" * 32],
        "private_oracle": ["b" * 32, "c" * 32],
    }
    assert resolution.evidence.evidence_run_ids["private_scoring"] == [
        "3" * 32,
        "f" * 32,
    ]
    assert checked == resolution.evidence
    assert checked.reason_codes == []


def _scoring_session_fixture(tmp_path, repository, monkeypatch, fault=None):
    from gpu_agent.benchmark.holdout import EvaluatorRecordBinding
    from gpu_agent.benchmark.holdout_scoring import HoldoutScoringResult
    from gpu_agent.benchmark.release import ReleaseEvidenceRoots, _ReleaseEvidenceResolver
    from gpu_agent.contracts import ExternalRunOrigin, RunBinding, RunStatus
    from gpu_agent.store import RunStore

    public = RunStore(tmp_path / "public")
    evaluator = RunStore(tmp_path / "evaluator", visibility="evaluator")
    roots = ReleaseEvidenceRoots(
        development_evaluation_run_id="1" * 32,
        holdout_evaluation_run_id="2" * 32,
        private_binding_run_id="3" * 32,
        release_test_run_id="4" * 32,
    )
    binding = RunBinding(
        repository=repository,
        purpose="evaluation",
        toolchain_lock_hash="5" * 64,
        prompt_version="v2",
        model_config_hash="6" * 64,
    )
    schedule_payload = {"schedule": "holdout"}
    records = [
        SimpleNamespace(blind=(lambda ordinal=ordinal: {"blind_id": f"{5000 + ordinal:064x}"}))
        for ordinal in range(120)
    ]
    record_hashes = {ordinal: f"{3000 + ordinal:064x}" for ordinal in range(120)}
    schedule = SimpleNamespace(
        corpus_cutoff=8,
        holdout_proof=SimpleNamespace(aliases_hash="7" * 64),
        model_dump=lambda mode: schedule_payload,
    )
    evidence = SimpleNamespace(
        run=SimpleNamespace(id=roots.holdout_evaluation_run_id),
        binding=binding,
        schedule=schedule,
        records=records,
        record_hashes=record_hashes,
    )
    bindings = [
        EvaluatorRecordBinding(
            evaluator_score_run_id=f"{1000 + ordinal:032x}",
            public_evaluation_run_id=roots.holdout_evaluation_run_id,
            public_record_id=f"{2000 + ordinal:032x}",
            public_record_hash=f"{3000 + ordinal:064x}",
            private_case_id=f"private-{ordinal % 8}",
            private_template_id=f"template-{ordinal % 8}",
            private_score_hash=f"{4000 + ordinal:064x}",
            corpus_cutoff=8,
        )
        for ordinal in range(120)
    ]
    metrics_payload = {"metrics": "exact"}
    monkeypatch.setattr(
        _ReleaseEvidenceResolver,
        "_scoring_metrics",
        lambda self, exact_bindings, selected_binding: metrics_payload,
        raising=False,
    )
    resolver = _ReleaseEvidenceResolver(roots, public, evaluator, object(), repository)
    if fault == "no_session":
        return resolver, evidence, bindings

    session_id = hashlib.sha256(
        (
            f"holdout-scoring-v1:{roots.holdout_evaluation_run_id}:{roots.private_binding_run_id}"
        ).encode()
    ).hexdigest()[:32]
    origin_run_id = "8" * 32 if fault == "other_pair" else roots.holdout_evaluation_run_id
    run = evaluator.create_run(
        "holdout_scoring",
        binding=binding,
        external_origin=ExternalRunOrigin(run_id=origin_run_id, visibility="public"),
        _run_id=session_id,
    )
    evaluator.transition(run.id, RunStatus.RUNNING, "PREPARING")
    input_payload = {
        "schema_version": 1,
        "evaluation_run_id": roots.holdout_evaluation_run_id,
        "private_binding_run_id": roots.private_binding_run_id,
        "schedule_hash": hashlib.sha256(
            json.dumps(schedule_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "aliases_hash": "7" * 64,
        "corpus_cutoff": 8,
        "expected_record_count": 120,
        "record_set_hash": hashlib.sha256(
            json.dumps(
                [
                    [
                        ordinal,
                        record_hashes[ordinal],
                        hashlib.sha256(
                            json.dumps(
                                records[ordinal].blind(),
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest(),
                    ]
                    for ordinal in range(120)
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "rubric_hash": "a" * 64,
        "package_hash": "b" * 64,
    }
    if fault == "input":
        input_payload["record_set_hash"] = "c" * 64
        input_payload["unexpected"] = True
    if fault == "other_pair":
        input_payload["evaluation_run_id"] = origin_run_id
    input_content = json.dumps(input_payload, sort_keys=True, separators=(",", ":")).encode()
    ordered = list(reversed(bindings)) if fault == "bindings_order" else bindings
    bindings_content = json.dumps(
        [item.model_dump(mode="json") for item in ordered],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    stored_metrics = {"metrics": "altered"} if fault == "metrics" else metrics_payload
    metrics_content = json.dumps(stored_metrics, sort_keys=True, separators=(",", ":")).encode()
    result = HoldoutScoringResult(
        scoring_run_id=session_id,
        evaluation_run_id=roots.holdout_evaluation_run_id,
        private_binding_run_id=roots.private_binding_run_id,
        package_hash="b" * 64,
        input_binding_hash=hashlib.sha256(input_content).hexdigest(),
        bindings_hash=hashlib.sha256(bindings_content).hexdigest(),
        metrics_hash=(
            "d" * 64 if fault == "result_hash" else hashlib.sha256(metrics_content).hexdigest()
        ),
    )
    for name, content in (
        ("input-binding", input_content),
        ("bindings", bindings_content),
        ("metrics", metrics_content),
        (
            "result",
            json.dumps(
                result.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        ),
    ):
        evaluator.put(run.id, f"holdout-scoring/{name}.json", content, "evaluator")
    if fault == "failed":
        evaluator.transition(run.id, RunStatus.FAILED, None)
    elif fault != "running":
        evaluator.transition(run.id, RunStatus.RUNNING, "FINALIZING")
        evaluator.transition(run.id, RunStatus.COMPLETED, None)
    return resolver, evidence, bindings


def test_release_resolver_requires_exact_completed_scoring_session(
    tmp_path, repository, monkeypatch
):
    resolver, holdout, bindings = _scoring_session_fixture(tmp_path, repository, monkeypatch)

    assert (
        resolver._scoring_session(holdout, bindings)
        == hashlib.sha256((f"holdout-scoring-v1:{'2' * 32}:{'3' * 32}").encode()).hexdigest()[:32]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "no_session",
        "running",
        "failed",
        "input",
        "bindings_order",
        "metrics",
        "result_hash",
        "other_pair",
    ],
)
def test_release_resolver_rejects_inexact_scoring_session(tmp_path, repository, monkeypatch, fault):
    resolver, holdout, bindings = _scoring_session_fixture(tmp_path, repository, monkeypatch, fault)

    with pytest.raises(ValueError) as error:
        resolver._scoring_session(holdout, bindings)

    assert getattr(error.value, "code", None) == "PRIVATE_SCORING_INCOMPLETE"


def test_release_resolver_rejects_duplicate_mode_e_diagnosis_ids(tmp_path, repository):
    from gpu_agent.benchmark.release import ReleaseEvidenceRoots, _ReleaseEvidenceResolver
    from gpu_agent.store import RunStore

    roots = ReleaseEvidenceRoots(
        development_evaluation_run_id="1" * 32,
        holdout_evaluation_run_id="2" * 32,
        private_binding_run_id="3" * 32,
        release_test_run_id="4" * 32,
    )
    resolver = _ReleaseEvidenceResolver(
        roots,
        RunStore(tmp_path / "public"),
        RunStore(tmp_path / "evaluator", visibility="evaluator"),
        object(),
        repository,
    )
    duplicate = "5" * 32
    evaluation = SimpleNamespace(
        schedule=SimpleNamespace(items=[SimpleNamespace(mode="E")]),
        records=[SimpleNamespace(lineage=SimpleNamespace(diagnosis_run_id=duplicate))],
    )
    public_cases = {"case": SimpleNamespace(validation_run_ids=["6" * 32, "7" * 32])}

    with pytest.raises(ValueError) as error:
        resolver._acceptance(
            public_cases,
            evaluation,
            evaluation,
            {"8" * 32},
        )

    assert getattr(error.value, "code", None) == "RELEASE_ACCEPTANCE_INVALID"


def _freeze_resolution(repository, evidence):
    from gpu_agent.benchmark.release import ReleaseEvidenceResolution

    return ReleaseEvidenceResolution(selection=_selection(repository), evidence=evidence)


def _freezer_stores(tmp_path):
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.store import RunStore

    public = RunStore(tmp_path / "public")
    evaluator = RunStore(tmp_path / "evaluator", visibility="evaluator")
    family = CorpusFamily.provision(
        tmp_path / "controller",
        public_store=public.root,
        evaluator_store=evaluator.root,
        repository=tmp_path / "repository",
    )
    return public, evaluator, family


def _freezer_roots():
    from gpu_agent.benchmark.release import ReleaseEvidenceRoots

    return ReleaseEvidenceRoots(
        development_evaluation_run_id="3" * 32,
        holdout_evaluation_run_id="4" * 32,
        private_binding_run_id="5" * 32,
        release_test_run_id="a" * 32,
    )


def test_freezer_gate_precedes_private_atomic_publication(
    tmp_path, repository, evidence, monkeypatch
):
    from gpu_agent.benchmark.release import ReleaseEvidenceFreezer, ReleaseEvidenceIndex

    public, evaluator, family = _freezer_stores(tmp_path)
    repository_path = tmp_path / "repository"
    repository_path.mkdir(exist_ok=True)
    output_parent = tmp_path / "output"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "selection.json"
    incomplete = ReleaseEvidenceIndex.model_validate(
        {**evidence.model_dump(), "public_case_count": 15}
    )
    freezer = ReleaseEvidenceFreezer()
    monkeypatch.setattr(
        freezer,
        "expected_development_commit",
        lambda *_args, **_kwargs: repository.commit,
        raising=False,
    )
    monkeypatch.setattr(
        freezer,
        "capture_repository",
        lambda *_args, **_kwargs: repository,
    )
    monkeypatch.setattr(
        freezer,
        "resolve",
        lambda *_args, **_kwargs: _freeze_resolution(repository, incomplete),
    )

    with pytest.raises(ValueError, match="CORPUS_COUNT_INSUFFICIENT"):
        freezer.freeze(_freezer_roots(), public, evaluator, family, repository_path, output)

    assert not output.exists()


def test_freezer_preserves_resolver_reason_code_without_publication(
    tmp_path, repository, monkeypatch
):
    from gpu_agent.benchmark.release import (
        ReleaseEvidenceFreezer,
        _ReleaseEvidenceError,
    )

    public, evaluator, family = _freezer_stores(tmp_path)
    repository_path = tmp_path / "repository"
    repository_path.mkdir(exist_ok=True)
    output_parent = tmp_path / "output"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "selection.json"
    freezer = ReleaseEvidenceFreezer()
    monkeypatch.setattr(
        freezer,
        "expected_development_commit",
        lambda *_args, **_kwargs: repository.commit,
    )
    monkeypatch.setattr(
        freezer,
        "capture_repository",
        lambda *_args, **_kwargs: repository,
    )

    def reject(*_args, **_kwargs):
        raise _ReleaseEvidenceError("PRIVATE_SCORING_INCOMPLETE")

    monkeypatch.setattr(freezer, "resolve", reject)

    with pytest.raises(ValueError, match="PRIVATE_SCORING_INCOMPLETE"):
        freezer.freeze(_freezer_roots(), public, evaluator, family, repository_path, output)

    assert not output.exists()


def test_freezer_requires_cutoff_before_publication(tmp_path, repository, evidence, monkeypatch):
    from gpu_agent.benchmark.release import ReleaseEvidenceFreezer, ReleaseEvidenceIndex

    public, evaluator, family = _freezer_stores(tmp_path)
    repository_path = tmp_path / "repository"
    repository_path.mkdir(exist_ok=True)
    output_parent = tmp_path / "output"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "selection.json"
    incomplete = ReleaseEvidenceIndex.model_validate(
        {**evidence.model_dump(), "corpus_cutoff": None}
    )
    freezer = ReleaseEvidenceFreezer()
    monkeypatch.setattr(
        freezer,
        "expected_development_commit",
        lambda *_args, **_kwargs: repository.commit,
    )
    monkeypatch.setattr(
        freezer,
        "capture_repository",
        lambda *_args, **_kwargs: repository,
    )
    monkeypatch.setattr(
        freezer,
        "resolve",
        lambda *_args, **_kwargs: _freeze_resolution(repository, incomplete),
    )

    with pytest.raises(ValueError, match="release evidence is incomplete"):
        freezer.freeze(_freezer_roots(), public, evaluator, family, repository_path, output)

    assert not output.exists()


def test_freezer_never_publishes_when_repository_changes(
    tmp_path, repository, evidence, monkeypatch
):
    from gpu_agent.benchmark.release import ReleaseEvidenceFreezer
    from gpu_agent.contracts import RepositorySnapshot

    public, evaluator, family = _freezer_stores(tmp_path)
    repository_path = tmp_path / "repository"
    repository_path.mkdir(exist_ok=True)
    output_parent = tmp_path / "output"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "selection.json"
    changed = RepositorySnapshot(
        commit=repository.commit,
        tracked_tree_hash="0" * 64,
        clean=True,
    )
    snapshots = iter((repository, changed))
    freezer = ReleaseEvidenceFreezer()
    monkeypatch.setattr(
        freezer,
        "expected_development_commit",
        lambda *_args, **_kwargs: repository.commit,
        raising=False,
    )
    monkeypatch.setattr(
        freezer,
        "capture_repository",
        lambda *_args, **_kwargs: next(snapshots),
    )
    monkeypatch.setattr(
        freezer,
        "resolve",
        lambda *_args, **_kwargs: _freeze_resolution(repository, evidence),
    )

    with pytest.raises(ValueError, match="release repository changed"):
        freezer.freeze(_freezer_roots(), public, evaluator, family, repository_path, output)

    assert not output.exists()


def test_freezer_publishes_canonical_private_selection_read_only(
    tmp_path, repository, evidence, monkeypatch
):
    from gpu_agent.benchmark.release import FrozenReleaseSelection, ReleaseEvidenceFreezer

    public, evaluator, family = _freezer_stores(tmp_path)
    repository_path = tmp_path / "repository"
    repository_path.mkdir(exist_ok=True)
    output_parent = tmp_path / "output"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "selection.json"
    resolution = _freeze_resolution(repository, evidence)
    freezer = ReleaseEvidenceFreezer()
    monkeypatch.setattr(
        freezer,
        "expected_development_commit",
        lambda *_args, **_kwargs: repository.commit,
        raising=False,
    )
    monkeypatch.setattr(
        freezer,
        "capture_repository",
        lambda *_args, **_kwargs: repository,
    )
    observed = []

    def resolve(*args):
        observed.append(args)
        return resolution

    monkeypatch.setattr(freezer, "resolve", resolve)
    before = (_tree_bytes(public.root), _tree_bytes(evaluator.root), _tree_bytes(family.root))

    frozen = freezer.freeze(_freezer_roots(), public, evaluator, family, repository_path, output)

    assert isinstance(frozen, FrozenReleaseSelection)
    assert observed == [(_freezer_roots(), public, evaluator, family, repository)]
    expected = resolution.selection.model_dump_json(indent=2).encode() + b"\n"
    assert output.read_bytes() == expected
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert frozen.output == output
    assert frozen.selection == resolution.selection
    assert frozen.selection_sha256 == hashlib.sha256(expected).hexdigest()
    assert frozen.corpus_cutoff == evidence.corpus_cutoff
    assert frozen.public_case_count == 16
    assert frozen.private_case_count == 8
    assert frozen.acceptance_run_count == sum(
        len(run_ids) for run_ids in resolution.selection.acceptance_run_ids.values()
    )
    assert (
        _tree_bytes(public.root),
        _tree_bytes(evaluator.root),
        _tree_bytes(family.root),
    ) == before


def _tree_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }
