"""Controller commands fail closed before provider construction and use durable scheduling."""

import hashlib
import stat

import pytest
from schedule_authority_support import schedule_client_for_test
from typer.testing import CliRunner


def _private_json(path):
    path.write_bytes(b"{}")
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("cap", [[], ["--max-cost-usd", "10"], ["--max-unit-cost-usd", "1"]])
def test_missing_caps_exit_before_service_construction(monkeypatch, cap):
    from gpu_agent.cli import app
    from gpu_agent.service import ApplicationService

    constructed = []

    def forbidden():
        constructed.append(True)
        raise AssertionError("provider construction reached")

    monkeypatch.setattr(ApplicationService, "configured", forbidden)
    result = CliRunner().invoke(
        app,
        ["benchmark", "evaluate", "--mode", "A", "--split", "development", "--repeats", "3", *cap],
    )
    assert result.exit_code == 2 and "COST_CAP_REQUIRED" in result.output
    assert not constructed


def test_benchmark_help_exposes_controller_commands():
    from gpu_agent.cli import app

    result = CliRunner().invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0 and "validate" in result.output and "evaluate" in result.output


@pytest.mark.parametrize("visibility", ["public", "evaluator"])
@pytest.mark.parametrize(
    "command,inside_option",
    [
        ("derive-manifest", "selection"),
        ("check", "selection"),
        ("check", "manifest"),
    ],
)
def test_release_commands_reject_artifacts_inside_configured_runstores_before_reading(
    tmp_path, monkeypatch, visibility, command, inside_option
):
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.cli import app

    repository = tmp_path / "repository"
    repository.mkdir()
    public = tmp_path / "public"
    evaluator = tmp_path / "evaluator"
    family = CorpusFamily.provision(
        tmp_path / "controller",
        public_store=public,
        evaluator_store=evaluator,
        repository=repository,
    )
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    forbidden = public if visibility == "public" else evaluator
    inside = forbidden / "PRIVATE_CONTROLLER_BYTES.json"
    inside.write_bytes(b"PRIVATE_CONTROLLER_BYTES")
    inside.chmod(0o600)
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    selection = inside if inside_option == "selection" else external / "selection.json"
    manifest = inside if inside_option == "manifest" else external / "manifest.json"
    if selection != inside:
        selection.write_bytes(b"{}")
        selection.chmod(0o600)
    if manifest != inside:
        manifest.write_bytes(b"{}")
        manifest.chmod(0o600)
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
    assert "PRIVATE_CONTROLLER_BYTES" not in result.output


@pytest.mark.parametrize(
    "command,reason_code",
    [
        ("derive-manifest", "RELEASE_EVIDENCE_INCOMPLETE"),
        ("check", "RELEASE_EVIDENCE_INVALID"),
    ],
)
def test_release_commands_sanitize_family_setup_oserrors(
    tmp_path, monkeypatch, command, reason_code
):
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.cli import app

    repository = tmp_path / "repository"
    repository.mkdir()
    family = CorpusFamily.provision(
        tmp_path / "PRIVATE_FAMILY_ROOT",
        public_store=tmp_path / "public",
        evaluator_store=tmp_path / "evaluator",
        repository=repository,
    )
    ledger = family.root / "ledger"
    for child in ledger.iterdir():
        child.unlink()
    ledger.rmdir()
    ledger.write_bytes(b"PRIVATE_FAMILY_BYTES")
    ledger.chmod(0o600)
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    selection = _private_json(external / "selection.json")
    manifest = _private_json(external / "manifest.json")
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
    assert reason_code in result.output
    assert str(family.root) not in result.output
    assert "PRIVATE_FAMILY" not in result.output


@pytest.mark.parametrize("visibility", ["public", "evaluator"])
def test_validate_refuses_unattested_serialized_claims(tmp_path, monkeypatch, visibility):
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.cli import app

    forged = tmp_path / "PRIVATE_SECRET.json"
    forged.write_text('{"oracle_passed":true,"run_ids":["forged"]}')
    corpus_root = tmp_path / "corpus"
    family = CorpusFamily.provision(
        tmp_path / "controller",
        public_store=corpus_root if visibility == "public" else tmp_path / "public",
        evaluator_store=corpus_root if visibility == "evaluator" else tmp_path / "evaluator",
        repository=tmp_path / "repository",
    )
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "validate",
            str(forged),
            str(forged),
            "--corpus-root",
            str(corpus_root),
            "--visibility",
            visibility,
        ],
    )
    assert result.exit_code == 2 and "CASE_EXECUTION_ATTESTATION_UNAVAILABLE" in result.output
    assert "PRIVATE_SECRET" not in result.output
    corpus = tmp_path / "corpus"
    assert corpus.is_dir()
    assert not [path for path in corpus.iterdir() if len(path.name) == 32]
    assert b"PRIVATE" not in (corpus / ".corpus-family.json").read_bytes()


@pytest.mark.parametrize("cap", ["0", "10"])
def test_production_evaluation_requires_attested_cost_before_construction(
    tmp_path, monkeypatch, cap
):
    from gpu_agent.benchmark.models import CaseManifest
    from gpu_agent.cli import app
    from gpu_agent.service import ApplicationService
    from gpu_agent.store import RunStore

    corpus = RunStore(tmp_path / "corpus")
    case = CaseManifest(
        id="case_0100",
        source_hash="1" * 64,
        harness_hash="2" * 64,
        mutation_id="delete-guard",
        template_id="vector-add",
        split="public",
        oracle_id="vector-add-cpu-v1",
        target_tool="memcheck",
        expected_finding="out of bounds",
        validation_run_ids=["a", "b"],
        toolchain_hash="3" * 64,
        input_set_hash="4" * 64,
    )
    run = corpus.create_run("benchmark_case")
    corpus.put(run.id, "case-manifest.json", case.model_dump_json().encode(), "public")
    corpus.transition(run.id, "RUNNING", "FINALIZING")
    corpus.transition(run.id, "COMPLETED", None)

    constructed = []

    def forbidden():
        constructed.append(True)
        raise AssertionError("configured service reached")

    monkeypatch.setattr(ApplicationService, "configured", forbidden)
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "evaluate",
            "--mode",
            "A",
            "--split",
            "development",
            "--repeats",
            "3",
            "--max-cost-usd",
            cap,
            "--max-unit-cost-usd",
            cap,
            "--corpus-root",
            str(corpus.root),
            "--case-root",
            str(tmp_path),
            "--commit",
            "a" * 40,
            "--toolchain-hash",
            "3" * 64,
            "--model-config-hash",
            "5" * 64,
        ],
    )
    assert result.exit_code == 2 and "COST_BOUND_UNAVAILABLE" in result.output
    assert not constructed
    assert "Hard maximum" not in result.output


def test_evaluate_uses_injected_executor_and_prints_reservation(
    tmp_path, monkeypatch, native_evaluation_executor
):
    from gpu_agent import cli
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.service import ApplicationService

    executor = native_evaluation_executor
    service = executor.service
    binding = service.binding
    assert binding is not None
    calls = []
    native_execute = executor.execute_scheduled

    class InjectedExecutor:
        def validate_scheduled_record(self, record, item, attempt):
            return executor.validate_scheduled_record(record, item, attempt)

        def execute_scheduled(self, run_id, ordinal):
            record = native_execute(run_id, ordinal)
            calls.append((record.case_id, record.template_id, record.mode, record.repeat))
            return record

    def forbidden():
        raise AssertionError("offline path cannot construct configured services")

    monkeypatch.setattr(ApplicationService, "configured", forbidden)
    injected = InjectedExecutor()
    executor.execute_scheduled = injected.execute_scheduled
    runner = EvaluationRunner(
        service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "benchmark",
            "evaluate",
            "--mode",
            "A",
            "--split",
            "development",
            "--repeats",
            "3",
            "--max-cost-usd",
            "0",
            "--max-unit-cost-usd",
            "0",
            "--corpus-root",
            str(executor.corpus.root),
            "--case-root",
            str(tmp_path / "sources"),
            "--commit",
            "a" * 40,
            "--toolchain-hash",
            "3" * 64,
            "--model-config-hash",
            "5" * 64,
        ],
        obj=runner,
    )
    assert result.exit_code == 0, result.output
    assert "1 case × 1 mode × 3 repeats = 3 units" in result.output
    assert "Cost reservation: $0.00" in result.output
    assert "Hard maximum" not in result.output
    assert calls == []


def _holdout_cli_family(tmp_path, monkeypatch):
    from schedule_authority_support import TestScheduleCommitClient

    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.contracts import RepositorySnapshot, RunBinding

    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    signer = TestScheduleCommitClient.create(tmp_path / "test-only-schedule-authority")
    family = CorpusFamily.provision_production(
        tmp_path / "controller",
        public_store=tmp_path / "public",
        evaluator_store=tmp_path / "evaluator",
        repository=repository,
        schedule_public_key=signer.public_key,
    )
    public = family.corpus_store("public")
    binding = RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="evaluation",
    )
    evaluation_run_id = "1" * 32
    public.create_run("evaluation", binding=binding, _run_id=evaluation_run_id)
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    return family, repository, evaluation_run_id, "2" * 32


def _tree_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


@pytest.mark.parametrize("location", ["missing", "relative", "repository", "public", "evaluator"])
def test_score_holdout_rejects_unsafe_labels_before_store_mutation(tmp_path, monkeypatch, location):
    from gpu_agent.cli import app

    family, repository, evaluation_run_id, mapping_run_id = _holdout_cli_family(
        tmp_path, monkeypatch
    )
    public = family.corpus_store("public")
    evaluator = family.corpus_store("evaluator")
    if location == "missing":
        labels = tmp_path / "missing.json"
    elif location == "relative":
        labels = "labels.json"
    else:
        root = {"repository": repository, "public": public.root, "evaluator": evaluator.root}[
            location
        ]
        labels = root / "PRIVATE-HOLDOUT-CANARY.json"
        labels.write_bytes(b"PRIVATE-HOLDOUT-CANARY")
        labels.chmod(0o600)
    before = (_tree_bytes(public.root), _tree_bytes(evaluator.root))
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "score-holdout",
            "--evaluation-run-id",
            evaluation_run_id,
            "--private-binding-run-id",
            mapping_run_id,
            "--labels",
            str(labels),
            "--repository",
            str(repository.absolute()),
        ],
    )
    assert result.exit_code == 2
    assert "HOLDOUT_LABEL_PACKAGE_INVALID" in result.output
    assert "PRIVATE-HOLDOUT-CANARY" not in result.output
    assert (_tree_bytes(public.root), _tree_bytes(evaluator.root)) == before


@pytest.mark.parametrize(
    "failure,code",
    [
        (
            ValueError("holdout label package is incomplete or invalid"),
            "HOLDOUT_LABEL_PACKAGE_INVALID",
        ),
        (
            ValueError("holdout evaluation schedule is invalid"),
            "HOLDOUT_SCORING_EVIDENCE_MISMATCH",
        ),
        (ValueError("holdout scoring package conflicts"), "HOLDOUT_SCORING_CONFLICT"),
        (OSError("PRIVATE-HOLDOUT-CANARY"), "HOLDOUT_SCORING_FAILED"),
    ],
)
def test_score_holdout_maps_stable_private_errors(tmp_path, monkeypatch, failure, code):
    from gpu_agent.benchmark.holdout_scoring import HoldoutScoringController
    from gpu_agent.cli import app

    _, repository, evaluation_run_id, mapping_run_id = _holdout_cli_family(tmp_path, monkeypatch)
    labels = tmp_path / "labels.json"
    labels.write_bytes(b"{}")
    labels.chmod(0o600)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(HoldoutScoringController, "score", fail)
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "score-holdout",
            "--evaluation-run-id",
            evaluation_run_id,
            "--private-binding-run-id",
            mapping_run_id,
            "--labels",
            str(labels),
            "--repository",
            str(repository),
        ],
    )
    assert result.exit_code == 2 and code in result.output
    assert "PRIVATE-HOLDOUT-CANARY" not in result.output


def test_score_holdout_publishes_exact_private_metrics_and_safe_stdout(tmp_path, monkeypatch):
    from gpu_agent.benchmark.holdout_scoring import (
        HoldoutScoringController,
        HoldoutScoringResult,
    )
    from gpu_agent.cli import app
    from gpu_agent.contracts import ExternalRunOrigin, RunStatus

    family, repository, evaluation_run_id, mapping_run_id = _holdout_cli_family(
        tmp_path, monkeypatch
    )
    canary = b"PRIVATE-HOLDOUT-CANARY-8eaa"
    labels = tmp_path / "labels.json"
    labels.write_bytes(b'{"private":"' + canary + b'"}')
    labels.chmod(0o600)
    output_parent = tmp_path / "output"
    output_parent.mkdir(mode=0o700)
    metrics_output = output_parent / "metrics.json"
    metrics = b'{"safe":true}'

    def score(self, requested_evaluation, requested_mapping, labels_path, requested_repository):
        assert (requested_evaluation, requested_mapping) == (evaluation_run_id, mapping_run_id)
        assert labels_path == labels and requested_repository == repository
        run_id = hashlib.sha256(
            f"holdout-scoring-v1:{evaluation_run_id}:{mapping_run_id}".encode()
        ).hexdigest()[:32]
        run = self.evaluator.create_run(
            "holdout_scoring",
            binding=self.binding,
            external_origin=ExternalRunOrigin(run_id=evaluation_run_id, visibility="public"),
            _run_id=run_id,
        )
        self.evaluator.transition(run.id, RunStatus.RUNNING, "FINALIZING")
        self.evaluator.put(run.id, "holdout-scoring/metrics.json", metrics, "evaluator")
        self.evaluator.transition(run.id, RunStatus.COMPLETED, None)
        return HoldoutScoringResult(
            scoring_run_id=run_id,
            evaluation_run_id=evaluation_run_id,
            private_binding_run_id=mapping_run_id,
            package_hash="3" * 64,
            input_binding_hash="4" * 64,
            bindings_hash="5" * 64,
            metrics_hash=hashlib.sha256(metrics).hexdigest(),
        )

    monkeypatch.setattr(HoldoutScoringController, "score", score)
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "score-holdout",
            "--evaluation-run-id",
            evaluation_run_id,
            "--private-binding-run-id",
            mapping_run_id,
            "--labels",
            str(labels),
            "--repository",
            str(repository),
            "--metrics-output",
            str(metrics_output),
        ],
    )
    assert result.exit_code == 0, result.output
    expected_run_id = hashlib.sha256(
        f"holdout-scoring-v1:{evaluation_run_id}:{mapping_run_id}".encode()
    ).hexdigest()[:32]
    assert result.stdout == (
        f"scoring_run_id {expected_run_id}\n"
        "scored 120/120\n"
        f"metrics_sha256 {hashlib.sha256(metrics).hexdigest()}\n"
    )
    assert metrics_output.read_bytes() == metrics
    assert stat.S_IMODE(metrics_output.stat().st_mode) == 0o600
    for root in (repository, family.corpus_store("public").root):
        assert canary not in b"".join(
            path.read_bytes() for path in root.rglob("*") if path.is_file()
        )
    assert canary not in result.stdout_bytes and canary not in result.stderr_bytes
