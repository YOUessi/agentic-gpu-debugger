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


@pytest.mark.parametrize("cap", [["--max-cost-usd", "10"], ["--max-unit-cost-usd", "1"]])
def test_dollar_limit_options_are_removed(monkeypatch, cap):
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
    assert result.exit_code == 2 and "No such option" in result.output
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


def test_production_evaluation_requires_attested_cost_before_construction(tmp_path, monkeypatch):
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


@pytest.mark.parametrize(
    "fault", ["wrong_public", "wrong_evaluator_parent", "symlink", "replaced_inode", "visibility"]
)
def test_production_evaluate_rejects_store_fault_before_side_effect(tmp_path, monkeypatch, fault):
    import json

    from schedule_authority_support import TestScheduleCommitClient

    from gpu_agent.agent.provider import OpenAIResponsesProvider
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.cli import app

    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    public = tmp_path / "public"
    public.mkdir(mode=0o700)
    evaluator_root = tmp_path / "evaluator-root"
    evaluator_root.mkdir(mode=0o700)
    (evaluator_root / "runs").mkdir(mode=0o700)
    signer = TestScheduleCommitClient.create(tmp_path / "signer")
    family = CorpusFamily.provision_production(
        tmp_path / "controller",
        public_store=public,
        evaluator_store=evaluator_root / "runs",
        repository=repository,
        schedule_public_key=signer.public_key,
    )
    run_root = public
    configured_evaluator = evaluator_root
    if fault == "wrong_public":
        run_root = tmp_path / "missing-public"
    elif fault == "wrong_evaluator_parent":
        configured_evaluator = tmp_path / "missing-evaluator"
    elif fault == "symlink":
        configured_evaluator = tmp_path / "evaluator-link"
        configured_evaluator.symlink_to(evaluator_root, target_is_directory=True)
    elif fault == "replaced_inode":
        displaced = tmp_path / "public-old"
        public.rename(displaced)
        public.mkdir(mode=0o700)
        (public / ".corpus-family.json").write_bytes(
            (displaced / ".corpus-family.json").read_bytes()
        )
    else:
        config_path = family.root / "family.json"
        config = json.loads(config_path.read_text())
        config["public_store_pin"]["visibility"] = "evaluator"
        config_path.write_text(json.dumps(config))
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    monkeypatch.setenv("GPU_AGENT_RUN_ROOT", str(run_root))
    monkeypatch.setenv("GPU_AGENT_EVALUATOR_ROOT", str(configured_evaluator))
    monkeypatch.setenv("GPU_AGENT_SCHEDULE_AUTHORITY_COMMAND", str(tmp_path / "authority"))
    monkeypatch.setenv("GPU_AGENT_PRICING_ATTESTATION", str(tmp_path / "pricing.json"))
    provider_calls = []
    monkeypatch.setattr(
        OpenAIResponsesProvider,
        "ensure_available",
        lambda _self: provider_calls.append(True),
    )
    before = _tree_bytes(tmp_path)

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
            "--corpus-root",
            str(public),
            "--case-root",
            str(tmp_path / "cases"),
            "--repository",
            str(repository),
            "--commit",
            "a" * 40,
            "--toolchain-hash",
            "b" * 64,
            "--model-config-hash",
            "c" * 64,
        ],
    )

    assert result.exit_code == 2
    assert "COST_BOUND_UNAVAILABLE" in result.output
    assert provider_calls == []
    assert _tree_bytes(tmp_path) == before


@pytest.mark.parametrize("split", ["development", "holdout"])
def test_configured_evaluation_runner_builds_exact_split_services(tmp_path, monkeypatch, split):
    import shutil
    import subprocess
    from datetime import UTC, datetime

    from schedule_authority_support import TestScheduleCommitClient

    from gpu_agent.agent.prompts import PROMPT_VERSION
    from gpu_agent.agent.provider import OpenAIResponsesProvider
    from gpu_agent.benchmark.evaluation import EvaluationProviderPolicy, PricingAttestation
    from gpu_agent.benchmark.holdout import HoldoutBatch, HoldoutController
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.models import CaseManifest
    from gpu_agent.cli import _configured_evaluation_runner
    from gpu_agent.environment import load_toolchain_lock
    from gpu_agent.execution.isolated import LOCK_PATH
    from gpu_agent.provenance import capture_repository_snapshot

    repository = tmp_path / "repository"
    (repository / "containers").mkdir(parents=True)
    for source_name in ("toolchain.lock.json", "runner.py", "Dockerfile"):
        shutil.copy2(LOCK_PATH.with_name(source_name), repository / "containers" / source_name)
    for arguments in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "fixture"],
    ):
        subprocess.run(["git", "-C", str(repository), *arguments], check=True)
    snapshot = capture_repository_snapshot(repository)
    # This fixture contains only toolchain files. Runtime binding is exercised with
    # real package trees in provenance tests, not this service-wiring test.
    monkeypatch.setattr("gpu_agent.service.runtime_code_fingerprint", lambda _: "9" * 64)
    toolchain_hash = load_toolchain_lock(LOCK_PATH).lock_hash
    public = tmp_path / "public"
    evaluator_root = tmp_path / "evaluator-root"
    evaluator = evaluator_root / "runs"
    public.mkdir(mode=0o700)
    evaluator.mkdir(parents=True, mode=0o700)
    evaluator_root.chmod(0o700)
    signer = TestScheduleCommitClient.create(tmp_path / "signer")
    family = CorpusFamily.provision_production(
        tmp_path / "controller",
        public_store=public,
        evaluator_store=evaluator,
        repository=repository,
        schedule_public_key=signer.public_key,
    )
    authority = tmp_path / "authority"
    authority.write_text("#!/bin/sh\nexit 1\n")
    authority.chmod(0o700)
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    monkeypatch.setenv("GPU_AGENT_RUN_ROOT", str(public))
    monkeypatch.setenv("GPU_AGENT_EVALUATOR_ROOT", str(evaluator_root))
    monkeypatch.setenv("GPU_AGENT_SCHEDULE_AUTHORITY_COMMAND", str(authority))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setenv("GPU_AGENT_STORE_FALSE_SUPPORTED", "1")
    preliminary = PricingAttestation.reviewed(
        provider="deepseek-responses",
        model="deepseek-chat",
        commit=snapshot.commit,
        model_config_hash="0" * 64,
        input_usd_per_million=1,
        output_usd_per_million=1,
        source_uri="https://example.invalid/pricing",
        reviewed_at=datetime(2026, 9, 21, tzinfo=UTC),
        source_content_hash="1" * 64,
    )
    policy = EvaluationProviderPolicy(
        provider="deepseek-responses",
        endpoint_host="api.deepseek.com",
        configured_model="deepseek-chat",
        allowed_response_models=["deepseek-chat"],
        prompt_version=PROMPT_VERSION,
        pricing_hash=preliminary.rate_card_hash,
    )
    pricing = preliminary.model_copy(update={"model_config_hash": policy.sha256})
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(pricing.model_dump_json())
    pricing_path.chmod(0o600)
    monkeypatch.setenv("GPU_AGENT_PRICING_ATTESTATION", str(pricing_path))
    source = b'extern "C" __global__ void kernel() {}\n'
    case_root = tmp_path / "cases"
    selected = case_root / "case_0100" / "public_input"
    selected.mkdir(parents=True)
    (selected / "kernel.cu").write_bytes(source)
    case = CaseManifest(
        id="case_0100",
        source_hash=hashlib.sha256(source).hexdigest(),
        harness_hash="2" * 64,
        mutation_id="delete-guard",
        template_id="vector-add",
        split="public" if split == "development" else "private",
        oracle_id="vector-add-cpu-v1",
        target_tool="memcheck",
        expected_finding="out of bounds",
        validation_run_ids=["a" * 32, "b" * 32],
        toolchain_hash=toolchain_hash,
        input_set_hash="4" * 64,
    )
    monkeypatch.setattr(
        "gpu_agent.benchmark.executor.registered_cases",
        lambda *_args, **_kwargs: {case.id: case},
    )
    batch = HoldoutBatch(
        public_run_id="1" * 32,
        evaluator_run_id="2" * 32,
        aliases=["3" * 64],
        public_alias_hash="4" * 64,
        corpus_cutoff=1,
    )
    monkeypatch.setattr(HoldoutController, "prepare", lambda _self: batch)
    provider_ensures = []
    monkeypatch.setattr(
        OpenAIResponsesProvider,
        "ensure_available",
        lambda _self: provider_ensures.append(True),
    )

    runner = _configured_evaluation_runner(
        repository=repository,
        case_root=case_root,
        corpus_root=public if split == "development" else evaluator,
        split=split,
        commit=snapshot.commit,
        toolchain_hash=toolchain_hash,
        model_config_hash=policy.sha256,
    )

    assert provider_ensures == [True]
    assert runner.store.identity == family.corpus_store("public").identity
    if split == "development":
        assert runner.executor.holdout_service is None
        assert runner.holdout_controller is None and runner.holdout_batch is None
    else:
        assert runner.executor.holdout_service is not None
        assert (
            runner.executor.holdout_service.store.identity
            == family.corpus_store("evaluator").identity
        )
        assert (
            runner.executor.holdout_service.evaluator_store.identity
            == family.corpus_store("evaluator").identity
        )
        assert runner.holdout_controller is not None and runner.holdout_batch == batch
        assert runner.holdout_controller._schedule_verifier is runner.executor._schedule_verifier


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
        max_cost_usd=1,
        max_unit_cost_usd=0.1,
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
    assert "Usage and cost are recorded only; no dollar ceiling." in result.output
    assert "Hard maximum" not in result.output
    assert calls == []


def _holdout_cli_family(tmp_path, monkeypatch):
    from schedule_authority_support import TestScheduleCommitClient

    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.contracts import RepositorySnapshot, RunBinding

    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    (tmp_path / "public").mkdir(mode=0o700)
    (tmp_path / "evaluator").mkdir(mode=0o700)
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


def _freeze_cli_args(repository, output):
    return [
        "release",
        "freeze-selection",
        "--development-evaluation-run-id",
        "1" * 32,
        "--holdout-evaluation-run-id",
        "2" * 32,
        "--private-binding-run-id",
        "3" * 32,
        "--release-test-run-id",
        "4" * 32,
        "--output",
        str(output),
        "--repository",
        str(repository),
    ]


@pytest.mark.parametrize(
    "failure,code",
    [
        (ValueError("release roots are invalid"), "RELEASE_ROOTS_INVALID"),
        (
            ValueError("release roots are invalid: PRIVATE_SCORING_INCOMPLETE"),
            "RELEASE_ROOTS_INVALID",
        ),
        (
            ValueError("release evidence is incomplete: CORPUS_COUNT_INSUFFICIENT"),
            "RELEASE_EVIDENCE_INCOMPLETE",
        ),
        (ValueError("release repository changed"), "RELEASE_REPOSITORY_CHANGED"),
        (
            ValueError("external artifact output is unsafe"),
            "RELEASE_SELECTION_OUTPUT_UNSAFE",
        ),
        (
            ValueError("external artifact output conflicts"),
            "RELEASE_SELECTION_OUTPUT_CONFLICT",
        ),
    ],
)
def test_freeze_selection_maps_stable_private_errors(tmp_path, monkeypatch, failure, code):
    from gpu_agent.benchmark.release import ReleaseEvidenceFreezer
    from gpu_agent.cli import app

    _, repository, _, _ = _holdout_cli_family(tmp_path, monkeypatch)
    output_parent = tmp_path / "output"
    output_parent.mkdir(mode=0o700)

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(ReleaseEvidenceFreezer, "freeze", fail)
    result = CliRunner().invoke(
        app,
        _freeze_cli_args(repository, output_parent / "selection.json"),
    )

    assert result.exit_code == 2
    assert code in result.output
    assert "CORPUS_COUNT_INSUFFICIENT" not in result.output


def test_freeze_selection_prints_only_safe_aggregates(tmp_path, monkeypatch):
    from gpu_agent.benchmark.release import (
        FrozenReleaseSelection,
        ReleaseEvidenceFreezer,
        ReleaseEvidenceSelection,
    )
    from gpu_agent.cli import app
    from gpu_agent.contracts import RepositorySnapshot

    _, repository, _, _ = _holdout_cli_family(tmp_path, monkeypatch)
    output_parent = tmp_path / "output"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "selection.json"
    snapshot = RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True)
    selection = ReleaseEvidenceSelection(
        repository=snapshot,
        public_case_run_ids=["5" * 32],
        private_case_run_ids=["6" * 32],
        development_evaluation_run_id="1" * 32,
        holdout_evaluation_run_id="2" * 32,
        private_binding_run_id="3" * 32,
        acceptance_run_ids={
            "four_tools": ["7" * 32],
            "isolation": ["4" * 32],
            "live_llm": ["8" * 32],
            "private_oracle": ["9" * 32],
        },
        release_test_run_id="4" * 32,
    )
    frozen = FrozenReleaseSelection(
        output=output,
        selection=selection,
        selection_sha256="c" * 64,
        corpus_cutoff=24,
        public_case_count=16,
        private_case_count=8,
        acceptance_run_count=4,
    )
    canaries = (
        "PRIVATE_ALIAS_CANARY",
        "PRIVATE_NONCE_CANARY",
        "PRIVATE_LABEL_CANARY",
        "PRIVATE_CASE_CANARY",
        "PRIVATE_TEMPLATE_CANARY",
        "/private/evaluator/artifact/path",
    )

    monkeypatch.setattr(ReleaseEvidenceFreezer, "freeze", lambda *_args, **_kwargs: frozen)
    result = CliRunner().invoke(app, _freeze_cli_args(repository, output))

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        f"selection_path {output}\n"
        f"selection_sha256 {'c' * 64}\n"
        "corpus_cutoff 24\n"
        "public_case_count 16\n"
        "private_case_count 8\n"
        "acceptance_run_count 4\n"
    )
    assert all(canary not in result.stdout + result.stderr for canary in canaries)


@pytest.mark.parametrize("location", ["relative", "repository", "public", "evaluator"])
def test_freeze_selection_rejects_unsafe_output_before_freezing(tmp_path, monkeypatch, location):
    from gpu_agent.benchmark.release import ReleaseEvidenceFreezer
    from gpu_agent.cli import app

    family, repository, _, _ = _holdout_cli_family(tmp_path, monkeypatch)
    if location == "relative":
        output = "selection.json"
    else:
        root = {
            "repository": repository,
            "public": family.corpus_store("public").root,
            "evaluator": family.corpus_store("evaluator").root,
        }[location]
        output = root / "selection.json"
    calls = []
    monkeypatch.setattr(ReleaseEvidenceFreezer, "freeze", lambda *_args: calls.append(True))

    result = CliRunner().invoke(app, _freeze_cli_args(repository, output))

    assert result.exit_code == 2
    assert "RELEASE_SELECTION_OUTPUT_UNSAFE" in result.output
    assert not calls


def test_freeze_selection_round_trips_through_derive_and_check_read_only(tmp_path, monkeypatch):
    from gpu_agent import provenance
    from gpu_agent.benchmark.release import (
        ReleaseEvidenceFreezer,
        ReleaseEvidenceIndex,
        ReleaseEvidenceResolution,
        ReleaseEvidenceSelection,
        ReleaseManifest,
        TestCounts,
    )
    from gpu_agent.cli import _derive_release_evidence, app
    from gpu_agent.contracts import RepositorySnapshot

    family, repository, _, _ = _holdout_cli_family(tmp_path, monkeypatch)
    public = family.corpus_store("public")
    evaluator = family.corpus_store("evaluator")
    evaluator_canary = evaluator.root / ".PRIVATE-EVALUATOR-PATH-CANARY"
    evaluator_canary.write_bytes(b"PRIVATE-LABEL-NONCE-ALIAS-CANARY")
    snapshot = RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True)
    selection = ReleaseEvidenceSelection(
        repository=snapshot,
        public_case_run_ids=["5" * 32],
        private_case_run_ids=["6" * 32],
        development_evaluation_run_id="1" * 32,
        holdout_evaluation_run_id="2" * 32,
        private_binding_run_id="3" * 32,
        acceptance_run_ids={
            "four_tools": ["7" * 32],
            "isolation": ["4" * 32],
            "live_llm": ["8" * 32],
            "private_oracle": ["9" * 32],
        },
        release_test_run_id="4" * 32,
    )
    evidence_ids = {
        **selection.acceptance_run_ids,
        "five_mode_evaluation": ["1" * 32, "2" * 32],
        "public_corpus": selection.public_case_run_ids,
        "private_corpus": selection.private_case_run_ids,
        "private_scoring": ["3" * 32, "a" * 32],
        "release_tests": ["4" * 32],
    }
    evidence = ReleaseEvidenceIndex(
        repository=snapshot,
        toolchain_hash="c" * 64,
        corpus_hash="d" * 64,
        model_config_hash="e" * 64,
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
        evidence_run_ids=evidence_ids,
    )
    monkeypatch.setattr(
        ReleaseEvidenceFreezer,
        "expected_development_commit",
        staticmethod(lambda *_args, **_kwargs: snapshot.commit),
    )
    monkeypatch.setattr(
        ReleaseEvidenceFreezer,
        "capture_repository",
        staticmethod(lambda *_args, **_kwargs: snapshot),
    )
    monkeypatch.setattr(
        ReleaseEvidenceFreezer,
        "resolve",
        staticmethod(
            lambda *_args, **_kwargs: ReleaseEvidenceResolution(
                selection=selection,
                evidence=evidence,
            )
        ),
    )
    monkeypatch.setattr(
        provenance,
        "capture_repository_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    observed = []

    def derive(cls, parsed, *args):
        observed.append((parsed, args))
        return evidence

    monkeypatch.setattr(ReleaseEvidenceIndex, "derive", classmethod(derive))
    output_parent = tmp_path / "output"
    output_parent.mkdir(mode=0o700)
    selection_path = output_parent / "selection.json"
    before = (_tree_bytes(public.root), _tree_bytes(evaluator.root), _tree_bytes(family.root))

    frozen = CliRunner().invoke(app, _freeze_cli_args(repository, selection_path))
    derived = _derive_release_evidence(
        selection_path,
        repository,
        family=family,
        forbidden_roots=(public.root, evaluator.root),
    )
    manifest_path = output_parent / "manifest.json"
    manifest_path.write_bytes(ReleaseManifest.from_evidence(derived).model_dump_json().encode())
    manifest_path.chmod(0o600)
    checked = CliRunner().invoke(
        app,
        [
            "release",
            "check",
            "--manifest",
            str(manifest_path),
            "--selection",
            str(selection_path),
            "--repository",
            str(repository),
        ],
    )

    assert frozen.exit_code == 0, frozen.output
    assert checked.exit_code == 0, checked.output
    assert '"passed": true' in checked.stdout
    assert derived == evidence
    assert [item[0] for item in observed] == [selection, selection]
    assert b"PRIVATE-LABEL-NONCE-ALIAS-CANARY" not in selection_path.read_bytes()
    assert "PRIVATE" not in frozen.stdout + frozen.stderr
    assert (
        _tree_bytes(public.root),
        _tree_bytes(evaluator.root),
        _tree_bytes(family.root),
    ) == before
