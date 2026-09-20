"""Offline tests: GPU subprocesses are simulated; no output is release evidence."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import test_corpus_registration as corpus_test_support


@pytest.fixture
def native_case(tmp_path, monkeypatch):
    return corpus_test_support.native_case.__wrapped__(tmp_path, monkeypatch)


ROOT = Path(__file__).resolve().parents[2]


def test_seed_batch_module_exists():
    assert importlib.util.find_spec("gpu_agent.benchmark.batch") is not None


@pytest.fixture
def seed_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for directory in ("benchmarks", "containers"):
        shutil.copytree(ROOT / directory, root / directory)
    for args in (
        ["init", "-q"],
        ["config", "user.name", "Test"],
        ["config", "user.email", "test@localhost"],
        ["add", "."],
        ["commit", "-qm", "fixture"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    return root


def test_static_preflight_covers_all_four_seed_hashes_without_creating_runs(seed_repo, tmp_path):
    from gpu_agent.benchmark.batch import prepare_seed_batch

    data_root = tmp_path / "data"
    prepared = prepare_seed_batch(seed_repo, data_root)
    assert [seed.spec.case_id for seed in prepared.seeds] == [
        "case_0001",
        "case_0002",
        "case_0003",
        "case_0004",
    ]
    assert [seed.spec.sanitizer_repetitions for seed in prepared.seeds] == [1, 5, 1, 1]
    assert prepared.report.gpu_executed is False
    assert not data_root.exists()
    for seed in prepared.seeds:
        assert hashlib.sha256(seed.input_bytes).hexdigest() == seed.spec.input_set_hash
        assert len(seed.clean_plan.source_manifest) == 4
        assert len(seed.mutant_plan.source_manifest) == 4


def test_preflight_selects_subset_in_recipe_order(seed_repo, tmp_path):
    from gpu_agent.benchmark.batch import prepare_seed_batch

    prepared = prepare_seed_batch(seed_repo, tmp_path / "data", ("case_0004", "case_0002"))
    assert [seed.spec.case_id for seed in prepared.seeds] == ["case_0002", "case_0004"]


@pytest.mark.parametrize("selected", [("case_9999",), ("case_0001", "case_0001")])
def test_preflight_rejects_unknown_or_duplicate_case(seed_repo, tmp_path, selected):
    from gpu_agent.benchmark.batch import BatchInputError, prepare_seed_batch

    with pytest.raises(BatchInputError):
        prepare_seed_batch(seed_repo, tmp_path / "data", selected)
    assert not (tmp_path / "data").exists()


def test_preflight_rejects_dirty_repository(seed_repo, tmp_path):
    from gpu_agent.benchmark.batch import BatchInputError, prepare_seed_batch

    (seed_repo / "untracked.txt").write_text("change")
    with pytest.raises(BatchInputError, match="REPOSITORY_NOT_READY"):
        prepare_seed_batch(seed_repo, tmp_path / "data")


@pytest.mark.parametrize("kind", ["inside", "ancestor", "symlink"])
def test_preflight_rejects_overlapping_or_symlinked_data_root(seed_repo, tmp_path, kind):
    from gpu_agent.benchmark.batch import BatchInputError, prepare_seed_batch

    if kind == "inside":
        data_root = seed_repo / "runs"
    elif kind == "ancestor":
        data_root = tmp_path
    else:
        data_root = tmp_path / "linked"
        data_root.symlink_to(tmp_path / "outside")
    with pytest.raises(BatchInputError):
        prepare_seed_batch(seed_repo, data_root)


def test_preflight_rejects_changed_kernel_even_after_git_commit(seed_repo, tmp_path):
    from gpu_agent.benchmark.batch import BatchInputError, prepare_seed_batch

    source = seed_repo / "benchmarks/public/case_0001/public_input/kernel.cu"
    source.write_text(source.read_text() + "\n// changed\n")
    subprocess.run(["git", "add", "."], cwd=seed_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "changed"], cwd=seed_repo, check=True)
    with pytest.raises(BatchInputError, match="SOURCE_IDENTITY_MISMATCH"):
        prepare_seed_batch(seed_repo, tmp_path / "data")


def _prepared(native):
    from gpu_agent.benchmark.batch_models import PreparedSeed
    from gpu_agent.benchmark.models import CaseExecutionPlan

    _, controller, _, input_bytes = native
    spec = next(iter(controller.specs.values()))
    root = controller.backend.repo_root

    def plan(role):
        names = [
            f"{role}/kernel.cu",
            *[f"harness/{name}" for name in ("vector_io.cpp", "vector_api.h", "json.hpp")],
        ]
        return CaseExecutionPlan(
            case_id=spec.case_id,
            template_id=spec.template_id,
            mutation_id="clean" if role == "clean" else spec.mutation_id,
            role=role,
            split="public",
            source_manifest={
                name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
            },
            target_tool=spec.target_tool,
            expected_finding=spec.expected_finding,
            sanitizer_repetitions=spec.sanitizer_repetitions,
            case_registry_hash=controller.registry_hash,
            case_spec_hash=controller.spec_hash(spec),
            mutation_provenance_hash=spec.mutation_provenance_hash,
        )

    return PreparedSeed(spec, plan("clean"), plan("mutant"), input_bytes)


def _runner(native):
    from gpu_agent.benchmark.batch import SeedBatchRunner
    from gpu_agent.benchmark.builder import BenchmarkBuilder

    store, controller, _, _ = native
    return SeedBatchRunner(controller, BenchmarkBuilder(store))


def test_batch_persists_real_controller_children_and_registers_only_through_gate(native_case):
    store, controller, _, _ = native_case
    result = _runner(native_case).run((_prepared(native_case),), register=True)
    case = result.cases[0]
    assert result.status == "COMPLETED"
    assert case.status == "REGISTERED"
    assert case.clean.run_id != case.mutant.run_id
    assert case.clean.oracle_passed is True
    assert case.mutant.target_detections == [True]
    assert store.load(case.clean.run_id).parent_run_id == result.batch_run_id
    assert len(controller.family.ledger.committed_through()) == 1
    summary = next(
        ref
        for ref in store.load(result.batch_run_id).artifact_refs
        if ref.name == "batch/summary.json"
    )
    assert json.loads(store.read(summary))["cases"][0]["status"] == "REGISTERED"


def test_validation_without_register_does_not_change_corpus_ledger(native_case):
    _, controller, _, _ = native_case
    result = _runner(native_case).run((_prepared(native_case),), register=False)
    assert result.cases[0].status == "VALIDATED"
    assert controller.family.ledger.committed_through() == []


def test_failed_clean_build_keeps_run_id_and_skips_mutant(native_case):
    from gpu_agent.execution.process import ProcessCapture

    store, controller, _, _ = native_case
    controller.backend._container = lambda *a, **kw: (
        ProcessCapture(1, b"", b"compiler error", False),
        b"",
        b"",
    )
    result = _runner(native_case).run((_prepared(native_case),), register=True)
    case = result.cases[0]
    assert case.status == "FAILED"
    assert case.reason_code == "BUILD_FAILED"
    assert case.failure_stage == "BUILD"
    assert case.clean.run_id is not None
    assert store.load(case.clean.run_id).status == "FAILED"
    assert case.mutant is None
    assert controller.family.ledger.committed_through() == []


def test_clean_oracle_failure_is_not_registered(native_case):
    from gpu_agent.execution.process import ProcessCapture

    _, controller, _, _ = native_case
    original = controller.backend._container

    def wrong_output(path, operation, timeout, **kwargs):
        result = original(path, operation, timeout, **kwargs)
        if operation == "run":
            output = b'{"dtype":"float32","shape":[2],"values":[0.0,0.0]}'
            return ProcessCapture(0, output, b"", False), b"", b""
        return result

    controller.backend._container = wrong_output
    result = _runner(native_case).run((_prepared(native_case),), register=True)
    assert result.cases[0].reason_code == "CLEAN_ORACLE_FAILED"
    assert result.cases[0].mutant is None
    assert controller.family.ledger.committed_through() == []


def test_missing_target_finding_is_failure_even_when_program_is_clean(native_case):
    from gpu_agent.execution.process import ProcessCapture

    _, controller, _, _ = native_case
    original = controller.backend._container

    def no_finding(path, operation, timeout, **kwargs):
        if operation == "memcheck":
            output = b'{"dtype":"float32","shape":[2],"values":[3.0,3.0]}'
            return (
                ProcessCapture(0, output, b"", False),
                b"",
                b"========= ERROR SUMMARY: 0 errors\n",
            )
        return original(path, operation, timeout, **kwargs)

    controller.backend._container = no_finding
    result = _runner(native_case).run((_prepared(native_case),), register=True)
    assert result.cases[0].reason_code == "TARGET_FINDING_MISSING"
    assert controller.family.ledger.committed_through() == []


def test_duplicate_registration_is_reported_and_never_overwrites_previous_entry(native_case):
    _, controller, _, _ = native_case
    runner = _runner(native_case)
    seed = _prepared(native_case)
    assert runner.run((seed,), register=True).cases[0].status == "REGISTERED"
    second = runner.run((seed,), register=True)
    assert second.cases[0].status == "FAILED"
    assert second.cases[0].reason_code == "REGISTRATION_REJECTED"
    assert len(controller.family.ledger.committed_through()) == 1


def test_ctrl_c_keeps_failed_child_and_terminal_batch(native_case):
    store, controller, _, _ = native_case

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    controller.backend.build = interrupt
    result = _runner(native_case).run((_prepared(native_case),))
    assert result.status == "CANCELLED"
    assert result.cases[0].reason_code == "USER_CANCELLED"
    assert result.cases[0].clean.run_id is not None
    assert store.load(result.batch_run_id).status == "CANCELLED"


def test_export_contains_public_child_logs_but_never_controller_or_evaluator_files(
    native_case, tmp_path
):
    from gpu_agent.benchmark.batch_report import export_batch

    store, controller, _, _ = native_case
    result = _runner(native_case).run((_prepared(native_case),), register=True)
    (controller.family.root / "do-not-export.key").write_text("controller-secret-canary")
    output = tmp_path / "results.zip"
    export_batch(store, result.batch_run_id, output)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "summary.json" in names
        assert "report.md" in names
        assert any("memcheck.log" in name for name in names)
        assert not any("controller" in name or "evaluator" in name for name in names)
        assert not any(b"controller-secret-canary" in archive.read(name) for name in names)
    with pytest.raises(FileExistsError):
        export_batch(store, result.batch_run_id, output)


def test_cli_help_and_preflight_work_without_loading_model_provider(seed_repo, tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    command = (
        "from typer.testing import CliRunner; from gpu_agent.cli import app; "
        "import sys; r=CliRunner().invoke(app, ['benchmark','run-seeds',"
        f"'--repository',{str(seed_repo)!r},'--data-root',{str(tmp_path / 'data')!r},"
        "'--preflight-only']); print(r.stdout); "
        "assert r.exit_code == 0, (r.stdout, r.exception); "
        "assert 'gpu_agent.agent.provider' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_unavailable_backend_persists_summary_without_attempting_a_case(native_case):
    from gpu_agent.execution.isolated import Availability

    store, controller, _, _ = native_case
    result = _runner(native_case).run(
        (_prepared(native_case),),
        register=True,
        availability=lambda: Availability(ready=False, reason="Docker is unavailable"),
    )
    assert result.status == "FAILED"
    assert result.stopped_reason == "BACKEND_UNAVAILABLE"
    assert result.cases[0].status == "NOT_RUN"
    assert store.children(result.batch_run_id) == []
    assert controller.family.ledger.committed_through() == []


def test_production_wiring_requires_preconfigured_family(seed_repo, tmp_path, monkeypatch):
    from gpu_agent.benchmark import batch

    monkeypatch.delenv("GPU_AGENT_CORPUS_FAMILY_ROOT", raising=False)
    monkeypatch.setattr(batch, "PACKAGE_REPOSITORY", seed_repo)
    prepared = batch.prepare_seed_batch(seed_repo, tmp_path / "data", ("case_0001",))
    with pytest.raises(batch.BatchInputError, match="FAMILY_CONFIG_REQUIRED"):
        batch.run_public_seeds(prepared, register=True)
    assert not (tmp_path / "data").exists()


def test_root_lock_rejects_concurrent_batch(tmp_path):
    from gpu_agent.benchmark.batch import BatchInputError, _data_lock

    (tmp_path / "data").mkdir(mode=0o700)
    with _data_lock(tmp_path / "data"):
        with pytest.raises(BatchInputError, match="BATCH_BUSY"):
            with _data_lock(tmp_path / "data"):
                pytest.fail("concurrent batch entered")


def test_timeout_is_not_a_confirmed_mutation(native_case):
    from gpu_agent.execution.process import ProcessCapture

    _, controller, _, _ = native_case
    original = controller.backend._container

    def timeout(path, operation, limit, **kwargs):
        if operation == "run":
            return ProcessCapture(-9, b"", b"", True), b"", b""
        return original(path, operation, limit, **kwargs)

    controller.backend._container = timeout
    result = _runner(native_case).run((_prepared(native_case),), register=True)
    assert result.cases[0].reason_code == "RUNTIME_TIMEOUT"
    assert result.cases[0].mutant is None
    assert controller.family.ledger.committed_through() == []


def test_sanitizer_tool_error_is_not_a_clean_result(native_case):
    from gpu_agent.execution.process import ProcessCapture

    _, controller, _, _ = native_case
    original = controller.backend._container

    def error(path, operation, timeout, **kwargs):
        if operation == "memcheck":
            return (
                ProcessCapture(None, b"", b"", False, tool_error="CONTAINER_UNAVAILABLE"),
                b"",
                b"",
            )
        return original(path, operation, timeout, **kwargs)

    controller.backend._container = error
    result = _runner(native_case).run((_prepared(native_case),), register=True)
    assert result.cases[0].reason_code == "SANITIZER_TOOL_ERROR"
    assert controller.family.ledger.committed_through() == []


def test_cleanup_failure_is_visible_in_batch_report(native_case):
    _, controller, _, _ = native_case
    original = controller.backend.cleanup

    def failed_cleanup(handle):
        result = original(handle)
        return result.model_copy(update={"removed": False})

    controller.backend.cleanup = failed_cleanup
    result = _runner(native_case).run((_prepared(native_case),), register=True)
    assert result.cases[0].reason_code == "WORKSPACE_CLEANUP_FAILED"
    assert result.cases[0].clean.run_id is not None
    assert controller.family.ledger.committed_through() == []


def test_export_rejects_modified_artifact_and_removes_partial_archive(native_case, tmp_path):
    from gpu_agent.benchmark.batch_report import export_batch

    store, _, _, _ = native_case
    result = _runner(native_case).run((_prepared(native_case),))
    child = store.load(result.cases[0].clean.run_id)
    ref = next(ref for ref in child.artifact_refs if ref.name.startswith("sources/"))
    path = store.root / ref.relative_path
    path.chmod(0o600)
    path.write_bytes(b"tampered")
    output = tmp_path / "broken.zip"
    with pytest.raises(ValueError):
        export_batch(store, result.batch_run_id, output)
    assert not output.exists()


def test_batch_report_remains_readable_after_cpu_restart(native_case):
    from gpu_agent.benchmark.batch_report import load_batch_summary
    from gpu_agent.store import RunStore

    store, _, _, _ = native_case
    result = _runner(native_case).run((_prepared(native_case),))
    reopened = RunStore(store.root)
    assert load_batch_summary(reopened, result.batch_run_id) == result


@pytest.mark.parametrize("failure,expected", [(False, 0), (True, 1)])
def test_cli_exit_codes_reflect_case_results(seed_repo, tmp_path, monkeypatch, failure, expected):
    from typer.testing import CliRunner

    from gpu_agent.benchmark import batch
    from gpu_agent.benchmark.batch_models import BatchSummary, SeedResult
    from gpu_agent.cli import app

    summary = BatchSummary(
        batch_run_id="a" * 32,
        status="COMPLETED",
        cases=[
            SeedResult(
                case_id="case_0001",
                target_tool="memcheck",
                repetitions=1,
                status="FAILED" if failure else "VALIDATED",
            )
        ],
    )
    monkeypatch.setattr(batch, "run_public_seeds", lambda *a, **kw: summary)
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "run-seeds",
            "--repository",
            str(seed_repo),
            "--data-root",
            str(tmp_path / "data"),
            "--case",
            "case_0001",
        ],
    )
    assert result.exit_code == expected, (result.stdout, result.exception)


def test_batch_render_has_a_contiguous_markdown_table(native_case):
    from gpu_agent.benchmark.batch_report import render_batch

    summary = _runner(native_case).run((_prepared(native_case),))
    # A second view of the same result is sufficient to check Markdown layout only.
    summary.cases.append(summary.cases[0].model_copy(update={"case_id": "case_0101"}))
    text = render_batch(summary)
    assert " |\n\n| case_" not in text


def test_public_case_preflight_cannot_select_private_specs(seed_repo, tmp_path):
    from gpu_agent.benchmark.batch import BatchInputError, prepare_seed_batch

    path = seed_repo / "benchmarks/corpus-registry.json"
    registry = json.loads(path.read_text())
    registry["cases"][0]["split"] = "private"
    path.write_text(json.dumps(registry))
    subprocess.run(["git", "add", "."], cwd=seed_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "private spec"], cwd=seed_repo, check=True)
    with pytest.raises(BatchInputError, match="PUBLIC_SEED_REQUIRED"):
        prepare_seed_batch(seed_repo, tmp_path / "data", ("case_0001",))


def test_all_fixed_repetitions_are_retained(native_case):
    from gpu_agent.benchmark.models import AuthoritativeCaseRegistry
    from gpu_agent.benchmark.validation import CaseValidationController

    store, original, execute, input_bytes = native_case
    spec = next(iter(original.specs.values())).model_copy(update={"sanitizer_repetitions": 5})
    raw = AuthoritativeCaseRegistry(cases=[spec]).model_dump_json().encode()
    binding = original.binding.model_copy(
        update={"case_registry_hash": hashlib.sha256(raw).hexdigest()}
    )
    controller = CaseValidationController._for_test(store, original.backend, binding, raw)
    native = (store, controller, execute, input_bytes)
    result = _runner(native).run((_prepared(native),), register=True)
    assert result.cases[0].status == "REGISTERED"
    assert result.cases[0].clean.sanitizer_outcomes == ["CLEAN"] * 5
    assert result.cases[0].mutant.target_detections == [True] * 5
    assert len(result.cases[0].mutant.instrumented_oracles) == 5


def test_failed_case_does_not_prevent_next_case(native_case):
    from dataclasses import replace

    from gpu_agent.benchmark.batch_models import PreparedSeed
    from gpu_agent.benchmark.models import AuthoritativeCaseRegistry
    from gpu_agent.benchmark.validation import CaseValidationController
    from gpu_agent.execution.process import ProcessCapture

    store, original, execute, input_bytes = native_case
    spec1 = next(iter(original.specs.values()))
    spec2 = spec1.model_copy(update={"case_id": "case_0101", "template_id": "vector-add-second"})
    raw = AuthoritativeCaseRegistry(cases=[spec1, spec2]).model_dump_json().encode()
    binding = original.binding.model_copy(
        update={"case_registry_hash": hashlib.sha256(raw).hexdigest()}
    )
    controller = CaseValidationController._for_test(store, original.backend, binding, raw)
    native = (store, controller, execute, input_bytes)
    seed1 = _prepared(native)
    seed2 = replace(
        seed1,
        spec=spec2,
        clean_plan=seed1.clean_plan.model_copy(
            update={
                "case_id": spec2.case_id,
                "template_id": spec2.template_id,
                "case_spec_hash": controller.spec_hash(spec2),
            }
        ),
        mutant_plan=seed1.mutant_plan.model_copy(
            update={
                "case_id": spec2.case_id,
                "template_id": spec2.template_id,
                "case_spec_hash": controller.spec_hash(spec2),
            }
        ),
    )
    assert isinstance(seed2, PreparedSeed)
    old_container = controller.backend._container
    first = True

    def fail_once(path, operation, timeout, **kwargs):
        nonlocal first
        if operation == "build" and first:
            first = False
            return ProcessCapture(1, b"", b"synthetic compile failure", False), b"", b""
        return old_container(path, operation, timeout, **kwargs)

    controller.backend._container = fail_once
    result = _runner(native).run((seed1, seed2), register=True)
    assert result.status == "COMPLETED"
    assert [case.status for case in result.cases] == ["FAILED", "REGISTERED"]
    assert not result.all_passed
    assert len(controller.family.ledger.committed_through()) == 1


def test_family_environment_conflict_is_not_silently_overridden(seed_repo, tmp_path, monkeypatch):
    from gpu_agent.benchmark.batch import BatchInputError, prepare_seed_batch

    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(tmp_path / "elsewhere"))
    with pytest.raises(BatchInputError, match="FAMILY_CONFIG_CONFLICT"):
        prepare_seed_batch(seed_repo, tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_export_rejects_a_non_batch_run(native_case, tmp_path):
    from gpu_agent.benchmark.batch_report import export_batch

    store, _, execute, _ = native_case
    run_id = execute("clean")
    with pytest.raises(ValueError, match="not a seed batch"):
        export_batch(store, run_id, tmp_path / "bad.zip")
