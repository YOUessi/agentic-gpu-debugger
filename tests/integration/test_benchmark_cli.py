"""Controller commands fail closed before provider construction and use durable scheduling."""

import pytest
from typer.testing import CliRunner


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


def test_validate_refuses_unattested_serialized_claims(tmp_path, monkeypatch):
    from gpu_agent.cli import app

    forged = tmp_path / "PRIVATE_SECRET.json"
    forged.write_text('{"oracle_passed":true,"run_ids":["forged"]}')
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "validate",
            str(forged),
            str(forged),
            "--corpus-root",
            str(tmp_path / "corpus"),
        ],
    )
    assert result.exit_code == 2 and "CASE_EXECUTION_ATTESTATION_UNAVAILABLE" in result.output
    assert "PRIVATE_SECRET" not in result.output
    corpus = tmp_path / "corpus"
    assert corpus.is_dir() and not list(corpus.iterdir())


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


def test_evaluate_uses_injected_executor_and_prints_reservation(tmp_path, monkeypatch, oob_service):
    from gpu_agent import cli
    from gpu_agent.benchmark.evaluation import EvaluationRecord, EvaluationRunner
    from gpu_agent.contracts import RepositorySnapshot, RunBinding
    from gpu_agent.service import ApplicationService

    service = oob_service[0]
    calls = []

    class InjectedExecutor:
        def __init__(self, service, corpus, sources):
            pass

        def execute(self, case_id, template_id, mode, repeat):
            calls.append((case_id, template_id, mode, repeat))
            return EvaluationRecord(
                record_id=str(repeat),
                case_id=case_id,
                template_id=template_id,
                mode=mode,
                repeat=repeat,
                input_hash="a" * 64,
                evidence_hash="b" * 64,
                executed_checks={},
                status="INCONCLUSIVE",
                diagnosis={},
                latency_ms=1,
                cost_usd=0.1,
            )

    from gpu_agent.benchmark.models import CaseManifest
    from gpu_agent.store import RunStore

    corpus = RunStore(tmp_path / "corpus")
    run = corpus.create_run("benchmark_case")
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
    corpus.put(run.id, "case-manifest.json", case.model_dump_json().encode(), "public")
    corpus.transition(run.id, "RUNNING", "FINALIZING")
    corpus.transition(run.id, "COMPLETED", None)

    def forbidden():
        raise AssertionError("offline path cannot construct configured services")

    monkeypatch.setattr(ApplicationService, "configured", forbidden)
    executor = InjectedExecutor(service, corpus, {})
    runner = EvaluationRunner(
        service.store,
        {"case_0100": "vector-add"},
        executor.execute,
        commit="a" * 40,
        prompt_version="test",
        toolchain_hash="3" * 64,
        model_config_hash="5" * 64,
        binding=RunBinding(
            repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
            purpose="evaluation",
            toolchain_lock_hash="3" * 64,
            prompt_version="test",
            model_config_hash="5" * 64,
        ),
        max_cost_usd=3,
        max_unit_cost_usd=1,
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
            "3",
            "--max-unit-cost-usd",
            "1",
            "--corpus-root",
            str(corpus.root),
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
    assert "Cost reservation: $3.00" in result.output
    assert "Hard maximum" not in result.output
    assert len(calls) == 3 and {call[3] for call in calls} == {0, 1, 2}
