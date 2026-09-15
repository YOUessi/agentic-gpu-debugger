"""Fake acceptance is deliberately separate from required live acceptance."""

import os

import pytest
from typer.testing import CliRunner


def test_fake_diagnose_single_candidate_verify_and_report(oob_service):
    service, provider, source = oob_service
    run = service.diagnose(source)
    candidate_id = service.candidates(run.id)[0]
    result = service.verify(run.id, candidate_id)
    assert result.verdict.value == "INCONCLUSIVE"  # Fake baseline cannot attest GPU execution.
    assert len(service.candidates(run.id)) == 1
    assert provider.kinds.count("patch") == 1
    report = service.report(run.id)
    assert "INCONCLUSIVE" in report and "Single candidate" in report
    assert "Observed facts" in report and "Model inferences" in report


def test_cli_no_credentials_persists_typed_result(tmp_path, monkeypatch):
    from gpu_agent.cli import app

    for name in ["OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GPU_AGENT_RUN_ROOT", str(tmp_path / "runs"))
    source = tmp_path / "kernel.cu"
    source.write_text("int main() { return 0; }\n")
    result = CliRunner().invoke(app, ["diagnose", str(source)])
    assert result.exit_code == 0, result.output
    assert "LLM_UNAVAILABLE" in result.output
    run_id = result.output.splitlines()[0].split()[-1]
    reported = CliRunner().invoke(app, ["report", run_id])
    assert reported.exit_code == 0 and "LLM_UNAVAILABLE" in reported.output


def test_cli_candidate_selectors_are_mutually_exclusive():
    from gpu_agent.cli import app

    result = CliRunner().invoke(
        app, ["verify", "a" * 32, "candidate.diff", "--generated-candidate", "--strict"]
    )
    assert result.exit_code == 2 and "exactly one" in result.output


@pytest.mark.gpu
@pytest.mark.container
@pytest.mark.live_llm
def test_live_oob_flow(tmp_path, monkeypatch):
    from pathlib import Path

    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.service import ApplicationService
    from gpu_agent.store import RunStore

    store = RunStore(tmp_path / "runs")
    backend = IsolatedGPUBackend(store, tmp_path, tmp_path / "tasks")
    availability = backend.availability()
    missing = []
    if not availability.ready:
        missing.append(availability.reason)
    if not all(os.environ.get(k) for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")):
        missing.append("LLM_UNAVAILABLE: explicit provider configuration absent")
    if missing:
        pytest.skip("; ".join(missing))
    monkeypatch.setenv("GPU_AGENT_RUN_ROOT", str(store.root))
    monkeypatch.setenv("GPU_AGENT_EVALUATOR_ROOT", str(tmp_path / "evaluator"))
    service = ApplicationService.configured()
    source = Path(__file__).resolve().parents[2] / "benchmarks/public/case_0001/public_input"
    run = service.diagnose(source)
    result = service.verify(run.id)
    assert result.verdict.value == "VERIFIED_FIXED", service.report(run.id)
