"""Offline graph/legacy parity tests: real policy and RunStore, synthetic GPU/model."""

import json

import pytest


@pytest.mark.parametrize("engine", ["langgraph", "legacy"])
def test_d_e_controller_parity_for_public_tools(oob_service, monkeypatch, engine):
    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_ORCHESTRATOR", engine)
    run = service.diagnose(source)
    result = service.diagnosis(run.id)

    assert result.diagnostic_outcome == "DIAGNOSED"
    assert provider.kinds == ["plan", "plan", "plan", "diagnose", "patch"]
    refs = {ref.name: ref for ref in run.artifact_refs}
    engine_info = json.loads(service.store.read(refs["agent/orchestration-engine.json"]))
    assert engine_info == {"schema_version": 1, "engine": engine}
    for name in (
        "agent/controller-lineage.json",
        "agent/acquisition-usage.json",
        "agent/budget-audit.json",
        "actions/0/decision.json",
        "actions/1/decision.json",
        "actions/2/decision.json",
    ):
        assert name in refs
    assert len(service.candidates(run.id)) == 1


@pytest.mark.parametrize("engine", ["langgraph", "legacy"])
def test_policy_denials_consume_steps_without_executing_tools(oob_service, monkeypatch, engine):
    from gpu_agent.agent.models import MemcheckAction

    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_ORCHESTRATOR", engine)
    provider.actions = [MemcheckAction(), MemcheckAction(), MemcheckAction()]
    run = service.diagnose(source)

    assert "DUPLICATE_NO_BENEFIT" in service.diagnosis(run.id).limitations
    assert provider.kinds == ["plan", "plan", "plan"]
    assert provider.inputs[-1]["controller_feedback"] == {
        "rejected_previous_action": ["DUPLICATE_NO_BENEFIT"]
    }
    assert not service.candidates(run.id)


@pytest.mark.parametrize("engine", ["langgraph", "legacy"])
def test_finish_without_required_evidence_cannot_skip_policy(oob_service, monkeypatch, engine):
    from gpu_agent.agent.models import FinishAction

    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_ORCHESTRATOR", engine)
    provider.actions = [FinishAction(), FinishAction()]
    run = service.diagnose(source)

    assert "MANDATORY_EVIDENCE_MISSING" in service.diagnosis(run.id).limitations
    assert "diagnose" not in provider.kinds


def test_rule_router_uses_graph_without_planner_model(oob_service, monkeypatch):
    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_ORCHESTRATOR", "langgraph")
    run = service.diagnose(source, mode="D")

    assert "plan" not in provider.kinds
    assert "agent/orchestration-engine.json" in {ref.name for ref in run.artifact_refs}


def test_invalid_engine_is_rejected_before_model_planning(oob_service, monkeypatch):
    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_ORCHESTRATOR", "unknown")
    with pytest.raises(ValueError, match="GPU_AGENT_ORCHESTRATOR"):
        service.diagnose(source)
    assert "plan" not in provider.kinds
