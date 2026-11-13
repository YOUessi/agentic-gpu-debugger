"""Evidence-gated planning must save calls without bypassing CUDA safety policy."""

import json

import pytest

from gpu_agent.agent.models import MemcheckAction, RetrieveDocsAction


def test_adaptive_planner_uses_one_model_plan_and_preserves_final_patch(oob_service, monkeypatch):
    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_ORCHESTRATOR", "langgraph")
    monkeypatch.setenv("GPU_AGENT_PLANNER_POLICY", "adaptive-v1")
    provider.actions = [RetrieveDocsAction(typed_arguments={"query": "out of bounds", "k": 3})]
    run = service.diagnose(source, mode="E")
    assert service.diagnosis(run.id).diagnostic_outcome == "DIAGNOSED"
    assert provider.kinds == ["plan", "diagnose", "patch"]
    assert len(service.candidates(run.id)) == 1
    refs = {ref.name: ref for ref in run.artifact_refs}
    assert json.loads(service.store.read(refs["agent/final-budget.json"]))["llm_calls"] == 3
    decisions = [
        json.loads(service.store.read(refs[f"agent/adaptive-policy/{step}.json"]))["selection"]
        for step in range(3)
    ]
    assert decisions == ["MANDATORY_MEMCHECK", "LLM_PLANNER", "EVIDENCE_COMPLETE"]
    assert {f"actions/{i}/decision.json" for i in range(3)} <= set(refs)


def test_model_duplicate_is_replaced_by_existing_rule_router(oob_service, monkeypatch):
    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_PLANNER_POLICY", "adaptive-v1")
    provider.actions = [
        MemcheckAction(),
    ]
    run = service.diagnose(source, mode="E")
    assert service.diagnosis(run.id).diagnostic_outcome == "DIAGNOSED"
    assert len(service.candidates(run.id)) == 1
    refs = {r.name: r for r in run.artifact_refs}
    assert (
        json.loads(service.store.read(refs["agent/adaptive-policy/2.json"]))["selection"]
        == "REJECTED_ACTION_FALLBACK"
    )
    assert provider.kinds == ["plan", "diagnose", "patch"]


def test_invalid_schema_uses_rule_but_never_retries_uncertain_calls(oob_service, monkeypatch):
    from gpu_agent.agent.provider import ProviderError

    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_PLANNER_POLICY", "adaptive-v1")

    def schema_failure(*args, **kwargs):
        provider._record("plan", {"schema_failure": True})
        raise ProviderError("LLM_INVALID_OUTPUT")

    monkeypatch.setattr(provider, "plan", schema_failure)
    run = service.diagnose(source)
    assert service.diagnosis(run.id).diagnostic_outcome == "DIAGNOSED"
    assert provider.kinds == ["plan", "diagnose", "patch"]
    refs = {r.name: r for r in run.artifact_refs}
    assert (
        json.loads(service.store.read(refs["agent/adaptive-policy/1.json"]))["selection"]
        == "INVALID_OUTPUT_FALLBACK"
    )


def test_uncertain_call_is_not_automatically_replayed(oob_service, monkeypatch):
    from gpu_agent.agent.provider import ProviderError

    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_PLANNER_POLICY", "adaptive-v1")

    def timeout(*args, **kwargs):
        provider._record("plan", {"timeout": True})
        raise ProviderError("LLM_TIMEOUT", state="UNCERTAIN")

    monkeypatch.setattr(provider, "plan", timeout)
    run = service.diagnose(source)
    assert service.diagnosis(run.id).diagnostic_outcome == "INCONCLUSIVE"
    assert provider.kinds == ["plan"]
    assert not service.candidates(run.id)


@pytest.mark.parametrize("mode", ["A", "B", "C", "D"])
def test_adaptive_experiment_cannot_change_other_evaluation_modes(oob_service, monkeypatch, mode):
    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_PLANNER_POLICY", "adaptive-v1")
    with pytest.raises(ValueError, match="restricted"):
        service.diagnose(source, mode=mode)
    assert not provider.kinds


def test_adaptive_rejected_on_legacy_and_evaluator(oob_service, monkeypatch):
    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_PLANNER_POLICY", "adaptive-v1")
    monkeypatch.setenv("GPU_AGENT_ORCHESTRATOR", "legacy")
    with pytest.raises(ValueError, match="restricted"):
        service.diagnose(source)
    monkeypatch.setenv("GPU_AGENT_ORCHESTRATOR", "langgraph")
    service._binding = object()
    with pytest.raises(ValueError, match="restricted"):
        service.diagnose(source)
    assert not provider.kinds


def test_unknown_policy_fails_closed(oob_service, monkeypatch):
    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_PLANNER_POLICY", "unknown")
    with pytest.raises(ValueError, match="GPU_AGENT_PLANNER_POLICY"):
        service.diagnose(source)
    assert not provider.kinds
