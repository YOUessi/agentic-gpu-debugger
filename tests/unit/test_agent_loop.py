"""Controller policy, physical budgets and privacy are observable boundaries."""

import json

import pytest
from pydantic import ValidationError


def test_action_registry_rejects_commands_and_cross_typed_arguments():
    from gpu_agent.agent.models import ACTION_ADAPTER

    for data in [
        {"action_type": "run_memcheck", "action_id": "../escape"},
        {"action_type": "shell", "typed_arguments": {"command": "id"}},
        {"action_type": "run_memcheck", "typed_arguments": {"path": "/private"}},
        {"action_type": "inspect_source", "typed_arguments": {"source_id": "../private"}},
        {"action_type": "retrieve_official_docs", "typed_arguments": {"url": "https://x"}},
    ]:
        with pytest.raises(ValidationError):
            ACTION_ADAPTER.validate_python(data)


def test_memcheck_retrieve_finish_flow(oob_service):
    service, provider, source = oob_service
    run = service.diagnose(source)
    result = service.diagnosis(run.id)
    assert result.diagnostic_outcome == "DIAGNOSED"
    assert result.tool_findings and result.documentation_evidence and result.observed_facts
    assert provider.kinds == ["plan", "plan", "plan", "diagnose", "patch"]
    assert len(service.candidates(run.id)) == 1
    assert run.status.value == "COMPLETED"
    final_budget = next(r for r in run.artifact_refs if r.name == "agent/final-budget.json")
    assert json.loads(service.store.read(final_budget))["llm_calls"] == 5


def test_provider_public_projection_excludes_private_and_controller_data(oob_service):
    service, provider, source = oob_service
    service.diagnose(source)
    wire = json.dumps(provider.inputs)
    for forbidden in [
        "reference.cu",
        "ground_truth",
        "private_seed",
        "checker",
        "evaluation_label",
        "secret-canary",
        "OPENAI_API_KEY",
        str(service.store.root),
        "vector_io.cpp",
    ]:
        assert forbidden not in wire
    assert "kernel.cu" in wire and "Invalid __global__ write" in wire


def test_repeated_no_benefit_action_stops_without_second_tool(oob_service):
    from gpu_agent.agent.models import MemcheckAction

    service, provider, source = oob_service
    provider.actions = [MemcheckAction(), MemcheckAction()]
    run = service.diagnose(source)
    assert "DUPLICATE_NO_BENEFIT" in service.diagnosis(run.id).limitations
    assert provider.kinds == ["plan", "plan"]
    assert not service.candidates(run.id)


def test_finish_cannot_bypass_mandatory_evidence(oob_service):
    from gpu_agent.agent.models import FinishAction

    service, provider, source = oob_service
    provider.actions = [FinishAction()]
    run = service.diagnose(source)
    assert service.diagnosis(run.id).diagnostic_outcome == "INCONCLUSIVE"
    assert "MANDATORY_EVIDENCE_MISSING" in service.diagnosis(run.id).limitations
    assert "diagnose" not in provider.kinds


def test_unsupported_action_is_typed_and_never_executes(oob_service):
    from gpu_agent.agent.models import RacecheckAction

    service, provider, source = oob_service
    provider.actions = [RacecheckAction()]
    run = service.diagnose(source)
    assert "ACTION_UNSUPPORTED" in service.diagnosis(run.id).limitations


def test_authoritative_budget_reserves_diagnosis_and_patch():
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import ProviderError

    gate = LLMCallGate()
    for _ in range(4):
        gate.reserve("plan")
    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        gate.reserve("plan")
    gate.reserve("diagnose")
    gate.reserve("patch")
    with pytest.raises(ProviderError):
        gate.reserve("patch")
    assert gate.snapshot().llm_calls == 6


def test_wall_budget_prevents_send():
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import ProviderError

    times = iter([0.0, 601.0])
    gate = LLMCallGate(clock=lambda: next(times))
    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        gate.reserve("plan")


def test_only_one_format_retry_is_available_for_the_run():
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import ProviderError

    gate = LLMCallGate()
    gate.reserve("plan")
    gate.reserve("plan", attempt=1)
    gate.reserve("diagnose")
    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        gate.reserve("diagnose", attempt=1)
    assert gate.snapshot().llm_calls == 3


def test_zero_remaining_tool_timeout_is_typed():
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import ProviderError

    times = iter([0.0, 600.0])
    gate = LLMCallGate(clock=lambda: next(times))
    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        gate.timeout(120)


def test_provider_patch_error_keeps_diagnosis_and_never_creates_candidate(oob_service):
    service, provider, source = oob_service
    provider.diff = "```diff\nnot a diff\n```"
    run = service.diagnose(source)
    result = service.diagnosis(run.id)
    assert result.diagnostic_outcome == "DIAGNOSED"
    assert "LLM_INVALID_OUTPUT" in result.limitations
    assert not service.candidates(run.id)
    assert provider.kinds.count("patch") == 1


def test_second_candidate_registration_is_refused(oob_service, tmp_path):
    service, provider, source = oob_service
    run = service.diagnose(source)
    diff_path = tmp_path / "candidate.diff"
    diff_path.write_text(provider.diff)
    with pytest.raises(ValueError, match="only one candidate"):
        service.register_patch(run.id, diff_path)
    assert len(service.candidates(run.id)) == 1


def test_high_confidence_does_not_make_forged_citation_valid(oob_service):
    service, provider, source = oob_service
    provider.forge_citations = True
    run = service.diagnose(source)
    assert service.diagnosis(run.id).diagnostic_outcome == "INCONCLUSIVE"
    assert "INVALID_DIAGNOSIS_EVIDENCE" in service.diagnosis(run.id).limitations
    assert "patch" not in provider.kinds


def test_diagnosis_locations_must_match_tool_evidence(oob_service):
    from gpu_agent.agent.orchestrator import public_evidence
    from gpu_agent.agent.policy import validate_diagnosis
    from gpu_agent.execution.models import SourceLocation

    service, _, source = oob_service
    run = service.diagnose(source)
    result = service.diagnosis(run.id).model_copy(
        update={"source_locations": [SourceLocation(path="kernel.cu", line=1)]}
    )
    assert not validate_diagnosis(result, public_evidence(service.store, run.id))
