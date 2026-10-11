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
    audit = next(r for r in run.artifact_refs if r.name == "agent/budget-audit.json")
    states = {event["state"] for event in json.loads(service.store.read(audit))}
    assert {"ATTEMPTED", "STARTED", "COMPLETED"} <= states
    source_ref = next(ref for ref in run.artifact_refs if ref.name == "sources/kernel.cu")
    assert source_ref.visibility == "public"


def test_provider_public_projection_excludes_private_and_controller_data(oob_service):
    service, provider, source = oob_service
    run = service.diagnose(source)
    source_ref = next(ref for ref in run.artifact_refs if ref.name == "sources/kernel.cu")
    full_source = service.store.read(source_ref).decode()
    for payload in provider.inputs:
        if "evidence" in payload:
            assert payload["evidence"]["sources"][0]["content"] == full_source
    assert not any(ref.name.startswith("source-reads/") for ref in run.artifact_refs)
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


def test_public_projection_deduplicates_repeated_identical_findings():
    from gpu_agent.agent.models import PublicFinding
    from gpu_agent.agent.orchestrator import _deduplicate_findings
    from gpu_agent.execution.models import SourceLocation

    repeated = PublicFinding(
        artifact_id="a" * 32,
        category="Invalid __global__ read",
        source_location=SourceLocation(path="kernel.cu", line=9),
    )

    assert _deduplicate_findings([repeated, repeated]) == [repeated]


def test_plan_prompt_states_constraints_not_the_rule_router_procedure():
    """E must differ from D only in acquisition policy, so the planner is not handed D's
    fixed sequence; it gets the controller's hard constraints and a goal instead."""
    from gpu_agent.agent.prompts import PROMPTS

    prompt = PROMPTS["plan"]
    assert "requires memcheck before any other sanitizer" in prompt
    assert "rejects actions that repeat evidence" in prompt
    assert "sanitizer_outcomes has no memcheck" not in prompt
    assert "retrieve official docs for that finding and then finish" not in prompt


def test_diagnosis_prompt_names_the_exact_citation_and_location_constraints():
    from gpu_agent.agent.prompts import PROMPTS

    prompt = PROMPTS["diagnose"]
    assert "tool_findings may cite only artifact_id" in prompt
    assert "documentation_evidence may cite only chunk_id" in prompt
    assert "copy one of those locations exactly" in prompt
    # The gate is evidence-derived: runs without findings or docs can still be DIAGNOSED.
    assert "if and only if the evidence contains tool_findings" in prompt
    assert "if and only if the evidence contains documentation" in prompt


def test_patch_prompt_forbids_observed_input_size_hardcoding():
    from gpu_agent.agent.prompts import PROMPTS

    prompt = PROMPTS["patch"]
    assert "Inspect the entire public source" in prompt
    assert "hard-code or narrow the accepted input sizes" in prompt
    # Generic across cases: no case_0001-specific guard wording leaks into the prompt.
    assert "257" not in prompt and "n != integer" not in prompt


def test_repeated_no_benefit_action_stops_without_second_tool(oob_service):
    from gpu_agent.agent.models import MemcheckAction

    service, provider, source = oob_service
    # The first denial earns one replan; repeating the denied action again is final.
    provider.actions = [MemcheckAction(), MemcheckAction(), MemcheckAction()]
    run = service.diagnose(source)
    assert "DUPLICATE_NO_BENEFIT" in service.diagnosis(run.id).limitations
    assert provider.kinds == ["plan", "plan", "plan"]
    assert provider.inputs[2]["controller_feedback"] == {
        "rejected_previous_action": ["DUPLICATE_NO_BENEFIT"]
    }
    assert not service.candidates(run.id)


def test_finish_cannot_bypass_mandatory_evidence(oob_service):
    from gpu_agent.agent.models import FinishAction

    service, provider, source = oob_service
    provider.actions = [FinishAction(), FinishAction()]
    run = service.diagnose(source)
    assert service.diagnosis(run.id).diagnostic_outcome == "INCONCLUSIVE"
    assert "MANDATORY_EVIDENCE_MISSING" in service.diagnosis(run.id).limitations
    assert "diagnose" not in provider.kinds


def test_unsupported_action_is_typed_and_never_executes(oob_service):
    from gpu_agent.agent.models import RunProgramAction

    service, provider, source = oob_service
    provider.actions = [RunProgramAction(), RunProgramAction()]
    run = service.diagnose(source)
    assert "ACTION_UNSUPPORTED" in service.diagnosis(run.id).limitations


def test_authoritative_budget_reserves_diagnosis_and_patch():
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import ProviderError

    gate = LLMCallGate()
    for _ in range(38):
        gate.reserve("plan")
    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        gate.reserve("plan")
    gate.reserve("diagnose")
    gate.reserve("patch")
    with pytest.raises(ProviderError):
        gate.reserve("patch")
    assert gate.snapshot().llm_calls == 40


def test_wall_budget_prevents_send():
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import ProviderError

    times = iter([0.0, 601.0])
    gate = LLMCallGate(clock=lambda: next(times))
    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        gate.reserve("plan")


def test_each_call_kind_has_its_own_single_format_retry():
    """case_0005: a diagnose retry used to consume the only retry, leaving the patch none."""
    from gpu_agent.agent.models import AgentBudget
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import ProviderError

    gate = LLMCallGate()
    gate.reserve("plan")
    gate.reserve("plan", attempt=1)
    gate.reserve("plan")
    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        gate.reserve("plan", attempt=1)
    gate.reserve("diagnose")
    gate.reserve("diagnose", attempt=1)
    gate.reserve("patch")
    gate.reserve("patch", attempt=1)
    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        gate.reserve("patch", attempt=1)
    assert gate.snapshot().llm_calls == 7
    # The total call bound still applies to retries.
    bounded = LLMCallGate(AgentBudget(max_llm_calls=1))
    bounded.reserve("patch")
    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        bounded.reserve("patch", attempt=1)


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


def test_planner_sees_missing_evidence_executed_actions_and_last_rejection(oob_service):
    """E gets controller progress each step; it still chooses the next action itself."""
    from gpu_agent.agent.models import FinishAction, MemcheckAction

    service, provider, source = oob_service
    provider.actions = [MemcheckAction(), FinishAction(), FinishAction()]
    run = service.diagnose(source)
    states = [p["controller_state"] for p in provider.inputs if "controller_state" in p]
    assert states[0]["missing_evidence"] == ["memcheck_outcome", "tool_finding"]
    assert states[0]["executed_actions"] == [] and states[0]["rejected_previous_action"] == []
    # After memcheck reported a finding, only documentation is missing before finishing.
    assert states[1]["missing_evidence"] == ["documentation_for_finding"]
    assert [a["action_type"] for a in states[1]["executed_actions"]] == ["run_memcheck"]
    assert states[2]["rejected_previous_action"] == ["MANDATORY_EVIDENCE_MISSING"]
    assert "MANDATORY_EVIDENCE_MISSING" in service.diagnosis(run.id).limitations


def test_full_source_contract_rejects_injected_obsolete_read(oob_service):
    from gpu_agent.agent.models import InspectSourceAction, MemcheckAction, SourceArguments

    service, provider, source = oob_service
    scripted = type(provider).plan
    states = []

    def plan(self, evidence, budget, feedback=None, state=None):
        states.append(state)
        if len(states) == 1:
            kernel_id = evidence.sources[0].source_id
            return InspectSourceAction(
                typed_arguments=SourceArguments(source_id=kernel_id, start_line=1, end_line=8)
            )
        return scripted(self, evidence, budget, feedback, state)

    provider.plan = plan.__get__(provider)
    provider.actions = [MemcheckAction()]
    service.diagnose(source)
    ranges = states[1].source_ranges_read
    assert ranges == []
    assert states[1].executed_actions == []
    assert states[1].rejected_previous_action == ["SOURCE_ALREADY_AVAILABLE"]


def test_legacy_source_action_remains_parseable_and_replayable():
    from gpu_agent.agent.models import (
        ACTION_ADAPTER,
        AgentBudget,
        PublicEvidence,
        PublicSource,
    )
    from gpu_agent.agent.policy import action_policy_for_prompt, decide_action
    from gpu_agent.contracts import CurrentPhase

    action = ACTION_ADAPTER.validate_python(
        {
            "action_type": "inspect_source",
            "typed_arguments": {"source_id": "a" * 32, "start_line": 1, "end_line": 2},
        }
    )
    evidence = PublicEvidence(sources=[PublicSource(source_id="a" * 32, content="line1\nline2\n")])
    old = decide_action(
        action,
        evidence,
        AgentBudget(),
        CurrentPhase.DIAGNOSING,
        set(),
        policy_version=action_policy_for_prompt("m3-2026-09-26-v9"),
    )
    new = decide_action(action, evidence, AgentBudget(), CurrentPhase.DIAGNOSING, set())
    assert old.allowed and old.policy_version == "diagnosis-m1-v1"
    assert not new.allowed and new.reason_codes == ["SOURCE_ALREADY_AVAILABLE"]


def test_current_prompt_selects_current_action_policy():
    from gpu_agent.agent.policy import CURRENT_ACTION_POLICY, action_policy_for_prompt
    from gpu_agent.agent.prompts import PROMPT_VERSION

    assert action_policy_for_prompt(PROMPT_VERSION) == CURRENT_ACTION_POLICY
    with pytest.raises(ValueError, match="bound prompt"):
        action_policy_for_prompt(None)


def test_current_wire_rejects_obsolete_read_and_retains_full_source():
    from pydantic import ValidationError

    from gpu_agent.agent.models import PlannerOutput, PublicEvidence, PublicSource

    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(
            {
                "action": {
                    "action_type": "inspect_source",
                    "typed_arguments": {"source_id": "a" * 32, "start_line": 1, "end_line": 2},
                }
            }
        )
    # Public evidence serialization remains full-source, not a hidden partial-source variant.
    evidence = PublicEvidence(sources=[PublicSource(source_id="a" * 32, content="entire kernel")])
    assert evidence.model_dump(mode="json")["sources"][0]["content"] == "entire kernel"


def test_missing_evidence_matches_the_finish_gate():
    from gpu_agent.agent.models import AgentBudget, FinishAction, PublicEvidence
    from gpu_agent.agent.policy import decide_action, missing_evidence
    from gpu_agent.contracts import CurrentPhase

    evidence = PublicEvidence()
    decision = decide_action(
        FinishAction(), evidence, AgentBudget(), CurrentPhase.DIAGNOSING, set()
    )
    assert missing_evidence(evidence) == ["memcheck_outcome", "tool_finding"]
    assert decision.reason_codes == ["MANDATORY_EVIDENCE_MISSING"]
