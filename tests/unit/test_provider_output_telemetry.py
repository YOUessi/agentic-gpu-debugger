"""Planner wire contract and value-free output telemetry (no network, no SDK)."""

import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

CANARY = "VALUE-CANARY-7f3a"


def test_parse_wire_text_classifies_without_leaking_values():
    from gpu_agent.agent.provider import PlannerOutput, _OutputRejected, parse_wire_text

    with pytest.raises(_OutputRejected) as not_json:
        parse_wire_text("plan", "not json " + CANARY, PlannerOutput)
    assert not_json.value.diagnostics.failure_class == "NOT_JSON"
    assert not_json.value.diagnostics.output_chars == len("not json " + CANARY)

    bad = json.dumps({"action": {"action_type": "run_shell", "rationale": CANARY}})
    with pytest.raises(_OutputRejected) as invalid:
        parse_wire_text("plan", bad, PlannerOutput)
    diagnostics = invalid.value.diagnostics
    assert diagnostics.failure_class == "SCHEMA_INVALID"
    assert diagnostics.issues and all(issue.loc for issue in diagnostics.issues)
    assert CANARY not in diagnostics.model_dump_json()
    assert diagnostics.output_sha256 and len(diagnostics.output_sha256) == 64


def test_constraint_names_are_recorded_for_bounded_fields():
    from gpu_agent.agent.provider import PlannerOutput, _OutputRejected, parse_wire_text

    text = json.dumps(
        {"action": {"action_type": "inspect_source", "typed_arguments": {"source_id": "x"}}}
    )
    with pytest.raises(_OutputRejected) as rejected:
        parse_wire_text("plan", text, PlannerOutput)
    issues = rejected.value.diagnostics.issues
    assert any(
        "source_id" in issue.loc and issue.constraint and "pattern" in issue.constraint
        for issue in issues
    )


def test_controller_owned_fields_are_dropped_and_reassigned():
    from gpu_agent.agent.models import PlannerOutput
    from gpu_agent.agent.provider import normalize_wire_value

    raw = {
        "action": {
            "action_id": "not-a-valid-id",
            "budget_snapshot": {"anything": 1},
            "action_type": "retrieve_official_docs",
            "rationale": "r" * 900,
            "typed_arguments": {"query": "q" * 900, "k": 12},
        }
    }
    wire = PlannerOutput.model_validate(normalize_wire_value("plan", raw))
    action = wire.to_action_output("c" * 32).action
    assert action.action_id == "c" * 32
    assert action.budget_snapshot is None
    assert len(action.rationale) == 300
    assert len(action.typed_arguments.query) == 500 and action.typed_arguments.k == 5


def test_planner_schema_exposes_only_supported_actions_and_no_controller_fields():
    from gpu_agent.agent.models import PlannerOutput
    from gpu_agent.agent.policy import SUPPORTED

    schema = json.dumps(PlannerOutput.model_json_schema())
    assert "action_id" not in schema and "budget_snapshot" not in schema
    for unsupported in ("run_program", "inspect_environment", "request_more_evidence"):
        assert unsupported not in schema
    for supported in SUPPORTED:
        assert supported in schema


def _deepseek_client(texts):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        text = texts.pop(0)
        envelope = {
            "id": "resp_ds",
            "model": "deepseek-test",
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        }
        return SimpleNamespace(content=json.dumps(envelope), request_id="req_ds")

    def factory(**_kwargs):
        return SimpleNamespace(
            responses=SimpleNamespace(with_raw_response=SimpleNamespace(create=create)),
            close=lambda: None,
        )

    return factory, calls


def _worker_request(attempt=0, hints=()):
    from gpu_agent.agent.provider import WorkerRequest

    return WorkerRequest(
        endpoint="https://api.deepseek.com",
        model="deepseek-test",
        api_key=SecretStr("secret-canary"),
        kind="plan",
        payload={"evidence": {}},
        client_request_id="a" * 32,
        timeout_seconds=30,
        attempt=attempt,
        correction_hints=list(hints),
    )


def test_deepseek_worker_reports_schema_locations_and_sends_reduced_schema():
    from gpu_agent.agent.provider import invoke_sdk

    bad = json.dumps({"action": {"action_type": "run_memcheck", "extra": CANARY}})
    factory, calls = _deepseek_client([bad])
    result = invoke_sdk(_worker_request(), factory)
    assert result.error_code == "LLM_INVALID_OUTPUT" and result.value is None
    assert result.diagnostics is not None
    assert result.diagnostics.failure_class == "SCHEMA_INVALID"
    assert any("extra" in issue.loc for issue in result.diagnostics.issues)
    assert CANARY not in result.model_dump_json()
    sent_schema = json.dumps(calls[0]["text"]["format"]["schema"])
    assert "action_id" not in sent_schema and "budget_snapshot" not in sent_schema


def test_deepseek_worker_accepts_minimal_planner_output():
    from gpu_agent.agent.provider import invoke_sdk

    good = json.dumps({"action": {"action_type": "run_memcheck", "rationale": "precheck"}})
    factory, _ = _deepseek_client([good])
    result = invoke_sdk(_worker_request(), factory)
    assert result.error_code is None and result.value is not None
    assert result.diagnostics is not None and result.diagnostics.failure_class is None


def test_retry_instructions_name_rejected_fields():
    from gpu_agent.agent.provider import invoke_sdk

    good = json.dumps({"action": {"action_type": "run_memcheck"}})
    factory, calls = _deepseek_client([good])
    invoke_sdk(_worker_request(attempt=1, hints=["action.extra: extra_forbidden"]), factory)
    assert "action.extra: extra_forbidden" in calls[0]["instructions"]


class _ScriptedPort:
    def __init__(self, results):
        self.results, self.requests = list(results), []

    def call(self, request):
        self.requests.append(request)
        return self.results.pop(0)


def test_call_persists_diagnostics_and_feeds_hints_into_the_single_retry(store):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import (
        OpenAIProviderSettings,
        OpenAIResponsesProvider,
        OutputDiagnostics,
        ResponseMetadata,
        SDKResult,
        Usage,
        ValidationIssue,
    )

    rejected = SDKResult(
        error_code="LLM_INVALID_OUTPUT",
        state="FAILED",
        metadata=ResponseMetadata(response_model="m", usage=Usage(output_tokens=250)),
        diagnostics=OutputDiagnostics(
            failure_class="SCHEMA_INVALID",
            output_chars=900,
            issues=[ValidationIssue(loc="action.rationale", type="string_too_long")],
        ),
    )
    accepted = SDKResult(
        value={"action": {"action_type": "run_memcheck", "rationale": "ok"}},
        metadata=ResponseMetadata(response_model="m", usage=Usage(output_tokens=40)),
    )
    port = _ScriptedPort([rejected, accepted])
    run = store.create_run("telemetry-test")
    provider = OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.openai.com/v1", model="m", api_key=SecretStr("k")
        ),
        LLMCallGate(),
        store,
        run.id,
        port=port,
    )
    action = provider.plan(PublicEvidence(), AgentBudget())
    assert action.action_type == "run_memcheck"
    assert port.requests[1].correction_hints == ["action.rationale: string_too_long"]
    failed = [i for i in provider.invocations() if i.state == "FAILED"]
    assert failed and failed[0].output_diagnostics is not None
    assert failed[0].output_diagnostics.issues[0].loc == "action.rationale"


def _bounded_provider(store, port, max_llm_calls):
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import (
        DevelopmentCallPolicy,
        OpenAIProviderSettings,
        OpenAIResponsesProvider,
    )

    run = store.create_run("dev-cap-test")
    return OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.openai.com/v1", model="m", api_key=SecretStr("k")
        ),
        LLMCallGate(),
        store,
        run.id,
        port=port,
        call_policy=DevelopmentCallPolicy(max_llm_calls=max_llm_calls),
    )


def test_development_call_limit_counts_format_retries(store):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError, SDKResult

    port = _ScriptedPort([SDKResult(value={"invalid": True})])
    provider = _bounded_provider(store, port, max_llm_calls=1)
    with pytest.raises(ProviderError) as refused:
        provider.plan(PublicEvidence(), AgentBudget())
    assert refused.value.code == "AGENT_BUDGET_EXHAUSTED"
    assert len(port.requests) == len(provider.invocations()) == 1


def test_development_call_limit_records_usage_and_stops_before_next_request(store):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError, ResponseMetadata, SDKResult, Usage

    accepted = SDKResult(
        value={"action": {"action_type": "run_memcheck"}},
        metadata=ResponseMetadata(
            response_model="m", usage=Usage(input_tokens=100, output_tokens=10)
        ),
    )
    port = _ScriptedPort([accepted])
    provider = _bounded_provider(store, port, max_llm_calls=1)
    assert provider.plan(PublicEvidence(), AgentBudget()).action_type == "run_memcheck"
    assert len(port.requests) == 1
    assert provider.invocations()[0].usage.input_tokens == 100
    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        provider.plan(PublicEvidence(), AgentBudget())
    assert len(port.requests) == 1


def test_development_paid_calls_are_refused_on_evaluation_bound_services(oob_service):
    from gpu_agent.agent.provider import DevelopmentCallPolicy
    from gpu_agent.contracts import RepositorySnapshot, RunBinding

    service = oob_service[0]
    service._binding = RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="evaluation",
    )
    with pytest.raises(ValueError):
        service.allow_development_paid_calls(DevelopmentCallPolicy(max_llm_calls=40))
