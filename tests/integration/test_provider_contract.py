"""Offline contract tests exercise the official adapter without network access."""

from types import SimpleNamespace

import pytest
from pydantic import SecretStr


def response(value=None, **changes):
    fields = dict(
        output_parsed=value,
        id="resp_1",
        _request_id="req_1",
        model="configured",
        status="completed",
        error=None,
        incomplete_details=None,
        usage=None,
        output=[],
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


@pytest.fixture
def provider_factory(store):
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import OpenAIProviderSettings, OpenAIResponsesProvider

    def make(script, **settings):
        calls, clients = [], []
        run = store.create_run("provider-test")

        def parse(**kwargs):
            import json

            from pydantic import BaseModel

            calls.append(kwargs)
            item = script.pop(0)
            if isinstance(item, Exception):
                raise item
            content = json.dumps(
                vars(item),
                default=lambda value: (
                    value.model_dump(mode="json") if isinstance(value, BaseModel) else vars(value)
                ),
            )
            return SimpleNamespace(content=content, request_id=item._request_id, parse=lambda: item)

        def factory(**kwargs):
            clients.append(kwargs)
            return SimpleNamespace(
                responses=SimpleNamespace(with_raw_response=SimpleNamespace(parse=parse)),
                close=lambda: None,
            )

        config = dict(
            endpoint="https://api.openai.com/v1",
            model="configured",
            api_key=SecretStr("secret-canary"),
        )
        config.update(settings)
        provider = OpenAIResponsesProvider(
            OpenAIProviderSettings(**config), LLMCallGate(), store, run.id, client_factory=factory
        )
        return provider, calls, clients

    return make


def test_missing_credentials_never_constructs_client_or_reserves(provider_factory):
    from gpu_agent.agent.models import AgentBudget, DiagnosisResult, PublicEvidence, PublicSource
    from gpu_agent.agent.provider import ProviderError

    provider, calls, clients = provider_factory([], api_key=None)
    for invoke in [
        lambda: provider.plan(PublicEvidence(), AgentBudget()),
        lambda: provider.diagnose(PublicEvidence()),
        lambda: provider.propose_patch(
            PublicSource(source_id="a" * 32, content="int x;\n"),
            DiagnosisResult.inconclusive("TEST"),
        ),
    ]:
        with pytest.raises(ProviderError, match="LLM_UNAVAILABLE"):
            invoke()
    assert not calls and not clients and not provider.invocations()
    assert provider.gate.snapshot().llm_calls == 0
    assert "secret-canary" not in repr(provider.settings)


def test_official_parse_parameters_metadata_and_no_implicit_retry(provider_factory):
    from gpu_agent.agent.models import (
        AgentActionOutput,
        AgentBudget,
        MemcheckAction,
        PublicEvidence,
    )

    provider, calls, clients = provider_factory(
        [response(AgentActionOutput(action=MemcheckAction()))]
    )
    action = provider.plan(PublicEvidence(), AgentBudget())
    assert action.action_type == "run_memcheck"
    assert clients[0]["max_retries"] == 0 and clients[0]["timeout"] <= 60
    assert clients[0]["base_url"] == "https://api.openai.com/v1"
    call = calls[0]
    assert call["store"] is False and call["model"] == "configured"
    assert call["text_format"] is AgentActionOutput
    assert call["extra_headers"]["X-Client-Request-Id"]
    assert not {"tools", "background", "conversation", "previous_response_id"}.intersection(call)
    record = provider.invocations()[-1]
    assert record.state == "COMPLETED"
    assert record.response_id == "resp_1" and record.provider_request_id == "req_1"
    assert record.usage is None and record.response_model == "configured"
    assert "secret-canary" not in record.model_dump_json()


def test_invalid_output_retries_once_and_charges_physical_calls(provider_factory):
    from gpu_agent.agent.models import (
        AgentActionOutput,
        AgentBudget,
        MemcheckAction,
        PublicEvidence,
    )

    provider, calls, _ = provider_factory(
        [
            response({"action": {"action_type": "shell"}}),
            response(AgentActionOutput(action=MemcheckAction())),
        ]
    )
    assert provider.plan(PublicEvidence(), AgentBudget()).action_type == "run_memcheck"
    assert len(calls) == provider.gate.snapshot().llm_calls == 2
    assert len({c["extra_headers"]["X-Client-Request-Id"] for c in calls}) == 2


def test_timeout_is_uncertain_and_never_replayed(provider_factory):
    import httpx2
    import openai

    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError

    provider, calls, _ = provider_factory(
        [
            openai.APITimeoutError(
                request=httpx2.Request("POST", "https://api.openai.com/v1/responses")
            )
        ]
    )
    with pytest.raises(ProviderError, match="LLM_TIMEOUT"):
        provider.plan(PublicEvidence(), AgentBudget())
    assert len(calls) == 1
    assert provider.invocations()[-1].state == "UNCERTAIN"
    assert provider.invocations()[-1].client_request_id


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"status": "incomplete"}, "LLM_INCOMPLETE"),
        ({"output": [SimpleNamespace(content=[SimpleNamespace(type="refusal")])]}, "LLM_REFUSED"),
    ],
)
def test_refusal_and_incomplete_do_not_retry(provider_factory, changes, code):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError

    provider, calls, _ = provider_factory([response(**changes)])
    with pytest.raises(ProviderError, match=code):
        provider.plan(PublicEvidence(), AgentBudget())
    assert len(calls) == 1 and provider.invocations()[-1].response_id == "resp_1"


def test_endpoint_secrets_and_unproven_store_capability_fail_closed():
    from gpu_agent.agent.provider import OpenAIProviderSettings

    for endpoint in [
        "https://user:password@api.openai.com/v1",
        "https://api.openai.com/v1?key=x",
        "https://api.openai.com/v1#x",
    ]:
        with pytest.raises(ValueError):
            OpenAIProviderSettings(endpoint=endpoint, model="configured")


def test_actual_installed_sdk_structured_surface():
    import inspect

    import openai
    from openai.types.responses.parsed_response import ParsedResponse

    client = openai.OpenAI(
        api_key="offline-contract-placeholder",
        max_retries=0,
        timeout=60,
        base_url="https://api.openai.com/v1",
    )
    try:
        parameters = inspect.signature(client.responses.parse).parameters
        assert {"text_format", "store", "timeout", "extra_headers"} <= set(parameters)
        assert hasattr(ParsedResponse, "output_parsed")
        assert client.max_retries == 0
    finally:
        client.close()


@pytest.mark.parametrize(
    "status,code",
    [
        (400, "LLM_INVALID_REQUEST"),
        (401, "LLM_AUTHENTICATION_FAILED"),
        (403, "LLM_PERMISSION_DENIED"),
        (404, "LLM_MODEL_OR_ENDPOINT_NOT_FOUND"),
        (409, "LLM_PROVIDER_ERROR"),
        (422, "LLM_INVALID_REQUEST"),
        (429, "LLM_RATE_LIMITED"),
        (500, "LLM_PROVIDER_ERROR"),
    ],
)
def test_http_error_mapping_redacts_body_and_preserves_request_id(provider_factory, status, code):
    import httpx2
    import openai

    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError

    raw = httpx2.Response(
        status,
        headers={"x-request-id": "req_failure"},
        request=httpx2.Request("POST", "https://api.openai.com/v1/responses"),
    )
    error = openai.APIStatusError(
        "secret-canary response body", response=raw, body={"key": "secret-canary"}
    )
    provider, calls, _ = provider_factory([error])
    with pytest.raises(ProviderError, match=code) as caught:
        provider.plan(PublicEvidence(), AgentBudget())
    assert "secret-canary" not in str(caught.value)
    assert len(calls) == 1
    assert provider.invocations()[-1].provider_request_id == "req_failure"
    assert provider.invocations()[-1].http_status == status
    assert "secret-canary" not in provider.invocations()[-1].model_dump_json()


def test_recovered_started_invocation_is_uncertain_and_not_replayed(provider_factory):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import Invocation, ProviderError
    from gpu_agent.contracts import now

    provider, calls, clients = provider_factory([])
    started = Invocation(
        invocation_id="a" * 32,
        run_id=provider.run_id,
        kind="plan",
        attempt=0,
        state="STARTED",
        started_at=now(),
        configured_model="configured",
        endpoint_host="api.openai.com",
        client_request_id="b" * 32,
    )
    provider.store.put(
        provider.run_id,
        f"provider/{started.invocation_id}/STARTED.json",
        started.model_dump_json().encode(),
        "public",
    )
    assert provider.invocations()[-1].state == "UNCERTAIN"
    with pytest.raises(ProviderError, match="LLM_UNCERTAIN_INVOCATION"):
        provider.plan(PublicEvidence(), AgentBudget())
    assert not clients and not calls


def test_usage_is_actual_and_model_difference_is_preserved(provider_factory):
    from gpu_agent.agent.models import (
        AgentActionOutput,
        AgentBudget,
        MemcheckAction,
        PublicEvidence,
    )

    usage = SimpleNamespace(
        input_tokens=7,
        output_tokens=11,
        total_tokens=18,
        input_tokens_details=SimpleNamespace(cached_tokens=3),
        output_tokens_details=SimpleNamespace(reasoning_tokens=2),
    )
    provider, _, _ = provider_factory(
        [response(AgentActionOutput(action=MemcheckAction()), usage=usage, model="actual")]
    )
    provider.plan(PublicEvidence(), AgentBudget())
    record = provider.invocations()[-1]
    assert record.usage.total_tokens == 18 and record.usage.cached_tokens == 3
    assert record.usage.cache_write_tokens is None and record.usage.reasoning_tokens == 2
    assert record.configured_model == "configured" and record.response_model == "actual"


def test_unproven_custom_endpoint_capability_does_not_send(provider_factory):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError

    provider, calls, clients = provider_factory([], endpoint="https://compatible.example/v1")
    with pytest.raises(ProviderError, match="LLM_CAPABILITY_UNAVAILABLE"):
        provider.plan(PublicEvidence(), AgentBudget())
    assert not clients and not calls


@pytest.mark.parametrize("diff", ["", "```diff\nx\n```", "--- a/private.cu\n+++ b/private.cu\n"])
def test_invalid_patch_format_is_retried_only_once(provider_factory, diff):
    from gpu_agent.agent.models import DiagnosisResult, PublicSource
    from gpu_agent.agent.provider import ProviderError

    provider, calls, _ = provider_factory(
        [response({"unified_diff": diff}), response({"unified_diff": diff})]
    )
    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        provider.propose_patch(
            PublicSource(source_id="a" * 32, content="int x;\n"),
            DiagnosisResult.inconclusive("TEST"),
        )
    assert len(calls) == 2 and provider.gate.snapshot().llm_calls == 2


def test_diff_scope_validator_runs_before_acceptance(provider_factory):
    from gpu_agent.agent.models import DiagnosisResult, PublicSource
    from gpu_agent.agent.provider import ProviderError

    diff = "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -1 +1 @@\n-context mismatch\n+int y;\n"
    provider, calls, _ = provider_factory(
        [response({"unified_diff": diff}), response({"unified_diff": diff})]
    )

    def invalid(_):
        raise ValueError("private host error details")

    provider._diff_validator = invalid
    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        provider.propose_patch(
            PublicSource(source_id="a" * 32, content="int x;\n"),
            DiagnosisResult.inconclusive("TEST"),
        )
    assert len(calls) == 2
    assert "private host error details" not in str(calls)


def test_real_sdk_offline_transport_sends_strict_schema_and_parses_result(store):
    import json

    import httpx2
    import openai

    from gpu_agent.agent.models import (
        AgentActionOutput,
        AgentBudget,
        MemcheckAction,
        PublicEvidence,
    )
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import OpenAIProviderSettings, OpenAIResponsesProvider

    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        payload = {
            "id": "resp_transport",
            "object": "response",
            "created_at": 123,
            "status": "completed",
            "model": "configured",
            "error": None,
            "incomplete_details": None,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": AgentActionOutput(action=MemcheckAction()).model_dump_json(),
                            "annotations": [],
                        }
                    ],
                }
            ],
        }
        return httpx2.Response(200, json=payload, headers={"x-request-id": "req_transport"})

    def factory(**kwargs):
        return openai.OpenAI(
            **kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
        )

    run = store.create_run("offline-sdk-transport")
    provider = OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.openai.com/v1",
            model="configured",
            api_key=SecretStr("offline-placeholder"),
        ),
        LLMCallGate(),
        store,
        run.id,
        client_factory=factory,
    )
    assert provider.plan(PublicEvidence(), AgentBudget()).action_type == "run_memcheck"
    assert len(requests) == 1
    schema = requests[0]["text"]["format"]["schema"]
    assert requests[0]["text"]["format"]["strict"] is True
    assert '"oneOf"' not in json.dumps(schema)
    assert '"discriminator"' not in json.dumps(schema)
    assert requests[0]["store"] is False
    assert provider.invocations()[-1].provider_request_id == "req_transport"


def test_sdk_format_failure_keeps_response_id_request_id_and_usage(store):
    import httpx2
    import openai

    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import (
        OpenAIProviderSettings,
        OpenAIResponsesProvider,
        ProviderError,
    )

    count = 0

    def handler(request):
        nonlocal count
        count += 1
        payload = {
            "id": f"resp_bad_{count}",
            "object": "response",
            "created_at": 123,
            "status": "completed",
            "model": "configured",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"action":{"action_type":"shell"}}',
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        return httpx2.Response(200, json=payload, headers={"x-request-id": f"req_bad_{count}"})

    def factory(**kwargs):
        return openai.OpenAI(
            **kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
        )

    run = store.create_run("offline-invalid-sdk-transport")
    provider = OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.openai.com/v1",
            model="configured",
            api_key=SecretStr("offline-placeholder"),
        ),
        LLMCallGate(),
        store,
        run.id,
        client_factory=factory,
    )
    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        provider.plan(PublicEvidence(), AgentBudget())
    assert count == 2
    assert [r.response_id for r in provider.invocations()] == ["resp_bad_1", "resp_bad_2"]
    assert [r.provider_request_id for r in provider.invocations()] == ["req_bad_1", "req_bad_2"]
    assert all(r.usage.total_tokens == 10 for r in provider.invocations())


def test_sdk_request_logging_is_disabled_even_when_host_enables_debug(
    provider_factory, caplog, monkeypatch
):
    import logging

    from gpu_agent.agent.models import (
        AgentActionOutput,
        AgentBudget,
        MemcheckAction,
        PublicEvidence,
    )

    provider, _, _ = provider_factory([response(AgentActionOutput(action=MemcheckAction()))])
    logger = logging.getLogger("openai")
    monkeypatch.setattr(logger, "disabled", False)
    original = provider._factory

    def noisy_factory(**kwargs):
        logger.debug("request body contains secret-canary")
        return original(**kwargs)

    provider._factory = noisy_factory
    with caplog.at_level(logging.DEBUG):
        provider.plan(PublicEvidence(), AgentBudget())
    assert "secret-canary" not in caplog.text
