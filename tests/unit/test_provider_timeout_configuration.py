"""Request deadlines are configurable, but never escape the run deadline."""

import pytest
from pydantic import ValidationError

from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.agent.provider import OpenAIProviderSettings, WorkerRequest


def test_environment_timeout_and_gate_do_not_reintroduce_sixty_second_cap(monkeypatch):
    monkeypatch.setenv("GPU_AGENT_LLM_TIMEOUT_SECONDS", "120")
    settings = OpenAIProviderSettings.from_environment()
    clock = [0.0]
    gate = LLMCallGate(clock=lambda: clock[0])
    assert min(settings.timeout_seconds, gate.reserve("plan")) == 120
    clock[0] = 580.0
    assert min(settings.timeout_seconds, gate.reserve("plan")) == 20
    assert gate.timeout(120) == 20


@pytest.mark.parametrize("value", ["0", "-1", "601", "nan", "inf", "not-a-number"])
def test_invalid_environment_timeout_is_rejected(monkeypatch, value):
    monkeypatch.setenv("GPU_AGENT_LLM_TIMEOUT_SECONDS", value)
    with pytest.raises((ValidationError, ValueError)):
        OpenAIProviderSettings.from_environment()


def test_default_timeout_is_preserved_and_worker_accepts_explicit_longer_deadline(monkeypatch):
    monkeypatch.delenv("GPU_AGENT_LLM_TIMEOUT_SECONDS", raising=False)
    assert OpenAIProviderSettings.from_environment().timeout_seconds == 60
    request = WorkerRequest(
        endpoint="https://api.deepseek.com",
        model="test",
        api_key="test",
        kind="plan",
        payload={},
        client_request_id="a" * 32,
        timeout_seconds=120,
        attempt=0,
    )
    assert request.timeout_seconds == 120



def test_local_model_http_requires_literal_loopback_and_opt_in(monkeypatch) -> None:
    from pydantic import ValidationError

    from gpu_agent.agent.provider import OpenAIProviderSettings

    local = "http://127.0.0.1:18731/v1"
    with pytest.raises(ValidationError, match="explicit operator opt-in"):
        OpenAIProviderSettings(endpoint=local, model="qwen3-8b-awq")
    valid = OpenAIProviderSettings(
        endpoint=local,
        model="qwen3-8b-awq",
        allow_loopback_http=True,
        supports_store_false=True,
    )
    assert valid.endpoint == local
    for forbidden in (
        "http://localhost:18731/v1",
        "http://0.0.0.0:18731/v1",
        "http://192.168.1.8:18731/v1",
        "http://127.0.0.1:18731/other",
        "http://127.0.0.1/v1",
        "http://root:password@127.0.0.1:18731/v1",
        "http://127.0.0.1:18731/v1?token=bad",
    ):
        with pytest.raises(ValidationError):
            OpenAIProviderSettings(
                endpoint=forbidden,
                model="qwen3-8b-awq",
                allow_loopback_http=True,
            )
    monkeypatch.setenv("OPENAI_BASE_URL", local)
    monkeypatch.setenv("OPENAI_MODEL", "qwen3-8b-awq")
    monkeypatch.setenv("OPENAI_API_KEY", "local-not-a-real-api-key")
    monkeypatch.setenv("GPU_AGENT_STORE_FALSE_SUPPORTED", "1")
    monkeypatch.delenv("GPU_AGENT_ALLOW_LOOPBACK_HTTP", raising=False)
    with pytest.raises(ValidationError):
        OpenAIProviderSettings.from_environment()
    monkeypatch.setenv("GPU_AGENT_ALLOW_LOOPBACK_HTTP", "1")
    actual = OpenAIProviderSettings.from_environment()
    assert actual.allow_loopback_http
    assert actual.endpoint == local
