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
