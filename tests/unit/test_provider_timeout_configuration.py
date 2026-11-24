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


def test_repository_roles_have_independent_durable_format_retry_quotas(tmp_path):
    """The real provider call loop runs against a local deterministic port."""
    from gpu_agent.agent.models import AgentBudget, ProviderError
    from gpu_agent.agent.policy import RepositoryCallBudget
    from gpu_agent.agent.provider import (
        DevelopmentCallPolicy,
        OpenAIResponsesProvider,
        SDKResult,
    )
    from gpu_agent.harness.models import SessionSnapshot
    from gpu_agent.store import RunStore

    budget = RepositoryCallBudget.from_session(
        max_steps=3, max_model_calls=9, hierarchical=True
    )
    assert budget.executor_format_retries == 2
    assert budget.total == 9

    class Port:
        def call(self, request):
            if request.attempt == 0:
                return SDKResult(error_code="LLM_INVALID_OUTPUT", state="FAILED")
            if request.payload.get("gpu_repo_task_plan"):
                return SDKResult(value={
                    "milestones": [
                        {"gate": gate, "objective": f"Execute {gate} with evidence",
                         "success_evidence": "Recorded controller evidence"}
                        for gate in ("inspect", "edit", "public_check", "finish")
                    ],
                    "failure_risk": "Avoid unsupported tool arguments",
                })
            if request.payload.get("gpu_repo_reflection"):
                return SDKResult(value={
                    "observed_failure": "No further observed failure",
                    "avoid_next": "Avoid repeated source reads",
                    "next_priority": "Proceed to the next allowed action",
                })
            return SDKResult(value={
                "action": {"tool": "repo_read_file", "path": "kernel.cu"},
                "reason": "Read public source",
            })

    store = RunStore(tmp_path / "public")
    run = store.create_run("gpu_repository_agent")
    store.transition(run.id, "RUNNING", "DIAGNOSING")
    settings = OpenAIProviderSettings(
        endpoint="https://api.openai.com/v1", model="test-model", api_key="fixture"
    )
    snapshot = SessionSnapshot(
        goal="Read public source", case_id="repository", source_hash="a" * 64,
        functional_requirement="Inspect the authorized public source",
        allowed_tools=("repo_read_file",), remaining_steps=3, history=(),
    )

    def provider():
        return OpenAIResponsesProvider(
            settings, LLMCallGate(AgentBudget(max_llm_calls=9), reserve_repair_stages=False),
            store, run.id, port=Port(),
            call_policy=DevelopmentCallPolicy(max_llm_calls=9), repository_budget=budget,
        )

    original = provider()
    original.choose_repository_plan(snapshot)
    original.choose_harness_v1(snapshot)
    # A resumed provider must account for old physical calls and format retries.
    resumed = provider()
    resumed.reflect_repository_task(snapshot)
    resumed.choose_harness_v1(snapshot)
    records = resumed.invocations()
    assert len(records) == 8
    assert sum(r.repository_role == "executor" and r.attempt == 1 for r in records) == 2
    assert sum(r.repository_role == "planner" and r.attempt == 1 for r in records) == 1
    assert sum(r.repository_role == "reflect" and r.attempt == 1 for r in records) == 1
    with pytest.raises(ProviderError, match="LLM_REPOSITORY_ROLE_BUDGET_EXHAUSTED"):
        resumed.choose_repository_plan(snapshot)
    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED|LLM_REPOSITORY_ROLE"):
        resumed.choose_harness_v1(snapshot)
    assert len(resumed.invocations()) == 9


def test_repository_budget_requires_auxiliary_reserve_and_preserves_legacy_retry_gate():
    from gpu_agent.agent.models import AgentBudget, ProviderError
    from gpu_agent.agent.policy import RepositoryCallBudget

    with pytest.raises(ValueError, match="reserve two executor retries"):
        RepositoryCallBudget.from_session(max_steps=12, max_model_calls=14, hierarchical=True)
    react = RepositoryCallBudget.from_session(max_steps=12, max_model_calls=14, hierarchical=False)
    planner = RepositoryCallBudget.from_session(max_steps=12, max_model_calls=18, hierarchical=True)
    assert react.executor_decisions == planner.executor_decisions == 12
    assert react.executor_format_retries == planner.executor_format_retries == 2
    old_gate = LLMCallGate(AgentBudget(max_llm_calls=10))
    old_gate.reserve("plan", attempt=1)
    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        old_gate.reserve("plan", attempt=1)
