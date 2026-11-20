"""No-network contract tests for HardSet V2 audited planner transport."""

import json

import pytest
from pydantic import SecretStr, ValidationError

from gpu_agent.agent.models import AgentBudget, PlannerOutput
from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.agent.prompts import (
    HARDSET_V2_PROBE_PROMPT_VERSION,
    PROMPT_VERSION,
    select_prompt,
)
from gpu_agent.agent.provider import (
    WIRE_MODELS,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
    SDKResult,
)
from gpu_agent.benchmark.hardset_v2_graph import (
    FinishProbeAction,
    HardsetProbePlanOutput,
    PlannerSnapshot,
)
from gpu_agent.benchmark.hardset_v2_probe import FunctionalProbeAction
from gpu_agent.store import RunStore


class MockPort:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def call(self, request):
        self.calls.append(request)
        index = min(len(self.calls) - 1, len(self.values) - 1)
        return SDKResult(value=self.values[index])


def provider(tmp_path, port):
    store = RunStore(tmp_path / "public")
    run = store.create_run("hardset_v2_test_planner")
    agent = OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.deepseek.com",
            model="test-model",
            api_key=SecretStr("not-a-real-secret"),
            supports_store_false=True,
        ),
        LLMCallGate(AgentBudget(max_llm_calls=8)),
        store,
        run.id,
        port=port,
    )
    return agent, store, run.id


def snapshot():
    return PlannerSnapshot(
        case_id="h001",
        remaining_probes=3,
        observations=(),
        public_source="__global__ void foo() { /* expf(x) */ }",
        functional_requirement="stable group-wise softmax",
        available_sizes=(1, 32, 33, 65),
    )


def test_hardset_model_uses_existing_budgeted_and_audited_worker(tmp_path):
    port = MockPort([{"action": {"action_type": "run_functional_probe", "size": 33}}])
    agent, store, run_id = provider(tmp_path, port)
    result = agent.plan_hardset_v2(snapshot())
    assert result == FunctionalProbeAction(size=33)
    assert len(port.calls) == 1
    request = port.calls[0]
    assert request.kind == "plan"
    assert request.payload["hardset_probe_contract"] == "hardset-v2"
    assert request.payload["snapshot"]["functional_requirement"] == "stable group-wise softmax"
    assert "api_key" not in json.dumps(request.payload)
    inv = agent.invocations()
    assert len(inv) == 1
    assert inv[0].kind == "plan"
    assert inv[0].prompt_version == HARDSET_V2_PROBE_PROMPT_VERSION
    assert inv[0].state == "COMPLETED"
    refs = {r.name for r in store.load(run_id).artifact_refs}
    assert any(x.startswith("provider/") for x in refs)


def test_hardset_plan_can_finish_without_shell_or_private_inputs(tmp_path):
    port = MockPort([{"action": {"action_type": "finish_probe"}}])
    agent, _, _ = provider(tmp_path, port)
    assert agent.plan_hardset_v2(snapshot()) == FinishProbeAction()
    assert len(port.calls) == 1


def test_invalid_hardset_wire_action_never_executes_a_tool(tmp_path):
    from gpu_agent.agent.models import ProviderError

    port = MockPort([{"action": {"action_type": "run_shell", "command": "cat /tmp/secret"}}])
    agent, _, _ = provider(tmp_path, port)
    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        agent.plan_hardset_v2(snapshot())
    # Only the pre-existing bounded single format-retry is permitted.
    assert len(port.calls) <= 2


def test_wire_model_and_trusted_prompt_are_opt_in():
    assert WIRE_MODELS["plan"] is PlannerOutput
    assert select_prompt("plan", {})[0] == PROMPT_VERSION
    version, instructions = select_prompt(
        "plan",
        {"hardset_probe_contract": "hardset-v2", "snapshot": snapshot().model_dump()},
    )
    assert version == HARDSET_V2_PROBE_PROMPT_VERSION
    assert "run_functional_probe" in instructions
    assert "shell" in instructions.lower()
    assert "ground truth" not in instructions.lower()
    from openai.lib._pydantic import to_strict_json_schema

    schema = to_strict_json_schema(HardsetProbePlanOutput)
    assert "action" in schema["properties"]


def test_hardset_wire_response_denies_arbitrary_size_and_extra_keys():
    with pytest.raises(ValidationError):
        HardsetProbePlanOutput.model_validate(
            {"action": {"action_type": "run_functional_probe", "size": 514}}
        )
    with pytest.raises(ValidationError):
        HardsetProbePlanOutput.model_validate(
            {
                "action": {
                    "action_type": "run_functional_probe",
                    "size": 33,
                    "execute_command": "cat private oracle",
                }
            }
        )
