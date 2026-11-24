"""GPU coding harness: real model-tool-model loop contracts, no GPU/API calls."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from gpu_agent.agent.models import AgentBudget
from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.agent.prompts import GPU_CODING_HARNESS_V1_PROMPT_VERSION, select_prompt
from gpu_agent.agent.provider import (
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
    SDKResult,
)
from gpu_agent.harness.hardset_tools import HardsetWorkspaceTools
from gpu_agent.harness.models import (
    ApplyPatch,
    FinishTask,
    HarnessDecision,
    ReadFile,
    RunPublicChecks,
    SessionSnapshot,
    ToolObservation,
)
from gpu_agent.harness.runner import run_tool_agent
from gpu_agent.store import RunStore

REPO = Path(__file__).resolve().parents[2]
SHA = "a" * 64


class FakeHost:
    def __init__(self):
        self.calls = []
        self.approved = False

    def snapshot(self, goal, history, steps):
        return SessionSnapshot(
            goal=goal,
            case_id="h001",
            source_hash=SHA,
            functional_requirement="Stable group-wise Softmax for every length",
            allowed_tools=(
                "read_file",
                "apply_patch",
                "run_public_checks",
                "finish",
            ),
            remaining_steps=steps,
            history=tuple(history),
            public_checks_passed=self.approved,
        )

    def invoke(self, action):
        self.calls.append(action.tool)
        if isinstance(action, ReadFile):
            return ToolObservation(tool=action.tool, status="OK", content="1 out[i] = expf(x);")
        if isinstance(action, ApplyPatch):
            return ToolObservation(tool=action.tool, status="OK", content="Staged candidate")
        if isinstance(action, RunPublicChecks):
            self.approved = True
            return ToolObservation(tool=action.tool, status="OK", content="PUBLIC_CHECKS_PASSED")
        if isinstance(action, FinishTask):
            return ToolObservation(
                tool=action.tool,
                status="OK" if self.approved else "REJECTED",
                content="Complete" if self.approved else "Must run GPU checks first",
            )
        raise ValueError("not allowed")


class DecisionSequence:
    def __init__(self, actions):
        self.actions = actions
        self.turns = 0
        self.snapshots = []

    def choose_harness_v1(self, snapshot):
        self.snapshots.append(snapshot)
        idx = min(self.turns, len(self.actions) - 1)
        self.turns += 1
        return HarnessDecision.model_validate(
            {"action": self.actions[idx], "reason": "Choose the next useful action"}
        )


def test_agent_can_choose_tools_in_any_order_and_follow_feedback():
    host = FakeHost()
    model = DecisionSequence(
        [
            {"tool": "read_file", "path": "kernel.cu"},
            {"tool": "finish", "summary": "The patch is complete"},
            {
                "tool": "apply_patch",
                "unified_diff": "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -1 +1 @@\n-a\n+b\n",
            },
            {"tool": "run_public_checks"},
            {"tool": "finish", "summary": "Public tests now pass"},
        ]
    )
    result = run_tool_agent(model, host, goal="Fix unstable Softmax32", max_steps=8)
    assert result["reason"] == "PUBLIC_CANDIDATE_ACCEPTED"
    assert result["turns"] == 5
    assert host.calls == [
        "read_file",
        "finish",
        "apply_patch",
        "run_public_checks",
        "finish",
    ]
    assert result["history"][1]["status"] == "REJECTED"
    assert model.snapshots[2].history[-1]["content"] == "Must run GPU checks first"


def test_model_cannot_force_completion_without_public_verification():
    host = FakeHost()
    model = DecisionSequence([{"tool": "finish", "summary": "Complete without testing"}])
    result = run_tool_agent(model, host, goal="Fix GPU kernel", max_steps=3)
    assert result["reason"] == "STEP_BUDGET_EXHAUSTED"
    assert result["turns"] == 3
    assert not host.approved


@pytest.mark.parametrize(
    "payload",
    [
        {"action": {"tool": "shell", "command": "cat /etc/passwd"}, "reason": "try"},
        {"action": {"tool": "read_file", "path": "../../.env"}, "reason": "try"},
        {"action": {"tool": "functional_probe", "size": 514}, "reason": "try"},
        {"action": {"tool": "finish", "summary": "ok"}, "reason": "try"},
        {"action": {"tool": "run_public_checks", "command": "rm -rf ~"}, "reason": "try"},
    ],
)
def test_untrusted_model_cannot_expand_tool_capabilities(payload):
    with pytest.raises(ValidationError):
        HarnessDecision.model_validate(payload)


def test_hardset_host_has_no_unrestricted_fs_or_terminal_capability(tmp_path):
    host = HardsetWorkspaceTools(REPO, tmp_path / "workspace", "h001")
    assert "shell" not in host.ALLOWED_TOOLS
    assert "run_command" not in host.ALLOWED_TOOLS
    with pytest.raises(ValueError, match="read permission denied"):
        host._file_bytes("../../private.json")
    with pytest.raises(ValueError, match="RunStore has not been initialized"):
        host._execute(FinishTask(summary="Everything is complete"))


class MockPort:
    def __init__(self):
        self.requests = []

    def call(self, request):
        self.requests.append(request)
        return SDKResult(
            value={
                "action": {"tool": "read_file", "path": "kernel.cu"},
                "reason": "Read source before choosing a debugging action",
            }
        )


def test_harness_uses_existing_audited_model_provider(tmp_path):
    store = RunStore(tmp_path / "store")
    run = store.create_run("harness_model_contract_test")
    port = MockPort()
    provider = OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.deepseek.com",
            model="mock-model",
            api_key=SecretStr("fake-secret"),
            supports_store_false=True,
        ),
        LLMCallGate(AgentBudget(max_llm_calls=8)),
        store,
        run.id,
        port=port,
    )
    state = FakeHost().snapshot("Fix CUDA kernel", [], 6)
    decision = provider.choose_harness_v1(state)
    assert isinstance(decision.action, ReadFile)
    assert port.requests[0].payload["gpu_coding_harness"] == "v1"
    assert port.requests[0].kind == "plan"
    assert provider.invocations()[0].prompt_version == GPU_CODING_HARNESS_V1_PROMPT_VERSION
    assert select_prompt("plan", {})[0] != GPU_CODING_HARNESS_V1_PROMPT_VERSION
    assert len(provider.invocations()) == 1
