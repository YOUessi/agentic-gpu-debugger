"""Model-directed agent loop: any legal tool can follow any observation.

No hardcoded investigate -> diagnose -> patch graph. LangGraph only manages
a generic model/tool/stop cycle, while policy and GPU authority live in host.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from gpu_agent.agent.models import AgentBudget
from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.agent.provider import (
    DevelopmentCallPolicy,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
)
from gpu_agent.benchmark.hardset_v2_verifier import run_strict_hardset_verification
from gpu_agent.harness.hardset_tools import HardsetWorkspaceTools
from gpu_agent.harness.models import (
    FinishTask,
    HarnessDecision,
    HarnessOutcome,
    SessionSnapshot,
    ToolObservation,
)


class DecisionModel(Protocol):
    def choose_harness_v1(self, snapshot: SessionSnapshot) -> HarnessDecision: ...


class ToolHost(Protocol):
    @property
    def accepted_diff(self) -> str | None: ...

    @property
    def tool_calls(self) -> int: ...

    def snapshot(self, goal: str, history: list[dict[str, str]], steps: int) -> SessionSnapshot: ...

    def invoke(self, action: Any) -> ToolObservation: ...


class AgentLoopState(TypedDict, total=False):
    history: list[dict[str, str]]
    decision: dict[str, Any]
    observation: dict[str, Any]
    turns: int
    stop: bool
    reason: str


def run_tool_agent(
    model: DecisionModel,
    host: ToolHost,
    *,
    goal: str,
    max_steps: int = 16,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> AgentLoopState:
    """Model chooses all actions; policy gates tool execution and completion."""
    if type(max_steps) is not int or not 1 <= max_steps <= 40:
        raise ValueError("agent step budget must be between 1 and 40")
    if not goal or len(goal) > 1500:
        raise ValueError("invalid goal")

    def observe(state: AgentLoopState) -> dict[str, Any]:
        n = state.get("turns", 0)
        remaining = max_steps - n
        snapshot = host.snapshot(goal, state.get("history", []), remaining)
        decision = model.choose_harness_v1(snapshot)
        if not isinstance(decision, HarnessDecision):
            raise ValueError("model must return exactly one typed action")
        if on_event:
            on_event(
                {
                    "stage": "model",
                    "turn": n + 1,
                    "tool": decision.action.tool,
                    "reason": decision.reason,
                }
            )
        return {
            "turns": n + 1,
            "decision": decision.model_dump(mode="json"),
        }

    def tool(state: AgentLoopState) -> dict[str, Any]:
        decision = HarnessDecision.model_validate(state["decision"])
        obs = host.invoke(decision.action)
        entry = {
            "tool": obs.tool,
            "status": obs.status,
            "content": obs.content[:2000],
        }
        if on_event:
            on_event(
                {
                    "stage": "tool",
                    "turn": state["turns"],
                    "tool": obs.tool,
                    "status": obs.status,
                    "observation_sha256": hashlib.sha256(
                        obs.model_dump_json().encode()
                    ).hexdigest(),
                }
            )
        done = isinstance(decision.action, FinishTask) and obs.status == "OK"
        return {
            "history": [*state.get("history", []), entry][-30:],
            "observation": obs.model_dump(mode="json"),
            "stop": done or state["turns"] >= max_steps,
            "reason": (
                "PUBLIC_CANDIDATE_ACCEPTED"
                if done
                else "STEP_BUDGET_EXHAUSTED"
                if state["turns"] >= max_steps
                else "CONTINUE"
            ),
        }

    graph = StateGraph(AgentLoopState)
    graph.add_node("model", observe)
    graph.add_node("tool", tool)
    graph.add_edge(START, "model")
    graph.add_edge("model", "tool")
    graph.add_conditional_edges(
        "tool",
        lambda state: "stop" if state.get("stop") else "again",
        {"stop": END, "again": "model"},
    )
    return cast(
        AgentLoopState,
        graph.compile().invoke(
            {"history": [], "turns": 0, "stop": False},
            config={"recursion_limit": max_steps * 3 + 12},
        ),
    )


def run_gpu_coding_harness(
    repository: Path,
    output: Path,
    *,
    case_id: str,
    goal: str,
    evaluator_secret: bytes,
    max_steps: int = 16,
    max_model_calls: int = 24,
    max_probes: int = 6,
    max_public_checks: int = 3,
    allow_paid_calls: bool = False,
    model_factory: Callable[[HardsetWorkspaceTools], DecisionModel] | None = None,
) -> HarnessOutcome:
    """One real-session end-to-end runtime; no hidden tests become Agent tools.

    The model factory is test-only; production loads the existing authenticated
    provider. The independent verifier runs *after* the agent loop finishes.
    """
    if model_factory is None and not allow_paid_calls:
        raise ValueError("real model usage requires explicit approval")
    if not isinstance(evaluator_secret, bytes) or len(evaluator_secret) < 32:
        raise ValueError("operator-owned evaluator secret is missing")
    if type(max_model_calls) is not int or not 4 <= max_model_calls <= 40:
        raise ValueError("model-call budget is out of bounds")
    if output.exists() or output.is_symlink():
        raise ValueError("do not overwrite previous harness evidence")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    events: list[dict[str, Any]] = []
    with HardsetWorkspaceTools(
        repository,
        output / "session",
        case_id,
        max_probes=max_probes,
        max_checks=max_public_checks,
    ) as tools:
        if model_factory is None:
            provider = OpenAIResponsesProvider(
                OpenAIProviderSettings.from_environment(),
                LLMCallGate(
                    AgentBudget(
                        max_llm_calls=max_model_calls,
                        max_wall_time_seconds=600,
                    )
                ),
                tools._probe.public_store,
                tools.run_id,
                call_policy=DevelopmentCallPolicy(max_llm_calls=max_model_calls),
            )
            provider.ensure_available()
            model: DecisionModel = provider
        else:
            model = model_factory(tools)
        state = run_tool_agent(model, tools, goal=goal, max_steps=max_steps, on_event=events.append)
        diff = tools.accepted_diff
        candidate_sha = tools._candidate_hash if diff else None
        run_id = tools.run_id
        tool_calls = tools.tool_calls
        inv = getattr(model, "invocations", None)
        records = inv() if callable(inv) else []
        total_tokens = [
            i.usage.total_tokens
            for i in records
            if i.usage is not None and i.usage.total_tokens is not None
        ]
        tokens = sum(total_tokens) if len(total_tokens) == len(records) else None
        for i, event in enumerate(events, 1):
            tools._probe.public_store.put(
                run_id,
                f"harness/v1/trace-{i:03d}.json",
                json.dumps(event, ensure_ascii=False, sort_keys=True).encode(),
                "public",
            )
    reason = state.get("reason", "UNKNOWN")
    verdict: Literal["VERIFIED_FIXED", "NOT_VERIFIED", "INCONCLUSIVE"] = "INCONCLUSIVE"
    verified = False
    if diff:
        projection = run_strict_hardset_verification(
            repository,
            output / "strict",
            case_id=case_id,
            unified_diff=diff,
            evaluator_secret=evaluator_secret,
        )
        verdict = projection.verdict
        verified = verdict == "VERIFIED_FIXED"
        reason = projection.reason_code
    outcome = HarnessOutcome(
        case_id=case_id,
        status=verdict,
        model_tool_turns=state.get("turns", 0),
        candidate_sha256=candidate_sha,
        tool_calls=tool_calls,
        model_physical_calls=len(records),
        model_tokens=tokens,
        reason=reason,
        public_run_id=run_id,
        private_verification=verified,
    )
    (output / "result.json").write_text(outcome.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return outcome
