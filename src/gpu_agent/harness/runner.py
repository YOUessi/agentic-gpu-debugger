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
from gpu_agent.harness.journal import EventKind, SessionJournal
from gpu_agent.harness.models import (
    FinishTask,
    HarnessDecision,
    HarnessOutcome,
    SessionSnapshot,
    TaskPlan,
    TaskReflection,
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
    task_plan: dict[str, Any]
    task_progress: dict[str, Any]
    task_reflection: dict[str, Any] | None


def run_tool_agent(
    model: DecisionModel,
    host: ToolHost,
    *,
    goal: str,
    max_steps: int = 16,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    on_checkpoint: Callable[[AgentLoopState], None] | None = None,
    initial_state: AgentLoopState | None = None,
    pause_after_turns: int | None = None,
    hierarchical_mode: bool = False,
) -> AgentLoopState:
    """Model chooses all actions; policy gates tool execution and completion."""
    if type(max_steps) is not int or not 1 <= max_steps <= 40:
        raise ValueError("agent step budget must be between 1 and 40")
    if not goal or len(goal) > 1500:
        raise ValueError("invalid goal")
    if pause_after_turns is not None and (
        type(pause_after_turns) is not int or not 1 <= pause_after_turns < max_steps
    ):
        raise ValueError("pause boundary must precede total step budget")
    start_state: AgentLoopState = {"history": [], "turns": 0, "stop": False}
    if initial_state is not None:
        previous = initial_state.get("turns", -1)
        if type(previous) is not int or not 1 <= previous <= max_steps:
            raise ValueError("invalid resumed turn count")
        if len(initial_state.get("history", [])) != previous:
            raise ValueError("resumed history length is inconsistent")
        start_state = initial_state.copy()
        if start_state.get("reason") == "SESSION_PAUSED":
            start_state["stop"] = False
            start_state["reason"] = "CONTINUE"
        elif start_state.get("stop"):
            # A successful finish is never executed twice after a crash.
            return initial_state
        if previous == max_steps:
            raise ValueError("cannot exceed the persisted total step budget")

    def planner(state: AgentLoopState) -> dict[str, Any]:
        if not hierarchical_mode or state.get("task_plan") is not None:
            return {}
        from gpu_agent.harness.planning import progress_from_tools

        callback = getattr(model, "choose_repository_plan", None)
        if not callable(callback):
            raise ValueError("hierarchical model must implement global task planning")
        snapshot = host.snapshot(goal, state.get("history", []), max_steps)
        plan = callback(snapshot)
        if not isinstance(plan, TaskPlan):
            raise ValueError("global Planner returned an invalid task plan")
        progress = progress_from_tools(
            state.get("history", []),
            current_source_visible=snapshot.context_mode == "evidence_first",
        )
        return {
            "task_plan": plan.model_dump(mode="json"),
            "task_progress": progress.model_dump(mode="json"),
            "task_reflection": None,
        }

    def reflect(state: AgentLoopState) -> dict[str, Any]:
        from gpu_agent.harness.models import TaskProgress

        callback = getattr(model, "reflect_repository_task", None)
        if not callable(callback):
            raise ValueError("hierarchical model must support concise reflection")
        used = state.get("turns", 0)
        snapshot = host.snapshot(goal, state.get("history", []), max_steps - used)
        snapshot = snapshot.model_copy(
            update={
                "task_plan": TaskPlan.model_validate(state["task_plan"]),
                "task_progress": TaskProgress.model_validate(state["task_progress"]),
            }
        )
        result = callback(snapshot)
        if not isinstance(result, TaskReflection):
            raise ValueError("Reflect node returned an invalid correction")
        return {"task_reflection": result.model_dump(mode="json")}

    def observe(state: AgentLoopState) -> dict[str, Any]:
        n = state.get("turns", 0)
        remaining = max_steps - n
        snapshot = host.snapshot(goal, state.get("history", []), remaining)
        if hierarchical_mode:
            from gpu_agent.harness.models import TaskProgress
            from gpu_agent.harness.planning import progress_from_tools, reflection_context

            reflection = (
                TaskReflection.model_validate(state["task_reflection"])
                if state.get("task_reflection") is not None
                else None
            )
            progress = progress_from_tools(
                state.get("history", []),
                current_source_visible=snapshot.context_mode == "evidence_first",
            )
            recent = (
                reflection_context(snapshot.history, reflection=reflection)
                if reflection is not None
                else snapshot.history
            )
            snapshot = snapshot.model_copy(
                update={
                    "task_plan": TaskPlan.model_validate(state["task_plan"]),
                    "task_progress": TaskProgress.model_validate(progress),
                    "task_reflection": reflection,
                    "history": recent,
                }
            )
        if on_event:
            on_event({"stage": "model_started", "turn": n + 1})
        try:
            decision = model.choose_harness_v1(snapshot)
        except Exception as exc:
            if on_event:
                on_event(
                    {
                        "stage": "model_uncertain",
                        "turn": n + 1,
                        "error_type": type(exc).__name__,
                    }
                )
            raise
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

    def tool(state: AgentLoopState) -> AgentLoopState:
        decision = HarnessDecision.model_validate(state["decision"])
        if on_event:
            on_event(
                {
                    "stage": "tool_started",
                    "turn": state["turns"],
                    "tool": decision.action.tool,
                }
            )
        try:
            obs = host.invoke(decision.action)
        except Exception as exc:
            if on_event:
                on_event(
                    {
                        "stage": "tool_uncertain",
                        "turn": state["turns"],
                        "tool": decision.action.tool,
                        "error_type": type(exc).__name__,
                    }
                )
            raise
        entry = {
            "tool": obs.tool,
            "status": obs.status,
            "content": obs.content[:3900],
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
        completion_reason = (
            str(getattr(host, "completion_reason", "PUBLIC_CANDIDATE_ACCEPTED"))
            if done
            else "CONTINUE"
        )
        paused = (
            not done
            and pause_after_turns is not None
            and state["turns"] >= pause_after_turns
            and state["turns"] < max_steps
        )
        update: AgentLoopState = {
            "history": [*state.get("history", []), entry][-max_steps:],
            "observation": obs.model_dump(mode="json"),
            "stop": done or paused or state["turns"] >= max_steps,
            "reason": (
                completion_reason
                if done
                else "STEP_BUDGET_EXHAUSTED"
                if state["turns"] >= max_steps
                else "SESSION_PAUSED"
                if paused
                else "CONTINUE"
            ),
        }
        if hierarchical_mode:
            from gpu_agent.harness.planning import progress_from_tools

            progress = progress_from_tools(
                update["history"],
                current_source_visible=getattr(host, "_context_mode", "") == "evidence_first",
            )
            update["task_progress"] = progress.model_dump(mode="json")
        if on_checkpoint is not None:
            on_checkpoint({**state, **update})
        return update

    def needs_reflection(state: AgentLoopState) -> bool:
        return (
            hierarchical_mode
            and state.get("task_plan") is not None
            and state.get("task_reflection") is None
            and state.get("turns", 0) >= max(2, max_steps // 2)
            and not state.get("stop", False)
        )

    graph = StateGraph(AgentLoopState)
    graph.add_node("planner", planner)
    graph.add_node("reflect", reflect)
    graph.add_node("model", observe)
    graph.add_node("tool", tool)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        lambda state: "reflect" if needs_reflection(state) else "model",
        {"reflect": "reflect", "model": "model"},
    )
    graph.add_edge("reflect", "model")
    graph.add_edge("model", "tool")
    graph.add_conditional_edges(
        "tool",
        lambda state: (
            "stop" if state.get("stop") else "reflect" if needs_reflection(state) else "again"
        ),
        {"stop": END, "reflect": "reflect", "again": "model"},
    )
    return cast(
        AgentLoopState,
        graph.compile().invoke(
            start_state,
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
    max_steps: int = 24,
    max_model_calls: int = 32,
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
    journal = SessionJournal(output / "session-events.jsonl")
    try:
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
            journal.append(
                "session_started",
                {
                    "case_id": case_id,
                    "public_run_id": tools.run_id,
                    "source_sha256": tools.source_hash,
                    "goal_sha256": hashlib.sha256(goal.encode()).hexdigest(),
                },
            )

            def persist_event(event: dict[str, Any]) -> None:
                events.append(event)
                index = len(events)
                kind_map: dict[str, EventKind] = {
                    "model_started": "model_started",
                    "model": "model_selected",
                    "tool_started": "tool_started",
                    "tool": "tool_completed",
                }
                stage = str(event.get("stage", ""))
                kind = kind_map.get(stage)
                if kind is not None:
                    journal.append(
                        kind,
                        {
                            "turn": event.get("turn", 0),
                            "tool": str(event.get("tool", ""))[:64],
                            "status": str(event.get("status", ""))[:32],
                            "evidence_sha256": str(event.get("observation_sha256", ""))[:64],
                        },
                    )
                tools._probe.public_store.put(
                    tools.run_id,
                    f"harness/v2/trace-{index:03d}.json",
                    json.dumps(event, ensure_ascii=False, sort_keys=True).encode(),
                    "public",
                )

            state = run_tool_agent(
                model, tools, goal=goal, max_steps=max_steps, on_event=persist_event
            )
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
        (output / "result.json").write_text(
            outcome.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        journal.append(
            "session_finished",
            {
                "status": outcome.status,
                "turns": outcome.model_tool_turns,
                "candidate_sha256": outcome.candidate_sha256,
            },
        )
        return outcome
    except Exception as exc:
        journal.append("session_failed", {"error_type": type(exc).__name__})
        raise
    finally:
        journal.close()
