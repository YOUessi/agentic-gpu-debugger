"""Graph-native repository Coding Agent with durable LangGraph execution.

LangGraph owns routing, task DAGs, evidence state, budgets and pause/resume.
RepositoryWorkspaceTools and the existing provider retain authority over
physical effects. Never replay an admitted but uncertain external effect.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from gpu_agent.harness.graph_plan import (
    TOOL_GATES,
    GraphTaskPlan,
    compatibility_plan,
    compatibility_progress,
    evaluate_tasks,
)
from gpu_agent.harness.journal import (
    SessionJournal,
    read_verified_journal,
    recovery_disposition,
)
from gpu_agent.harness.models import HarnessDecision, SessionSnapshot, ToolObservation
from gpu_agent.harness.runner import AgentLoopState

VERSION = "repository-langgraph-v2"
DB_NAME = "langgraph-checkpoints.sqlite3"
HEADER_NAME = "langgraph-runtime.json"
EFFECT_NAME = "langgraph-effects.jsonl"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def merge_events(old: list[dict], new: list[dict]) -> list[dict]:
    """Idempotent reducer: subgraph output may include already merged events."""
    by_id = {row["id"]: row for row in old}
    for row in new:
        if row["id"] in by_id and by_id[row["id"]] != row:
            raise ValueError("graph event identity collision")
        by_id[row["id"]] = row
    return list(by_id.values())


class NativeState(TypedDict, total=False):
    identity: dict[str, Any]
    deadline: float
    max_steps: int
    max_model_calls: int
    max_replans: int
    hierarchical: bool
    history: list[dict[str, str]]
    events: Annotated[list[dict[str, Any]], merge_events]
    evidence: list[dict[str, Any]]
    snapshot: dict[str, Any]
    decision: dict[str, Any]
    observation: dict[str, Any]
    turns: int
    logical_model_calls: int
    observed_physical_calls: int
    effects_completed: int
    plan: dict[str, Any]
    plan_revision: int
    plan_revisions: list[dict[str, Any]]
    task_status: list[dict[str, Any]]
    task_progress: dict[str, Any]
    last_failure: dict[str, Any] | None
    failure_streak: int
    last_plan_turn: int
    edited_files: list[str]
    finish_accepted: bool
    stop: bool
    reason: str
    pause_requested: bool
    reviewed_candidate: str | None
    review_kind: str


def max_replans_for(max_steps: int, max_model_calls: int, hierarchical: bool) -> int:
    if not hierarchical:
        return 0
    # Reserve one planner decision+format retry, two executor format retries,
    # and two physical admissions per replan. Unused slots are not reallocated.
    slots = max_model_calls - max_steps - 4
    if slots < 2:
        raise ValueError("graph planning requires at least six auxiliary/retry call slots")
    return min(3, slots // 2)


def _private_file(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("graph persistence cannot use a symlink")
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("graph persistence must be a private regular file")


def _new_private(path: Path, content: bytes = b"") -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_native_header(output: Path) -> dict[str, Any]:
    p = output / HEADER_NAME
    _private_file(p)
    value = json.loads(p.read_bytes())
    if value.get("version") != VERSION:
        raise ValueError("native runtime version mismatch")
    return value


class EffectJournal:
    """Separate write-ahead journal for every planner/executor/tool effect.

    The native checkpointer handles graph durability, not exactly-once external
    I/O. A graph may be behind this ledger after a crash. Such a graph is not
    replayable. Even a completed logical effect is not automatically repeated.
    """

    def __init__(self, path: Path, run_id: str, *, resume: bool):
        self.keys: set[str] = set()
        self.completed = 0
        if resume:
            _private_file(path)
            records = read_verified_journal(path)
            if (
                records[0]["payload"].get("run_id") != run_id
                or recovery_disposition(records) != "QUIESCENT_INSPECTION_ONLY"
            ):
                raise ValueError("graph has uncertain effects or mismatched identity")
            pending: tuple[str, int, str] | None = None
            for row in records[1:]:
                data = row["payload"]
                key = data["effect_key"]
                index = data["turn"]
                if row["kind"] in ("model_started", "tool_started"):
                    if pending is not None or key in self.keys or index != self.completed + 1:
                        raise ValueError("invalid graph effect admission")
                    self.keys.add(key)
                    pending = (key, index, row["kind"])
                else:
                    expected = (
                        "model_started" if row["kind"] == "model_selected"
                        else "tool_started" if row["kind"] == "tool_completed"
                        else ""
                    )
                    if pending != (key, index, expected) or not data.get("result_sha256"):
                        raise ValueError("invalid graph effect completion")
                    self.completed += 1
                    pending = None
            if pending is not None:
                raise ValueError("graph contains an uncertain effect")
            self.journal = SessionJournal.reopen(path, expected_head=records[-1]["sha256"])
        else:
            self.journal = SessionJournal(path)
            self.journal.append("session_started", {"run_id": run_id, "version": VERSION})

    def perform(self, key: str, role: str, operation: Callable[[], Any]) -> Any:
        if key in self.keys:
            raise ValueError("refusing to replay an admitted graph effect")
        index = self.completed + 1
        model = role != "tool"
        self.journal.append(
            "model_started" if model else "tool_started",
            {"effect_key": key, "turn": index, "role": role},
        )
        self.keys.add(key)
        value = operation()
        result = value.model_dump(mode="json")
        self.journal.append(
            "model_selected" if model else "tool_completed",
            {"effect_key": key, "turn": index, "role": role, "result_sha256": digest(result)},
        )
        self.completed += 1
        return value

    def close(self) -> None:
        self.journal.close()


def project_state(state: NativeState) -> AgentLoopState:
    """Compatibility view used solely to validate/reopen external workspace."""
    view: AgentLoopState = {
        "history": state.get("history", []),
        "decision": state["decision"],
        "observation": state["observation"],
        "turns": state["turns"],
        "stop": state.get("stop", False),
        "reason": state.get("reason", "CONTINUE"),
    }
    if state.get("hierarchical"):
        view.update(
            task_plan=compatibility_plan(GraphTaskPlan.model_validate(state["plan"])).model_dump(mode="json"),
            task_progress=state["task_progress"],
            task_reflection=None,
        )
    return view


def _event(state: NativeState, name: str, **details: Any) -> list[dict]:
    return [{
        "id": f"{name}:{state.get('turns', 0)}:{state.get('plan_revision', 0)}",
        "kind": name, **details,
    }]


def _physical_calls(model: Any, fallback: int) -> int:
    invocations = getattr(model, "invocations", None)
    return len(invocations()) if callable(invocations) else fallback


def _failure(observation: ToolObservation, turn: int) -> dict[str, Any] | None:
    if observation.status == "OK":
        return None
    codes = re.findall(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b", observation.content)
    return {
        "turn": turn,
        "tool": observation.tool,
        "status": observation.status,
        "code": codes[0][:100] if codes else observation.status,
        "evidence": observation.content[:700],
    }


def build_native_graph(
    *,
    model: Any,
    host: Any,
    goal: str,
    ledger: EffectJournal,
    on_event: Callable[[dict[str, Any]], None],
    on_checkpoint: Callable[[AgentLoopState], None],
    pause_after_turns: int | None,
    review_public_pass: bool,
    checkpointer: Any,
):
    """Compile parent planning/recovery graph and per-turn Executor subgraph."""
    def current_snapshot(state: NativeState) -> dict[str, Any]:
        snapshot = host.snapshot(
            goal, state.get("history", []), max(0, state["max_steps"] - state.get("turns", 0))
        )
        return snapshot.model_dump(mode="json")

    def refresh(state: NativeState) -> dict[str, Any]:
        snap = current_snapshot(state)
        edited = list(host.workspace.edited_files)
        evidence = state.get("evidence", [])
        progress = compatibility_progress(snap, evidence, edited, state.get("last_failure"))
        tasks = (
            evaluate_tasks(GraphTaskPlan.model_validate(state["plan"]), evidence, snap, edited)
            if state.get("plan") else []
        )
        return {
            "snapshot": snap,
            "edited_files": edited,
            "task_progress": progress.model_dump(mode="json"),
            "task_status": tasks,
            "observed_physical_calls": _physical_calls(model, state.get("logical_model_calls", 0)),
        }

    def model_snapshot(state: NativeState) -> SessionSnapshot:
        data = dict(state["snapshot"])
        if state.get("plan"):
            plan = GraphTaskPlan.model_validate(state["plan"])
            data["task_plan"] = compatibility_plan(plan).model_dump(mode="json")
            data["task_progress"] = state["task_progress"]
            ready = [x for x in state["task_status"] if x["status"] in ("READY", "STALE")]
            objective_by_id = {t.task_id: t.objective for t in plan.tasks}
            priorities = [
                {**x, "objective": objective_by_id[x["task_id"]]} for x in ready
            ]
            graph_context = json.dumps({
                "plan_revision": state["plan_revision"],
                "ready_or_stale_tasks": priorities,
                "task_status": state["task_status"],
                "last_failure": state.get("last_failure"),
                "revision_reason": plan.revision_reason,
            }, ensure_ascii=False)
            # Preserve the most relevant previous controller guidance. The full
            # evidence and plan stay in persisted graph state, not prompt text.
            data["context_summary"] = (
                "GRAPH_PLAN (advisory; allowed_tools remains authoritative):\n"
                + graph_context[:1700] + "\n" + data.get("context_summary", "")
            )[:2800]
        return SessionSnapshot.model_validate_json(json.dumps(data))

    def plan_node(state: NativeState) -> dict[str, Any]:
        old = state.get("plan")
        revision = state.get("plan_revision", 0) + 1
        callback = getattr(model, "choose_graph_plan", None)
        if not callable(callback):
            raise ValueError("graph planner requires typed choose_graph_plan")
        role = "reflect" if old else "planner"
        plan = ledger.perform(
            f"plan-{revision}", role,
            lambda: callback(
                model_snapshot(state), previous_plan=old,
                task_status=state.get("task_status", []),
                failure=state.get("last_failure"),
            ),
        )
        if not isinstance(plan, GraphTaskPlan):
            raise ValueError("graph planner returned an invalid DAG")
        plan.validate_scope(state["snapshot"])
        value = plan.model_dump(mode="json")
        calls = state.get("logical_model_calls", 0) + 1
        return {
            "plan": value, "plan_revision": revision,
            "plan_revisions": [
                *state.get("plan_revisions", []),
                {"revision": revision, "turn": state.get("turns", 0),
                 "plan": value, "trigger": state.get("last_failure")},
            ],
            "last_plan_turn": state.get("turns", 0),
            "logical_model_calls": calls, "observed_physical_calls": _physical_calls(model, calls),
            "effects_completed": ledger.completed,
            "events": _event(state, f"{role}_completed", new_revision=revision),
        }

    def gate(state: NativeState) -> dict[str, Any]:
        reason = (
            "GRAPH_DEADLINE_EXHAUSTED" if time.time() >= state["deadline"]
            else "STEP_BUDGET_EXHAUSTED" if state["turns"] >= state["max_steps"]
            else "GRAPH_MODEL_BUDGET_EXHAUSTED" if state["observed_physical_calls"] >= state["max_model_calls"]
            else None
        )
        if reason:
            return {"stop": True, "reason": reason}
        return {}

    def route_gate(state: NativeState) -> str:
        if state.get("stop"):
            return "terminal"
        if state["hierarchical"] and not state.get("plan"):
            return "planner"
        if (
            state["hierarchical"]
            and state.get("last_failure") is not None
            and state["turns"] > state.get("last_plan_turn", 0)
            and state.get("plan_revision", 0) <= state["max_replans"]
            and state["max_steps"] - state["turns"] >= 2
            and state["max_model_calls"] - state["observed_physical_calls"] >= 2
        ):
            return "replan"
        return "executor"

    def choose(state: NativeState) -> dict[str, Any]:
        if time.time() >= state["deadline"]:
            return {"stop": True, "reason": "GRAPH_DEADLINE_EXHAUSTED"}
        turn = state["turns"] + 1
        on_event({"stage": "model_started", "turn": turn})
        decision = ledger.perform(
            f"decision-{turn}", "executor", lambda: model.choose_harness_v1(model_snapshot(state))
        )
        if not isinstance(decision, HarnessDecision):
            raise ValueError("executor must return one typed HarnessDecision")
        on_event({"stage": "model", "turn": turn, "tool": decision.action.tool, "reason": decision.reason})
        calls = state.get("logical_model_calls", 0) + 1
        return {
            "decision": decision.model_dump(mode="json"),
            "logical_model_calls": calls,
            "observed_physical_calls": _physical_calls(model, calls),
            "effects_completed": ledger.completed,
            "events": _event(state, "decision", tool=decision.action.tool, decision_turn=turn),
        }

    def dispatch(state: NativeState) -> str:
        if state.get("stop"):
            return "halt"
        # An unknown/unauthorized action still goes through the existing host's
        # denial and audit path. Dispatch never grants a tool capability.
        name = state["decision"]["action"]["tool"]
        return TOOL_GATES.get(name, "denied")

    def perform_tool(state: NativeState) -> dict[str, Any]:
        if time.time() >= state["deadline"]:
            # The model was admitted but no new external tool may start after
            # expiration. This terminal incomplete turn is not resumable.
            return {"stop": True, "reason": "GRAPH_DEADLINE_EXHAUSTED"}
        turn = state["turns"] + 1
        decision = HarnessDecision.model_validate(state["decision"])
        on_event({"stage": "tool_started", "turn": turn, "tool": decision.action.tool})
        observation = ledger.perform(f"tool-{turn}", "tool", lambda: host.invoke(decision.action))
        if not isinstance(observation, ToolObservation):
            raise ValueError("tool host returned an invalid observation")
        if observation.tool != decision.action.tool:
            raise ValueError("tool observation does not bind to the decision")
        on_event({
            "stage": "tool", "turn": turn, "tool": observation.tool, "status": observation.status,
            "observation_sha256": hashlib.sha256(observation.model_dump_json().encode()).hexdigest(),
        })
        history = [
            *state.get("history", []),
            {"tool": observation.tool, "status": observation.status, "content": observation.content[:3900]},
        ]
        after = host.snapshot(goal, history, max(0, state["max_steps"] - turn)).model_dump(mode="json")
        failure = _failure(observation, turn)
        path = getattr(decision.action, "path", None) or after.get("registered_kernel_path") or ""
        evidence = {
            "turn": turn, "tool": observation.tool, "gate": TOOL_GATES.get(observation.tool, "denied"),
            "path": path, "status": observation.status,
            "candidate_before": state["snapshot"].get("current_candidate_sha256"),
            "candidate": after.get("current_candidate_sha256"),
            "artifact_id": observation.artifact_id,
            "observation_sha256": digest(observation.model_dump(mode="json")),
        }
        return {
            "turns": turn, "history": history, "observation": observation.model_dump(mode="json"),
            "evidence": [*state.get("evidence", []), evidence], "snapshot": after,
            "last_failure": failure,
            "failure_streak": state.get("failure_streak", 0) + 1 if failure else 0,
            "finish_accepted": observation.tool == "finish" and observation.status == "OK",
            "effects_completed": ledger.completed,
            "events": _event(state, "tool_observed", evidence=evidence, failure=failure),
        }

    executor = StateGraph(NativeState)
    executor.add_node("choose", choose)
    executor.add_edge(START, "choose")
    tool_nodes = ("inspect", "edit", "compile", "probe", "public_check", "revert", "finish", "denied")
    for node in tool_nodes:
        executor.add_node(node, perform_tool)
        executor.add_edge(node, END)
    executor.add_conditional_edges("choose", dispatch, {**{n: n for n in tool_nodes}, "halt": END})

    def save_boundary(state: NativeState) -> dict[str, Any]:
        if state.get("stop") and state.get("reason") == "GRAPH_DEADLINE_EXHAUSTED":
            # Incomplete model-only turn: do not forge a completed host checkpoint.
            return {}
        turn = state["turns"]
        snap = state["snapshot"]
        done = state.get("finish_accepted", False)
        failure = state.get("last_failure") or {}
        infra_failure = failure.get("code") in {"BUILD_UNAVAILABLE", "PUBLIC_EXECUTION_UNAVAILABLE", "PUBLIC_TOOL_IO_FAILED"}
        manual_review = (
            review_public_pass
            and snap.get("public_checks_passed")
            and snap.get("current_candidate_sha256") != state.get("reviewed_candidate")
        )
        pause = (
            not done and not infra_failure and turn < state["max_steps"]
            and ((pause_after_turns is not None and turn >= pause_after_turns) or manual_review)
        )
        reason = (
            host.completion_reason if done else "GRAPH_INFRASTRUCTURE_FAILURE" if infra_failure
            else "STEP_BUDGET_EXHAUSTED" if turn >= state["max_steps"]
            else "SESSION_PAUSED" if pause else "CONTINUE"
        )
        update = {
            "stop": done or infra_failure or turn >= state["max_steps"] or pause,
            "reason": reason, "pause_requested": pause,
            "review_kind": "public_candidate_review" if manual_review else "operator_pause",
        }
        on_checkpoint(project_state(cast(NativeState, {**state, **update})))
        return {**update, "events": _event(state, "boundary", reason=reason)}

    def route_boundary(state: NativeState) -> str:
        if state.get("pause_requested"):
            return "review"
        return "terminal" if state.get("stop") else "refresh"

    def review(state: NativeState) -> Command[Literal["refresh", "terminal"]]:
        # interrupt must precede any new external effect. The resume command is
        # explicit operator input, not a model-issued approval.
        response = interrupt({
            "kind": state["review_kind"], "turn": state["turns"],
            "candidate": state["snapshot"].get("current_candidate_sha256"),
            "public_checks_passed": state["snapshot"].get("public_checks_passed"),
            "task_status": state.get("task_status", []),
        })
        if not isinstance(response, dict) or type(response.get("approved")) is not bool:
            raise ValueError("operator review requires explicit boolean approval")
        if not response["approved"]:
            return Command(update={"stop": True, "reason": "GRAPH_OPERATOR_REJECTED", "pause_requested": False}, goto="terminal")
        return Command(
            update={
                "stop": False, "reason": "CONTINUE", "pause_requested": False,
                "reviewed_candidate": state["snapshot"].get("current_candidate_sha256"),
                "events": _event(state, "operator_review", approved=True),
            },
            goto="refresh",
        )

    def terminal(state: NativeState) -> dict[str, Any]:
        return {"stop": True, "events": _event(state, "terminal", reason=state.get("reason"))}

    graph = StateGraph(NativeState)
    graph.add_node("refresh", refresh)
    graph.add_node("budget_gate", gate)
    graph.add_node("planner", plan_node)
    graph.add_node("replan", plan_node)
    graph.add_node("executor", executor.compile())
    graph.add_node("reconcile", refresh)
    graph.add_node("checkpoint", save_boundary)
    graph.add_node("review", review)
    graph.add_node("terminal", terminal)
    graph.add_edge(START, "refresh")
    graph.add_edge("refresh", "budget_gate")
    graph.add_conditional_edges(
        "budget_gate", route_gate,
        {n: n for n in ("planner", "replan", "executor", "terminal")},
    )
    graph.add_edge("planner", "refresh")
    graph.add_edge("replan", "refresh")
    graph.add_edge("executor", "reconcile")
    graph.add_edge("reconcile", "checkpoint")
    graph.add_conditional_edges("checkpoint", route_boundary, {n: n for n in ("review", "terminal", "refresh")})
    graph.add_edge("terminal", END)
    return graph.compile(checkpointer=checkpointer)


def run_native_repository_agent(
    model: Any,
    host: Any,
    *,
    output: Path,
    goal: str,
    max_steps: int,
    max_model_calls: int,
    hierarchical_mode: bool,
    deadline: float,
    on_event: Callable[[dict[str, Any]], None],
    on_checkpoint: Callable[[AgentLoopState], None],
    initial_state: AgentLoopState | None,
    pause_after_turns: int | None,
    review_public_pass: bool = False,
    approve_review: bool = True,
) -> AgentLoopState:
    """Run or resume the real repository graph using persistent native state."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    resume = initial_state is not None
    db_path = output / DB_NAME
    header = {
        "version": VERSION, "run_id": host.run_id,
        "goal_sha256": hashlib.sha256(goal.encode()).hexdigest(),
        "base_sha": host.workspace.base_sha,
        "deadline": deadline, "max_steps": max_steps, "max_model_calls": max_model_calls,
        "hierarchical": hierarchical_mode, "review_public_pass": review_public_pass,
    }
    if resume:
        if read_native_header(output) != header:
            raise ValueError("native graph identity or budget differs")
        _private_file(db_path)
    else:
        _new_private(output / HEADER_NAME, json.dumps(header, sort_keys=True).encode() + b"\n")
        _new_private(db_path)
    if time.time() >= deadline:
        raise ValueError("native session deadline exhausted; no effects admitted")
    ledger = EffectJournal(output / EFFECT_NAME, host.run_id, resume=resume)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        conn.execute("PRAGMA synchronous=FULL")
        saver = SqliteSaver(conn)
        graph = build_native_graph(
            model=model, host=host, goal=goal, ledger=ledger,
            on_event=on_event, on_checkpoint=on_checkpoint,
            pause_after_turns=pause_after_turns, review_public_pass=review_public_pass,
            checkpointer=saver,
        )
        config = {"configurable": {"thread_id": host.run_id}, "recursion_limit": max_steps * 12 + 40}
        if resume:
            saved = graph.get_state(config)
            state = cast(NativeState, saved.values)
            if (
                state.get("identity") != header
                or state.get("effects_completed") != ledger.completed
                or state.get("turns") != initial_state["turns"]
                or state.get("history") != initial_state["history"]
                or state.get("decision") != initial_state["decision"]
                or state.get("observation") != initial_state["observation"]
                or host.tool_calls != state.get("turns")
            ):
                raise ValueError("native checkpoint, effect ledger and workspace disagree; no replay")
            if state.get("reason") == "SESSION_PAUSED":
                if saved.next != ("review",) or not any(t.interrupts for t in saved.tasks):
                    raise ValueError("native pause is not at an authenticated interrupt boundary")
                graph_input = Command(resume={"approved": approve_review})
            elif state.get("stop"):
                raise ValueError("terminal native session cannot be resumed")
            elif saved.next not in (("refresh",), ("budget_gate",), ("planner",), ("replan",), ("checkpoint",), ("executor",)):
                raise ValueError("native graph has no safe continuation boundary")
            else:
                graph_input = None
        else:
            graph_input = {
                "identity": header, "deadline": deadline,
                "max_steps": max_steps, "max_model_calls": max_model_calls,
                "hierarchical": hierarchical_mode,
                "max_replans": max_replans_for(max_steps, max_model_calls, hierarchical_mode),
                "history": [], "events": [], "evidence": [], "turns": 0,
                "logical_model_calls": 0, "observed_physical_calls": 0, "effects_completed": 0,
                "plan_revisions": [], "plan_revision": 0, "last_plan_turn": 0,
                "last_failure": None, "failure_streak": 0,
                "finish_accepted": False, "stop": False, "reason": "CONTINUE", "pause_requested": False,
            }
        # Sync durability guarantees each native superstep is saved before the
        # next node. Do not use async writes for external side effects.
        result = cast(NativeState, graph.invoke(graph_input, config=config, durability="sync"))
        saved = graph.get_state(config)
        if result.get("effects_completed", 0) != ledger.completed:
            raise ValueError("graph state did not durably account for all effects")
        if saved.next and not (result.get("reason") == "SESSION_PAUSED" and saved.next == ("review",)):
            raise ValueError("graph ended at an unexpected continuation")
        if result.get("turns", 0) == 0:
            return {"history": [], "turns": 0, "stop": True, "reason": result.get("reason", "UNKNOWN")}
        return project_state(result)
    finally:
        conn.close()
        ledger.close()


def inspect_native_graph(output: Path) -> dict[str, Any]:
    """Operator-only native checkpoint inspection, no model/host constructed."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    header = read_native_header(output)
    path = output / DB_NAME
    _private_file(path)
    # A read-only connection ensures inspection cannot advance the runtime.
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        # Setup is normally implicit and writes DDL. The database already has
        # a schema; mark only this read-only saver instance initialized.
        saver.is_setup = True
        config = {"configurable": {"thread_id": header["run_id"]}}
        current = saver.get_tuple(config)
        if current is None:
            raise ValueError("native checkpoint unavailable")
        values = current.checkpoint["channel_values"]
        return {
            "version": VERSION, "thread_id": header["run_id"],
            "checkpoint_id": current.checkpoint["id"],
            "checkpoint_count": sum(1 for _ in saver.list(config)),
            "turns": values.get("turns", 0),
            "reason": values.get("reason"), "plan_revision": values.get("plan_revision", 0),
            "task_status": values.get("task_status", []),
            "observed_physical_calls": values.get("observed_physical_calls", 0),
            "effects_completed": values.get("effects_completed", 0),
            "graph_events": values.get("events", []),
            "deadline": header["deadline"],
        }
    finally:
        conn.close()
