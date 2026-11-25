"""Dependency-driven task scheduling for the native Coding Agent.

Pure frontier scoring may fan out in parallel. Only the single selected
task's executor can admit a model decision or mutate the workspace.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from gpu_agent.harness.graph_plan import GraphTask, GraphTaskPlan, TOOL_GATES

_INSPECT = {"repo_read_file", "repo_list_files", "repo_inspect_syntax", "repo_git_diff"}
_EDIT = {"repo_replace_snippet", "repo_replace_many", "repo_replace_syntax"}
_PRIMARY = {
    "inspect": _INSPECT,
    "edit": _EDIT,
    "compile": {"repo_compile_candidate"},
    "probe": {"repo_probe_public"},
    "public_check": {"repo_run_public_checks"},
    "finish": {"finish"},
}
_SUPPORT = {
    "inspect": set(),
    "edit": _INSPECT | {"repo_revert_last_edit", "repo_compile_candidate", "repo_probe_public"},
    "compile": _INSPECT,
    "probe": _INSPECT | {"repo_compile_candidate"},
    "public_check": _INSPECT | {"repo_compile_candidate", "repo_probe_public"},
    "finish": {"repo_git_diff"},
}


def task_fingerprint(task: GraphTask) -> str:
    """Conservative proof reuse across plan revisions, not just task IDs."""
    return hashlib.sha256(
        json.dumps(
            [task.gate, task.target, task.objective], ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def frontier_inputs(state: dict[str, Any]) -> list[dict[str, Any]]:
    plan = GraphTaskPlan.model_validate(state["plan"])
    statuses = {row["task_id"]: row for row in state["task_status"]}
    descendants = {task.task_id: set() for task in plan.tasks}
    for task in reversed(plan.tasks):
        for dep in task.depends_on:
            descendants[dep].add(task.task_id)
            descendants[dep].update(descendants[task.task_id])
    ready = [
        task for task in plan.tasks
        if statuses[task.task_id]["status"] in {"READY", "STALE"}
    ]
    unfinished = sum(row["status"] != "DONE" for row in statuses.values())
    # Each explicit milestone needs at least one distinct observation. This is
    # a lower bound, not a promise that a task takes only one model turn.
    return [
        {
            "task": task.model_dump(mode="json"),
            "round": [state["plan_revision"], state["turns"]],
            "position": index,
            "descendants": len(descendants[task.task_id]),
            "remaining_steps": state["max_steps"] - state["turns"],
            "unfinished": unfinished,
            "allowed": list(state["snapshot"]["allowed_tools"]),
            "recovery": state.get("recovery_state"),
        }
        for index, task in enumerate(ready)
    ]


def assess_frontier_task(item: dict[str, Any]) -> dict[str, Any]:
    """Pure worker: no model, store write, Git read, GPU or tool call."""
    task = GraphTask.model_validate(item["task"])
    allowed = set(item["allowed"])
    primary = allowed & _PRIMARY[task.gate]
    helpers = allowed & _SUPPORT[task.gate]
    # A failed candidate must be repaired by an edit task, not by rerunning
    # its old checker or compile. Revert, if available, is an edit helper.
    repair = (item.get("recovery") or {}).get("strategy")
    repair_priority = int(
        task.gate == "edit"
        and repair in {"AST_REBIND", "COMPILE_REPAIR", "NUMERIC_REPAIR", "REPLAN"}
    )
    prerequisite = bool(
        (task.gate == "edit" and helpers & {"repo_inspect_syntax", "repo_read_file", "repo_revert_last_edit", "repo_compile_candidate"})
        or (task.gate in {"probe", "public_check"} and "repo_compile_candidate" in helpers)
    )
    return {
        "id": f'{item["round"][0]}:{item["round"][1]}:{task.task_id}',
        "round": item["round"],
        "task": task.model_dump(mode="json"),
        "task_key": task_fingerprint(task),
        "primary": sorted(primary),
        "scope": sorted(primary | helpers),
        "feasible": item["remaining_steps"] >= item["unfinished"] and bool(primary or prerequisite),
        "priority": [repair_priority, int(bool(primary)), item["descendants"], -item["position"]],
    }


def select_frontier(
    results: list[dict[str, Any]], revision: int, turn: int,
) -> dict[str, Any] | None:
    current = [
        row for row in results
        if row["round"] == [revision, turn] and row["scope"] and row["feasible"]
    ]
    return max(current, key=lambda row: tuple(row["priority"])) if current else None


def evidence_task_key(selection: dict[str, Any] | None, tool: str, path: str) -> str | None:
    if not selection:
        return None
    task = selection["task"]
    if TOOL_GATES.get(tool) != task["gate"]:
        return None
    if task["target"] and path != task["target"]:
        return None
    return selection["task_key"]
