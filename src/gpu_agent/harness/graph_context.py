"""Bounded model context projection from graph evidence, not free-form memory."""

from __future__ import annotations

from typing import Any

from gpu_agent.harness.graph_memory import STRATEGIES


def compact_projection(
    state: dict[str, Any], *, limit: int = 2800,
) -> dict[str, Any]:
    """Pack complete evidence units by priority; never truncate JSON mid-record.

    Current AST binding, workspace source, typed tool schema, and host history
    remain separate SessionSnapshot fields. Only context_summary is compacted.
    """
    if not 800 <= limit <= 2800:
        raise ValueError("context projection budget out of bounds")
    parts: list[str] = []
    included: list[str] = []
    omitted: list[str] = []
    used = 0

    def add(label: str, text: str, cap: int):
        nonlocal used
        line = label + " " + text.replace("\n", " ")
        if len(line) > cap:
            line = line[:cap - 13] + " [truncated]"
        if used + len(line) + 1 <= limit:
            parts.append(line)
            included.append(label)
            used += len(line) + 1
        else:
            omitted.append(label)

    snap = state["snapshot"]
    old = str(snap.get("context_summary", ""))
    # Preserve controller budget guidance before discretionary memory/context.
    if old.startswith("BUDGET_ALERT:"):
        alert, _, old = old.partition("\n")
        add("HOST_BUDGET", alert, 380)
    add("GRAPH_BUDGET", (
        f"steps={state['max_steps'] - state['turns']}; "
        f"physical_calls={state['max_model_calls'] - state.get('observed_physical_calls', 0)}; "
        f"public_checks={snap.get('remaining_public_checks', 0)}; "
        f"probes={snap.get('remaining_public_probes', 0)}"
    ), 220)
    active = state.get("active_task")
    if active:
        task = active["task"]
        add("ACTIVE_TASK", f"{task['task_id']} {task['gate']} {task['target']} — {task['objective']}", 480)
    if state.get("plan"):
        statuses = state.get("task_status", [])
        add("GRAPH_PLAN", "revision=" + str(state.get("plan_revision", 0)) + " "
            + " ".join(f"{x['task_id']}:{x['status']}" for x in statuses), 360)
    recovery = state.get("recovery_state") or {}
    if recovery:
        add("RECOVERY", str(recovery.get("strategy")) + ": " + str(recovery.get("guidance")), 440)
    failure = state.get("last_failure")
    if failure:
        add("UNTRUSTED_LAST_OBSERVATION", f"turn={failure.get('turn')} {failure.get('code')}: "
            + str(failure.get("evidence", "")), 500)
    # Retrieve prior episodes as generic, controller-generated suggestions.
    # Never paste raw Store text, source, old aliases, patches or private checks.
    for index, lesson in enumerate(state.get("retrieved_lessons", [])[:3], 1):
        text = "; ".join(STRATEGIES[s] for s in lesson.get("strategies", []) if s in STRATEGIES)
        if text:
            add(f"PRIOR_EPISODE_{index}", (
                f"{lesson['outcome']} (not strict proof; advisory): {text}"
            ), 260)
            matching = [e for e in lesson.get("episodes", [])
                        if e["strategy"] == recovery.get("strategy")]
            if matching:
                sequence = " > ".join(f"{x['tool']}:{x['status']}"
                                      for x in matching[-1]["after_failure"][:4])
                add("OBSERVED_PRIOR_SEQUENCE", sequence + " (association, not proof)", 240)
    for evidence in reversed(state.get("evidence", [])[-6:]):
        if evidence.get("gate") in {"edit", "compile", "probe", "public_check", "revert"}:
            add("EVIDENCE", (
                f"turn={evidence['turn']} {evidence['tool']} {evidence['status']} "
                f"candidate={(evidence.get('candidate') or 'none')[:12]}"
            ), 125)
    if old:
        add("HOST_CONTEXT", old, 600)
    summary = "\n".join(parts)
    assert len(summary) <= limit
    return {
        "summary": summary, "chars": len(summary), "budget_chars": limit,
        "included": included, "omitted": omitted,
        "retained_full_history_turns": len(state.get("history", [])),
        "retained_ast_nodes": len((snap.get("syntax_state") or {}).get("observed_nodes", [])),
        "memory_episode_ids": [x["run_id"] for x in state.get("retrieved_lessons", [])[:3]],
    }
