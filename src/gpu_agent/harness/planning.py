"""Controller-derived task progress, not LLM self-reported completion.

The Planner proposes objectives; only actual successful tool observations can
check a milestone. Reflection changes a compact, model-visible working view,
never the immutable canonical event log.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from gpu_agent.harness.models import TaskProgress, TaskReflection

_INSPECT = {"repo_read_file", "repo_inspect_syntax"}
_EDIT = {"repo_replace_snippet", "repo_replace_many", "repo_replace_syntax"}
_REDUNDANT = {"REDUNDANT_READ", "REDUNDANT_LIST", "REDUNDANT_DIFF"}


def progress_from_tools(
    history: Sequence[Mapping[str, str]], *, current_source_visible: bool = False
) -> TaskProgress:
    inspected = current_source_visible
    edited = False
    public_checked = False
    finished = False
    last_failure: str | None = None
    for event in history:
        tool = event.get("tool", "")
        status = event.get("status", "")
        content = event.get("content", "")
        if status == "OK":
            if tool in _INSPECT:
                inspected = True
            elif tool in _EDIT:
                edited = True
                # A new edit invalidates public verification of the previous diff.
                public_checked = False
            elif tool == "repo_run_public_checks":
                public_checked = True
            elif tool == "finish":
                finished = True
        elif status in {"REJECTED", "FAILED"}:
            if status == "FAILED" and tool == "repo_run_public_checks":
                public_checked = False
            last_failure = content.split(":", 2)[-1].strip()[:160] if content else status
    stage = (
        "done"
        if finished
        else "inspect"
        if not inspected
        else "edit"
        if not edited
        else "public_check"
        if not public_checked
        else "finish"
    )
    return TaskProgress(
        inspected=inspected,
        edited=edited,
        public_checked=public_checked,
        finished=finished,
        active_gate=stage,
        last_failure_code=last_failure,
    )


def reflection_context(
    history: Sequence[Mapping[str, str]],
    *,
    reflection: TaskReflection,
    max_recent: int = 4,
) -> tuple[dict[str, str], ...]:
    """Preserve failures and novel observations; hide redundant success noise.

    This is a projection for the model only; RunStore and checkpoint retain
    full canonical history. Avoid deleting all bad cases from memory.
    """
    selected: list[dict[str, str]] = []
    redundant_seen: set[str] = set()
    for event in reversed(history):
        tool = event.get("tool", "")
        status = event.get("status", "")
        content = event.get("content", "")
        if status == "REJECTED":
            code = next((name for name in _REDUNDANT if name in content), None)
            if code is not None:
                if code in redundant_seen:
                    continue
                redundant_seen.add(code)
        limit = 3500 if tool == "repo_inspect_syntax" and status == "OK" else 900
        selected.append({"tool": tool[:50], "status": status[:20], "content": content[:limit]})
        if len(selected) >= max_recent:
            break
    # Model-visible reflection is structured separately in SessionSnapshot.
    # Keep the function signature explicit to avoid accidentally ignoring it.
    _ = reflection
    return tuple(reversed(selected))
