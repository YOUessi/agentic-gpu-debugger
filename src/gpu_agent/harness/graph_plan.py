"""Graph-native variable-length task DAG and controller-attested task evidence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from gpu_agent.harness.models import (
    PlanMilestone,
    StrictModel,
    TaskPlan,
    TaskProgress,
)

Gate = Literal["inspect", "edit", "compile", "probe", "public_check", "finish"]

TOOL_GATES = {
    "repo_list_files": "inspect",
    "repo_read_file": "inspect",
    "repo_inspect_syntax": "inspect",
    "repo_git_diff": "inspect",
    "repo_replace_snippet": "edit",
    "repo_replace_many": "edit",
    "repo_replace_syntax": "edit",
    "repo_compile_candidate": "compile",
    "repo_probe_public": "probe",
    "repo_run_public_checks": "public_check",
    "repo_revert_last_edit": "revert",
    "finish": "finish",
}


class GraphTask(StrictModel):
    task_id: str = Field(pattern=r"^t[1-9][0-9]?$")
    objective: str = Field(min_length=8, max_length=220)
    gate: Gate
    target: str = Field(max_length=300)
    depends_on: list[str] = Field(max_length=9)


class GraphTaskPlan(StrictModel):
    """Model proposes objectives, not completion or tool permissions.

    Tasks are topologically ordered. The graph can replace the DAG after an
    observed failure. Completion is always derived from public tool evidence.
    """

    tasks: list[GraphTask] = Field(min_length=2, max_length=10)
    revision_reason: str = Field(min_length=8, max_length=280)
    failure_risk: str = Field(min_length=8, max_length=230)

    @model_validator(mode="after")
    def validate_dag(self) -> GraphTaskPlan:
        seen: set[str] = set()
        ancestors: dict[str, set[str]] = {}
        for task in self.tasks:
            deps = set(task.depends_on)
            if task.task_id in seen or len(deps) != len(task.depends_on) or not deps <= seen:
                raise ValueError("task IDs and dependencies must form a topologically ordered DAG")
            ancestors[task.task_id] = deps | set().union(*(ancestors[dep] for dep in deps))
            seen.add(task.task_id)
        if (
            self.tasks[-1].gate != "finish"
            or sum(t.gate == "finish" for t in self.tasks) != 1
            or ancestors[self.tasks[-1].task_id] != seen - {self.tasks[-1].task_id}
        ):
            raise ValueError("exactly one terminal finish task must depend on every prior task")
        if not any(t.gate == "edit" for t in self.tasks):
            raise ValueError("a coding plan must include an edit task")
        return self

    def validate_scope(self, snapshot: dict[str, Any]) -> None:
        path = snapshot.get("registered_kernel_path")
        if path:
            if any(t.target not in ("", path) for t in self.tasks):
                raise ValueError("registered CUDA plan may only target the registered kernel")
            if not any(t.gate == "public_check" for t in self.tasks):
                raise ValueError("registered CUDA plan requires public verification")
        # The plan never grants capabilities. The executor's tool host remains
        # the authoritative admission check, even if an objective is wrong.


def evaluate_tasks(
    plan: GraphTaskPlan,
    evidence: list[dict[str, Any]],
    snapshot: dict[str, Any],
    edited_files: list[str],
) -> list[dict[str, Any]]:
    """Use exact candidate identity for build/probe/check/finish evidence."""
    candidate = snapshot.get("current_candidate_sha256")
    # A successful observation may discharge only one distinct milestone.
    # Proofs explicitly bound to another objective are not silently reused.
    from gpu_agent.harness.graph_schedule import task_fingerprint

    results: dict[str, dict[str, Any]] = {}
    used_turns: set[int] = set()
    failed_candidate = any(
        e.get("candidate") == candidate and candidate
        and e.get("gate") in {"compile", "public_check"}
        and e.get("status") == "FAILED"
        for e in evidence
    )
    if snapshot.get("public_checks_passed"):
        failed_candidate = False
    for task in plan.tasks:
        matches = [
            e
            for e in evidence
            if e["gate"] == task.gate
            and e["status"] == "OK"
            and (not task.target or e.get("path") == task.target)
            and e.get("task_key") in (None, task_fingerprint(task))
            and e["turn"] not in used_turns
        ]
        if (
            task.gate == "inspect"
            and snapshot.get("registered_public_case")
            and snapshot.get("context_mode") == "evidence_first"
            and task.target in ("", snapshot.get("registered_kernel_path"))
            and 0 not in used_turns
        ):
            matches.insert(
                0,
                {
                    "turn": 0,
                    "candidate": candidate,
                    "artifact_id": None,
                    "source": "registered_context",
                },
            )
        if task.gate == "edit":
            matches = [
                e
                for e in matches
                if candidate
                and (
                    e.get("path") in edited_files
                    or (not e.get("path") and e.get("candidate") == candidate)
                )
            ]
        if task.gate == "edit" and failed_candidate:
            # A failed current candidate cannot satisfy the repair milestone,
            # even when a previous revision also changed this file.
            matches = []
        if task.gate == "edit":
            last_revert = max(
                (e["turn"] for e in evidence if e.get("gate") == "revert"
                 and e.get("status") == "OK"
                 and (not task.target or e.get("path") == task.target)),
                default=-1,
            )
            matches = [e for e in matches if e["turn"] > last_revert]
        if task.gate in ("compile", "probe", "public_check", "finish"):
            current = [e for e in matches if e.get("candidate") == candidate]
        else:
            current = matches
        if task.gate == "public_check" and not snapshot.get("public_checks_passed"):
            current = []
        if task.gate == "compile":
            recovery = snapshot.get("candidate_recovery")
            if recovery is not None and recovery.get("compile_status") != "BUILD_PASSED":
                current = []
        if task.gate == "finish" and not candidate:
            current = []
        dependencies_done = all(results[d]["status"] == "DONE" for d in task.depends_on)
        if task.depends_on and dependencies_done:
            prerequisite_turn = max(results[d]["evidence_turn"] for d in task.depends_on)
            current = [e for e in current if e["turn"] > prerequisite_turn]
        status = (
            "DONE"
            if current and dependencies_done
            else "BLOCKED"
            if not dependencies_done
            else "STALE"
            if matches
            else "READY"
        )
        if status == "DONE":
            used_turns.add(current[-1]["turn"])
        results[task.task_id] = {
            "task_id": task.task_id,
            "gate": task.gate,
            "target": task.target,
            "status": status,
            "evidence_turn": current[-1]["turn"] if status == "DONE" else None,
            "artifact_id": current[-1].get("artifact_id") if status == "DONE" else None,
        }
    return list(results.values())


def compatibility_plan(plan: GraphTaskPlan) -> TaskPlan:
    """Old snapshot/host checkpoint schema is a projection, not graph state."""
    defaults = {
        "inspect": "Inspect the relevant source and observable failure",
        "edit": "Modify the required source with a scoped candidate",
        "public_check": "Verify the exact current candidate if supported",
        "finish": "Stage the reviewed current candidate for independent verification",
    }
    milestones: list[PlanMilestone] = []
    for gate in ("inspect", "edit", "public_check", "finish"):
        objectives = [t.objective for t in plan.tasks if t.gate == gate]
        milestones.append(
            PlanMilestone(
                gate=gate,
                objective=("; ".join(objectives)[:220] if objectives else defaults[gate]),
                success_evidence="Controller evidence for the exact candidate where applicable",
            )
        )
    return TaskPlan(milestones=milestones, failure_risk=plan.failure_risk)


def compatibility_progress(
    snapshot: dict[str, Any],
    evidence: list[dict[str, Any]],
    edited_files: list[str],
    failure: dict[str, Any] | None,
) -> TaskProgress:
    inspected = bool(
        snapshot.get("registered_public_case") and snapshot.get("context_mode") == "evidence_first"
    ) or any(e["gate"] == "inspect" and e["status"] == "OK" for e in evidence)
    edited = bool(edited_files and snapshot.get("current_candidate_sha256"))
    checked = bool(snapshot.get("public_checks_passed"))
    finished = bool(
        evidence and evidence[-1]["gate"] == "finish" and evidence[-1]["status"] == "OK"
    )
    gate = (
        "done"
        if finished
        else "inspect"
        if not inspected
        else "edit"
        if not edited
        else "public_check"
        if snapshot.get("registered_public_case") and not checked
        else "finish"
    )
    return TaskProgress(
        inspected=inspected,
        edited=edited,
        public_checked=checked,
        finished=finished,
        active_gate=gate,
        last_failure_code=(failure["code"][:160] if failure else None),
    )
