"""End-to-end CPU tests of native graph with the real Git tool host.

No physical model calls, CUDA execution, or private evaluator access.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from gpu_agent.harness.graph_plan import GraphTaskPlan, evaluate_tasks
from gpu_agent.harness.journal import SessionJournal, read_verified_journal
from gpu_agent.harness.models import HarnessDecision
from gpu_agent.harness.native_graph import (
    DB_NAME,
    EFFECT_NAME,
    inspect_native_graph,
    max_replans_for,
)
from gpu_agent.harness.repo_runner import run_repository_coding_session


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True, capture_output=True, check=True, timeout=10,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "kernel.cu").write_text("float x = 1.0f;\n")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
        "commit", "-q", "-m", "test source")
    return repo, git(repo, "rev-parse", "HEAD")


def plan(revised=False):
    return GraphTaskPlan.model_validate({
        "tasks": [
            {"task_id": "t1", "objective": "Read the kernel source for exact edit evidence",
             "gate": "inspect", "target": "kernel.cu", "depends_on": []},
            {"task_id": "t2", "objective": "Correct the original float literal using current evidence"
             if revised else "Change the float literal in the kernel",
             "gate": "edit", "target": "kernel.cu", "depends_on": ["t1"]},
            {"task_id": "t3", "objective": "Stage the changed file for operator review",
             "gate": "finish", "target": "", "depends_on": ["t2"]},
        ],
        "revision_reason": "Correct rejected old-text binding" if revised else "Initial task plan",
        "failure_risk": "Incorrect old text may reject the proposed source edit",
    })


def action(tool, **kw):
    return {"tool": tool, **kw}


READ = action("repo_read_file", path="kernel.cu")
BAD = action("repo_replace_snippet", path="kernel.cu", old_text="9.0f", new_text="2.0f")
EDIT = action("repo_replace_snippet", path="kernel.cu", old_text="1.0f", new_text="2.0f")
FINISH = action("finish", summary="Stage the edited CUDA file for review")


class Model:
    def __init__(self, actions):
        self.actions = list(actions)
        self.decisions = []
        self.plans = []

    def choose_graph_plan(self, snapshot, *, previous_plan, task_status, failure):
        self.plans.append({
            "previous": previous_plan, "task_status": task_status, "failure": failure,
        })
        return plan(previous_plan is not None)

    def choose_harness_v1(self, snapshot):
        assert self.actions, "unexpected repeated model decision"
        self.decisions.append(snapshot)
        return HarnessDecision.model_validate({
            "action": self.actions.pop(0),
            "reason": "Perform the next evidence-backed task action",
        })


def args(repository, out, model, **kw):
    repo, sha = repository
    return {
        "repository": repo, "base_sha": sha, "output": out,
        "goal": "Correct the float literal and stage the CUDA source",
        "max_steps": 8, "max_model_calls": 16, "runtime_mode": "langgraph",
        "hierarchical_mode": True, "model_factory": lambda _: model, **kw,
    }


def test_task_dag_validation_and_stale_evidence():
    p = plan()
    data = p.model_dump()
    data["tasks"][0]["depends_on"] = ["t3"]
    with pytest.raises(ValidationError):
        GraphTaskPlan.model_validate(data)
    data = p.model_dump()
    data["tasks"][-1]["depends_on"] = ["t1"]
    with pytest.raises(ValidationError):
        GraphTaskPlan.model_validate(data)
    cuda_plan = GraphTaskPlan.model_validate({
        "tasks": [
            {"task_id": "t1", "objective": "Edit the registered CUDA kernel",
             "gate": "edit", "target": "kernel.cu", "depends_on": []},
            {"task_id": "t2", "objective": "Run the full public numerical checker",
             "gate": "public_check", "target": "kernel.cu", "depends_on": ["t1"]},
            {"task_id": "t3", "objective": "Stage the verified public candidate",
             "gate": "finish", "target": "", "depends_on": ["t2"]},
        ],
        "revision_reason": "Initial CUDA verification plan",
        "failure_risk": "A stale candidate may invalidate verification",
    })
    evidence = [
        {"turn": 1, "gate": "edit", "status": "OK", "path": "kernel.cu", "candidate": "a"},
        {"turn": 2, "gate": "public_check", "status": "OK", "path": "kernel.cu", "candidate": "a"},
    ]
    snapshot = {"current_candidate_sha256": "a", "public_checks_passed": True}
    assert [x["status"] for x in evaluate_tasks(cuda_plan, evidence, snapshot, ["kernel.cu"])] == [
        "DONE", "DONE", "READY"
    ]
    snapshot.update(current_candidate_sha256="b", public_checks_passed=False)
    assert [x["status"] for x in evaluate_tasks(cuda_plan, evidence, snapshot, ["kernel.cu"])] == [
        "DONE", "STALE", "BLOCKED"
    ]
    assert max_replans_for(8, 16, True) == 2


def test_real_host_failure_triggers_native_replan(repository, tmp_path):
    out = tmp_path / "native"
    model = Model([READ, BAD, EDIT, FINISH])
    result = run_repository_coding_session(**args(repository, out, model))
    assert result.status == "DIFF_STAGED_UNVERIFIED"
    assert result.turns == result.tool_calls == 4
    assert len(model.plans) == 2
    assert model.plans[1]["previous"] is not None
    assert model.plans[1]["failure"]["tool"] == "repo_replace_snippet"
    assert model.plans[1]["failure"]["status"] == "REJECTED"
    assert "GRAPH_PLAN" in model.decisions[-1].context_summary
    assert (out / "worktree/kernel.cu").read_text() == "float x = 2.0f;\n"
    assert (repository[0] / "kernel.cu").read_text() == "float x = 1.0f;\n"
    assert git(repository[0], "status", "--porcelain") == ""
    info = inspect_native_graph(out)
    assert info["plan_revision"] == 2
    assert info["turns"] == 4 and info["effects_completed"] == 10
    assert info["checkpoint_count"] > result.turns
    assert [task["status"] for task in info["task_status"]] == ["DONE"] * 3
    journal = read_verified_journal(out / EFFECT_NAME)
    started = [x["payload"]["effect_key"] for x in journal if x["kind"].endswith("started") and x["kind"] != "session_started"]
    assert len(started) == len(set(started)) == 10
    assert len([e for e in info["graph_events"] if e["kind"] == "tool_observed"]) == 4
    assert result.model_calls == 0  # injected model is not a physical provider


def test_native_sqlite_interrupt_and_cross_process_style_resume(repository, tmp_path):
    out = tmp_path / "paused"
    first = Model([READ, EDIT])
    report = run_repository_coding_session(**args(
        repository, out, first, pause_after_turns=2,
    ))
    assert report.status == "PAUSED_RESUMABLE"
    assert report.turns == 2
    info = inspect_native_graph(out)
    assert info["reason"] == "SESSION_PAUSED"
    assert info["effects_completed"] == 5
    db_before = hashlib.sha256((out / DB_NAME).read_bytes()).hexdigest()
    assert inspect_native_graph(out)["checkpoint_id"] == info["checkpoint_id"]
    assert hashlib.sha256((out / DB_NAME).read_bytes()).hexdigest() == db_before
    second = Model([FINISH])
    final = run_repository_coding_session(**args(repository, out, second, resume=True))
    assert final.status == "DIFF_STAGED_UNVERIFIED"
    assert final.turns == final.tool_calls == 3
    assert second.plans == [] and len(second.decisions) == 1
    resumed = inspect_native_graph(out)
    assert resumed["thread_id"] == info["thread_id"]
    assert resumed["deadline"] == info["deadline"]
    assert resumed["effects_completed"] == 7
    assert resumed["checkpoint_count"] > info["checkpoint_count"]
    assert any(e["kind"] == "operator_review" for e in resumed["graph_events"])
    entries = read_verified_journal(out / EFFECT_NAME)
    ids = [e["payload"]["effect_key"] for e in entries if e["kind"] in ("model_started", "tool_started")]
    assert len(ids) == len(set(ids)) == 7


def test_native_operator_rejects_at_interrupt_without_more_effects(repository, tmp_path):
    out = tmp_path / "reject"
    first = Model([READ, EDIT])
    assert run_repository_coding_session(**args(repository, out, first, pause_after_turns=2)).status == "PAUSED_RESUMABLE"
    before = inspect_native_graph(out)
    second = Model([])
    result = run_repository_coding_session(**args(
        repository, out, second, resume=True, approve_review=False,
    ))
    assert result.status == "INCONCLUSIVE"
    assert result.reason == "GRAPH_OPERATOR_REJECTED"
    assert not second.decisions and not second.plans
    assert inspect_native_graph(out)["effects_completed"] == before["effects_completed"]


def test_uncertain_external_effect_blocks_native_resume(repository, tmp_path):
    out = tmp_path / "uncertain"
    first = Model([READ, EDIT])
    assert run_repository_coding_session(**args(repository, out, first, pause_after_turns=2)).status == "PAUSED_RESUMABLE"
    rows = read_verified_journal(out / EFFECT_NAME)
    with SessionJournal.reopen(out / EFFECT_NAME, expected_head=rows[-1]["sha256"]) as journal:
        journal.append("model_started", {
            "turn": 6, "effect_key": "decision-3", "role": "executor",
        })
    second = Model([])
    with pytest.raises(ValueError, match="uncertain"):
        run_repository_coding_session(**args(repository, out, second, resume=True))
    assert not second.decisions and not second.plans
    assert inspect_native_graph(out)["effects_completed"] == 5


def test_cannot_resume_legacy_graph_or_change_runtime(repository, tmp_path):
    out = tmp_path / "mode"
    first = Model([READ, EDIT])
    assert run_repository_coding_session(**args(repository, out, first, pause_after_turns=2)).status == "PAUSED_RESUMABLE"
    with pytest.raises(ValueError, match="switch runtime"):
        run_repository_coding_session(**args(
            repository, out, Model([]), resume=True, runtime_mode="legacy",
            hierarchical_mode=False,
        ))
