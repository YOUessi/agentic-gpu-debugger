"""End-to-end CPU tests of native graph with the real Git tool host.

No physical model calls, CUDA execution, or private evaluator access.
"""

from __future__ import annotations

import hashlib
import subprocess

import pytest
from pydantic import ValidationError

from gpu_agent.harness.graph_plan import GraphTaskPlan, evaluate_tasks
from gpu_agent.harness.journal import SessionJournal, read_verified_journal
from gpu_agent.harness.models import HarnessDecision
from gpu_agent.harness.native_graph import (
    DB_NAME,
    EFFECT_NAME,
    classify_recovery,
    inspect_native_graph,
    max_replans_for,
)
from gpu_agent.harness.repo_runner import run_repository_coding_session


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "kernel.cu").write_text("float x = 1.0f;\n")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "-q",
        "-m",
        "test source",
    )
    return repo, git(repo, "rev-parse", "HEAD")


def plan(revised=False):
    return GraphTaskPlan.model_validate(
        {
            "tasks": [
                {
                    "task_id": "t1",
                    "objective": "Read the kernel source for exact edit evidence",
                    "gate": "inspect",
                    "target": "kernel.cu",
                    "depends_on": [],
                },
                {
                    "task_id": "t2",
                    "objective": "Correct the original float literal using current evidence"
                    if revised
                    else "Change the float literal in the kernel",
                    "gate": "edit",
                    "target": "kernel.cu",
                    "depends_on": ["t1"],
                },
                {
                    "task_id": "t3",
                    "objective": "Stage the changed file for operator review",
                    "gate": "finish",
                    "target": "",
                    "depends_on": ["t2"],
                },
            ],
            "revision_reason": "Correct rejected old-text binding"
            if revised
            else "Initial task plan",
            "failure_risk": "Incorrect old text may reject the proposed source edit",
        }
    )


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
        self.plans.append(
            {
                "previous": previous_plan,
                "task_status": task_status,
                "failure": failure,
            }
        )
        return plan(previous_plan is not None)

    def choose_harness_v1(self, snapshot):
        assert self.actions, "unexpected repeated model decision"
        self.decisions.append(snapshot)
        return HarnessDecision.model_validate(
            {
                "action": self.actions.pop(0),
                "reason": "Perform the next evidence-backed task action",
            }
        )


def args(repository, out, model, **kw):
    repo, sha = repository
    return {
        "repository": repo,
        "base_sha": sha,
        "output": out,
        "goal": "Correct the float literal and stage the CUDA source",
        "max_steps": 8,
        "max_model_calls": 16,
        "runtime_mode": "langgraph",
        "hierarchical_mode": True,
        "model_factory": lambda _: model,
        **kw,
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
    cuda_plan = GraphTaskPlan.model_validate(
        {
            "tasks": [
                {
                    "task_id": "t1",
                    "objective": "Edit the registered CUDA kernel",
                    "gate": "edit",
                    "target": "kernel.cu",
                    "depends_on": [],
                },
                {
                    "task_id": "t2",
                    "objective": "Run the full public numerical checker",
                    "gate": "public_check",
                    "target": "kernel.cu",
                    "depends_on": ["t1"],
                },
                {
                    "task_id": "t3",
                    "objective": "Stage the verified public candidate",
                    "gate": "finish",
                    "target": "",
                    "depends_on": ["t2"],
                },
            ],
            "revision_reason": "Initial CUDA verification plan",
            "failure_risk": "A stale candidate may invalidate verification",
        }
    )
    evidence = [
        {"turn": 1, "gate": "edit", "status": "OK", "path": "kernel.cu", "candidate": "a"},
        {"turn": 2, "gate": "public_check", "status": "OK", "path": "kernel.cu", "candidate": "a"},
    ]
    snapshot = {"current_candidate_sha256": "a", "public_checks_passed": True}
    assert [x["status"] for x in evaluate_tasks(cuda_plan, evidence, snapshot, ["kernel.cu"])] == [
        "DONE",
        "DONE",
        "READY",
    ]
    snapshot.update(current_candidate_sha256="b", public_checks_passed=False)
    assert [x["status"] for x in evaluate_tasks(cuda_plan, evidence, snapshot, ["kernel.cu"])] == [
        "DONE",
        "STALE",
        "BLOCKED",
    ]
    assert max_replans_for(8, 16, True) == 2


def test_real_host_failure_triggers_native_replan(repository, tmp_path):
    out = tmp_path / "native"
    model = Model([READ, BAD, EDIT, FINISH])
    stream = []
    result = run_repository_coding_session(
        **args(repository, out, model, on_graph_update=stream.append)
    )
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
    assert any(e["node"] == "replan" for e in stream)
    assert any(e["node"] == "edit" and e["namespace"] for e in stream)
    assert any(e["node"] == "recovery" for e in stream)
    assert info["effects_head_sha256"] == read_verified_journal(out / EFFECT_NAME)[-1]["sha256"]
    assert info["turns"] == 4 and info["effects_completed"] == 10
    assert info["checkpoint_count"] > result.turns
    assert [task["status"] for task in info["task_status"]] == ["DONE"] * 3
    journal = read_verified_journal(out / EFFECT_NAME)
    started = [
        x["payload"]["effect_key"]
        for x in journal
        if x["kind"].endswith("started") and x["kind"] != "session_started"
    ]
    assert len(started) == len(set(started)) == 10
    assert len([e for e in info["graph_events"] if e["kind"] == "tool_observed"]) == 4
    assert result.model_calls == 0  # injected model is not a physical provider


def test_native_sqlite_interrupt_and_cross_process_style_resume(repository, tmp_path):
    out = tmp_path / "paused"
    first = Model([READ, EDIT])
    report = run_repository_coding_session(
        **args(
            repository,
            out,
            first,
            pause_after_turns=2,
        )
    )
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
    ids = [
        e["payload"]["effect_key"]
        for e in entries
        if e["kind"] in ("model_started", "tool_started")
    ]
    assert len(ids) == len(set(ids)) == 7


def test_native_operator_rejects_at_interrupt_without_more_effects(repository, tmp_path):
    out = tmp_path / "reject"
    first = Model([READ, EDIT])
    assert (
        run_repository_coding_session(**args(repository, out, first, pause_after_turns=2)).status
        == "PAUSED_RESUMABLE"
    )
    before = inspect_native_graph(out)
    second = Model([])
    result = run_repository_coding_session(
        **args(
            repository,
            out,
            second,
            resume=True,
            approve_review=False,
        )
    )
    assert result.status == "INCONCLUSIVE"
    assert result.reason == "GRAPH_OPERATOR_REJECTED"
    assert not second.decisions and not second.plans
    assert inspect_native_graph(out)["effects_completed"] == before["effects_completed"]


def test_uncertain_external_effect_blocks_native_resume(repository, tmp_path):
    out = tmp_path / "uncertain"
    first = Model([READ, EDIT])
    assert (
        run_repository_coding_session(**args(repository, out, first, pause_after_turns=2)).status
        == "PAUSED_RESUMABLE"
    )
    rows = read_verified_journal(out / EFFECT_NAME)
    with SessionJournal.reopen(out / EFFECT_NAME, expected_head=rows[-1]["sha256"]) as journal:
        journal.append(
            "model_started",
            {
                "turn": 6,
                "effect_key": "decision-3",
                "role": "executor",
            },
        )
    second = Model([])
    with pytest.raises(ValueError, match="uncertain"):
        run_repository_coding_session(**args(repository, out, second, resume=True))
    assert not second.decisions and not second.plans
    assert inspect_native_graph(out)["effects_completed"] == 5


def test_cannot_resume_legacy_graph_or_change_runtime(repository, tmp_path):
    out = tmp_path / "mode"
    first = Model([READ, EDIT])
    assert (
        run_repository_coding_session(**args(repository, out, first, pause_after_turns=2)).status
        == "PAUSED_RESUMABLE"
    )
    with pytest.raises(ValueError, match="switch runtime"):
        run_repository_coding_session(
            **args(
                repository,
                out,
                Model([]),
                resume=True,
                runtime_mode="legacy",
                hierarchical_mode=False,
            )
        )


def test_failure_classes_are_distinct_and_bound_to_candidate():
    snapshot = {
        "current_candidate_sha256": "diff-identity",
        "allowed_tools": ["repo_revert_last_edit", "repo_replace_syntax"],
        "candidate_recovery": {"source_sha256": "source-identity"},
        "remaining_public_checks": 1,
        "remaining_public_probes": 2,
    }
    for tool, code, expected in (
        ("repo_compile_candidate", "BUILD_FAILED", "COMPILE_REPAIR"),
        ("repo_run_public_checks", "NUMERIC_MISMATCH", "NUMERIC_REPAIR"),
        ("repo_replace_syntax", "CUDA_AST_TARGET_BINDING_MISMATCH", "AST_REBIND"),
        ("repo_read_file", "REDUNDANT_READ", "AVOID_DUPLICATE"),
        ("repo_run_public_checks", "PUBLIC_EXECUTION_UNAVAILABLE", "HALT_INFRASTRUCTURE"),
    ):
        result = classify_recovery({"tool": tool, "code": code, "status": "FAILED"}, snapshot)
        assert result["strategy"] == expected
        assert result["candidate"] == "diff-identity"
        assert result["source_sha256"] == "source-identity"
        assert result["revert_available"] is True


def test_registered_cuda_public_failure_replan_and_native_review(tmp_path, monkeypatch):
    from test_gpu_harness_repo_public import (
        GOAL,
        KERNEL,
        FakePublicChecker,
        _check,
        _edit_denominator,
        _edit_numerator,
        _finish,
        _registered_repository,
    )

    source, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([False, True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)

    class CudaModel(Model):
        def choose_graph_plan(self, snapshot, *, previous_plan, task_status, failure):
            self.plans.append({"previous": previous_plan, "failure": failure})
            tasks = [
                {
                    "task_id": "t1",
                    "gate": "inspect",
                    "target": KERNEL,
                    "objective": "Inspect the registered Softmax source and failure",
                    "depends_on": [],
                },
                {
                    "task_id": "t2",
                    "gate": "edit",
                    "target": KERNEL,
                    "objective": "Stabilize Softmax numerator and denominator",
                    "depends_on": ["t1"],
                },
                {
                    "task_id": "t3",
                    "gate": "public_check",
                    "target": KERNEL,
                    "objective": "Run public checker on the exact edited candidate",
                    "depends_on": ["t2"],
                },
                {
                    "task_id": "t4",
                    "gate": "finish",
                    "target": "",
                    "objective": "Stage the publicly passing candidate for independent review",
                    "depends_on": ["t3"],
                },
            ]
            return GraphTaskPlan.model_validate(
                {
                    "tasks": tasks,
                    "revision_reason": "Revise numeric repair"
                    if previous_plan
                    else "Initial numeric repair",
                    "failure_risk": "One-sided stabilization does not preserve the correct ratio",
                }
            )

    out = tmp_path / "native-public"
    first = CudaModel([_edit_numerator(), _check(), _edit_denominator(), _check()])
    options = dict(
        repository=source,
        base_sha=sha,
        output=out,
        goal=GOAL,
        public_case_id="h001",
        max_steps=8,
        max_model_calls=16,
        hierarchical_mode=True,
        runtime_mode="langgraph",
        review_public_pass=True,
    )
    report = run_repository_coding_session(**options, model_factory=lambda _: first)
    assert report.status == "PAUSED_RESUMABLE" and report.turns == 4
    assert len(checker.calls) == 2 and checker.calls[0] != checker.calls[1]
    assert len(first.plans) == 2
    info = inspect_native_graph(out)
    recoveries = [
        e["recovery"]["strategy"] for e in info["graph_events"] if e["kind"] == "failure_routed"
    ]
    assert "NUMERIC_REPAIR" in recoveries
    assert info["plan_revision"] == 2
    second = CudaModel([_finish()])
    final = run_repository_coding_session(**options, model_factory=lambda _: second, resume=True)
    assert final.status == "DIFF_STAGED_UNVERIFIED"
    assert final.public_checks_passed and final.turns == 5
    assert len(checker.calls) == 2 and len(second.plans) == 0
    assert inspect_native_graph(out)["task_status"][-1]["status"] == "DONE"


def test_native_ast_resume_preserves_exact_node_binding(tmp_path, monkeypatch):
    from test_gpu_harness_repo_public import GOAL, KERNEL, FakePublicChecker, _registered_repository

    from gpu_agent.harness.recovery import read_repository_checkpoint

    repo, sha = _registered_repository(tmp_path)
    out = tmp_path / "native-ast"
    checker = FakePublicChecker([True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    options = dict(
        repository=repo,
        output=out,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        runtime_mode="langgraph",
        syntax_mode=True,
        ast_target_binding=True,
        state_scoped_actions=True,
        max_steps=8,
        max_model_calls=10,
    )
    first = Model([action("repo_inspect_syntax", path=KERNEL, symbol="softmax32", limit=20)])
    result = run_repository_coding_session(
        **options,
        pause_after_turns=1,
        model_factory=lambda _: first,
    )
    assert result.status == "PAUSED_RESUMABLE"
    state = read_repository_checkpoint(out / "session-checkpoint.json").syntax_state
    assert state is not None
    d = next(node for node in state.observed_nodes if node.code == "denominator += expf(x);")
    n = next(
        node for node in state.observed_nodes if node.code == "out[i] = expf(own) / denominator;"
    )
    edits = [
        {
            "node_id": d.id,
            "expected_old_code": d.code,
            "replacement_code": "denominator += expf(x - maximum);",
        },
        {
            "node_id": n.id,
            "expected_old_code": n.code,
            "replacement_code": "out[i] = expf(own - maximum) / denominator;",
        },
    ]
    bad_edits = [{**edits[0], "expected_old_code": n.code}, edits[1]]

    def replace(rows):
        return action(
            "repo_replace_syntax",
            path=KERNEL,
            expected_source_sha256=state.source_sha256,
            edits=rows,
        )

    second = Model([replace(bad_edits), replace(edits), action("repo_run_public_checks"), FINISH])
    result = run_repository_coding_session(**options, resume=True, model_factory=lambda _: second)
    assert result.status == "DIFF_STAGED_UNVERIFIED"
    assert result.turns == 5 and len(checker.calls) == 1
    assert second.decisions[0].syntax_state == state
    info = inspect_native_graph(out)
    assert any(
        e["kind"] == "failure_routed" and e["recovery"]["strategy"] == "AST_REBIND"
        for e in info["graph_events"]
    )
    from gpu_agent.harness.repo_audit import read_repository_tool_events
    from gpu_agent.store import RunStore

    rows = read_repository_tool_events(
        RunStore(out / "public"),
        result.public_run_id,
        expected_count=5,
        require_action=True,
        require_observation=True,
    )
    assert rows[1]["status"] == "REJECTED"
    assert rows[1]["rejection_code"] == "CUDA_AST_TARGET_BINDING_MISMATCH"
    assert rows[1]["diff_sha256"] == rows[0]["diff_sha256"]
    assert rows[2]["status"] == "OK"


def test_real_provider_adapter_admits_planner_and_replanner_roles(repository, tmp_path):
    from pydantic import SecretStr

    from gpu_agent.agent.models import AgentBudget
    from gpu_agent.agent.policy import LLMCallGate, RepositoryCallBudget
    from gpu_agent.agent.provider import (
        DevelopmentCallPolicy,
        OpenAIProviderSettings,
        OpenAIResponsesProvider,
        SDKResult,
    )
    from gpu_agent.harness.repo_tools import RepositoryWorkspaceTools
    from gpu_agent.store import RunStore

    calls = []

    class Port:
        def call(self, request):
            calls.append(request)
            return SDKResult(
                value=plan(bool(request.payload["previous_plan"])).model_dump(mode="json")
            )

    store = RunStore(tmp_path / "wire")
    run = store.create_run("native_plan_wire")
    provider = OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.deepseek.com",
            model="mock-only",
            api_key=SecretStr("fixture-only"),
            supports_store_false=True,
        ),
        LLMCallGate(AgentBudget(max_llm_calls=10), reserve_repair_stages=False),
        store,
        run.id,
        port=Port(),
        call_policy=DevelopmentCallPolicy(max_llm_calls=10),
        repository_budget=RepositoryCallBudget(
            executor_decisions=4,
            executor_format_retries=2,
            planner_decisions=1,
            planner_format_retries=1,
            reflect_decisions=1,
            reflect_format_retries=1,
        ),
    )
    repo, sha = repository
    with RepositoryWorkspaceTools(origin=repo, output=tmp_path / "host", sha=sha) as host:
        snapshot = host.snapshot("Correct the float literal in source", [], 4)
        initial = provider.choose_graph_plan(
            snapshot, previous_plan=None, task_status=[], failure=None
        )
        revised = provider.choose_graph_plan(
            snapshot,
            previous_plan=initial.model_dump(mode="json"),
            task_status=[],
            failure={"code": "REJECTED"},
        )
    assert isinstance(initial, GraphTaskPlan) and isinstance(revised, GraphTaskPlan)
    assert len(calls) == 2
    assert all(req.payload["gpu_repo_graph_plan"] == "v1" for req in calls)
    assert [i.repository_role for i in provider.invocations()] == ["planner", "reflect"]
    assert all(i.prompt_version == "gpu-repository-graph-dag-v1" for i in provider.invocations())
