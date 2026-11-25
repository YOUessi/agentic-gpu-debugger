"""Targeted regression gates from the 5/15 Native DAG development failures.

Real model requests and Strict CUDA validation are intentionally NOT retried.
These are test fixtures for runtime liveness, typed plans and audit boundaries.
"""

from __future__ import annotations

import json

import pytest
from pydantic import SecretStr, ValidationError
from test_native_langgraph_runtime import BAD, READ, Model, args, plan
from test_native_langgraph_runtime import repository as repository

from gpu_agent.agent.models import AgentBudget, ProviderError
from gpu_agent.agent.policy import LLMCallGate, RepositoryCallBudget
from gpu_agent.agent.provider import (
    DevelopmentCallPolicy,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
    SDKResult,
)
from gpu_agent.harness.graph_plan import GraphTaskPlan, scoped_graph_plan_model
from gpu_agent.harness.graph_schedule import assess_frontier_task
from gpu_agent.harness.journal import read_verified_journal, recovery_disposition
from gpu_agent.harness.native_graph import EFFECT_NAME, inspect_native_graph, merge_events
from gpu_agent.harness.repo_runner import run_repository_coding_session


def test_frontier_with_one_step_and_multiple_open_milestones_is_live():
    row = assess_frontier_task(
        {
            "task": plan().tasks[1].model_dump(mode="json"),
            "round": [2, 13],
            "position": 0,
            "descendants": 3,
            "remaining_steps": 1,
            "unfinished": 6,
            "allowed": ["repo_replace_snippet", "repo_read_file"],
            "recovery": {"strategy": "AST_REBIND"},
        }
    )
    assert row["feasible"] and "repo_replace_snippet" in row["scope"]
    assert row["round"] == [2, 13]


def test_duplicate_graph_events_are_idempotent_not_colliding():
    one = {"id": "task_dispatched:1:2", "kind": "task_dispatched", "task": "t2"}
    assert merge_events([one], [one]) == [one]
    with pytest.raises(ValueError, match="collision"):
        merge_events([one], [{**one, "task": "t3"}])


def test_registered_dynamic_wire_schema_is_path_bound_and_generic_gates_filtered():
    kernel = "benchmarks/hardset_v2/h001/mutant/kernel.cu"
    registered = scoped_graph_plan_model(
        {
            "registered_public_case": "h001",
            "registered_kernel_path": kernel,
        }
    )
    task_schema = registered.model_json_schema()["$defs"]["RegisteredGraphTask"]
    assert task_schema["properties"]["target"]["enum"] == ["", kernel]
    sample = {
        "tasks": [
            {
                "task_id": "t1",
                "objective": "Edit only the registered CUDA kernel",
                "gate": "edit",
                "target": kernel,
                "depends_on": [],
            },
            {
                "task_id": "t2",
                "objective": "Check the exact candidate publicly",
                "gate": "public_check",
                "target": kernel,
                "depends_on": ["t1"],
            },
            {
                "task_id": "t3",
                "objective": "Stage the checked candidate for review",
                "gate": "finish",
                "target": "",
                "depends_on": ["t2"],
            },
        ],
        "revision_reason": "Initial registered CUDA plan",
        "failure_risk": "The numeric check might fail",
    }
    assert isinstance(registered.model_validate(sample), GraphTaskPlan)
    sample["tasks"][0]["target"] = "benchmarks/hardset_v2/h002/clean/kernel.cu"
    with pytest.raises(ValidationError):
        registered.model_validate(sample)
    generic = scoped_graph_plan_model(
        {
            "registered_public_case": None,
            "registered_kernel_path": None,
        }
    )
    gate_enum = generic.model_json_schema()["$defs"]["UnregisteredGraphTask"]["properties"]["gate"][
        "enum"
    ]
    assert gate_enum == ["inspect", "edit", "finish"]
    sample["tasks"][0]["target"] = kernel
    with pytest.raises(ValidationError):
        generic.model_validate(sample)


def _provider(host, port, max_calls=16):
    return OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.deepseek.com",
            model="fixture-only",
            api_key=SecretStr("fixture-key-not-sent"),
            supports_store_false=True,
        ),
        LLMCallGate(AgentBudget(max_llm_calls=max_calls), reserve_repair_stages=False),
        host.store,
        host.run_id,
        port=port,
        call_policy=DevelopmentCallPolicy(max_llm_calls=max_calls),
        repository_budget=RepositoryCallBudget(
            executor_decisions=8,
            executor_format_retries=2,
            planner_decisions=1,
            planner_format_retries=1,
            reflect_decisions=2,
            reflect_format_retries=2,
        ),
    )


def test_provider_invalid_scope_uses_exactly_one_existing_planner_retry(repository, tmp_path):
    from gpu_agent.harness.repo_tools import RepositoryWorkspaceTools

    sample = plan().model_dump(mode="json")

    class Port:
        def __init__(self):
            self.calls = []

        def call(self, request):
            self.calls.append(request)
            value = json.loads(json.dumps(sample))
            if len(self.calls) == 1:
                value["tasks"][0]["gate"] = "public_check"
            return SDKResult(value=value)

    port = Port()
    repo, sha = repository
    with RepositoryWorkspaceTools(origin=repo, output=tmp_path / "host", sha=sha) as host:
        provider = _provider(host, port)
        snap = host.snapshot("Correct the CUDA source", [], 8)
        actual = provider.choose_graph_plan(snap, previous_plan=None, task_status=[], failure=None)
        assert actual == plan()
        calls = provider.invocations()
    assert len(port.calls) == 2 and len(calls) == 2
    assert [v.attempt for v in calls] == [0, 1]
    assert all(v.repository_role == "planner" for v in calls)
    assert calls[0].error_code == "LLM_INVALID_OUTPUT"
    assert calls[1].state == "COMPLETED"
    assert calls[0].output_diagnostics.failure_class == "DOMAIN_REJECTED"


@pytest.mark.parametrize("stage", ["planner", "executor", "reflect"])
def test_failed_provider_role_produces_final_result_and_provenance(repository, tmp_path, stage):
    class RoleError(Model):
        def choose_graph_plan(self, snapshot, *, previous_plan, task_status, failure):
            if stage == "planner" or stage == "reflect" and previous_plan is not None:
                raise ProviderError("LLM_INVALID_OUTPUT", state="FAILED")
            return plan()

        def choose_harness_v1(self, snapshot):
            if stage == "executor":
                raise ProviderError("LLM_INVALID_OUTPUT", state="FAILED")
            return super().choose_harness_v1(snapshot)

    out = tmp_path / stage
    model = RoleError([READ, BAD])
    report = run_repository_coding_session(**args(repository, out, model))
    assert report.status == "INCONCLUSIVE"
    assert (out / "result.json").is_file()
    assert report.reason == "GRAPH_" + stage.upper() + "_INVALID_OUTPUT"
    metadata = json.loads((out / "terminal-diagnostics.json").read_text())
    assert metadata["failure_stage"] == ("replanner" if stage == "reflect" else stage)
    assert metadata["reason"] == report.reason
    assert metadata["model_failure"]["error_code"] == "LLM_INVALID_OUTPUT"
    assert metadata["model_failure"]["replay_allowed"] is False
    assert metadata["no_external_effect_replay"] is True
    assert metadata["effect_journal_file_sha256"]
    if stage == "reflect":
        assert report.turns == report.tool_calls == 2
        assert metadata["last_complete_checkpoint_turn"] == 2
        assert metadata["final_source_sha256_by_path"]["kernel.cu"]
    else:
        assert report.turns == 0
        assert metadata["last_complete_checkpoint_turn"] is None
    effects = read_verified_journal(out / EFFECT_NAME)
    assert effects[-1]["kind"] == "model_failed"
    assert effects[-1]["payload"]["role"] == stage
    assert recovery_disposition(read_verified_journal(out / "session-events.jsonl")) == "FINISHED"
    assert report.reason == inspect_native_graph(out)["reason"]
    with pytest.raises(ValueError):
        run_repository_coding_session(**args(repository, out, RoleError([]), resume=True))


def test_uncertain_provider_failure_is_finalized_but_never_resumable(repository, tmp_path):
    class Unknown(Model):
        def choose_graph_plan(self, snapshot, **kw):
            raise ProviderError("LLM_WORKER_ERROR", state="UNCERTAIN")

    out = tmp_path / "uncertain"
    report = run_repository_coding_session(**args(repository, out, Unknown([])))
    assert report.status == "INCONCLUSIVE"
    d = json.loads((out / "terminal-diagnostics.json").read_text())
    assert d["model_failure"]["delivery_state"] == "UNCERTAIN"
    assert recovery_disposition(read_verified_journal(out / EFFECT_NAME)) == (
        "UNCERTAIN_EXTERNAL_EFFECT_RECONCILE_MANUALLY"
    )
    with pytest.raises(ValueError):
        run_repository_coding_session(**args(repository, out, Unknown([]), resume=True))


def test_dag_frontier_no_admissible_tool_ends_audited_without_any_tool(repository, tmp_path):
    from test_gpu_harness_repo_public import GOAL, KERNEL, _registered_repository

    source, sha = _registered_repository(tmp_path)

    class WrongGateModel(Model):
        def choose_graph_plan(self, snapshot, **kw):
            return GraphTaskPlan.model_validate(
                {
                    "tasks": [
                        {
                            "task_id": "t1",
                            "objective": "Inspect the mutant CUDA kernel",
                            "gate": "inspect",
                            "target": KERNEL,
                            "depends_on": [],
                        },
                        {
                            "task_id": "t2",
                            "objective": "Run public checks before any edit",
                            "gate": "public_check",
                            "target": KERNEL,
                            "depends_on": ["t1"],
                        },
                        {
                            "task_id": "t3",
                            "objective": "Edit kernel after premature checks",
                            "gate": "edit",
                            "target": KERNEL,
                            "depends_on": ["t2"],
                        },
                        {
                            "task_id": "t4",
                            "objective": "Finish and stage the kernel candidate",
                            "gate": "finish",
                            "target": "",
                            "depends_on": ["t3"],
                        },
                    ],
                    "revision_reason": "A deliberately unusable dependency chain",
                    "failure_risk": "Checks require a modified candidate",
                }
            )

        def choose_harness_v1(self, snapshot):
            raise AssertionError("no tool should be admitted")

    out = tmp_path / "blocked"
    result = run_repository_coding_session(
        repository=source,
        output=out,
        base_sha=sha,
        goal=GOAL,
        max_steps=8,
        max_model_calls=16,
        public_case_id="h001",
        hierarchical_mode=True,
        runtime_mode="langgraph",
        model_factory=lambda _: WrongGateModel([]),
    )
    assert result.status == "INCONCLUSIVE"
    assert result.tool_calls == 0
    assert result.reason == "GRAPH_TASK_NO_ADMISSIBLE_TOOL"
    assert (out / "result.json").is_file()
    d = json.loads((out / "terminal-diagnostics.json").read_text())
    assert d["failure_stage"] == "dag_scheduler"
    g = inspect_native_graph(out)
    assert any(e["kind"] == "task_frontier_terminal" for e in g["graph_events"])


def test_structured_scope_invalid_model_fixture_is_terminal_with_source_hash(tmp_path):
    from test_gpu_harness_repo_public import GOAL, KERNEL, _registered_repository

    source, sha = _registered_repository(tmp_path)

    class InvalidScope(Model):
        def choose_graph_plan(self, snapshot, **kw):
            tasks = [
                {
                    "task_id": "t1",
                    "objective": "Inspect the registered kernel",
                    "gate": "inspect",
                    "target": KERNEL,
                    "depends_on": [],
                },
                {
                    "task_id": "t2",
                    "objective": "Edit an illegal clean reference",
                    "gate": "edit",
                    "target": "benchmarks/hardset_v2/h002/clean/kernel.cu",
                    "depends_on": ["t1"],
                },
                {
                    "task_id": "t3",
                    "objective": "Publicly check the current candidate",
                    "gate": "public_check",
                    "target": KERNEL,
                    "depends_on": ["t2"],
                },
                {
                    "task_id": "t4",
                    "objective": "Finish the candidate after public validation",
                    "gate": "finish",
                    "target": "",
                    "depends_on": ["t3"],
                },
            ]
            return GraphTaskPlan.model_validate(
                {
                    "tasks": tasks,
                    "revision_reason": "Attempt out-of-scope source edit",
                    "failure_risk": "Source does not match registered scope",
                }
            )

    out = tmp_path / "invalid-plan"
    report = run_repository_coding_session(
        repository=source,
        output=out,
        base_sha=sha,
        goal=GOAL,
        max_steps=8,
        max_model_calls=16,
        public_case_id="h001",
        runtime_mode="langgraph",
        hierarchical_mode=True,
        model_factory=lambda _: InvalidScope([]),
    )
    assert report.reason == "GRAPH_PLAN_INVALID_SCOPE"
    assert report.turns == 0 and report.status == "INCONCLUSIVE"
    d = json.loads((out / "terminal-diagnostics.json").read_text())
    assert d["failure_stage"] == "planner"
    assert KERNEL in d["final_source_sha256_by_path"]
    assert d["last_complete_checkpoint_sha256"] is None


@pytest.mark.parametrize(
    "role,expected_calls",
    [
        ("planner", 2),
        ("executor", 3),
        ("reflect", 5),
    ],
)
def test_real_audited_provider_adapter_exhausts_only_role_retry_and_finalizes(
    repository, tmp_path, role, expected_calls
):
    """Exercise actual provider audit and _call role budgets, never paid API."""

    class Port:
        def __init__(self):
            self.calls = []
            self.executor_turn = 0

        def call(self, request):
            self.calls.append(request)
            payload = request.payload
            if payload.get("gpu_repo_graph_plan") == "v1":
                value = plan().model_dump(mode="json")
                failing = (
                    role == "planner"
                    and payload.get("previous_plan") is None
                    or role == "reflect"
                    and payload.get("previous_plan") is not None
                )
                if failing:
                    value["revision_reason"] = "x" * 400  # Schema-invalid twice.
                return SDKResult(value=value)
            if payload.get("gpu_repo_harness") == "v1":
                if role == "executor":
                    return SDKResult(value={"invalid": True})
                action = READ if self.executor_turn == 0 else BAD
                self.executor_turn += 1
                return SDKResult(
                    value={
                        "action": action,
                        "reason": "Evidence-backed model action",
                    }
                )
            raise AssertionError("unrecognized provider role")

    out = tmp_path / f"real-adapter-{role}"
    port = Port()
    options = args(repository, out, None)
    options["model_factory"] = lambda host: _provider(host, port)
    report = run_repository_coding_session(**options)
    assert report.status == "INCONCLUSIVE"
    assert report.reason == f"GRAPH_{role.upper()}_INVALID_OUTPUT"
    assert report.model_calls == expected_calls
    assert (out / "result.json").is_file()
    info = json.loads((out / "terminal-diagnostics.json").read_text())
    assert info["failure_stage"] == ("replanner" if role == "reflect" else role)
    assert info["role_format_retry_counts"][role] == 1
    assert info["role_call_counts"][role] == 2
    assert info["last_complete_checkpoint_turn"] == (2 if role == "reflect" else None)
    assert (
        sum(
            x["count"]
            for x in info["role_error_counts"]
            if x["role"] == role and x["code"] == "LLM_INVALID_OUTPUT"
        )
        == 2
    )
    assert (
        info["final_source_sha256_by_path"].get("kernel.cu") is not None
        if role == "reflect"
        else True
    )
    assert recovery_disposition(read_verified_journal(out / "session-events.jsonl")) == "FINISHED"
    assert inspect_native_graph(out)["reason"] == report.reason
    with pytest.raises(ValueError):
        run_repository_coding_session(**{**options, "resume": True})


def test_cli_defaults_to_native_react_and_dag_is_explicit_opt_in(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from gpu_agent.cli import app

    captured = []

    class Result:
        status = "DIFF_STAGED_UNVERIFIED"

        def model_dump_json(self):
            return '{"status":"DIFF_STAGED_UNVERIFIED"}'

    def fake_run(**kwargs):
        captured.append(kwargs)
        return Result()

    monkeypatch.setattr(
        "gpu_agent.harness.repo_runner.run_repository_coding_session",
        fake_run,
    )
    args = [
        "harness",
        "repo-run",
        "--output",
        str(tmp_path / "dummy"),
        "--goal",
        "Fix the registered CUDA defect",
        "--base-sha",
        "a" * 40,
        "--allow-paid-calls",
    ]
    cli = CliRunner()
    default = cli.invoke(app, args)
    assert default.exit_code == 0, default.output
    assert captured[-1]["runtime_mode"] == "langgraph"
    assert captured[-1]["hierarchical_mode"] is False
    dag = cli.invoke(app, [*args, "--hierarchical"])
    assert dag.exit_code == 0, dag.output
    assert captured[-1]["runtime_mode"] == "langgraph"
    assert captured[-1]["hierarchical_mode"] is True
