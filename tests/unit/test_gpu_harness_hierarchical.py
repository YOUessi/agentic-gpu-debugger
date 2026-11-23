"""Zero-cost integration for Planner/Executor/Reflect and CUDA tree-sitter edits.

All syntax/decision tests use fixed public source or CPU fakes, never a paid
model, GPU checker, clean reference, or private evaluator.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from gpu_agent.harness.models import (
    HarnessDecision,
    SessionSnapshot,
    TaskPlan,
    TaskReflection,
    ToolObservation,
)
from gpu_agent.harness.planning import progress_from_tools, reflection_context
from gpu_agent.harness.runner import run_tool_agent
from gpu_agent.harness.syntax_tools import rewrite_syntax_nodes, syntax_nodes

ROOT = Path(__file__).resolve().parents[2]
CUDA_FILE = ROOT / "benchmarks/hardset_v2/h001/mutant/kernel.cu"


def _plan() -> TaskPlan:
    return TaskPlan.model_validate(
        {
            "milestones": [
                {
                    "gate": "inspect",
                    "objective": "Inspect the registered CUDA source for numerical faults",
                    "success_evidence": "Registered CUDA source inspected or supplied",
                },
                {
                    "gate": "edit",
                    "objective": "Create a SHA-bound scoped candidate patch with correct math",
                    "success_evidence": "An exact staged candidate diff appears in the workspace",
                },
                {
                    "gate": "public_check",
                    "objective": "Execute registered GPU functional and sanitizer public checks",
                    "success_evidence": "Controller public checks pass on that exact candidate",
                },
                {
                    "gate": "finish",
                    "objective": "Finish only after the public candidate is verifiably accepted",
                    "success_evidence": "Finish action accepted with no private verification claim",
                },
            ],
            "failure_risk": "Model may confuse visible line numbers with raw source edits",
        }
    )


def test_cuda_ast_nodes_match_exact_stable_function_statements() -> None:
    source = CUDA_FILE.read_bytes()
    indexed = syntax_nodes(source, symbol="softmax32")
    assert indexed["schema_version"] == "gpu-cuda-syntax-map-v1"
    assert indexed["source_sha256"] == hashlib.sha256(source).hexdigest()
    nodes = indexed["nodes"]
    selected = {
        item["code"]: item
        for item in nodes
        if "denominator += expf(x)" in item["code"] or "out[i] = expf(own)" in item["code"]
    }
    assert len(selected) == 2
    before = source
    updates = [
        (
            selected["denominator += expf(x);"]["id"],
            "denominator += expf(x - maximum);",
        ),
        (
            selected["out[i] = expf(own) / denominator;"]["id"],
            "out[i] = expf(own - maximum) / denominator;",
        ),
    ]
    candidate = rewrite_syntax_nodes(
        source,
        expected_source_sha256=indexed["source_sha256"],
        edits=updates,
    )
    assert candidate != source
    assert b"denominator += expf(x - maximum);" in candidate
    assert b"out[i] = expf(own - maximum) / denominator;" in candidate
    assert b"denominator += expf(x);" in before
    assert syntax_nodes(candidate, symbol="softmax32")["source_sha256"] != indexed["source_sha256"]
    with pytest.raises(ValueError, match="CUDA_SOURCE_SHA_STALE"):
        rewrite_syntax_nodes(
            candidate, expected_source_sha256=indexed["source_sha256"], edits=updates
        )
    with pytest.raises(ValueError, match="CUDA_SYNTAX_DUPLICATE_NODE"):
        rewrite_syntax_nodes(
            source,
            expected_source_sha256=indexed["source_sha256"],
            edits=[updates[0], updates[0]],
        )
    with pytest.raises(ValueError, match="CUDA_SYNTAX_REPLACEMENT_PARSE_ERROR"):
        rewrite_syntax_nodes(
            source,
            expected_source_sha256=indexed["source_sha256"],
            edits=[(updates[0][0], "denominator += (")],
        )


def test_ast_inspection_rejects_nonexistent_function_and_invalid_source() -> None:
    with pytest.raises(ValueError, match="CUDA_SYNTAX_SYMBOL_NOT_FOUND"):
        syntax_nodes(CUDA_FILE.read_bytes(), symbol="madeUpSymbol")
    with pytest.raises(ValueError, match="CUDA_SYNTAX_SOURCE_HAS_ERRORS"):
        syntax_nodes(b"__global__ void broken( {\n")


class FakeRepositoryHost:
    completion_reason = "WORKSPACE_DIFF_STAGED_UNVERIFIED"

    def __init__(self) -> None:
        self.tools: list[str] = []
        self.current_source = "denominator += expf(x);"

    @property
    def accepted_diff(self) -> str:
        return "diff-candidate"

    @property
    def tool_calls(self) -> int:
        return len(self.tools)

    def snapshot(self, goal: str, history: list[dict[str, str]], steps: int) -> SessionSnapshot:
        return SessionSnapshot(
            goal=goal,
            case_id="repository",
            source_hash="0" * 64,
            functional_requirement="Stable CUDA softmax32 of a+b",
            registered_public_case="h001",
            context_mode="evidence_first",
            allowed_tools=(
                "repo_read_file",
                "repo_replace_syntax",
                "repo_run_public_checks",
                "finish",
            ),
            remaining_steps=steps,
            workspace_source=self.current_source,
            history=tuple(history),
        )

    def invoke(self, action: Any) -> ToolObservation:
        self.tools.append(action.tool)
        return ToolObservation(tool=action.tool, status="OK", content="Registered tool succeeded")


class FakeTwoLevelModel:
    def __init__(self) -> None:
        self.planner_calls = 0
        self.reflection_calls = 0
        self.seen: list[SessionSnapshot] = []

    def choose_repository_plan(self, snapshot: SessionSnapshot) -> TaskPlan:
        self.planner_calls += 1
        assert snapshot.task_plan is None
        return _plan()

    def reflect_repository_task(self, snapshot: SessionSnapshot) -> TaskReflection:
        self.reflection_calls += 1
        assert snapshot.task_progress is not None
        assert snapshot.task_progress.public_checked
        return TaskReflection(
            observed_failure="No observed public GPU failures in this scripted CPU test",
            avoid_next="Do not repeat already successful source inspections",
            next_priority="Submit finish because the public gate is accepted",
        )

    def choose_harness_v1(self, snapshot: SessionSnapshot) -> HarnessDecision:
        self.seen.append(snapshot)
        assert snapshot.task_plan is not None
        assert snapshot.task_progress is not None
        n = len(self.seen)
        if n == 1:
            action: dict[str, Any] = {"tool": "repo_read_file", "path": "kernel.cu"}
        elif n == 2:
            action = {
                "tool": "repo_replace_syntax",
                "path": "kernel.cu",
                "expected_source_sha256": "a" * 64,
                "edits": [{"node_id": "a" * 24, "replacement_code": "x = 2;"}],
            }
        elif n == 3:
            action = {"tool": "repo_run_public_checks"}
        else:
            assert snapshot.task_reflection is not None
            assert snapshot.task_reflection.next_priority.startswith("Submit finish")
            assert snapshot.task_progress.active_gate == "finish"
            action = {"tool": "finish", "summary": "Public GPU check accepted the candidate"}
        return HarnessDecision.model_validate(
            {"action": action, "reason": "Follow the global milestone and current state"}
        )


def test_planner_executor_reflect_graph_updates_verified_milestones() -> None:
    host = FakeRepositoryHost()
    model = FakeTwoLevelModel()
    checkpoints: list[dict[str, Any]] = []
    result = run_tool_agent(
        model,
        host,
        goal="Fix numerical Softmax fault in public registered CUDA kernel",
        max_steps=6,
        hierarchical_mode=True,
        on_checkpoint=lambda state: checkpoints.append(dict(state)),
    )
    assert result["reason"] == "WORKSPACE_DIFF_STAGED_UNVERIFIED"
    assert host.tools == [
        "repo_read_file",
        "repo_replace_syntax",
        "repo_run_public_checks",
        "finish",
    ]
    assert model.planner_calls == 1
    assert model.reflection_calls == 1
    assert len(model.seen) == 4
    assert model.seen[2].task_reflection is None
    assert model.seen[3].task_reflection is not None
    assert checkpoints[0]["task_progress"]["inspected"]
    assert checkpoints[1]["task_progress"]["edited"]
    assert checkpoints[2]["task_progress"]["public_checked"]
    assert checkpoints[-1]["task_progress"]["finished"]


def test_reflection_compacts_repeated_rejections_without_erasing_failures() -> None:
    reason = TaskReflection(
        observed_failure="The same file was repeatedly read",
        avoid_next="Reusing the same read is not new evidence",
        next_priority="Apply the already identified syntax replacement",
    )
    history = [
        {"tool": "repo_read_file", "status": "OK", "content": "source map"},
        *[
            {
                "tool": "repo_read_file",
                "status": "REJECTED",
                "content": "ValueError: REDUNDANT_READ: already read",
            }
            for _ in range(5)
        ],
        {
            "tool": "repo_replace_syntax",
            "status": "REJECTED",
            "content": "ValueError: CUDA_SOURCE_SHA_STALE",
        },
    ]
    projection = reflection_context(history, reflection=reason, max_recent=4)
    assert sum("REDUNDANT_READ" in item["content"] for item in projection) == 1
    assert any("CUDA_SOURCE_SHA_STALE" in item["content"] for item in projection)
    assert len(history) == 7
    progress = progress_from_tools(history, current_source_visible=True)
    assert progress.active_gate == "edit"
    assert not progress.edited
    with pytest.raises(ValidationError):
        TaskPlan.model_validate(
            {
                "milestones": [
                    {
                        "gate": "finish",
                        "objective": "Just finish immediately without any public checks",
                        "success_evidence": "Unverified operator claim",
                    }
                ]
                * 4,
                "failure_risk": "No independent verification available",
            }
        )
