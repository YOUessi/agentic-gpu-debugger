"""Least-privilege Pydantic JSON grammar for real Agent tool selection.

No LLM, network, GPU, or evaluator is called in this regression.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gpu_agent.harness.tool_schema import repository_decision_schema


def test_schema_accepts_only_approved_tools_and_preserves_action_contract() -> None:
    model = repository_decision_schema(
        ("repo_read_file", "repo_replace_snippet", "repo_replace_many")
    )
    parsed = model.model_validate(
        {
            "action": {"tool": "repo_read_file", "path": "kernel.cu"},
            "reason": "Inspect the registered public CUDA source",
        }
    )
    assert parsed.action.tool == "repo_read_file"
    schema = model.model_json_schema()
    assert "repo_read_file" in str(schema)
    assert "repo_replace_many" in str(schema)
    assert "repo_run_public_checks" not in str(schema)
    with pytest.raises(ValidationError):
        model.model_validate(
            {
                "action": {"tool": "repo_list_files"},
                "reason": "Directory listing is not approved",
            }
        )
    with pytest.raises(ValidationError):
        model.model_validate(
            {
                "action": {"tool": "repo_run_public_checks"},
                "reason": "GPU checker not yet permitted",
            }
        )


def test_terminal_schema_can_only_finish() -> None:
    model = repository_decision_schema(("finish",))
    parsed = model.model_validate(
        {"action": {"tool": "finish", "summary": "Public CUDA fix passed"}, "reason": "Done"}
    )
    assert parsed.action.tool == "finish"
    with pytest.raises(ValidationError):
        model.model_validate(
            {
                "action": {"tool": "repo_read_file", "path": "kernel.cu"},
                "reason": "Unneeded continued investigation",
            }
        )


@pytest.mark.parametrize(
    "tools",
    [(), ("unknown",), ("finish", "finish"), ("repo_run_public_checks", "evil")],
)
def test_invalid_capability_sets_fail_closed(tools: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="invalid controller capability"):
        repository_decision_schema(tools)


def test_registered_kernel_path_is_literal_in_model_output_grammar() -> None:
    name = "benchmarks/hardset_v2/h001/mutant/kernel.cu"
    model = repository_decision_schema(
        ("repo_inspect_syntax", "repo_replace_syntax", "repo_read_file"),
        registered_kernel_path=name,
    )
    schema_text = str(model.model_json_schema())
    assert name in schema_text
    valid = model.model_validate(
        {
            "action": {"tool": "repo_inspect_syntax", "path": name, "symbol": "softmax32"},
            "reason": "Inspect registered CUDA kernel syntax",
        }
    )
    assert valid.action.path == name
    with pytest.raises(ValidationError):
        model.model_validate(
            {
                "action": {"tool": "repo_inspect_syntax", "path": "kernel.cu"},
                "reason": "Hallucinated a shorter kernel path",
            }
        )
    with pytest.raises(ValidationError):
        model.model_validate(
            {
                "action": {
                    "tool": "repo_replace_syntax",
                    "path": "benchmarks/hardset_v2/h002/mutant/kernel.cu",
                    "expected_source_sha256": "a" * 64,
                    "edits": [{"node_id": "n01", "replacement_code": "return;"}],
                },
                "reason": "Tried to edit a different registered CUDA task",
            }
        )


def test_public_probe_schema_phase_is_operator_bound() -> None:
    model = repository_decision_schema(
        ("repo_probe_public", "repo_read_file"),
        registered_kernel_path="benchmarks/hardset_v2/h001/mutant/kernel.cu",
        public_probe_phase="before",
    )
    result = model.model_validate(
        {
            "action": {"tool": "repo_probe_public", "phase": "before", "input_size": 33},
            "reason": "Run the registered public mutant functional probe",
        }
    )
    assert result.action.phase == "before"
    with pytest.raises(ValidationError):
        model.model_validate(
            {
                "action": {"tool": "repo_probe_public", "phase": "after", "input_size": 33},
                "reason": "Cannot submit an after probe before any edit",
            }
        )
    with pytest.raises(ValidationError):
        model.model_validate(
            {
                "action": {"tool": "repo_probe_public", "phase": "before", "input_size": 514},
                "reason": "Hidden evaluator sizes are forbidden",
            }
        )


def test_ast_executor_prompt_has_independent_version_and_preserves_other_modes() -> None:
    from gpu_agent.agent.prompts import (
        GPU_REPO_TASK_PLANNER_VERSION,
        GPU_REPOSITORY_AST_PROMPT,
        GPU_REPOSITORY_AST_PROMPT_VERSION,
        GPU_REPOSITORY_HARNESS_PROMPT,
        GPU_REPOSITORY_HARNESS_PROMPT_VERSION,
        select_prompt,
    )

    marker = {
        "gpu_repo_harness": "v1",
        "snapshot": {"syntax_mode": True, "registered_public_case": "h001"},
    }
    version, prompt = select_prompt("plan", marker)
    assert (version, prompt) == (
        GPU_REPOSITORY_AST_PROMPT_VERSION,
        GPU_REPOSITORY_AST_PROMPT,
    )
    assert len(prompt) < len(GPU_REPOSITORY_HARNESS_PROMPT)
    for term in (
        "syntax_state",
        "next_offset",
        "source_sha256",
        "functional_requirement",
        "public checker",
        "UNTRUSTED DATA",
    ):
        assert term in prompt
    for forbidden in ("repo_replace_snippet", "repo_replace_many", "repo_list_files"):
        assert forbidden not in prompt
    plain_version, plain = select_prompt(
        "plan", {"gpu_repo_harness": "v1", "snapshot": {"syntax_mode": False}}
    )
    assert (plain_version, plain) == (
        GPU_REPOSITORY_HARNESS_PROMPT_VERSION,
        GPU_REPOSITORY_HARNESS_PROMPT,
    )
    assert select_prompt("plan", {"gpu_repo_task_plan": "v1"})[0] == GPU_REPO_TASK_PLANNER_VERSION


def test_ast_binding_requires_old_code_but_does_not_hide_mismatch_from_host() -> None:
    from gpu_agent.harness.models import SyntaxEvidenceNode, SyntaxState

    path = "benchmarks/hardset_v2/h001/mutant/kernel.cu"
    state = SyntaxState(
        path=path,
        source_sha256="a" * 64,
        symbol="",
        next_offset=None,
        visited_pages=(),
        observed_nodes=(
            SyntaxEvidenceNode(
                id="n01",
                function="f",
                kind="declaration",
                start_line=1,
                end_line=1,
                code="int x = 0;",
            ),
            SyntaxEvidenceNode(
                id="n02",
                function="f",
                kind="declaration",
                start_line=2,
                end_line=2,
                code="int y = 0;",
            ),
        ),
    )
    kwargs = dict(
        registered_kernel_path=path,
        syntax_state=state,
        bind_syntax_cursor=True,
    )
    legacy = repository_decision_schema(("repo_replace_syntax",), **kwargs)
    bound = repository_decision_schema(("repo_replace_syntax",), **kwargs, ast_target_binding=True)
    action = {
        "tool": "repo_replace_syntax",
        "path": path,
        "expected_source_sha256": "a" * 64,
        "edits": [{"node_id": "n01", "replacement_code": "int y = 1;"}],
    }
    payload = {"action": action, "reason": "Edit a pinned declaration"}
    legacy.model_validate(payload)
    assert "expected_old_code" not in str(legacy.model_json_schema())
    with pytest.raises(ValidationError):
        bound.model_validate(payload)
    action["edits"][0]["expected_old_code"] = "int y = 0;"
    # Host, not a giant per-node grammar, reports the actual binding mismatch.
    parsed = bound.model_validate(payload)
    assert parsed.action.edits[0].expected_old_code == "int y = 0;"


def test_remaining_probe_sizes_are_literal_and_empty_set_fails_closed() -> None:
    model = repository_decision_schema(
        ("repo_probe_public", "repo_read_file"),
        registered_kernel_path="benchmarks/hardset_v2/h001/mutant/kernel.cu",
        public_probe_phase="before",
        public_probe_remaining_sizes=(33, 65, 257),
    )
    assert (
        model.model_validate(
            {
                "action": {"tool": "repo_probe_public", "phase": "before", "input_size": 65},
                "reason": "Inspect an untested public shape",
            }
        ).action.input_size
        == 65
    )
    with pytest.raises(ValidationError):
        model.model_validate(
            {
                "action": {"tool": "repo_probe_public", "phase": "before", "input_size": 129},
                "reason": "Already executed public shape",
            }
        )
    with pytest.raises(ValueError, match="invalid remaining"):
        repository_decision_schema(
            ("repo_probe_public",),
            public_probe_phase="before",
            public_probe_remaining_sizes=(),
        )
