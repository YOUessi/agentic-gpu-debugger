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
