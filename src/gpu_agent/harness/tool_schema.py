"""Trusted state-scoped structured-output grammars for repository Coding Agents.

A small local model MUST NOT see actions that cannot run in the current
controller state. This is schema-level least privilege, not scripted planning:
the model still selects an action and all source edits itself.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from gpu_agent.harness.models import (
    FinishTask,
    RepoGitDiff,
    RepoInspectSyntax,
    RepoReplaceSyntax,
    RepoListFiles,
    RepoReadFile,
    RepoReplaceMany,
    RepoReplaceSnippet,
    RepoRunPublicChecks,
)

_REPO_ACTION_CLASSES: dict[str, type[BaseModel]] = {
    "repo_list_files": RepoListFiles,
    "repo_read_file": RepoReadFile,
    "repo_replace_snippet": RepoReplaceSnippet,
    "repo_replace_many": RepoReplaceMany,
    "repo_git_diff": RepoGitDiff,
    "repo_inspect_syntax": RepoInspectSyntax,
    "repo_replace_syntax": RepoReplaceSyntax,
    "repo_run_public_checks": RepoRunPublicChecks,
    "finish": FinishTask,
}


def repository_decision_schema(allowed_tools: tuple[str, ...]) -> type[BaseModel]:
    """Return a strict Pydantic response schema for EXACTLY approved tools.

    The backend independently verifies these capabilities at execution time;
    narrowing model output only removes impossible requests at generation time.
    """
    if (
        not allowed_tools
        or len(allowed_tools) > len(_REPO_ACTION_CLASSES)
        or len(set(allowed_tools)) != len(allowed_tools)
        or any(name not in _REPO_ACTION_CLASSES for name in allowed_tools)
    ):
        raise ValueError("registered model tool grammar has invalid controller capability set")
    options = tuple(_REPO_ACTION_CLASSES[name] for name in allowed_tools)
    action: Any
    if len(options) == 1:
        action = options[0]
    else:
        # Pydantic compiles a tagged union into a JSON-Schema discriminator,
        # so the model cannot choose a tool omitted by the operator.
        combined: Any = options[0]
        for variant in options[1:]:
            combined = combined | variant
        action = Annotated[combined, Field(discriminator="tool")]
    return create_model(
        "ApprovedRepositoryHarnessDecision",
        action=(action, ...),
        reason=(str, Field(min_length=1, max_length=300)),
        __config__=ConfigDict(extra="forbid", frozen=True, strict=True),
    )
