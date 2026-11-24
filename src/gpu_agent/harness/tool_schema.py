"""Trusted state-scoped structured-output grammars for repository Coding Agents.

A small local model MUST NOT see actions that cannot run in the current
controller state. This is schema-level least privilege, not scripted planning:
the model still selects an action and all source edits itself.
"""

from __future__ import annotations

from types import GenericAlias
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from gpu_agent.harness.models import (
    FinishTask,
    RepoCompileCandidate,
    RepoRevertLastEdit,
    RepoGitDiff,
    RepoInspectSyntax,
    RepoListFiles,
    RepoProbePublic,
    RepoReadFile,
    RepoReplaceMany,
    RepoReplaceSnippet,
    RepoReplaceSyntax,
    RepoRunPublicChecks,
    RepoSyntaxEdit,
    SyntaxState,
)

_REPO_ACTION_CLASSES: dict[str, type[BaseModel]] = {
    "repo_list_files": RepoListFiles,
    "repo_read_file": RepoReadFile,
    "repo_probe_public": RepoProbePublic,
    "repo_replace_snippet": RepoReplaceSnippet,
    "repo_replace_many": RepoReplaceMany,
    "repo_git_diff": RepoGitDiff,
    "repo_inspect_syntax": RepoInspectSyntax,
    "repo_replace_syntax": RepoReplaceSyntax,
    "repo_run_public_checks": RepoRunPublicChecks,
    "repo_compile_candidate": RepoCompileCandidate,
    "repo_revert_last_edit": RepoRevertLastEdit,
    "finish": FinishTask,
}


def repository_decision_schema(
    allowed_tools: tuple[str, ...],
    *,
    registered_kernel_path: str | None = None,
    public_probe_phase: Literal["before", "after"] | None = None,
    syntax_state: SyntaxState | None = None,
    bind_syntax_cursor: bool = False,
    recovery_source_sha256: str | None = None,
) -> type[BaseModel]:
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

    def bound_action(name: str) -> type[BaseModel]:
        base = _REPO_ACTION_CLASSES[name]
        fields: dict[str, Any] = {}
        if registered_kernel_path is not None and name in {
            "repo_inspect_syntax",
            "repo_replace_syntax",
            "repo_read_file",
        }:
            fields["path"] = (Literal[registered_kernel_path], ...)
        if name in {"repo_compile_candidate", "repo_revert_last_edit"}:
            if recovery_source_sha256 is None or registered_kernel_path is None:
                raise ValueError("candidate recovery exposed without bound source")
            fields["expected_source_sha256"] = (Literal[recovery_source_sha256], ...)
        if name == "repo_probe_public" and public_probe_phase is not None:
            fields["phase"] = (Literal[public_probe_phase], ...)
        if name == "repo_inspect_syntax" and bind_syntax_cursor:
            if syntax_state is not None and syntax_state.next_offset is None:
                raise ValueError("exhausted AST cursor exposed in model grammar")
            offset = syntax_state.next_offset if syntax_state else 0
            symbol = syntax_state.symbol if syntax_state else ""
            fields["offset"] = (Literal[offset], ...)
            fields["symbol"] = (Literal[symbol], ...)
            fields["limit"] = (Literal[8], ...)
        if name == "repo_replace_syntax" and bind_syntax_cursor:
            if syntax_state is None or not syntax_state.observed_nodes:
                raise ValueError("AST edit exposed without observed node evidence")
            aliases = tuple(
                node.id for node in syntax_state.observed_nodes if not node.code_truncated
            )
            if not aliases:
                raise ValueError("AST edit exposed without a complete node")
            if len(set(aliases)) != len(aliases):
                raise ValueError("duplicate AST aliases in model grammar")
            edit = create_model(
                "BoundRepoSyntaxEdit",
                __base__=RepoSyntaxEdit,
                node_id=(Literal[aliases], ...),
            )
            fields["expected_source_sha256"] = (Literal[syntax_state.source_sha256], ...)
            fields["edits"] = (GenericAlias(list, edit), Field(min_length=1, max_length=4))
        if fields:
            return create_model(f"Bound{name.title().replace('_', '')}", __base__=base, **fields)
        return base

    options = tuple(bound_action(name) for name in allowed_tools)
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
