"""A bounded GPU coding-agent tool vocabulary, independent of tool order.

Codex/Claude Code-like planning is about model ownership of the NEXT action;
it does not require granting a model an unrestricted host shell.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ListFiles(StrictModel):
    tool: Literal["list_files"] = "list_files"


class ReadFile(StrictModel):
    tool: Literal["read_file"] = "read_file"
    path: Literal["kernel.cu", "vector_api.h", "vector_io.cpp"] = "kernel.cu"
    start_line: int = Field(default=1, ge=1, le=100_000)
    lines: int = Field(default=100, ge=1, le=200)


class SearchCode(StrictModel):
    tool: Literal["search_code"] = "search_code"
    needle: str = Field(min_length=1, max_length=120)


class FunctionalProbe(StrictModel):
    tool: Literal["functional_probe"] = "functional_probe"
    size: int = Field(ge=1, le=513)


class ApplyPatch(StrictModel):
    tool: Literal["apply_patch"] = "apply_patch"
    unified_diff: str = Field(min_length=20, max_length=200_000)


class ReplaceSnippet(StrictModel):
    tool: Literal["replace_snippet"] = "replace_snippet"
    path: Literal["kernel.cu"] = "kernel.cu"
    old_text: str = Field(min_length=1, max_length=4000)
    new_text: str = Field(max_length=4000)


class RunPublicChecks(StrictModel):
    tool: Literal["run_public_checks"] = "run_public_checks"


class RepoListFiles(StrictModel):
    tool: Literal["repo_list_files"] = "repo_list_files"
    suffix: str = Field(default="", max_length=20)


class RepoReadFile(StrictModel):
    tool: Literal["repo_read_file"] = "repo_read_file"
    path: str = Field(min_length=1, max_length=300)
    start_line: int = Field(default=1, ge=1, le=100_000)
    lines: int = Field(default=120, ge=1, le=200)


class RepoReplaceSnippet(StrictModel):
    tool: Literal["repo_replace_snippet"] = "repo_replace_snippet"
    path: str = Field(min_length=1, max_length=300)
    old_text: str = Field(min_length=1, max_length=4000)
    new_text: str = Field(max_length=4000)


class RepoGitDiff(StrictModel):
    tool: Literal["repo_git_diff"] = "repo_git_diff"


class RepoRunPublicChecks(StrictModel):
    """Public GPU check for an operator-bound registered HardSet case only."""

    tool: Literal["repo_run_public_checks"] = "repo_run_public_checks"


class FinishTask(StrictModel):
    tool: Literal["finish"] = "finish"
    summary: str = Field(min_length=5, max_length=700)


HarnessAction = Annotated[
    ListFiles
    | ReadFile
    | SearchCode
    | FunctionalProbe
    | ApplyPatch
    | ReplaceSnippet
    | RunPublicChecks
    | RepoListFiles
    | RepoReadFile
    | RepoReplaceSnippet
    | RepoGitDiff
    | RepoRunPublicChecks
    | FinishTask,
    Field(discriminator="tool"),
]


class HarnessDecision(StrictModel):
    """Exactly one action is selected by the model per agent turn."""

    action: HarnessAction
    reason: str = Field(min_length=1, max_length=300)


class ToolObservation(StrictModel):
    tool: str
    status: Literal["OK", "REJECTED", "FAILED"]
    content: str = Field(max_length=4000)
    artifact_id: str | None = None


class SessionSnapshot(StrictModel):
    goal: str = Field(min_length=3, max_length=1500)
    case_id: str = Field(min_length=1, max_length=80)
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    functional_requirement: str = Field(min_length=5, max_length=2000)
    base_git_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    registered_public_case: str | None = Field(default=None, pattern=r"^h00[1-3]$")
    current_candidate_sha256: str | None = None
    public_checks_passed: bool = False
    allowed_tools: tuple[str, ...]
    allowed_probe_sizes: tuple[int, ...] = ()
    remaining_steps: int = Field(ge=0)
    history: tuple[dict[str, str], ...] = Field(max_length=30)
    workspace_source: str = Field(default="", max_length=13000)
    context_summary: str = Field(default="", max_length=2800)
    total_public_actions: int = Field(default=0, ge=0)
    remaining_public_checks: int = Field(default=3, ge=0)


class RepoHarnessOutcome(StrictModel):
    schema_version: Literal["gpu-repository-harness-v1"] = "gpu-repository-harness-v1"
    status: Literal["DIFF_STAGED_UNVERIFIED", "INCONCLUSIVE", "PAUSED_RESUMABLE"]
    base_sha: str = Field(pattern=r"^[a-f0-9]{40}$")
    diff_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    edited_files: tuple[str, ...]
    turns: int = Field(ge=0, le=40)
    tool_calls: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    model_tokens: int | None = Field(default=None, ge=0)
    reason: str = Field(max_length=200)
    public_run_id: str
    public_checks_attempted: int = Field(default=0, ge=0, le=3)
    public_checks_passed: bool = False
    public_checked_candidate_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class HarnessOutcome(StrictModel):
    schema_version: Literal["gpu-coding-harness-v1"] = "gpu-coding-harness-v1"
    case_id: str
    status: Literal["VERIFIED_FIXED", "NOT_VERIFIED", "INCONCLUSIVE"]
    model_tool_turns: int = Field(ge=0)
    candidate_sha256: str | None = None
    tool_calls: int = Field(ge=0)
    model_physical_calls: int = Field(ge=0)
    model_tokens: int | None = Field(default=None, ge=0)
    reason: str = Field(max_length=300)
    public_run_id: str
    private_verification: bool = False

    @model_validator(mode="after")
    def truth(self) -> HarnessOutcome:
        if self.status == "VERIFIED_FIXED" and not self.private_verification:
            raise ValueError("public tool success never equals a verified fix")
        return self
