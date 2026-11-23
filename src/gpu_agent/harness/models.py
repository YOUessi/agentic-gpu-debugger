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


class RepoSnippetEdit(StrictModel):
    old_text: str = Field(min_length=1, max_length=4000)
    new_text: str = Field(max_length=4000)


class RepoReplaceMany(StrictModel):
    """Atomic, non-overlapping snippet updates within one source file."""

    tool: Literal["repo_replace_many"] = "repo_replace_many"
    path: str = Field(min_length=1, max_length=300)
    edits: list[RepoSnippetEdit] = Field(min_length=2, max_length=4)


class RepoInspectSyntax(StrictModel):
    """Read registered CUDA syntax nodes without exposing source line decoration."""

    tool: Literal["repo_inspect_syntax"] = "repo_inspect_syntax"
    path: str = Field(min_length=1, max_length=300)
    symbol: str = Field(default="", max_length=80)


class RepoSyntaxEdit(StrictModel):
    node_id: str = Field(min_length=24, max_length=24, pattern=r"^[a-f0-9]{24}$")
    replacement_code: str = Field(min_length=1, max_length=1200)


class RepoReplaceSyntax(StrictModel):
    """Edit 1-4 AST-addressed statements in one frozen source revision."""

    tool: Literal["repo_replace_syntax"] = "repo_replace_syntax"
    path: str = Field(min_length=1, max_length=300)
    expected_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    edits: list[RepoSyntaxEdit] = Field(min_length=1, max_length=4)


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
    | RepoReplaceMany
    | RepoInspectSyntax
    | RepoReplaceSyntax
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


class PlanMilestone(StrictModel):
    gate: Literal["inspect", "edit", "public_check", "finish"]
    objective: str = Field(min_length=8, max_length=220)
    success_evidence: str = Field(min_length=8, max_length=180)


class TaskPlan(StrictModel):
    """A public, concise goal plan; never private chain-of-thought."""

    milestones: list[PlanMilestone] = Field(min_length=4, max_length=4)
    failure_risk: str = Field(min_length=8, max_length=230)

    @model_validator(mode="after")
    def stage_contract(self) -> "TaskPlan":
        gates = tuple(item.gate for item in self.milestones)
        if gates != ("inspect", "edit", "public_check", "finish"):
            raise ValueError("planner must cover evidence, patch, public check and finish")
        return self


class TaskReflection(StrictModel):
    """Concise decision correction, NOT model private reasoning traces."""

    observed_failure: str = Field(min_length=6, max_length=280)
    avoid_next: str = Field(min_length=6, max_length=280)
    next_priority: str = Field(min_length=6, max_length=280)


class TaskProgress(StrictModel):
    inspected: bool = False
    edited: bool = False
    public_checked: bool = False
    finished: bool = False
    active_gate: Literal["inspect", "edit", "public_check", "finish", "done"] = "inspect"
    last_failure_code: str | None = Field(default=None, max_length=160)


class SessionSnapshot(StrictModel):
    goal: str = Field(min_length=3, max_length=1500)
    case_id: str = Field(min_length=1, max_length=80)
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    functional_requirement: str = Field(min_length=5, max_length=2000)
    base_git_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    registered_public_case: str | None = Field(default=None, pattern=r"^h00[1-3]$")
    context_mode: Literal["legacy", "evidence_first"] = "legacy"
    task_plan: TaskPlan | None = None
    task_progress: TaskProgress | None = None
    task_reflection: TaskReflection | None = None
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
    context_mode: Literal["legacy", "evidence_first"] = "evidence_first"
    allow_batch_edits: bool = True
    hierarchical_mode: bool = False
    syntax_mode: bool = False
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
