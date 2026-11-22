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


class RunPublicChecks(StrictModel):
    tool: Literal["run_public_checks"] = "run_public_checks"


class FinishTask(StrictModel):
    tool: Literal["finish"] = "finish"
    summary: str = Field(min_length=5, max_length=700)


HarnessAction = Annotated[
    ListFiles | ReadFile | SearchCode | FunctionalProbe | ApplyPatch | RunPublicChecks | FinishTask,
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
    case_id: Literal["h001", "h002", "h003"]
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    functional_requirement: str = Field(min_length=5, max_length=2000)
    current_candidate_sha256: str | None = None
    public_checks_passed: bool = False
    allowed_tools: tuple[str, ...]
    remaining_steps: int = Field(ge=0)
    history: tuple[dict[str, str], ...] = Field(max_length=30)


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
