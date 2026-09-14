"""Typed local execution requests contain no commands or compiler flags."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.contracts import ArtifactRef, ToolResult


class ExecutionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class WorkspaceRequest(ExecutionModel):
    run_id: str
    source_manifest: dict[str, str]
    trust_level: str = "TRUSTED_LOCAL"


class WorkspaceHandle(ExecutionModel):
    id: str
    run_id: str
    path: Path


class CleanupResult(ExecutionModel):
    workspace_id: str
    removed: bool


class BuildRequest(ExecutionModel):
    workspace_id: str
    target_arch: str = Field(default="sm_89", pattern=r"^sm_[0-9]+$")


class ExecutionRequest(ExecutionModel):
    workspace_id: str
    stdin_ref: ArtifactRef
    timeout_seconds: float = Field(default=30, gt=0, le=300)


class SanitizerRequest(ExecutionModel):
    workspace_id: str
    tool: Literal["memcheck", "racecheck", "initcheck", "synccheck"] = "memcheck"
    stdin_ref: ArtifactRef | None = None
    timeout_seconds: float = Field(default=30, gt=0, le=300)


RuntimeStatus = Literal["SUCCESS", "FAILED", "TIMEOUT", "CANCELLED", "TOOL_ERROR", "TRUNCATED"]


class BuildPayload(ExecutionModel):
    argv: list[str]
    binary_ref: ArtifactRef | None


class BuildResult(ExecutionModel):
    success: bool
    binary_ref: ArtifactRef | None
    tool_result: ToolResult[BuildPayload]


class ExecutionPayload(ExecutionModel):
    runtime_status: RuntimeStatus
    output_ref: ArtifactRef
    stdin_ref: ArtifactRef
    binary_ref: ArtifactRef


class ExecutionResult(ExecutionModel):
    output_ref: ArtifactRef
    runtime_status: RuntimeStatus
    tool_result: ToolResult[ExecutionPayload]


class SanitizerTool(StrEnum):
    MEMCHECK = "memcheck"
    RACECHECK = "racecheck"
    INITCHECK = "initcheck"
    SYNCCHECK = "synccheck"


class SourceLocation(ExecutionModel):
    path: str
    line: int | None = Field(default=None, ge=1)
    function: str | None = None


class Finding(ExecutionModel):
    tool: SanitizerTool
    category: str
    kernel: str | None = None
    source_location: SourceLocation | None = None
    raw_ref: ArtifactRef | None = None


CheckOutcome = Literal["CLEAN", "FINDING", "TOOL_ERROR", "UNSUPPORTED"]


class SanitizerPayload(ExecutionModel):
    status: str = "UNSUPPORTED"
    tool: str
    findings: list[Finding] = Field(default_factory=list)
    completed: bool = False
    parser_version: str = "memcheck-1"
    check_outcome: CheckOutcome = "UNSUPPORTED"
    binary_ref: ArtifactRef | None = None
    stdin_ref: ArtifactRef | None = None
    program_output_ref: ArtifactRef | None = None
    program_stderr_ref: ArtifactRef | None = None


class SanitizerResult(ExecutionModel):
    status: str = "UNSUPPORTED"
    tool_result: ToolResult[SanitizerPayload] | None = None
    findings: list[Finding] = Field(default_factory=list)
    completed: bool = False
    parser_version: str = "memcheck-1"
    check_outcome: CheckOutcome = "UNSUPPORTED"
    program_output_ref: ArtifactRef | None = None
