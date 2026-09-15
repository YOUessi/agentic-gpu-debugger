"""Versioned run, evidence and typed tool contracts."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def new_id() -> str:
    return uuid4().hex


def now() -> datetime:
    return datetime.now(UTC)


class RunStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class CurrentPhase(StrEnum):
    PREPARING = "PREPARING"
    COMPILING = "COMPILING"
    EXECUTING = "EXECUTING"
    COLLECTING_EVIDENCE = "COLLECTING_EVIDENCE"
    DIAGNOSING = "DIAGNOSING"
    PATCH_GENERATING = "PATCH_GENERATING"
    VERIFYING = "VERIFYING"
    FINALIZING = "FINALIZING"


Visibility = Literal["public", "evaluator"]


class ArtifactRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    run_id: str
    name: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    visibility: Visibility
    relative_path: str
    byte_count: int = Field(ge=0)


class StateEvent(BaseModel):
    at: datetime = Field(default_factory=now)
    status: RunStatus
    phase: CurrentPhase | None


class RepositorySnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    commit: str = Field(pattern=r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
    tracked_tree_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    clean: Literal[True]


class RunBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    repository: RepositorySnapshot
    purpose: Literal["corpus_validation", "evaluation", "release_acceptance"]
    toolchain_lock_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    prompt_version: str | None = Field(default=None, min_length=1, max_length=128)
    model_config_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ExternalRunOrigin(BaseModel):
    """Typed cross-store edge; the consuming controller resolves it in the named store."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    visibility: Visibility


class RunManifest(BaseModel):
    schema_version: Literal[1] = 1
    id: str
    kind: str
    parent_run_id: str | None = None
    binding: RunBinding | None = Field(default=None, frozen=True)
    external_origin: ExternalRunOrigin | None = Field(default=None, frozen=True)
    status: RunStatus = RunStatus.QUEUED
    current_phase: CurrentPhase | None = None
    last_completed_phase: CurrentPhase | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    events: list[StateEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_phase(self) -> "RunManifest":
        if (self.status == RunStatus.RUNNING) != (self.current_phase is not None):
            raise ValueError("only RUNNING requires an active phase")
        return self


Payload = TypeVar("Payload")


class ToolResult(BaseModel, Generic[Payload]):
    tool_name: str
    request_id: str
    started_at: datetime
    finished_at: datetime
    elapsed_ms: float
    exit_code: int | None
    timed_out: bool
    cancelled: bool = False
    truncated: bool = False
    stdout_artifact: ArtifactRef
    stderr_artifact: ArtifactRef
    tool_error: str | None = None
    typed_payload: Payload
