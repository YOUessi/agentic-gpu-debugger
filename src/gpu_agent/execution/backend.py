"""Backend interface shared by orchestration and future isolated execution."""

from typing import Protocol

from gpu_agent.execution.models import (
    BuildRequest,
    BuildResult,
    CleanupResult,
    ExecutionRequest,
    ExecutionResult,
    SanitizerRequest,
    SanitizerResult,
    WorkspaceHandle,
    WorkspaceRequest,
)


class ExecutionBackend(Protocol):
    def prepare(self, request: WorkspaceRequest) -> WorkspaceHandle: ...

    def build(self, request: BuildRequest) -> BuildResult: ...

    def run(self, request: ExecutionRequest) -> ExecutionResult: ...

    def run_sanitizer(self, request: SanitizerRequest) -> SanitizerResult: ...

    def cleanup(self, workspace: WorkspaceHandle) -> CleanupResult: ...
