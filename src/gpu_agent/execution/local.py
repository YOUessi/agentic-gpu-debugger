"""Execute only exact, reviewed source sets using an operator-owned CUDA toolkit.

This backend is a trust gate, not isolation against the owning OS user. Candidates
and arbitrary user code require the future isolated backend.
"""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

from gpu_agent.config import Settings
from gpu_agent.contracts import ArtifactRef, RunStatus, ToolResult, new_id, now
from gpu_agent.execution.models import (
    BuildPayload,
    BuildRequest,
    BuildResult,
    CleanupResult,
    ExecutionPayload,
    ExecutionRequest,
    ExecutionResult,
    RuntimeStatus,
    SanitizerPayload,
    SanitizerRequest,
    SanitizerResult,
    WorkspaceHandle,
    WorkspaceRequest,
)
from gpu_agent.execution.process import ProcessCapture, ProcessExecutor
from gpu_agent.store import RunStore, read_regular, reject_symlinks

SOURCE_LIMIT = 4 * 1024 * 1024
BINARY_LIMIT = 64 * 1024 * 1024
LOG_LIMIT = 2 * 1024 * 1024
SOURCE_NAMES = {"kernel.cu", "vector_io.cpp", "vector_api.h", "json.hpp"}
Payload = TypeVar("Payload", bound=BaseModel)


def _load_trusted_sources() -> dict[str, dict[str, str]]:
    """Controller package data, never request-supplied authorization."""
    data = read_regular(Path(__file__).with_name("trusted_sources.json"), 1024 * 1024)
    return TypeAdapter(dict[str, dict[str, str]]).validate_json(data)


@dataclass
class _Workspace:
    handle: WorkspaceHandle
    hashes: dict[str, str] = field(default_factory=dict)
    binary_ref: ArtifactRef | None = None


class LocalBackend:
    def __init__(
        self, settings: Settings, store: RunStore, repo_root: Path, workspace_root: Path
    ) -> None:
        if store.visibility != "public":
            raise ValueError("local execution requires the public store")
        self.settings, self.store = settings, store
        self.repo_root = repo_root.absolute()
        self.workspace_root = workspace_root.absolute()
        reject_symlinks(self.repo_root)
        reject_symlinks(self.workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._workspaces: dict[str, _Workspace] = {}
        self._executor = ProcessExecutor()

    def _active_run(self, run_id: str) -> None:
        if self.store.load(run_id).status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
            raise ValueError("terminal run cannot execute")

    def _workspace(self, workspace_id: str, *, active: bool = True) -> _Workspace:
        state = self._workspaces.get(workspace_id)
        if state is None:
            raise ValueError("unknown workspace")
        reject_symlinks(state.handle.path)
        if not state.handle.path.is_dir() or state.handle.path.parent != self.workspace_root:
            raise ValueError("workspace path mismatch")
        if active:
            self._active_run(state.handle.run_id)
        return state

    def prepare(self, request: WorkspaceRequest) -> WorkspaceHandle:
        self._active_run(request.run_id)
        manifest = dict(request.source_manifest)
        if request.trust_level != "TRUSTED_LOCAL" or not any(
            manifest == entry for entry in _load_trusted_sources().values()
        ):
            raise ValueError("source set is not registered as trusted")
        if len(manifest) != 4 or {PurePosixPath(p).name for p in manifest} != SOURCE_NAMES:
            raise ValueError("incomplete trusted source set")
        snapshots: dict[str, bytes] = {}
        for relative, expected in manifest.items():
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative:
                raise ValueError("invalid source path")
            data = read_regular(self.repo_root / relative, SOURCE_LIMIT)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("source hash mismatch")
            snapshots[path.name] = data
        reject_symlinks(self.workspace_root)
        directory = Path(tempfile.mkdtemp(prefix="trusted-", dir=self.workspace_root))
        handle = WorkspaceHandle(id=new_id(), run_id=request.run_id, path=directory)
        try:
            for name, data in snapshots.items():
                fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                self.store.put(request.run_id, f"sources/{handle.id}/{name}", data, "public")
            self._workspaces[handle.id] = _Workspace(
                handle=handle,
                hashes={name: hashlib.sha256(data).hexdigest() for name, data in snapshots.items()},
            )
        except BaseException:
            shutil.rmtree(directory)
            raise
        return handle

    def _environment(self) -> dict[str, str]:
        return {"PATH": f"{self.settings.bin_dir}:/usr/bin:/bin", "LC_ALL": "C"}

    def _put(self, state: _Workspace, name: str, data: bytes) -> ArtifactRef:
        return self.store.put(state.handle.run_id, name, data, "public")

    def _tool_result(
        self,
        state: _Workspace,
        tool: str,
        capture: ProcessCapture,
        payload: Payload,
        stdout: ArtifactRef,
        stderr: ArtifactRef,
        request_id: str,
    ) -> ToolResult[Payload]:
        result = ToolResult(
            tool_name=tool,
            request_id=request_id,
            started_at=capture.started_at,
            finished_at=capture.finished_at,
            elapsed_ms=capture.elapsed_ms,
            exit_code=capture.exit_code,
            timed_out=capture.timed_out,
            cancelled=capture.cancelled,
            truncated=capture.truncated,
            stdout_artifact=stdout,
            stderr_artifact=stderr,
            tool_error=capture.tool_error,
            typed_payload=payload,
        )
        self._put(state, f"{tool}/{request_id}/result.json", result.model_dump_json().encode())
        return result

    def build(self, request: BuildRequest) -> BuildResult:
        state = self._workspace(request.workspace_id)
        state.binary_ref = None  # Any new build attempt revokes the previous executable.
        if request.target_arch != self.settings.target_arch:
            raise ValueError("target architecture must match operator settings")
        if {entry.name for entry in state.handle.path.iterdir()} - (SOURCE_NAMES | {"vector_add"}):
            raise ValueError("unexpected workspace files")
        for name, expected in state.hashes.items():
            data = read_regular(state.handle.path / name, SOURCE_LIMIT)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("snapshot hash mismatch")
        binary = state.handle.path / "vector_add"
        reject_symlinks(binary)
        binary.unlink(missing_ok=True)
        argv = [
            str(self.settings.bin_dir / "nvcc"),
            "-std=c++17",
            "-lineinfo",
            f"-arch={request.target_arch}",
            "-ccbin",
            str(self.settings.host_compiler),
            "kernel.cu",
            "vector_io.cpp",
            "-I",
            str(state.handle.path),
            "-o",
            str(binary),
        ]
        request_id = new_id()
        self._put(state, f"build/{request_id}/argv.json", json.dumps(argv).encode())
        capture = self._executor.execute(
            argv, state.handle.path, 120, LOG_LIMIT, env=self._environment()
        )
        stdout = self._put(state, f"build/{request_id}/stdout", capture.stdout)
        stderr = self._put(state, f"build/{request_id}/stderr", capture.stderr)
        success = self._runtime_status(capture) == "SUCCESS"
        if success:
            try:
                data = read_regular(binary, BINARY_LIMIT)
                if not data:
                    raise ValueError("empty compiler output")
            except ValueError:
                success = False
                capture = replace(capture, tool_error="BINARY_UNAVAILABLE")
            else:
                binary.chmod(0o500)
                state.binary_ref = self._put(state, f"build/{request_id}/binary", data)
        payload = BuildPayload(argv=argv, binary_ref=state.binary_ref)
        tool_result = self._tool_result(
            state, "build", capture, payload, stdout, stderr, request_id
        )
        return BuildResult(success=success, binary_ref=state.binary_ref, tool_result=tool_result)

    @staticmethod
    def _runtime_status(capture: ProcessCapture) -> RuntimeStatus:
        if capture.cancelled:
            return "CANCELLED"
        if capture.timed_out:
            return "TIMEOUT"
        if capture.tool_error:
            return "TOOL_ERROR"
        if capture.truncated:
            return "TRUNCATED"
        return "SUCCESS" if capture.exit_code == 0 else "FAILED"

    def _input(self, state: _Workspace, ref: ArtifactRef) -> bytes:
        if ref.run_id != state.handle.run_id or ref.visibility != "public":
            raise ValueError("input belongs to another run or store")
        if ref.byte_count > 32 * 1024 * 1024:
            raise ValueError("input exceeds process limit")
        return self.store.read(ref)

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        state = self._workspace(request.workspace_id)
        if state.binary_ref is None:
            raise ValueError("workspace has no successful build")
        binary = state.handle.path / "vector_add"
        data = read_regular(binary, BINARY_LIMIT)
        if hashlib.sha256(data).hexdigest() != state.binary_ref.sha256:
            raise ValueError("binary hash mismatch")
        self.store.read(state.binary_ref)
        stdin = self._input(state, request.stdin_ref)
        request_id = new_id()
        self._put(
            state,
            f"run/{request_id}/request.json",
            json.dumps(
                {
                    "workspace_id": state.handle.id,
                    "stdin_ref": request.stdin_ref.model_dump(mode="json"),
                    "binary_ref": state.binary_ref.model_dump(mode="json"),
                    "argv": [str(binary)],
                    "timeout_seconds": request.timeout_seconds,
                }
            ).encode(),
        )
        capture = self._executor.execute(
            [str(binary)],
            state.handle.path,
            request.timeout_seconds,
            LOG_LIMIT,
            stdin=stdin,
            env=self._environment(),
        )
        output = self._put(state, f"run/{request_id}/stdout", capture.stdout)
        stderr = self._put(state, f"run/{request_id}/stderr", capture.stderr)
        status = self._runtime_status(capture)
        payload = ExecutionPayload(
            runtime_status=status,
            output_ref=output,
            stdin_ref=request.stdin_ref,
            binary_ref=state.binary_ref,
        )
        tool_result = self._tool_result(state, "run", capture, payload, output, stderr, request_id)
        return ExecutionResult(output_ref=output, runtime_status=status, tool_result=tool_result)

    def run_sanitizer(self, request: SanitizerRequest) -> SanitizerResult:
        state = self._workspace(request.workspace_id)
        if request.stdin_ref is not None:
            self._input(state, request.stdin_ref)
        request_id = new_id()
        started = now()
        capture = ProcessCapture(None, b"", b"", False, 0, started, now(), tool_error="UNSUPPORTED")
        stdout = self._put(state, f"sanitizer/{request_id}/stdout", b"")
        stderr = self._put(state, f"sanitizer/{request_id}/stderr", b"")
        payload = SanitizerPayload(tool=request.tool)
        result = self._tool_result(state, "sanitizer", capture, payload, stdout, stderr, request_id)
        return SanitizerResult(tool_result=result)

    def cleanup(self, workspace: WorkspaceHandle) -> CleanupResult:
        state = self._workspace(workspace.id, active=False)
        if workspace != state.handle:
            raise ValueError("workspace handle mismatch")
        shutil.rmtree(state.handle.path)
        del self._workspaces[workspace.id]
        return CleanupResult(workspace_id=workspace.id, removed=True)
