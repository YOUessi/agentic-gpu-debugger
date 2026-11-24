"""Compile-only public candidate validation; no numeric or sanitizer execution."""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.benchmark.hardset_v2_selfcheck import HARNESS
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import BuildRequest, WorkspaceRequest
from gpu_agent.store import RunStore, read_regular

CompileStatus = Literal["BUILD_PASSED", "BUILD_FAILED", "BUILD_UNAVAILABLE"]


class CompileDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    line: int | None = Field(default=None, ge=1)
    column: int | None = Field(default=None, ge=1)
    message: str = Field(max_length=360)
    source_excerpt: str = Field(default="", max_length=300)


class PublicCompileResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    status: CompileStatus
    diagnostic: CompileDiagnostic | None = None


_NVCC = re.compile(
    r"^(?:/input/)?kernel\.cu\((\d+)(?:,\s*(\d+))?\):\s*"
    r"(?:fatal\s+)?error(?:\s*#\d+[^:]*)?:\s*(.*)$",
    re.IGNORECASE,
)
_GCC = re.compile(
    r"^(?:/input/)?kernel\.cu:(\d+)(?::(\d+))?:\s*(?:fatal\s+)?error:\s*(.*)$",
    re.IGNORECASE,
)
_GENERIC = re.compile(r"^(?:nvcc|ptxas|nvlink)(?:\s+fatal|\s+error)\s*:\s*(.*)$", re.I)


def public_compile_diagnostic(stderr: bytes, candidate: bytes) -> CompileDiagnostic:
    """Scan bounded complete stderr for the first candidate-source error.

    Never copy commands, environment, include stacks or arbitrary host paths.
    Source excerpt comes from the exact public candidate, not compiler prose.
    """
    lines = stderr[: 2 * 1024 * 1024].decode("utf-8", errors="replace").splitlines()
    source_lines = candidate.decode("utf-8", errors="replace").splitlines()
    for line in lines:
        match = _NVCC.match(line.strip()) or _GCC.match(line.strip())
        if match is None:
            continue
        number = int(match[1])
        column = int(match[2]) if match[2] else None
        if not 1 <= number <= len(source_lines):
            continue
        message = re.sub(r"(?<![\w])/(?:[^\s\"'<>:]+)", "<path>", match[3])
        message = "".join(c if c.isprintable() else " " for c in message)[:360]
        return CompileDiagnostic(
            line=number,
            column=column if column and column <= 100000 else None,
            message=message or "CUDA compilation error",
            source_excerpt=source_lines[number - 1].strip()[:300],
        )
    for line in lines:
        match = _GENERIC.match(line.strip())
        if match:
            message = re.sub(r"(?<![\w])/(?:[^\s\"'<>:]+)", "<path>", match[1])
            return CompileDiagnostic(message=message[:360])
    return CompileDiagnostic(
        message="Compilation failed; no bounded kernel.cu diagnostic available"
    )


def compile_public_candidate(
    *, store: RunStore, parent_run_id: str, repository: Path, candidate: bytes, output: Path
) -> PublicCompileResult:
    """Use the same isolated NVCC build contract as the public Checker, only build."""
    if store.visibility != "public" or not 0 < len(candidate) <= 1024 * 1024:
        raise ValueError("invalid public compile input")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    child = store.create_run("hardset_v2_public_compile", parent_run_id=parent_run_id)
    store.transition(child.id, "RUNNING", "PREPARING")
    digest = hashlib.sha256(candidate).hexdigest()
    result = PublicCompileResult(
        candidate_sha256=digest, run_id=child.id, status="BUILD_UNAVAILABLE"
    )
    backend = None
    handle = None
    try:
        with tempfile.TemporaryDirectory(prefix="gpu-public-compile-") as temporary:
            source = Path(temporary).resolve(strict=True)
            data = {"kernel.cu": candidate}
            for rel in HARNESS:
                data[Path(rel).name] = read_regular(repository / "benchmarks" / rel, 4 << 20)
            for name, raw in data.items():
                IsolatedGPUBackend._write_snapshot(source / name, raw)
            manifest = {name: hashlib.sha256(raw).hexdigest() for name, raw in data.items()}
            backend = IsolatedGPUBackend(store, source, output / "workspaces")
            if backend.availability().ready:
                handle = backend.prepare(
                    WorkspaceRequest(
                        run_id=child.id, source_manifest=manifest, trust_level="UNTRUSTED"
                    )
                )
                try:
                    build = backend.build(BuildRequest(workspace_id=handle.id))
                    if build.success:
                        result = result.model_copy(update={"status": "BUILD_PASSED"})
                    elif not (
                        build.tool_result.timed_out
                        or build.tool_result.cancelled
                        or build.tool_result.truncated
                        or build.tool_result.tool_error
                    ):
                        error = public_compile_diagnostic(
                            store.read(build.tool_result.stderr_artifact), candidate
                        )
                        result = result.model_copy(
                            update={"status": "BUILD_FAILED", "diagnostic": error}
                        )
                finally:
                    backend.cleanup(handle)
                    handle = None
    except (ValueError, OSError):
        result = result.model_copy(update={"status": "BUILD_UNAVAILABLE", "diagnostic": None})
    store.put(child.id, "compile/public-summary.json", result.model_dump_json().encode(), "public")
    store.transition(child.id, "COMPLETED" if result.status == "BUILD_PASSED" else "FAILED", None)
    return result
