"""Public, bounded pre-/post-edit CUDA functional probe.

Operator selects registered case, source, public input and hidden CPU oracle.
The LLM can choose only a whitelisted public size and phase. No model tool
can access reference source, host expected output, evaluator files or shell.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.benchmark.hardset_v2 import HARDSET_CASES, sample_input, verify_numeric
from gpu_agent.benchmark.hardset_v2_selfcheck import HARNESS
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import BuildRequest, ExecutionRequest, WorkspaceRequest
from gpu_agent.store import RunStore, read_regular, reject_symlinks
from gpu_agent.verification.oracle import parse_output

PUBLIC_PROBE_SIZES = (32, 33, 65, 129, 257)
ProbeOutcome = Literal[
    "PASS", "NUMERIC_MISMATCH", "OUTPUT_INVALID", "PROGRAM_FAILED",
    "BUILD_FAILED", "PROBE_INCONCLUSIVE",
]


class PublicProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["gpu-public-functional-probe-v1"] = "gpu-public-functional-probe-v1"
    case_id: Literal["h001", "h002", "h003"]
    phase: Literal["before", "after"]
    input_size: int = Field(ge=1, le=513)
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    outcome: ProbeOutcome
    diagnostic_region: Literal["full_warp", "cross_warp_tail", "multi_warp_tail"]


def _region(n: int) -> Literal["full_warp", "cross_warp_tail", "multi_warp_tail"]:
    return "full_warp" if n == 32 else "cross_warp_tail" if n == 33 else "multi_warp_tail"


def probe_candidate_publicly(
    *,
    store: RunStore,
    parent_run_id: str,
    repository: Path,
    case_id: str,
    phase: Literal["before", "after"],
    input_size: int,
    candidate: bytes,
    output: Path,
) -> PublicProbeResult:
    """Compile and test ONE public input with typed, non-oracle feedback."""
    if (
        store.visibility != "public"
        or case_id not in HARDSET_CASES
        or phase not in {"before", "after"}
        or type(input_size) is not int
        or input_size not in PUBLIC_PROBE_SIZES
        or not isinstance(candidate, bytes)
        or not 0 < len(candidate) <= 1024 * 1024
    ):
        raise ValueError("invalid registered public probe request")
    reject_symlinks(repository)
    reject_symlinks(output)
    if output.exists() or output.is_symlink():
        raise ValueError("public probe cannot overwrite prior evidence")
    payload = sample_input(case_id, input_size)
    candidate_sha = hashlib.sha256(candidate).hexdigest()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    child = store.create_run("hardset_v2_public_functional_probe", parent_run_id=parent_run_id)
    store.transition(child.id, "RUNNING", "PREPARING")
    result = PublicProbeResult(
        case_id=case_id,  # type: ignore[arg-type]
        phase=phase,
        input_size=input_size,
        input_sha256=hashlib.sha256(payload).hexdigest(),
        candidate_sha256=candidate_sha,
        run_id=child.id,
        outcome="PROBE_INCONCLUSIVE",
        diagnostic_region=_region(input_size),
    )
    backend = None
    handle = None
    try:
        with tempfile.TemporaryDirectory(prefix="gpu-agent-public-probe-") as scratch:
            source = Path(scratch)
            data = {"kernel.cu": candidate}
            bench = repository.resolve() / "benchmarks"
            for relative in HARNESS:
                data[Path(relative).name] = read_regular(bench / relative, 4 << 20)
            for name, raw in data.items():
                IsolatedGPUBackend._write_snapshot(source / name, raw)
            manifest = {k: hashlib.sha256(v).hexdigest() for k, v in data.items()}
            backend = IsolatedGPUBackend(store, source, output / "workspaces")
            if not backend.availability().ready:
                raise ValueError("PUBLIC_PROBE_BACKEND_UNAVAILABLE")
            handle = backend.prepare(
                WorkspaceRequest(
                    run_id=child.id, source_manifest=manifest, trust_level="UNTRUSTED"
                )
            )
            built = backend.build(BuildRequest(workspace_id=handle.id))
            if not built.success:
                result = result.model_copy(update={"outcome": "BUILD_FAILED"})
            else:
                input_ref = store.put(
                    child.id, "probe/public-input.json", payload, "public"
                )
                execution = backend.run(
                    ExecutionRequest(
                        workspace_id=handle.id,
                        stdin_ref=input_ref,
                        timeout_seconds=120,
                    )
                )
                if (
                    execution.tool_result.tool_error
                    or execution.tool_result.timed_out
                    or execution.tool_result.cancelled
                    or execution.tool_result.truncated
                ):
                    result = result.model_copy(update={"outcome": "PROBE_INCONCLUSIVE"})
                elif execution.runtime_status != "SUCCESS":
                    result = result.model_copy(update={"outcome": "PROGRAM_FAILED"})
                else:
                    observed = store.read(execution.output_ref)
                    try:
                        parse_output(observed)
                    except ValueError:
                        outcome: ProbeOutcome = "OUTPUT_INVALID"
                    else:
                        outcome = "PASS" if verify_numeric(case_id, payload, observed) else "NUMERIC_MISMATCH"
                    result = result.model_copy(update={"outcome": outcome})
    except (OSError, ValueError):
        result = result.model_copy(update={"outcome": "PROBE_INCONCLUSIVE"})
    finally:
        if backend is not None and handle is not None:
            try:
                backend.cleanup(handle)
            except (OSError, ValueError):
                result = result.model_copy(update={"outcome": "PROBE_INCONCLUSIVE"})
        store.put(
            child.id, "probe/public-summary.json", result.model_dump_json().encode(), "public"
        )
        store.transition(
            child.id, "FAILED" if result.outcome == "PROBE_INCONCLUSIVE" else "COMPLETED", None
        )
    return result
