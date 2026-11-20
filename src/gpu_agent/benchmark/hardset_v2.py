"""Trusted native-GPU acceptance for three self-authored functional-only CUDA faults.

This benchmark is intentionally separate from the seven-algorithm Repair V3
protocol. It proves that real, complex numerical failures can be distinguished
from sanitizer-only evidence *without* exposing a reference to the model.
It is not a signed holdout, not Agent repair evidence and not a release gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import (
    BuildRequest,
    ExecutionRequest,
    SanitizerRequest,
    SanitizerTool,
    WorkspaceRequest,
)
from gpu_agent.store import RunStore, read_regular, reject_symlinks
from gpu_agent.verification.oracle import NumericOracle, parse_output

HARDSET_CASES = {
    "h001": "stable-softmax32-v2",
    "h002": "tail-rmsnorm32-v2",
    "h003": "two-stage-group-center32-v2",
}
SIZES = (1, 31, 32, 33, 65, 129, 257)
HARNESS = ("harness/vector_io.cpp", "harness/vector_api.h", "harness/vendor/json.hpp")


def f32(value: float) -> float:
    return float(struct.unpack("f", struct.pack("f", value))[0])


class HardsetCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str = Field(pattern=r"^h00[1-3]$")
    oracle_id: str
    difficulty: str
    fault_class: str
    clean_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    mutant_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_visible_reference: bool = False

    @model_validator(mode="after")
    def valid(self) -> HardsetCase:
        if HARDSET_CASES.get(self.case_id) != self.oracle_id:
            raise ValueError("unregistered hardset case/oracle")
        if self.model_visible_reference:
            raise ValueError("benchmark cannot expose a CPU reference to the model")
        if self.clean_source_sha256 == self.mutant_source_sha256:
            raise ValueError("mutant is identical to clean source")
        return self


class HardsetRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str
    status: str
    cases: list[HardsetCase]

    @model_validator(mode="after")
    def check(self) -> HardsetRegistry:
        if self.schema_version != "hardset-v2-native-1":
            raise ValueError("unsupported hardset registry version")
        if self.status != "NOT_YET_GPU_VALIDATED":
            raise ValueError("hardset native acceptance is never pre-certified")
        if {c.case_id for c in self.cases} != set(HARDSET_CASES):
            raise ValueError("hardset manifest missing or duplicated case")
        if len(self.cases) != len(HARDSET_CASES):
            raise ValueError("duplicate hardset entries")
        return self


def load_registry(root: Path) -> tuple[HardsetRegistry, str]:
    raw = read_regular(root / "hardset_v2" / "manifest.json", 1024 * 1024)
    registry = HardsetRegistry.model_validate_json(raw)
    for case in registry.cases:
        for role in ("clean", "mutant"):
            path = root / "hardset_v2" / case.case_id / role / "kernel.cu"
            actual = hashlib.sha256(read_regular(path, 4 * 1024 * 1024)).hexdigest()
            if actual != getattr(case, f"{role}_source_sha256"):
                raise ValueError("hardset source does not match pinned registry")
    return registry, hashlib.sha256(raw).hexdigest()


def sample_input(case_id: str, n: int) -> bytes:
    if case_id not in HARDSET_CASES or n not in SIZES:
        raise ValueError("unsupported fixed acceptance input")
    base = 95.0 if case_id == "h001" else 0.0
    a = [f32(base + ((i * 7) % 17 - 8) / 4) for i in range(n)]
    b = [f32(((i * 5) % 11 - 5) / 8) for i in range(n)]
    return json.dumps({"n": n, "a": a, "b": b}, separators=(",", ":")).encode()


def expected_output(case_id: str, raw: bytes) -> list[float]:
    """Independent host semantics; the tested CUDA sources never implement this."""
    if case_id not in HARDSET_CASES:
        raise ValueError("unregistered oracle")
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {"n", "a", "b"}:
        raise ValueError("invalid input contract")
    n, a, b = data["n"], data["a"], data["b"]
    if (
        type(n) is not int
        or n < 1
        or n > 65536
        or not isinstance(a, list)
        or not isinstance(b, list)
        or len(a) != n
        or len(b) != n
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in [*a, *b])
    ):
        raise ValueError("invalid float32 inputs")
    x = [f32(f32(u) + f32(v)) for u, v in zip(a, b, strict=True)]
    result = [0.0] * n
    for start in range(0, n, 32):
        end = min(start + 32, n)
        group = x[start:end]
        if case_id == "h001":
            # Stable softmax reference: subtract max before exponentiation.
            maximum = max(group)
            exp = [math.exp(value - maximum) for value in group]
            scale = sum(exp)
            values = [f32(value / scale) for value in exp]
        elif case_id == "h002":
            mean_sq = sum(value * value for value in group) / len(group)
            inv_rms = 1.0 / math.sqrt(mean_sq + 1.0e-5)
            values = [f32(value * inv_rms) for value in group]
        else:
            # Two-stage CUDA pipeline must compute this group mean from BOTH inputs.
            average = sum(group) / len(group)
            values = [f32(value - average) for value in group]
        result[start:end] = values
    return result


def verify_numeric(case_id: str, raw: bytes, result: bytes) -> bool:
    try:
        actual = parse_output(result)
    except ValueError:
        return False
    reference = expected_output(case_id, raw)
    # CUDA expf/rsqrtf differ slightly from CPU math; tolerance is fixed
    # before GPU runs and the same for clean/mutant across all sizes.
    return NumericOracle(3e-5, 3e-5, False, False).check(actual, reference).passed


def manifest_for(root: Path, case: HardsetCase, role: str) -> dict[str, str]:
    if role not in {"clean", "mutant"}:
        raise ValueError("unapproved role")
    kernel = f"hardset_v2/{case.case_id}/{role}/kernel.cu"
    files = (kernel, *HARNESS)
    manifest = {
        relative: hashlib.sha256(read_regular(root / relative, 4 * 1024 * 1024)).hexdigest()
        for relative in files
    }
    if manifest[kernel] != getattr(case, f"{role}_source_sha256"):
        raise ValueError("source changed after preflight")
    return manifest


def exercise(
    backend: IsolatedGPUBackend,
    store: RunStore,
    root: Path,
    case: HardsetCase,
    role: str,
) -> dict[str, Any]:
    run = store.create_run(f"hardset_v2_{role}")
    handle = None
    checks: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "run_id": run.id,
        "role": role,
        "checks": checks,
        "passed": False,
    }
    failure_count = 0
    try:
        handle = backend.prepare(
            WorkspaceRequest(
                run_id=run.id,
                source_manifest=manifest_for(root, case, role),
                trust_level="UNTRUSTED",
            )
        )
        built = backend.build(BuildRequest(workspace_id=handle.id))
        if not built.success:
            raise ValueError("CLEAN_BUILD_FAILED" if role == "clean" else "MUTANT_BUILD_FAILED")
        for n in SIZES:
            raw = sample_input(case.case_id, n)
            ref = store.put(run.id, f"hardset/input-n{n}.json", raw, "public")
            executed = backend.run(
                ExecutionRequest(workspace_id=handle.id, stdin_ref=ref, timeout_seconds=120)
            )
            complete = not (
                executed.tool_result.tool_error
                or executed.tool_result.timed_out
                or executed.tool_result.cancelled
                or executed.tool_result.truncated
            )
            if not complete:
                raise ValueError("NATIVE_EXECUTION_INCOMPLETE")
            correct = False
            if executed.runtime_status == "SUCCESS" and executed.output_ref is not None:
                correct = verify_numeric(case.case_id, raw, store.read(executed.output_ref))
            checks.append(
                {
                    "size": n,
                    "input_sha256": ref.sha256,
                    "runtime_status": executed.runtime_status,
                    "cpu_oracle_passed": correct,
                }
            )
            if role == "clean" and not correct:
                raise ValueError("CLEAN_NUMERICAL_ORACLE_FAILED")
            if role == "mutant" and not correct:
                failure_count += 1
        if role == "clean":
            # Four NVIDIA tools still gate the clean reference. A mutant with
            # a wrong number is NOT required to produce a sanitizer finding.
            raw = sample_input(case.case_id, 65)
            ref = store.put(run.id, "hardset/sanitizer-input.json", raw, "public")
            for tool in SanitizerTool:
                result = backend.run_sanitizer(
                    SanitizerRequest(
                        workspace_id=handle.id,
                        stdin_ref=ref,
                        tool=tool.value,
                        timeout_seconds=120,
                    )
                )
                payload = result.tool_result.typed_payload if result.tool_result else None
                correct = bool(
                    payload
                    and payload.program_output_ref
                    and verify_numeric(case.case_id, raw, store.read(payload.program_output_ref))
                )
                checks.append(
                    {
                        "sanitizer": tool.value,
                        "outcome": result.check_outcome,
                        "cpu_oracle_passed": correct,
                    }
                )
                if not result.completed or result.check_outcome != "CLEAN" or not correct:
                    raise ValueError("CLEAN_SANITIZER_OR_ORACLE_FAILED")
        else:
            if not failure_count:
                raise ValueError("MUTATION_DID_NOT_BREAK_FUNCTION")
        report["passed"] = True
        report["observed_functional_failures"] = failure_count
    except (ValueError, OSError) as exc:
        report["error"] = str(exc)
    finally:
        if handle is not None:
            try:
                backend.cleanup(handle)
            except (ValueError, OSError) as exc:
                report.update(passed=False, cleanup_error=str(exc))
        store.put(run.id, "hardset/summary.json", json.dumps(report).encode(), "public")
        current = store.load(run.id)
        if current.status.value == "QUEUED":
            store.transition(run.id, "RUNNING", "FINALIZING")
        if store.load(run.id).status.value == "RUNNING":
            store.transition(run.id, "COMPLETED" if report["passed"] else "FAILED", None)
    return report


def run_hardset(
    repository: Path,
    output: Path,
    case_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    reject_symlinks(repository)
    reject_symlinks(output)
    if output.exists():
        raise ValueError("Output exists: do not overwrite native evidence")
    if len(set(case_ids)) != len(case_ids) or not set(case_ids) <= set(HARDSET_CASES):
        raise ValueError("unknown or duplicate hardset selection")
    root = repository.resolve() / "benchmarks"
    registry, registry_sha = load_registry(root)
    selected = [c for c in registry.cases if not case_ids or c.case_id in case_ids]
    # Preflight all hashes before creating a RunStore or touching the GPU.
    for case in selected:
        for role in ("clean", "mutant"):
            manifest_for(root, case, role)
    output.mkdir(parents=True, exist_ok=False)
    store = RunStore(output / "public")
    backend = IsolatedGPUBackend(store, root, output / "workspaces")
    availability = backend.availability()
    rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": "hardset-v2-native-result-1",
        "evidence_type": "REAL_NATIVE_GPU_ACCEPTANCE_ONLY_NOT_AGENT_FIX",
        "registry_sha256": registry_sha,
        "gpu_ready": availability.ready,
        "gpu_error": None if availability.ready else availability.reason,
        "api_calls": 0,
        "cases": rows,
        "passed": False,
    }
    try:
        if availability.ready:
            for case in selected:
                pair = []
                for role in ("clean", "mutant"):
                    entry = exercise(backend, store, root, case, role)
                    pair.append(entry)
                    print(
                        f"{case.case_id} {role}: "
                        f"{'PASS' if entry['passed'] else entry.get('error', 'FAIL')}",
                        flush=True,
                    )
                rows.append(
                    {
                        "case_id": case.case_id,
                        "oracle_id": case.oracle_id,
                        "results": pair,
                        "passed": all(part["passed"] for part in pair),
                    }
                )
        result["passed"] = bool(rows) and all(r["passed"] for r in rows)
    finally:
        (output / "report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result
