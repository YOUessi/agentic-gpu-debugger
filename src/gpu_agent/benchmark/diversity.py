"""Native public algorithm acceptance, separate from frozen corpus/evaluation authority.

No LLM, no implicit family creation, no registration and no release claims. The same
isolated backend and numeric oracles used in verification produce persistent artifacts.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

from gpu_agent.benchmark.models import AuthoritativeCaseRegistry, AuthoritativeCaseSpec
from gpu_agent.benchmark.validation import derive_oracle, source_identities
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import (
    BuildRequest,
    ExecutionRequest,
    SanitizerRequest,
    SanitizerTool,
    WorkspaceRequest,
)
from gpu_agent.store import RunStore, read_regular, reject_symlinks

HARNESS = ("harness/vector_io.cpp", "harness/vector_api.h", "harness/vendor/json.hpp")
CLEAN_NAMES = {
    "stencil2d-cpu-v1": "stencil2d",
    "segment-scan-cpu-v1": "segment_scan",
    "rotate-add-cpu-v1": "rotate",
    "stencil-cpu-v1": "stencil",
    "histogram-cpu-v1": "histogram",
    "warp-reduce-cpu-v1": "warp_reduce",
}
SIZES = (1, 31, 32, 33, 127, 128, 129, 257)


def input_bytes(n: int) -> bytes:
    return json.dumps(
        {
            "n": n,
            "a": [float(i % 13 - 6) for i in range(n)],
            "b": [float(i % 7 + 1) for i in range(n)],
        }
    ).encode()


def remove_launches(text: str) -> str:
    """Deliberately wrong ablation; not a proposed repair or an agent input."""
    result, count = re.subn(r"^\s*\w+<<<[^\n]+>>>[^\n]+;\n", "\n", text, flags=re.M)
    if not count:
        raise ValueError("ablation found no launches")
    return result


def checked_sources(root: Path, spec: AuthoritativeCaseSpec, role: str) -> dict[str, str]:
    if role == "clean":
        kernel = f"diverse_clean/{CLEAN_NAMES[spec.oracle_id]}/kernel.cu"
        expected = spec.clean_source_hash
    else:
        kernel = f"public/{spec.case_id}/public_input/kernel.cu"
        expected = spec.mutant_source_hash
    manifest = {
        name: hashlib.sha256(read_regular(root / name, 4 * 1024 * 1024)).hexdigest()
        for name in (kernel, *HARNESS)
    }
    actual, harness = source_identities(manifest)
    if (actual, harness) != (expected, spec.harness_hash):
        raise ValueError("diverse source or harness differs from manifest")
    payload = read_regular(root / f"public/{spec.case_id}/public_input/input.json", 4 * 1024 * 1024)
    if hashlib.sha256(payload).hexdigest() != spec.input_set_hash:
        raise ValueError("diverse input differs from manifest")
    return manifest


def _exercise(
    store: RunStore,
    backend: IsolatedGPUBackend,
    spec: AuthoritativeCaseSpec,
    manifest: dict[str, str],
    role: str,
    public_input: bytes,
) -> dict[str, object]:
    run = store.create_run(f"diversity_{role}")
    checks: list[dict[str, object]] = []
    row: dict[str, object] = {"run_id": run.id, "role": role, "checks": checks, "passed": False}
    handle = None
    try:
        handle = backend.prepare(
            WorkspaceRequest(run_id=run.id, source_manifest=manifest, trust_level="UNTRUSTED")
        )
        built = backend.build(BuildRequest(workspace_id=handle.id))
        if not built.success:
            raise ValueError("BUILD_FAILED")
        payloads = [input_bytes(n) for n in SIZES] if role == "clean" else [public_input]
        for index, payload in enumerate(payloads):
            ref = store.put(run.id, f"acceptance/input-{index}.json", payload, "public")
            result = backend.run(ExecutionRequest(workspace_id=handle.id, stdin_ref=ref))
            tool_result = result.tool_result
            # A synchronization mutant may hang before instrumentation. Preserve
            # that symptom, then require a completed target finding below; a timeout
            # alone never qualifies a mutation and never qualifies clean/ablation.
            if (
                role == "mutant"
                and tool_result.timed_out
                and result.runtime_status == "TIMEOUT"
                and not (tool_result.tool_error or tool_result.cancelled or tool_result.truncated)
            ):
                checks.append(
                    {
                        "kind": "ordinary",
                        "input_sha256": ref.sha256,
                        "runtime_status": "TIMEOUT",
                        "oracle_passed": False,
                    }
                )
                continue
            if (
                tool_result.tool_error
                or tool_result.timed_out
                or tool_result.cancelled
                or tool_result.truncated
                or result.runtime_status not in {"SUCCESS", "FAILED"}
            ):
                raise ValueError("ORDINARY_EXECUTION_INCOMPLETE")
            passed = (
                result.runtime_status == "SUCCESS"
                and derive_oracle(payload, store.read(result.output_ref), spec.oracle_id).passed
            )
            checks.append(
                {
                    "kind": "ordinary",
                    "input_sha256": ref.sha256,
                    "runtime_status": result.runtime_status,
                    "oracle_passed": passed,
                }
            )
            if role == "clean" and not passed:
                raise ValueError("CLEAN_ORACLE_FAILED")
            if role == "ablation" and passed:
                raise ValueError("DELETED_COMPUTATION_PASSED")
        if role != "ablation":
            ref = store.put(run.id, "acceptance/sanitizer-input.json", public_input, "public")
            tools = (
                list(SanitizerTool)
                if role == "clean"
                else [spec.target_tool] * spec.sanitizer_repetitions
            )
            for tool in tools:
                sanitized = backend.run_sanitizer(
                    SanitizerRequest(
                        workspace_id=handle.id, stdin_ref=ref, tool=tool.value, timeout_seconds=120
                    )
                )
                categories = [f.category for f in sanitized.findings]
                output = sanitized.tool_result.typed_payload if sanitized.tool_result else None
                oracle_passed = bool(
                    output
                    and output.program_output_ref
                    and derive_oracle(
                        public_input, store.read(output.program_output_ref), spec.oracle_id
                    ).passed
                )
                checks.append(
                    {
                        "kind": "sanitizer",
                        "tool": tool.value,
                        "outcome": sanitized.check_outcome,
                        "categories": categories,
                        "oracle_passed": oracle_passed,
                    }
                )
                print(spec.case_id, role, tool.value, sanitized.check_outcome, flush=True)
                if not sanitized.completed:
                    raise ValueError("SANITIZER_INCOMPLETE")
                if role == "clean" and (sanitized.check_outcome != "CLEAN" or not oracle_passed):
                    raise ValueError("CLEAN_SANITIZER_OR_ORACLE_FAILED")
                if role == "mutant" and (
                    sanitized.check_outcome != "FINDING" or spec.expected_finding not in categories
                ):
                    raise ValueError("TARGET_FINDING_MISSING")
        row["passed"] = True
    except (ValueError, OSError) as exc:
        row["error"] = str(exc)
    finally:
        if handle is not None:
            try:
                backend.cleanup(handle)
            except (ValueError, OSError) as exc:
                row.update(passed=False, cleanup_error=str(exc))
        store.put(run.id, "acceptance/summary.json", json.dumps(row).encode(), "public")
        current = store.load(run.id)
        if current.status.value == "QUEUED":
            store.transition(run.id, "RUNNING", "FINALIZING")
        if store.load(run.id).status.value == "RUNNING":
            store.transition(run.id, "COMPLETED" if row["passed"] else "FAILED", None)
    return row


def run_diversity(
    repository: Path,
    output: Path,
    case_ids: tuple[str, ...] = (),
    roles: tuple[str, ...] = ("clean", "mutant", "ablation"),
) -> dict[str, object]:
    """Operator-only acceptance; refuse overwrite and retain unsuccessful attempts."""
    reject_symlinks(repository)
    reject_symlinks(output)
    if (
        not roles
        or len(set(roles)) != len(roles)
        or not set(roles) <= {"clean", "mutant", "ablation"}
    ):
        raise ValueError("unknown, duplicate or empty role selection")
    if output.exists():
        raise ValueError("Output exists; select a new directory to preserve prior evidence")
    root = repository.resolve() / "benchmarks"
    raw = read_regular(root / "diverse-registry.json", 1024 * 1024)
    registry = AuthoritativeCaseRegistry.model_validate_json(raw)
    if len({s.case_id for s in registry.cases}) != len(registry.cases):
        raise ValueError("duplicate case identity")
    if len(set(case_ids)) != len(case_ids) or not set(case_ids) <= {
        s.case_id for s in registry.cases
    }:
        raise ValueError("unknown or duplicate case selection")
    selected = [s for s in registry.cases if not case_ids or s.case_id in case_ids]
    if not selected:
        raise ValueError("empty diversity registry")
    for spec in selected:
        checked_sources(root, spec, "clean")
        checked_sources(root, spec, "mutant")
    output.mkdir(parents=True, exist_ok=False)
    store = RunStore(output / "public")
    backend = IsolatedGPUBackend(store, root, output / "workspaces")
    availability = backend.availability()
    rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "schema_version": 1,
        "purpose": "diverse_public_native_acceptance_not_registration",
        "registry_sha256": hashlib.sha256(raw).hexdigest(),
        "case_definitions": [s.model_dump(mode="json") for s in selected],
        "roles": list(roles),
        "complete_acceptance_scope": set(roles) == {"clean", "mutant", "ablation"},
        "api_calls": 0,
        "gpu_ready": availability.ready,
        "cases": rows,
        "passed": False,
    }
    if not availability.ready:
        report["error"] = availability.reason
    else:
        for spec in selected:
            results = []
            payload = read_regular(
                root / f"public/{spec.case_id}/public_input/input.json", 4 * 1024 * 1024
            )
            for role in ("clean", "mutant"):
                if role not in roles:
                    continue
                results.append(
                    _exercise(
                        store, backend, spec, checked_sources(root, spec, role), role, payload
                    )
                )
            # Remove all launches; the actual output contract/oracle must reject
            # the result. Infrastructure failures do not count as rejection.
            if "ablation" in roles:
                ablation = output / "ablations" / spec.case_id
                ablation.mkdir(parents=True)
                clean_name = f"diverse_clean/{CLEAN_NAMES[spec.oracle_id]}/kernel.cu"
                (ablation / "kernel.cu").write_text(
                    remove_launches((root / clean_name).read_text())
                )
                for name in HARNESS:
                    shutil.copyfile(root / name, ablation / Path(name).name)
                manifest = {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ablation.iterdir()
                }
                negative = IsolatedGPUBackend(store, ablation, output / "ablation-workspaces")
                results.append(_exercise(store, negative, spec, manifest, "ablation", payload))
            rows.append(
                {
                    "case_id": spec.case_id,
                    "oracle_id": spec.oracle_id,
                    "results": results,
                    "passed": all(r["passed"] for r in results),
                }
            )
            (output / "report.json").write_text(json.dumps(report, indent=2))
        report["passed"] = all(row["passed"] for row in rows)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report
