"""Public-only GPU self-check of an actual model-produced HardSet V2 candidate.

The controller alone chooses the inputs and expected semantics. Candidate code
is never allowed to supply commands, scripts, a checker, or expected answers.
This is the feedback loop for Agent revisions, *not* the private verifier.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from gpu_agent.benchmark.hardset_v2 import sample_input, verify_numeric
from gpu_agent.benchmark.hardset_v2_agent_models import (
    HardsetPublicCheck,
    PublicSanitizerDiagnostic,
)
from gpu_agent.benchmark.public_numeric_feedback import classify_public_numeric
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import (
    BuildRequest,
    ExecutionRequest,
    SanitizerRequest,
    SanitizerTool,
    WorkspaceRequest,
)
from gpu_agent.store import RunStore, read_regular

PUBLIC_CHECK_SIZES = (32, 33, 65, 129, 257)
HARNESS = ("harness/vector_io.cpp", "harness/vector_api.h", "harness/vendor/json.hpp")


def _save_summary(store: RunStore, run_id: str, result: HardsetPublicCheck) -> None:
    store.put(run_id, "hardset/public-selfcheck.json", result.model_dump_json().encode(), "public")


def check_candidate_publicly(
    *,
    store: RunStore,
    parent_run_id: str,
    repository: Path,
    case_id: str,
    candidate: bytes,
    output: Path,
    diagnostic_mode: bool = False,
) -> HardsetPublicCheck:
    """Build/run and numerically check the *candidate*, not its original mutant."""
    if store.visibility != "public":
        raise ValueError("private store forbidden in public self-check")
    if output.exists() or output.is_symlink():
        raise ValueError("candidate output exists; cannot overwrite audit")
    if case_id not in {"h001", "h002", "h003"} or not candidate:
        raise ValueError("unknown or empty HardSet candidate")
    if len(candidate) > 4 * 1024 * 1024:
        raise ValueError("candidate exceeds trusted source size")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    run = store.create_run("hardset_v2_public_candidate", parent_run_id=parent_run_id)
    store.transition(run.id, "RUNNING", "PREPARING")
    candidate_hash = hashlib.sha256(candidate).hexdigest()
    store.put(
        run.id,
        "hardset/public-check-candidate.json",
        json.dumps({"candidate_sha256": candidate_hash, "case_id": case_id}).encode(),
        "public",
    )
    feedback = HardsetPublicCheck(
        candidate_sha256=candidate_hash,
        passed=False,
        failure_code="PUBLIC_EXECUTION_UNAVAILABLE",
        public_run_id=run.id,
    )
    with tempfile.TemporaryDirectory(prefix="hardset-v2-public-candidate-") as scratch:
        source = Path(scratch)
        data = {"kernel.cu": candidate}
        benchmark_root = repository.resolve() / "benchmarks"
        for relative in HARNESS:
            data[Path(relative).name] = read_regular(benchmark_root / relative, 4 * 1024 * 1024)
        for name, value in data.items():
            IsolatedGPUBackend._write_snapshot(source / name, value)
        manifest = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
        backend = IsolatedGPUBackend(store, source, output / "workspaces")
        handle = None
        try:
            if not backend.availability().ready:
                raise ValueError("PUBLIC_EXECUTION_UNAVAILABLE")
            handle = backend.prepare(
                WorkspaceRequest(run_id=run.id, source_manifest=manifest, trust_level="UNTRUSTED")
            )
            build = backend.build(BuildRequest(workspace_id=handle.id))
            if not build.success:
                stderr = store.read(build.tool_result.stderr_artifact).decode(
                    "utf-8", errors="replace"
                )[:900]
                feedback = feedback.model_copy(
                    update={"failure_code": "BUILD_FAILED", "public_error_excerpt": stderr}
                )
            else:
                store.put(
                    run.id,
                    "hardset/public-build-passed.json",
                    json.dumps({"candidate_sha256": candidate_hash}).encode(),
                    "public",
                )
                failure = None
                for size in PUBLIC_CHECK_SIZES:
                    payload = sample_input(case_id, size)
                    ref = store.put(run.id, f"hardset/public-input-n{size}.json", payload, "public")
                    if size == PUBLIC_CHECK_SIZES[0]:
                        # Records dispatch of a numeric workload, not proof that
                        # the GPU kernel completed. Build failures do not count.
                        store.put(
                            run.id,
                            "hardset/public-numeric-dispatch.json",
                            json.dumps(
                                {"candidate_sha256": candidate_hash, "input_size": size}
                            ).encode(),
                            "public",
                        )
                    execution = backend.run(
                        ExecutionRequest(workspace_id=handle.id, stdin_ref=ref, timeout_seconds=120)
                    )
                    if (
                        execution.tool_result.tool_error
                        or execution.tool_result.timed_out
                        or execution.tool_result.cancelled
                        or execution.tool_result.truncated
                    ):
                        failure = ("PUBLIC_EXECUTION_UNAVAILABLE", size)
                        break
                    if execution.runtime_status != "SUCCESS":
                        failure = ("PROGRAM_FAILED", size)
                        break
                    observed = store.read(execution.output_ref)
                    if not verify_numeric(case_id, payload, observed):
                        failure = ("NUMERIC_MISMATCH", size)
                        if diagnostic_mode:
                            feedback = feedback.model_copy(
                                update={
                                    "numeric_diagnostic": classify_public_numeric(
                                        case_id, payload, observed
                                    ),
                                }
                            )
                        break
                if failure is not None:
                    feedback = feedback.model_copy(
                        update={
                            "failure_code": failure[0],
                            "first_failing_size": failure[1],
                        }
                    )
                else:
                    feedback = feedback.model_copy(
                        update={"passed": True, "failure_code": "PUBLIC_CHECKS_PASSED"}
                    )
                # In diagnostic mode, still run the bounded sanitizer stage
                # after a numeric mismatch. An infrastructure failure does not
                # grant permission to start extra workloads.
                if failure is None or (diagnostic_mode and failure[0] == "NUMERIC_MISMATCH"):
                    input_data = sample_input(case_id, 65)
                    ref = store.put(
                        run.id, "hardset/public-sanitizer-input.json", input_data, "public"
                    )
                    for tool in SanitizerTool:
                        result = backend.run_sanitizer(
                            SanitizerRequest(
                                workspace_id=handle.id,
                                stdin_ref=ref,
                                tool=tool.value,
                                timeout_seconds=120,
                            )
                        )
                        output_ref = result.program_output_ref
                        numeric_match = (
                            verify_numeric(case_id, input_data, store.read(output_ref))
                            if result.completed and output_ref is not None
                            else None
                        )
                        clean = result.completed and result.check_outcome == "CLEAN"
                        if diagnostic_mode:
                            detail = PublicSanitizerDiagnostic(
                                tool=tool.value,
                                status=(
                                    "CLEAN"
                                    if clean
                                    else "FINDINGS"
                                    if result.completed and result.check_outcome == "FINDING"
                                    else "INCONCLUSIVE"
                                ),
                                numeric_output_matches=numeric_match,
                            )
                            feedback = feedback.model_copy(
                                update={
                                    "sanitizer_diagnostics": (
                                        *feedback.sanitizer_diagnostics,
                                        detail,
                                    )
                                }
                            )
                        if not clean or numeric_match is not True:
                            if failure is None and feedback.passed:
                                feedback = feedback.model_copy(
                                    update={
                                        "passed": False,
                                        "failure_code": "SANITIZER_FAILED",
                                        "public_error_excerpt": tool.value,
                                    }
                                )
                            if not diagnostic_mode:
                                break
        except (OSError, ValueError) as error:
            if feedback.failure_code == "PUBLIC_EXECUTION_UNAVAILABLE":
                feedback = feedback.model_copy(
                    update={"public_error_excerpt": type(error).__name__}
                )
        finally:
            if handle is not None:
                try:
                    backend.cleanup(handle)
                except (OSError, ValueError):
                    feedback = feedback.model_copy(
                        update={
                            "passed": False,
                            "failure_code": "PUBLIC_EXECUTION_UNAVAILABLE",
                            "public_error_excerpt": "CLEANUP_FAILED",
                        }
                    )
            _save_summary(store, run.id, feedback)
            store.transition(run.id, "COMPLETED" if feedback.passed else "FAILED", None)
    return feedback
