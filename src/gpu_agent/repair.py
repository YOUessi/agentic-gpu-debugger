"""Public-only candidate self-checks. No evaluator, truth or hidden-input capability."""

import hashlib
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from gpu_agent.agent.models import DiagnosisResult, PublicSource
from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.agent.prompts import REPAIR_PROMPT_VERSION
from gpu_agent.agent.provider import LLMProvider, ProviderError
from gpu_agent.execution.backend import ExecutionBackend
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import (
    BackendInfrastructureError,
    BuildRequest,
    ExecutionModel,
    ExecutionRequest,
    SanitizerRequest,
    SanitizerTool,
    WorkspaceRequest,
)
from gpu_agent.patching import (
    PatchCandidate,
    SourceSnapshot,
    apply_generated_candidate,
    materialize_candidate,
)
from gpu_agent.public_task import PublicTask, check_public_output
from gpu_agent.store import RunStore

if TYPE_CHECKING:
    from gpu_agent.repair_coordinator import RepairCoordinator

REPAIR_VERSION = "public-repair-v2"


class RepairPolicy(ExecutionModel):
    version: Literal["public-repair-v2", "public-repair-v3"] = "public-repair-v2"
    max_candidates: int = Field(default=3, ge=1, le=20, strict=True)
    max_reinvestigations: int = Field(default=1, ge=0, le=3, strict=True)


class PublicCheck(ExecutionModel):
    run_id: str
    status: Literal["PASSED", "FAILED", "UNAVAILABLE"]
    checks: dict[str, str]
    # Bounded public process output, never a verifier projection.
    feedback: list[dict[str, str]]


BackendFactory = Callable[[RunStore, Path, Path], ExecutionBackend]


def self_check(
    store: RunStore,
    parent_id: str,
    sources: dict[str, bytes],
    stdin: bytes,
    backend_factory: BackendFactory,
    gate: LLMCallGate,
    public_task: PublicTask | None = None,
) -> PublicCheck:
    if store.visibility != "public":
        raise ValueError("self-check requires a public store")
    run = store.create_run("repair_self_check", parent_run_id=parent_id)
    store.transition(run.id, "RUNNING", "PREPARING")
    checks: dict[str, str] = {}
    feedback: list[dict[str, str]] = []
    sanitizer_calls = 0
    status: Literal["PASSED", "FAILED", "UNAVAILABLE"] = "UNAVAILABLE"
    with tempfile.TemporaryDirectory(prefix="gpu-agent-self-check-") as tmp:
        root = Path(tmp)
        for name, content in sources.items():
            IsolatedGPUBackend._write_snapshot(root / name, content)
            store.put(run.id, f"sources/{name}", content, "public")
        backend = backend_factory(store, root, root / "tasks")
        handle = None
        try:
            handle = backend.prepare(
                WorkspaceRequest(
                    run_id=run.id,
                    source_manifest={n: hashlib.sha256(b).hexdigest() for n, b in sources.items()},
                    trust_level="UNTRUSTED",
                )
            )
            store.transition(run.id, "RUNNING", "COMPILING")
            built = backend.build(
                BuildRequest(workspace_id=handle.id, timeout_seconds=gate.timeout(120))
            )
            checks["build"] = "CLEAN" if built.success else "FAILED"
            if not built.success:
                status = (
                    "UNAVAILABLE"
                    if built.tool_result.tool_error
                    or built.tool_result.timed_out
                    or built.tool_result.cancelled
                    or built.tool_result.truncated
                    else "FAILED"
                )
                feedback.append(
                    {
                        "check": "build",
                        "stderr": store.read(built.tool_result.stderr_artifact).decode(
                            errors="replace"
                        )[:8000],
                    }
                )
            else:
                input_ref = store.put(run.id, "public-input.json", stdin, "public")
                store.transition(run.id, "RUNNING", "EXECUTING")
                execution = backend.run(
                    ExecutionRequest(
                        workspace_id=handle.id,
                        stdin_ref=input_ref,
                        timeout_seconds=gate.timeout(30),
                    )
                )
                checks["runtime"] = execution.runtime_status
                status = "PASSED" if execution.runtime_status == "SUCCESS" else "FAILED"
                if execution.runtime_status in {"TOOL_ERROR", "TIMEOUT", "CANCELLED", "TRUNCATED"}:
                    status = "UNAVAILABLE"
                feedback.append(
                    {
                        "check": "runtime",
                        "status": execution.runtime_status,
                        "stdout": store.read(execution.output_ref).decode(errors="replace")[:8000],
                        "stderr": store.read(execution.tool_result.stderr_artifact).decode(
                            errors="replace"
                        )[:8000],
                    }
                )
                # A concrete program failure is already actionable. Do not obscure it
                # with the sanitizer's non-complete summary for that failed program.
                if status == "PASSED":
                    if public_task is None:
                        checks["functional"] = "PUBLIC_TASK_UNAVAILABLE"
                        status = "UNAVAILABLE"
                    else:
                        store.put(
                            run.id,
                            "public-task.json",
                            public_task.model_dump_json().encode(),
                            "public",
                        )
                        try:
                            outcome = check_public_output(
                                public_task, stdin, store.read(execution.output_ref)
                            )
                        except (ValueError, OverflowError):
                            outcome = "PUBLIC_INPUT_INVALID"
                            status = "UNAVAILABLE"
                        checks["functional"] = outcome
                        feedback.append(
                            {
                                "check": "functional",
                                "outcome": outcome,
                                "requirement": public_task.requirement,
                            }
                        )
                        if outcome != "PASSED" and status != "UNAVAILABLE":
                            status = "FAILED"
                if status == "PASSED":
                    for tool in SanitizerTool:
                        request = SanitizerRequest(
                            workspace_id=handle.id,
                            tool=tool.value,
                            stdin_ref=input_ref,
                            timeout_seconds=gate.timeout(60),
                        )
                        sanitizer_calls += 1
                        result = backend.run_sanitizer(request)
                        checks[tool.value] = result.check_outcome
                        feedback.append(
                            {
                                "check": tool.value,
                                "outcome": result.check_outcome,
                                "findings": json.dumps(
                                    [
                                        f.model_dump(mode="json", exclude={"raw_ref"})
                                        for f in result.findings
                                    ]
                                )[:8000],
                            }
                        )
                        if not result.completed or result.check_outcome in {
                            "TOOL_ERROR",
                            "UNSUPPORTED",
                        }:
                            status = "UNAVAILABLE"
                            break
                        if result.check_outcome == "FINDING":
                            status = "FAILED"
                        if result.program_output_ref is not None and public_task is not None:
                            outcome = check_public_output(
                                public_task, stdin, store.read(result.program_output_ref)
                            )
                            checks[f"{tool.value}_functional"] = outcome
                            feedback.append(
                                {"check": f"{tool.value}_functional", "outcome": outcome}
                            )
                            if outcome != "PASSED":
                                status = "FAILED"
        except (BackendInfrastructureError, ProviderError) as exc:
            status = "UNAVAILABLE"
            checks["interruption"] = (
                exc.code
                if isinstance(exc, ProviderError)
                else "EXECUTION_INFRASTRUCTURE_UNAVAILABLE"
            )
        finally:
            if handle is not None:
                try:
                    backend.cleanup(handle)
                except (BackendInfrastructureError, OSError):
                    status = "UNAVAILABLE"
                    checks["interruption"] = "EXECUTION_INFRASTRUCTURE_UNAVAILABLE"
    report = PublicCheck(run_id=run.id, status=status, checks=checks, feedback=feedback)
    store.put(run.id, "self-check.json", report.model_dump_json().encode(), "public")
    store.put(
        run.id,
        "self-check-usage.json",
        json.dumps({"sanitizer_calls": sanitizer_calls}).encode(),
        "public",
    )
    store.transition(run.id, "COMPLETED", None)
    return report


def repair_candidates(
    store: RunStore,
    snapshot: SourceSnapshot,
    first: PatchCandidate,
    source: PublicSource,
    diagnosis: DiagnosisResult,
    stdin: bytes,
    provider: LLMProvider,
    backend_factory: BackendFactory,
    policy: RepairPolicy,
    public_task: PublicTask | None = None,
    *,
    coordinator: "RepairCoordinator | None" = None,
) -> PatchCandidate:
    """Select the last checked candidate; preserve all attempts against the original base."""
    if (policy.version == "public-repair-v3") != (coordinator is not None):
        raise ValueError("repair v3 requires its controller coordinator")
    run_id = snapshot.parent_run_id
    store.put(run_id, "repair/policy.json", policy.model_dump_json().encode(), "public")
    candidate = first
    selected = first
    seen: set[str] = set()
    rounds: list[dict[str, object]] = []
    stop = "CANDIDATE_LIMIT"
    for number in range(1, policy.max_candidates + 1):
        store.put(
            run_id,
            f"repair/{number}/candidate.json",
            candidate.model_dump_json().encode(),
            "public",
        )
        if candidate.patched_source_hash in seen:
            stop = "REPEATED_CANDIDATE"
            break
        seen.add(candidate.patched_source_hash)
        sources = materialize_candidate(snapshot, candidate)
        checked = self_check(
            store, run_id, sources, stdin, backend_factory, provider.gate, public_task
        )
        selected = candidate
        if coordinator is not None:
            coordinator.observe(checked)
        rounds.append(
            {
                "round": number,
                "candidate_hash": candidate.patched_source_hash,
                "check": checked.model_dump(mode="json"),
            }
        )
        store.put(
            run_id, f"repair/{number}/result.json", checked.model_dump_json().encode(), "public"
        )
        if checked.status != "FAILED":
            stop = (
                "PUBLIC_CHECKS_PASSED" if checked.status == "PASSED" else "PUBLIC_CHECK_UNAVAILABLE"
            )
            break
        if number == policy.max_candidates:
            break
        if coordinator is not None:
            decision = coordinator.decide(checked)
            store.put(
                run_id,
                f"repair/{number}/decision.json",
                decision.model_dump_json().encode(),
                "public",
            )
            rounds[-1]["decision"] = decision.model_dump(mode="json")
            if decision.action == "REINVESTIGATE":
                try:
                    updated = coordinator.investigate(number, sources, stdin, checked)
                except ProviderError as exc:
                    stop = exc.code
                    break
                if updated.diagnostic_outcome != "DIAGNOSED":
                    stop = "REINVESTIGATION_INCONCLUSIVE"
                    rounds[-1]["reinvestigation_limitations"] = updated.limitations
                    break
            diagnosis = coordinator.diagnosis
        feedback: dict[str, object] = {
            "contract": policy.version,
            "round": number,
            "previous_candidate_source": sources["kernel.cu"].decode(),
            "public_self_check": checked.model_dump(mode="json", exclude={"run_id"}),
        }
        if coordinator is not None:
            feedback.update(
                {
                    "diagnosis_source_sha256": coordinator.diagnosis_source_sha256,
                    "diagnosis_source": coordinator.diagnosis_source,
                    "original_source_sha256": coordinator.original_source_sha256,
                }
            )
        store.put(run_id, f"repair/{number}/feedback.json", json.dumps(feedback).encode(), "public")
        try:
            diff = provider.revise_patch(source, diagnosis, feedback)
            candidate = apply_generated_candidate(snapshot, diff).model_copy(
                update={
                    "generated_by": first.generated_by,
                    "provider": first.provider,
                    "model": first.model,
                    "prompt_version": (
                        REPAIR_PROMPT_VERSION if coordinator is not None else first.prompt_version
                    ),
                }
            )
        except ProviderError as exc:
            stop = exc.code
            break
        except ValueError:
            stop = "REVISION_REJECTED"
            break
    store.put(
        run_id,
        "repair/summary.json",
        json.dumps(
            {
                "version": policy.version,
                "stop_reason": stop,
                "selected_hash": selected.patched_source_hash,
                "rounds": rounds,
                **(coordinator.summary() if coordinator is not None else {}),
            }
        ).encode(),
        "public",
    )
    return selected
