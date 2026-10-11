"""Source-scoped public repair decisions and bounded candidate investigation."""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from gpu_agent.agent.models import DiagnosisResult, PublicRepairContext, PublicSource
from gpu_agent.agent.orchestrator import AgentOrchestrator
from gpu_agent.agent.prompts import REPAIR_PROMPT_VERSION
from gpu_agent.agent.provider import ProviderError
from gpu_agent.benchmark.evaluation import EvaluationMode
from gpu_agent.evidence.repository import _evidence
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import (
    BackendInfrastructureError,
    BuildRequest,
    ExecutionModel,
    ExecutionRequest,
    SanitizerTool,
    WorkspaceRequest,
)
from gpu_agent.public_task import PublicTask
from gpu_agent.repair_evidence import transfer_self_check_evidence
from gpu_agent.store import RunStore

if TYPE_CHECKING:
    from gpu_agent.repair import BackendFactory, PublicCheck


class RepairDecision(ExecutionModel):
    action: Literal["REVISE_PATCH", "REINVESTIGATE"]
    reason: str
    failure_signature: str


class RepairCoordinator:
    """Controller-owned state; a candidate diagnosis never replaces parent evidence."""

    def __init__(
        self,
        store: RunStore,
        orchestrator: AgentOrchestrator,
        backend_factory: "BackendFactory",
        public_task: PublicTask,
        original_source: PublicSource,
        diagnosis: DiagnosisResult,
        *,
        max_reinvestigations: int,
        mode: EvaluationMode,
    ) -> None:
        parent = store.load(orchestrator.handle.run_id)
        if store.visibility != "public" or parent.binding is not None:
            raise ValueError("repair coordination requires an unbound public run")
        if orchestrator.store is not store or mode not in {"D", "E"}:
            raise ValueError("repair coordinator scope mismatch")
        self.store, self.orchestrator = store, orchestrator
        self.backend_factory, self.public_task = backend_factory, public_task
        self.original_source, self.diagnosis = original_source, diagnosis
        self.diagnosis_source = original_source.content
        self.diagnosis_source_sha256 = hashlib.sha256(original_source.content.encode()).hexdigest()
        self.original_source_sha256 = self.diagnosis_source_sha256
        self.max_reinvestigations, self.mode = max_reinvestigations, mode
        self.reinvestigations = 0
        self.previous_signature: str | None = None
        self.investigation_runs: list[str] = []
        self.self_check_sanitizer_calls = 0

    def observe(self, checked: "PublicCheck") -> None:
        """Count native fixed-check attempts separately from acquisition calls."""
        refs = [
            ref
            for ref in self.store.load(checked.run_id).artifact_refs
            if ref.name == "self-check-usage.json"
        ]
        if refs:
            self.self_check_sanitizer_calls += int(
                json.loads(self.store.read(refs[-1]))["sanitizer_calls"]
            )

    def decide(self, checked: "PublicCheck") -> RepairDecision:
        if checked.status != "FAILED":
            raise ValueError("failure attribution requires a failed public check")
        signature = hashlib.sha256(
            json.dumps(checked.checks, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        repeated = signature == self.previous_signature
        self.previous_signature = signature
        reason = "PATCH_FAILURE"
        investigate = False
        if checked.checks.get("build") == "FAILED":
            reason = "BUILD_FAILURE"
        elif any(checked.checks.get(t.value) == "FINDING" for t in SanitizerTool):
            reason, investigate = "PUBLIC_SANITIZER_FAILURE", True
        elif any(
            (name == "functional" or name.endswith("_functional"))
            and outcome not in {"PASSED", "PUBLIC_TASK_UNAVAILABLE", "PUBLIC_INPUT_INVALID"}
            for name, outcome in checked.checks.items()
        ):
            reason, investigate = "PUBLIC_FUNCTIONAL_FAILURE", True
        elif repeated:
            reason, investigate = "REPEATED_PUBLIC_FAILURE", True
        if investigate and self.reinvestigations >= self.max_reinvestigations:
            reason, investigate = "REINVESTIGATION_LIMIT", False
        return RepairDecision(
            action="REINVESTIGATE" if investigate else "REVISE_PATCH",
            reason=reason,
            failure_signature=signature,
        )

    def _require_check_source(self, checked: "PublicCheck", sources: dict[str, bytes]) -> None:
        parent_id = self.orchestrator.handle.run_id
        run = self.store.load(checked.run_id)
        reports = [ref for ref in run.artifact_refs if ref.name == "self-check.json"]
        kernels = [ref for ref in run.artifact_refs if ref.name == "sources/kernel.cu"]
        if (
            run.parent_run_id != parent_id
            or run.kind != "repair_self_check"
            or run.status != "COMPLETED"
            or not reports
            or not kernels
            or json.loads(self.store.read(reports[-1])) != checked.model_dump(mode="json")
            or self.store.read(kernels[-1]) != sources["kernel.cu"]
        ):
            raise ProviderError("PUBLIC_REPAIR_SOURCE_MISMATCH")

    def investigate(
        self, number: int, sources: dict[str, bytes], stdin: bytes, checked: "PublicCheck"
    ) -> DiagnosisResult:
        if self.reinvestigations >= self.max_reinvestigations:
            raise ProviderError("REINVESTIGATION_LIMIT")
        self._require_check_source(checked, sources)
        self._save_shared_usage()
        self.reinvestigations += 1
        parent_id = self.orchestrator.handle.run_id
        run = self.store.create_run("repair_reinvestigation", parent_run_id=parent_id)
        self.investigation_runs.append(run.id)
        self.store.transition(run.id, "RUNNING", "PREPARING")
        source_sha256 = hashlib.sha256(sources["kernel.cu"]).hexdigest()
        context = PublicRepairContext(
            repair_round=number,
            original_source_sha256=self.original_source_sha256,
            candidate_source_sha256=source_sha256,
            previous_diagnosis_source_sha256=self.diagnosis_source_sha256,
            previous_diagnosis=self.diagnosis,
            public_checks=checked.checks,
            public_feedback=checked.feedback,
        )
        self.store.put(run.id, "repair/context.json", context.model_dump_json().encode(), "public")
        self.store.put(
            run.id,
            "agent/acquisition-policy.json",
            json.dumps({"mode": self.mode, "required_tools": ["memcheck"]}).encode(),
            "public",
        )
        self.store.put(
            run.id,
            "repair/lineage.json",
            json.dumps(
                {
                    "parent_run_id": parent_id,
                    "self_check_run_id": checked.run_id,
                    "source_sha256": source_sha256,
                    "original_source_sha256": self.original_source_sha256,
                    "prompt_version": REPAIR_PROMPT_VERSION,
                    "budget_before": self.orchestrator.budget.model_dump(mode="json"),
                }
            ).encode(),
            "public",
        )
        result = DiagnosisResult.inconclusive("REINVESTIGATION_NOT_COMPLETED")
        gate = self.orchestrator.provider.gate
        with tempfile.TemporaryDirectory(prefix="gpu-agent-reinvestigate-") as tmp:
            root = Path(tmp)
            for name, content in sources.items():
                IsolatedGPUBackend._write_snapshot(root / name, content)
            backend = self.backend_factory(self.store, root, root / "tasks")
            handle = None
            continuation = None
            try:
                handle = backend.prepare(
                    WorkspaceRequest(
                        run_id=run.id,
                        source_manifest={
                            n: hashlib.sha256(b).hexdigest() for n, b in sources.items()
                        },
                        trust_level="UNTRUSTED",
                    )
                )
                bundle = _evidence(self.store).view(run.id)
                _evidence(self.store).save(
                    run.id, bundle.model_copy(update={"public_task": self.public_task})
                )
                self.store.transition(run.id, "RUNNING", "COMPILING")
                built = backend.build(
                    BuildRequest(workspace_id=handle.id, timeout_seconds=gate.timeout(120))
                )
                if not built.success:
                    raise ProviderError("REINVESTIGATION_BUILD_UNAVAILABLE")
                stdin_ref = self.store.put(run.id, "public-input.json", stdin, "public")
                self.store.transition(run.id, "RUNNING", "EXECUTING")
                execution = backend.run(
                    ExecutionRequest(
                        workspace_id=handle.id,
                        stdin_ref=stdin_ref,
                        timeout_seconds=gate.timeout(30),
                    )
                )
                if execution.runtime_status in {"TOOL_ERROR", "TIMEOUT", "CANCELLED", "TRUNCATED"}:
                    raise ProviderError("EXECUTION_EVIDENCE_UNAVAILABLE")
                transfer_self_check_evidence(
                    self.store, run.id, checked.run_id, context.candidate_source_sha256, stdin
                )
                continuation = self.orchestrator.continue_in_workspace(backend, handle, stdin_ref)
                result = continuation.investigate(run.id, mode=self.mode)
            except (ProviderError, BackendInfrastructureError) as exc:
                result = DiagnosisResult.inconclusive(
                    exc.code
                    if isinstance(exc, ProviderError)
                    else "EXECUTION_INFRASTRUCTURE_UNAVAILABLE"
                )
            finally:
                if continuation is not None:
                    self.orchestrator.budget = continuation.budget
                    self.orchestrator.acquisition_usage = continuation.acquisition_usage
                if handle is not None:
                    try:
                        backend.cleanup(handle)
                    except (BackendInfrastructureError, OSError):
                        result = DiagnosisResult.inconclusive(
                            "EXECUTION_INFRASTRUCTURE_UNAVAILABLE"
                        )
                self._save_shared_usage()
        self.store.put(run.id, "diagnosis.json", result.model_dump_json().encode(), "public")
        self.store.transition(run.id, "COMPLETED", None)
        scoped = {
            "source_role": "failed_candidate",
            "source_sha256": source_sha256,
            "run_id": run.id,
            "diagnosis": result.model_dump(mode="json"),
            "budget_after": self.orchestrator.budget.model_dump(mode="json"),
        }
        self.store.put(
            parent_id,
            f"repair/{number}/reinvestigation.json",
            json.dumps(scoped).encode(),
            "public",
        )
        if result.diagnostic_outcome == "DIAGNOSED":
            self.diagnosis = result
            self.diagnosis_source = sources["kernel.cu"].decode()
            self.diagnosis_source_sha256 = source_sha256
        return result

    def _save_shared_usage(self) -> None:
        parent_id = self.orchestrator.handle.run_id
        self.orchestrator.budget = self.orchestrator.budget.model_copy(
            update={
                "llm_calls": self.orchestrator.provider.gate.snapshot().llm_calls,
                "remaining_seconds": self.orchestrator.provider.gate.remaining(),
            }
        )
        self.store.put(
            parent_id,
            "agent/budget.json",
            self.orchestrator.budget.model_dump_json().encode(),
            "public",
        )
        self.store.put(
            parent_id,
            "agent/acquisition-usage.json",
            self.orchestrator.acquisition_usage.model_dump_json().encode(),
            "public",
        )
        self.store.put(
            parent_id,
            "agent/budget-audit.json",
            json.dumps(self.orchestrator.ledger.audit).encode(),
            "public",
        )

    def summary(self) -> dict[str, object]:
        return {
            "reinvestigations": self.reinvestigations,
            "investigation_runs": self.investigation_runs,
            "diagnosis_source_sha256": self.diagnosis_source_sha256,
            "acquisition_usage": self.orchestrator.acquisition_usage.model_dump(mode="json"),
            "self_check_sanitizer_calls": self.self_check_sanitizer_calls,
        }
