"""Application workflows: immutable input -> diagnosis -> at most one registered candidate."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from gpu_agent.agent.models import AcquisitionUsage, AgentBudget, DiagnosisResult
from gpu_agent.agent.orchestrator import AgentOrchestrator, public_evidence
from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.agent.prompts import PROMPT_VERSION
from gpu_agent.agent.provider import (
    FakeProvider,
    LLMProvider,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
    ProviderError,
)
from gpu_agent.benchmark.evaluation import EvaluationMode
from gpu_agent.benchmark.ledger import CorpusFamily
from gpu_agent.contracts import RunBinding, RunManifest
from gpu_agent.environment import load_toolchain_lock
from gpu_agent.evidence.models import EvidenceBundle
from gpu_agent.evidence.repository import EvidenceRepository
from gpu_agent.execution.backend import ExecutionBackend
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import (
    BuildRequest,
    ExecutionRequest,
    SanitizerTool,
    WorkspaceHandle,
    WorkspaceRequest,
)
from gpu_agent.knowledge.models import KnowledgeError
from gpu_agent.knowledge.retrieve import KnowledgeIndex
from gpu_agent.patching import (
    PatchCandidate,
    SourceSnapshot,
    apply_candidate,
    apply_generated_candidate,
)
from gpu_agent.provenance import capture_repository_snapshot
from gpu_agent.store import RunStore, read_regular
from gpu_agent.verification.engine import VerificationEngine, register_candidate
from gpu_agent.verification.models import VerificationResult, VerificationVerdict

BackendFactory = Callable[[RunStore, Path, Path], ExecutionBackend]
BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "benchmarks"


class ApplicationService:
    def __init__(
        self,
        store: RunStore,
        evaluator_root: Path,
        *,
        provider: LLMProvider | None = None,
        backend_factory: BackendFactory = IsolatedGPUBackend,
        knowledge: KnowledgeIndex | None = None,
        knowledge_version: str = "",
        _binding: RunBinding | None = None,
    ) -> None:
        self.store, self.evaluator_root = store, evaluator_root
        self._provider, self._backend_factory = provider, backend_factory
        self.knowledge, self.knowledge_version = knowledge, knowledge_version
        self._binding = _binding

    @property
    def binding(self) -> RunBinding | None:
        return self._binding

    @classmethod
    def configured(cls) -> "ApplicationService":
        root = Path(os.environ.get("GPU_AGENT_RUN_ROOT", ".gpu-agent/runs")).absolute()
        evaluator = Path(os.environ.get("GPU_AGENT_EVALUATOR_ROOT", str(root.parent / "evaluator")))
        knowledge = None
        if cache := os.environ.get("GPU_AGENT_KNOWLEDGE_INDEX"):
            try:
                knowledge = KnowledgeIndex.load(Path(cache))
            except KnowledgeError:
                pass  # The typed knowledge limitation is emitted if retrieval is requested.
        return cls(
            RunStore(root),
            evaluator.absolute(),
            knowledge=knowledge,
            knowledge_version=os.environ.get("GPU_AGENT_KNOWLEDGE_VERSION", ""),
        )

    @classmethod
    def for_release(
        cls,
        repository: Path,
        *,
        purpose: Literal["corpus_validation", "evaluation", "release_acceptance"],
        expected_commit: str | None = None,
        prompt_version: str | None = None,
        model_config_hash: str | None = None,
    ) -> "ApplicationService":
        """Construct a bound service only from controller-observed repository state."""
        snapshot = capture_repository_snapshot(repository, expected_commit=expected_commit)
        toolchain = load_toolchain_lock(repository.absolute() / "containers/toolchain.lock.json")
        registry_hash = (
            hashlib.sha256(
                read_regular(repository.absolute() / "benchmarks/corpus-registry.json", 1024 * 1024)
            ).hexdigest()
            if purpose == "corpus_validation"
            else None
        )
        family = (
            CorpusFamily.open(Path(os.environ["GPU_AGENT_CORPUS_FAMILY_ROOT"]))
            if purpose == "corpus_validation" and "GPU_AGENT_CORPUS_FAMILY_ROOT" in os.environ
            else None
        )
        if purpose == "corpus_validation" and family is None:
            raise ValueError("trusted corpus family configuration is required")
        if family is not None:
            family.reject_repository_overlap(repository)
        confirmed = capture_repository_snapshot(repository, expected_commit=snapshot.commit)
        if confirmed != snapshot:
            raise ValueError("repository changed while release configuration was captured")
        binding = RunBinding(
            repository=snapshot,
            purpose=purpose,
            toolchain_lock_hash=toolchain.lock_hash,
            prompt_version=prompt_version,
            model_config_hash=model_config_hash,
            case_registry_hash=registry_hash,
            corpus_ledger_namespace_hash=(family.namespace_hash if family else None),
        )
        ordinary = cls.configured()
        if family is not None:
            family.require_store(ordinary.store)
        return cls(
            ordinary.store,
            ordinary.evaluator_root,
            provider=ordinary._provider,
            backend_factory=ordinary._backend_factory,
            knowledge=ordinary.knowledge,
            knowledge_version=ordinary.knowledge_version,
            _binding=binding,
        )

    def _save_diagnosis(self, run_id: str, result: DiagnosisResult) -> None:
        self.store.put(run_id, "diagnosis.json", result.model_dump_json().encode(), "public")

    def diagnosis(self, run_id: str) -> DiagnosisResult:
        refs = [r for r in self.store.load(run_id).artifact_refs if r.name == "diagnosis.json"]
        if not refs:
            return DiagnosisResult.inconclusive("DIAGNOSIS_NOT_COMPLETED")
        return DiagnosisResult.model_validate_json(self.store.read(refs[-1]))

    def candidates(self, run_id: str) -> list[str]:
        self.store.load(run_id)
        found = []
        for path in sorted(self.store.root.iterdir()):
            if path.is_dir() and len(path.name) == 32:
                manifest = self.store.load(path.name)
                if manifest.kind == "candidate" and manifest.parent_run_id == run_id:
                    found.append(manifest.id)
        return found

    def _snapshot(self, run_id: str, root: Path) -> SourceSnapshot:
        bundle = EvidenceRepository(self.store).public_view(run_id)
        hashes = {}
        for ref in bundle.source_snapshot:
            name = Path(ref.name).name
            if (
                name not in {"kernel.cu", "vector_io.cpp", "vector_api.h", "json.hpp"}
                or name in hashes
            ):
                raise ValueError("unsupported or ambiguous source snapshot")
            IsolatedGPUBackend._write_snapshot(root / name, self.store.read(ref))
            hashes[name] = ref.sha256
        return SourceSnapshot(parent_run_id=run_id, root=root, hashes=hashes)

    def diagnose(
        self,
        source: Path,
        *,
        mode: EvaluationMode = "E",
        required_tools: tuple[SanitizerTool, ...] = (SanitizerTool.MEMCHECK,),
        expected_source_hash: str | None = None,
    ) -> RunManifest:
        if mode not in {"A", "B", "C", "D", "E"}:
            raise ValueError("invalid acquisition mode")
        selected = source / "kernel.cu" if source.is_dir() else source
        data = read_regular(selected.absolute(), 4 * 1024 * 1024)
        if (
            expected_source_hash is not None
            and hashlib.sha256(data).hexdigest() != expected_source_hash
        ):
            raise ValueError("registered source hash mismatch")
        text = data.decode("utf-8")
        run = self.store.create_run("diagnosis", binding=self._binding)
        self.store.transition(run.id, "RUNNING", "PREPARING")
        self.store.put(
            run.id, "agent/acquisition-policy.json", json.dumps({"mode": mode}).encode(), "public"
        )
        ref = self.store.put(run.id, "sources/kernel.cu", data, "public")
        EvidenceRepository(self.store).save(run.id, EvidenceBundle(source_snapshot=[ref]))
        gate = LLMCallGate()
        handle: WorkspaceHandle | None = None
        backend: ExecutionBackend | None = None
        result = DiagnosisResult.inconclusive("DIAGNOSIS_NOT_COMPLETED")
        with tempfile.TemporaryDirectory(prefix="gpu-agent-source-") as temporary:
            root = Path(temporary)
            snapshot = SourceSnapshot(
                parent_run_id=run.id, root=root, hashes={"kernel.cu": ref.sha256}
            )
            IsolatedGPUBackend._write_snapshot(root / "kernel.cu", data)
            try:
                # Explicitly injected fakes are test-only; ambient config cannot select them.
                provider = self._provider
                if provider is None:
                    provider = OpenAIResponsesProvider(
                        OpenAIProviderSettings.from_environment(),
                        gate,
                        self.store,
                        run.id,
                        diff_validator=lambda diff: apply_generated_candidate(snapshot, diff),
                    )
                elif isinstance(provider, FakeProvider):
                    provider.gate = gate
                provider.ensure_available()
                gate = provider.gate
                vector = '#include "vector_api.h"' in text
                hashes = {"kernel.cu": ref.sha256}
                if vector:
                    for name, path in {
                        "vector_io.cpp": BENCHMARK_ROOT / "harness/vector_io.cpp",
                        "vector_api.h": BENCHMARK_ROOT / "harness/vector_api.h",
                        "json.hpp": BENCHMARK_ROOT / "harness/vendor/json.hpp",
                    }.items():
                        content = read_regular(path, 4 * 1024 * 1024)
                        IsolatedGPUBackend._write_snapshot(root / name, content)
                        hashes[name] = hashlib.sha256(content).hexdigest()
                snapshot = snapshot.model_copy(update={"hashes": hashes})
                backend = self._backend_factory(self.store, root, root / "tasks")
                handle = backend.prepare(
                    WorkspaceRequest(run_id=run.id, source_manifest=hashes, trust_level="UNTRUSTED")
                )
                self.store.transition(run.id, "RUNNING", "COMPILING")
                build = backend.build(
                    BuildRequest(workspace_id=handle.id, timeout_seconds=gate.timeout(120))
                )
                if not build.success:
                    raise ProviderError(build.tool_result.tool_error or "BUILD_FAILED")
                stdin = (
                    json.dumps({"n": 257, "a": [1.0] * 257, "b": [2.0] * 257}).encode()
                    if vector
                    else b""
                )
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
                result = AgentOrchestrator(
                    self.store,
                    provider,
                    backend,
                    handle,
                    stdin_ref,
                    self.knowledge,
                    self.knowledge_version,
                ).investigate(run.id, mode=mode, required_tools=required_tools)
                if result.diagnostic_outcome == "DIAGNOSED":
                    self.store.transition(run.id, "RUNNING", "PATCH_GENERATING")
                    public_source = public_evidence(self.store, run.id).sources[0]
                    diff = provider.propose_patch(public_source, result)
                    try:
                        candidate = apply_generated_candidate(snapshot, diff)
                    except ValueError:
                        raise ProviderError("LLM_INVALID_OUTPUT") from None
                    candidate = candidate.model_copy(
                        update={
                            "generated_by": "agent",
                            "provider": provider.provider_name,
                            "model": provider.model_name,
                            "prompt_version": PROMPT_VERSION,
                        }
                    )
                    register_candidate(self.store, candidate)
            except ProviderError as exc:
                if result.diagnostic_outcome == "DIAGNOSED":
                    result = result.model_copy(
                        update={"limitations": [*result.limitations, exc.code]}
                    )
                else:
                    result = DiagnosisResult.inconclusive(exc.code)
            finally:
                if backend is not None and handle is not None:
                    backend.cleanup(handle)
        self._save_diagnosis(run.id, result)
        if not any(
            ref.name == "agent/acquisition-usage.json"
            for ref in self.store.load(run.id).artifact_refs
        ):
            # Preparation failed before the orchestrator could invoke acquisition dependencies.
            self.store.put(
                run.id,
                "agent/acquisition-usage.json",
                AcquisitionUsage(sanitizer_calls=0, retrieval_calls=0).model_dump_json().encode(),
                "public",
            )
        budget_refs = [
            r for r in self.store.load(run.id).artifact_refs if r.name == "agent/budget.json"
        ]
        budget = (
            AgentBudget.model_validate_json(self.store.read(budget_refs[-1]))
            if budget_refs
            else (AgentBudget())
        )
        final_budget = budget.model_copy(
            update={"llm_calls": gate.snapshot().llm_calls, "remaining_seconds": gate.remaining()}
        )
        self.store.put(
            run.id, "agent/final-budget.json", final_budget.model_dump_json().encode(), "public"
        )
        self.store.put(
            run.id,
            "agent/usage-summary.json",
            json.dumps(
                {
                    "physical_calls": gate.snapshot().llm_calls,
                    "synthetic": isinstance(self._provider, FakeProvider),
                }
            ).encode(),
            "public",
        )
        self.store.transition(run.id, "RUNNING", "FINALIZING")
        return self.store.transition(run.id, "COMPLETED", None)

    def register_patch(self, run_id: str, candidate_path: Path) -> str:
        diff = read_regular(candidate_path.absolute(), 4 * 1024 * 1024).decode("utf-8")
        with tempfile.TemporaryDirectory(prefix="gpu-agent-patch-") as directory:
            snapshot = self._snapshot(run_id, Path(directory))
            candidate = apply_candidate(snapshot, diff, ["kernel.cu"])
        return register_candidate(self.store, candidate)

    def verify(
        self, run_id: str, candidate_id: str | None = None, strict: bool = False
    ) -> VerificationResult:
        self.store.load(run_id)
        if candidate_id is None:
            ids = self.candidates(run_id)
            if len(ids) != 1:
                return self._inconclusive_verification(run_id, "CANDIDATE_UNAVAILABLE")
            candidate_id = ids[0]
            candidate_ref = next(
                r for r in self.store.load(candidate_id).artifact_refs if r.name == "candidate.json"
            )
            generated = PatchCandidate.model_validate_json(self.store.read(candidate_ref))
            if generated.generated_by != "agent":
                return self._inconclusive_verification(run_id, "CANDIDATE_UNAVAILABLE")
        registration = self.store.load(candidate_id)
        if registration.kind != "candidate" or registration.parent_run_id != run_id:
            raise ValueError("candidate does not belong to original run")
        bundle = EvidenceRepository(self.store).public_view(run_id)
        if len(bundle.source_snapshot) != 4:
            return self._inconclusive_verification(run_id, "ORACLE_UNAVAILABLE", candidate_id)
        return VerificationEngine(self.store, self.evaluator_root).verify(
            run_id, candidate_id, "full" if strict else "standard"
        )

    def _inconclusive_verification(
        self, run_id: str, code: str, candidate_id: str | None = None
    ) -> VerificationResult:
        candidate_hash = ""
        if candidate_id:
            ref = next(
                r for r in self.store.load(candidate_id).artifact_refs if r.name == "candidate.json"
            )
            candidate_hash = PatchCandidate.model_validate_json(
                self.store.read(ref)
            ).patched_source_hash
        result = VerificationResult(
            verdict=VerificationVerdict.INCONCLUSIVE,
            failure_stage="precondition",
            reason_code=code,
            original_finding_present=None,
            public_oracle_passed=None,
            private_holdout_passed=None,
            required_checks={"oracle": "NOT_RUN"},
            candidate_hash=candidate_hash,
            suite_hash="",
            not_run_count=1,
            limitations=[code],
        )
        verification = self.store.create_run("verification", run_id)
        self.store.put(
            verification.id, "verification/result.json", result.model_dump_json().encode(), "public"
        )
        self.store.transition(verification.id, "RUNNING", "FINALIZING")
        self.store.transition(verification.id, "COMPLETED", None)
        return result

    def report(self, run_id: str) -> str:
        from gpu_agent.reporting import render_report

        return render_report(self.store, run_id)
