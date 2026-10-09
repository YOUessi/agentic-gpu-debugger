"""Application workflows: immutable input -> diagnosis -> at most one registered candidate."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

from gpu_agent.agent.models import AcquisitionUsage, AgentBudget, DiagnosisResult
from gpu_agent.agent.orchestrator import AgentOrchestrator, public_evidence
from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.agent.prompts import PROMPT_VERSION
from gpu_agent.agent.provider import (
    DevelopmentCallPolicy,
    FakeProvider,
    Invocation,
    LLMProvider,
    MockResponsesProvider,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
    ProviderError,
    provider_invocations,
)
from gpu_agent.benchmark.evaluation import (
    EvaluationMode,
    EvaluationProviderPolicy,
    EvaluationUnitBinding,
    PricingAttestation,
)
from gpu_agent.benchmark.ledger import CorpusFamily
from gpu_agent.contracts import RunBinding, RunManifest, RunStatus
from gpu_agent.environment import load_toolchain_lock
from gpu_agent.evidence.models import EvidenceBundle
from gpu_agent.evidence.repository import _evidence
from gpu_agent.execution.backend import ExecutionBackend
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import (
    BackendInfrastructureError,
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
from gpu_agent.provenance import capture_repository_snapshot, runtime_code_fingerprint
from gpu_agent.public_task import PublicRepairInputError, load_public_task, public_expected_output
from gpu_agent.repair import RepairPolicy, repair_candidates
from gpu_agent.store import RunStore, read_regular
from gpu_agent.verification.engine import (
    VerificationEngine,
    candidate_run_id,
    register_candidate,
    verification_run_id,
)
from gpu_agent.verification.models import VerificationResult, VerificationVerdict

BackendFactory = Callable[[RunStore, Path, Path], ExecutionBackend]
BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "benchmarks"

if TYPE_CHECKING:
    from gpu_agent.benchmark.holdout import (
        HoldoutBatch,
        HoldoutController,
        PreparedHoldoutExecution,
    )
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier


class ApplicationService:
    def __init__(
        self,
        store: RunStore,
        evaluator_store: RunStore,
        *,
        provider: LLMProvider | None = None,
        backend_factory: BackendFactory = IsolatedGPUBackend,
        knowledge: KnowledgeIndex | None = None,
        knowledge_version: str = "",
        _binding: RunBinding | None = None,
        _evaluation_schedule_verifier: "EvaluationScheduleVerifier | None" = None,
    ) -> None:
        if evaluator_store.visibility != "evaluator":
            raise ValueError("verification requires an evaluator RunStore")
        self.store, self.evaluator_store = store, evaluator_store
        self._provider, self._backend_factory = provider, backend_factory
        self.knowledge, self.knowledge_version = knowledge, knowledge_version
        self._binding = _binding
        self._evaluation_schedule_verifier = _evaluation_schedule_verifier
        self._pricing_attestation: PricingAttestation | None = None
        self._paid_calls: DevelopmentCallPolicy | None = None
        self._repository_root: Path | None = None

    @property
    def evaluator_root(self) -> Path:
        """Compatibility path for callers not yet migrated to the exact store capability."""
        return self.evaluator_store.root.parent

    def provider_invocations(self, run_id: str) -> list[Invocation]:
        return provider_invocations(self.store, run_id)

    def _bind_evaluation_schedule_verifier(self, verifier: "EvaluationScheduleVerifier") -> None:
        from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

        EvaluationScheduleVerifier.require_store(verifier, self.store)
        if (
            self._evaluation_schedule_verifier is not None
            and self._evaluation_schedule_verifier is not verifier
        ):
            raise ValueError("evaluation service authority is already bound")
        self._evaluation_schedule_verifier = verifier

    def allow_development_paid_calls(self, policy: DevelopmentCallPolicy) -> None:
        """Opt one non-evaluation service into a bounded number of physical calls.

        Runs produced this way carry `agent/development-mode.json` and are never evaluation
        or release evidence.
        """
        if self._binding is not None:
            raise ValueError("development paid calls are forbidden on a bound service")
        self._paid_calls = DevelopmentCallPolicy.model_validate(policy.model_dump())

    def _attest_runtime_code(self) -> str | None:
        """Re-hash the executing code for every evaluation unit; any drift stops the batch."""
        expected = self._binding.runtime_code_hash if self._binding else None
        if expected is None:
            return None
        if self._repository_root is None:
            raise ValueError("runtime code binding has no repository root")
        if runtime_code_fingerprint(self._repository_root) != expected:
            raise ValueError("executing code changed after the evaluation binding")
        return expected

    def _bind_pricing_attestation(self, attestation: PricingAttestation) -> None:
        """Bind one controller-reviewed rate card before any evaluation provider call."""
        if self._binding is None or self._binding.purpose != "evaluation":
            raise ValueError("pricing attestation requires an evaluation-bound service")
        if (
            attestation.repository_commit != self._binding.repository.commit
            or attestation.model_config_hash != self._binding.model_config_hash
        ):
            raise ValueError("pricing attestation differs from evaluation binding")
        if self._pricing_attestation is not None and self._pricing_attestation != attestation:
            raise ValueError("evaluation pricing is already bound")
        self._pricing_attestation = attestation

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
            RunStore(evaluator.absolute() / "runs", visibility="evaluator"),
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
        require_corpus_family: bool = False,
        cost_policy: Literal["capped", "record_only"] = "record_only",
        workflow_visibility: Literal["public", "evaluator"] = "public",
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
            if "GPU_AGENT_CORPUS_FAMILY_ROOT" in os.environ
            else None
        )
        if (purpose == "corpus_validation" or require_corpus_family) and family is None:
            raise ValueError("trusted corpus family configuration is required")
        if family is not None:
            family.reject_repository_overlap(repository)
        if workflow_visibility == "evaluator" and purpose != "evaluation":
            raise ValueError("evaluator workflow is reserved for evaluation")
        confirmed = capture_repository_snapshot(repository, expected_commit=snapshot.commit)
        if confirmed != snapshot:
            raise ValueError("repository changed while release configuration was captured")
        binding = RunBinding(
            repository=snapshot,
            cost_policy=cost_policy,
            purpose=purpose,
            runtime_code_hash=(
                runtime_code_fingerprint(repository) if purpose == "evaluation" else None
            ),
            toolchain_lock_hash=toolchain.lock_hash,
            prompt_version=prompt_version,
            model_config_hash=model_config_hash,
            case_registry_hash=registry_hash,
            corpus_ledger_namespace_hash=(family.namespace_hash if family else None),
        )
        if family is not None and purpose == "evaluation":
            from gpu_agent.benchmark.controller_config import (
                validate_production_store_configuration,
            )

            public, evaluator = validate_production_store_configuration(family, repository)
            knowledge = None
            if cache := os.environ.get("GPU_AGENT_KNOWLEDGE_INDEX"):
                try:
                    knowledge = KnowledgeIndex.load(Path(cache))
                except KnowledgeError:
                    pass
            ordinary = cls(
                public if workflow_visibility == "public" else evaluator,
                evaluator,
                knowledge=knowledge,
                knowledge_version=os.environ.get("GPU_AGENT_KNOWLEDGE_VERSION", ""),
            )
        else:
            ordinary = cls.configured()
            if family is not None:
                family.require_store(ordinary.store)
        bound = cls(
            ordinary.store,
            ordinary.evaluator_store,
            provider=ordinary._provider,
            backend_factory=ordinary._backend_factory,
            knowledge=ordinary.knowledge,
            knowledge_version=ordinary.knowledge_version,
            _binding=binding,
        )
        bound._repository_root = repository.absolute()
        return bound

    def _save_diagnosis(self, run_id: str, result: DiagnosisResult) -> None:
        self.store.put(
            run_id,
            "diagnosis.json",
            result.model_dump_json().encode(),
            self.store.visibility,
        )

    def diagnosis(self, run_id: str) -> DiagnosisResult:
        refs = [r for r in self.store.load(run_id).artifact_refs if r.name == "diagnosis.json"]
        if not refs:
            return DiagnosisResult.inconclusive("DIAGNOSIS_NOT_COMPLETED")
        return DiagnosisResult.model_validate_json(self.store.read(refs[-1]))

    def candidates(self, run_id: str) -> list[str]:
        self.store.load(run_id)
        exact_id = candidate_run_id(run_id)
        if not (self.store.root / exact_id).exists():
            return []
        candidate = self.store.load(exact_id)
        if candidate.kind != "candidate" or candidate.parent_run_id != run_id:
            raise ValueError("candidate slot is invalid")
        return [exact_id]

    def _snapshot(self, run_id: str, root: Path) -> SourceSnapshot:
        bundle = _evidence(self.store).view(run_id)
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
        expected_input_hash: str | None = None,
        evaluation_unit: EvaluationUnitBinding | None = None,
    ) -> RunManifest:
        return self._diagnose(
            source,
            mode=mode,
            required_tools=required_tools,
            expected_source_hash=expected_source_hash,
            expected_input_hash=expected_input_hash,
            evaluation_unit=evaluation_unit,
        )

    def repair(
        self,
        source: Path,
        *,
        policy: RepairPolicy | None = None,
        mode: EvaluationMode = "E",
    ) -> tuple[RunManifest, VerificationResult | None]:
        """Public-only iterative repair, then one independent verification; not evaluation."""
        if self._binding is not None or self.store.visibility != "public":
            raise ValueError("iterative repair is a public, non-evaluation workflow")
        if mode not in {"D", "E"}:
            raise ValueError("iterative repair supports D/E; A-C ablations remain single-candidate")
        run = self._diagnose(
            source,
            mode=mode,
            required_tools=(SanitizerTool.MEMCHECK,),
            expected_source_hash=None,
            evaluation_unit=None,
            repair_policy=policy or RepairPolicy(),
        )
        summaries = [ref for ref in run.artifact_refs if ref.name == "repair/summary.json"]
        verified = None
        if (
            summaries
            and json.loads(self.store.read(summaries[-1]))["stop_reason"] == "PUBLIC_CHECKS_PASSED"
        ):
            verified = self.verify_exact(run.id, strict=True)[0]
        return run, verified

    @staticmethod
    def _public_input(
        kernel: Path, expected_hash: str | None, evaluation_unit: EvaluationUnitBinding | None
    ) -> bytes:
        """The case's registered public input (`input.json` beside kernel.cu).

        Evaluation units must use exactly the input the corpus validated for that case; a
        single hard-coded size would miss defects that only a case's own input triggers.
        Ad-hoc development runs without an input file keep the historical default.
        """
        path = kernel.parent / "input.json"
        if path.exists():
            content = read_regular(path.absolute(), 4 * 1024 * 1024)
        elif expected_hash is None and evaluation_unit is None:
            return json.dumps({"n": 257, "a": [1.0] * 257, "b": [2.0] * 257}).encode()
        else:
            raise ValueError("registered public input is unavailable")
        if expected_hash is not None and hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError("registered public input hash mismatch")
        return content

    def _diagnose_reserved(
        self,
        source: Path,
        *,
        controller: "HoldoutController",
        batch: "HoldoutBatch",
        prepared: "PreparedHoldoutExecution",
        mode: EvaluationMode,
        required_tools: tuple[SanitizerTool, ...],
        expected_source_hash: str,
        expected_input_hash: str | None = None,
    ) -> RunManifest:
        from gpu_agent.benchmark.holdout import (
            HoldoutBatch,
            HoldoutController,
            PreparedHoldoutExecution,
        )

        if (
            type(controller) is not HoldoutController
            or type(batch) is not HoldoutBatch
            or type(prepared) is not PreparedHoldoutExecution
        ):
            raise ValueError("reserved diagnosis authority is invalid")
        return self._diagnose(
            source,
            mode=mode,
            required_tools=required_tools,
            expected_source_hash=expected_source_hash,
            expected_input_hash=expected_input_hash,
            evaluation_unit=prepared.evaluation_unit,
            _reserved_controller=controller,
            _reserved_batch=batch,
            _reserved_prepared=prepared,
        )

    def _diagnose(
        self,
        source: Path,
        *,
        mode: EvaluationMode,
        required_tools: tuple[SanitizerTool, ...],
        expected_source_hash: str | None,
        evaluation_unit: EvaluationUnitBinding | None,
        expected_input_hash: str | None = None,
        _reserved_controller: object | None = None,
        _reserved_batch: object | None = None,
        _reserved_prepared: object | None = None,
        repair_policy: RepairPolicy | None = None,
    ) -> RunManifest:
        if mode not in {"A", "B", "C", "D", "E"}:
            raise ValueError("invalid acquisition mode")
        if evaluation_unit is not None and (
            self._binding is None
            or self._binding.purpose != "evaluation"
            or evaluation_unit.mode != mode
        ):
            raise ValueError("evaluation unit requires an evaluation-bound service")
        runtime_code_hash = self._attest_runtime_code() if evaluation_unit is not None else None
        reserved_values = (
            _reserved_controller,
            _reserved_batch,
            _reserved_prepared,
        )
        reserved_started = any(value is not None for value in reserved_values)
        if reserved_started:
            from gpu_agent.benchmark.holdout import (
                HoldoutBatch,
                HoldoutController,
                PreparedHoldoutExecution,
            )

            if (
                type(_reserved_controller) is not HoldoutController
                or type(_reserved_batch) is not HoldoutBatch
                or type(_reserved_prepared) is not PreparedHoldoutExecution
                or evaluation_unit != _reserved_prepared.evaluation_unit
            ):
                raise ValueError("reserved diagnosis authority is invalid")
            run = _reserved_controller.authorize_and_start_reserved(
                _reserved_batch, _reserved_prepared, self
            )
        else:
            run = None
        selected = source / "kernel.cu" if source.is_dir() else source
        data = read_regular(selected.absolute(), 4 * 1024 * 1024)
        if (
            expected_source_hash is not None
            and hashlib.sha256(data).hexdigest() != expected_source_hash
        ):
            raise ValueError("registered source hash mismatch")
        public_input = self._public_input(selected, expected_input_hash, evaluation_unit)
        public_task = None
        if repair_policy is not None:
            try:
                public_task = load_public_task(selected, data)
            except (ValueError, OSError):
                raise PublicRepairInputError("PUBLIC_TASK_INVALID") from None
            if public_task is None:
                raise PublicRepairInputError("PUBLIC_TASK_UNAVAILABLE")
            if '#include "vector_api.h"' not in data.decode("utf-8"):
                raise PublicRepairInputError("PUBLIC_INTERFACE_UNSUPPORTED")
            try:
                public_expected_output(public_task, public_input)
            except (ValueError, OverflowError):
                raise PublicRepairInputError("PUBLIC_INPUT_INVALID") from None
        text = data.decode("utf-8")
        if evaluation_unit is not None and not reserved_started:
            if self._evaluation_schedule_verifier is None:
                raise ValueError("evaluation unit requires signed schedule authority")
            run = self.store.validate_and_create_evaluation_child(
                self._evaluation_schedule_verifier, evaluation_unit
            )
            if run.status == RunStatus.COMPLETED:
                return run
        elif evaluation_unit is None:
            run = self.store.create_run("diagnosis", binding=self._binding)
        if run is None:
            raise ValueError("diagnosis run authorization is missing")
        if not reserved_started:
            self.store.transition(run.id, "RUNNING", "PREPARING")
        self.store.put(
            run.id,
            "agent/acquisition-policy.json",
            json.dumps(
                {
                    "mode": mode,
                    "required_tools": [tool.value for tool in required_tools],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            self.store.visibility,
        )
        if runtime_code_hash is not None:
            self.store.put(
                run.id,
                "agent/runtime-code.json",
                json.dumps({"runtime_code_hash": runtime_code_hash}).encode(),
                self.store.visibility,
            )
        ref = self.store.put(run.id, "sources/kernel.cu", data, self.store.visibility)
        if public_task is not None:
            self.store.put(
                run.id, "public-task.json", public_task.model_dump_json().encode(), "public"
            )
        _evidence(self.store).save(
            run.id, EvidenceBundle(source_snapshot=[ref], public_task=public_task)
        )
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
                provider: LLMProvider
                if self._provider is None:
                    provider = OpenAIResponsesProvider(
                        OpenAIProviderSettings.from_environment(),
                        gate,
                        self.store,
                        run.id,
                        diff_validator=lambda diff: apply_generated_candidate(snapshot, diff),
                        idempotency_key=(
                            evaluation_unit.idempotency_key if evaluation_unit else None
                        ),
                        call_policy=self._paid_calls if evaluation_unit is None else None,
                    )
                else:
                    provider = self._provider
                if isinstance(provider, FakeProvider):
                    provider.gate = gate
                # Capability validation is local-only. It must finish before pricing checks,
                # while every physical provider invocation remains behind both gates.
                provider.ensure_available()
                if evaluation_unit is None and not isinstance(provider, FakeProvider):
                    # Outside evaluation a real provider needs the explicit development opt-in.
                    if self._paid_calls is None:
                        raise ProviderError("PAID_CALLS_NOT_ALLOWED")
                    self.store.put(
                        run.id,
                        "agent/development-mode.json",
                        self._paid_calls.model_dump_json().encode(),
                        self.store.visibility,
                    )
                elif not isinstance(provider, FakeProvider):
                    settings = getattr(provider, "settings", None)
                    if not isinstance(settings, OpenAIProviderSettings):
                        raise ProviderError("MODEL_CONFIG_MISMATCH")
                    if (
                        self._binding is None
                        or self._binding.prompt_version != PROMPT_VERSION
                        or not provider.model_name
                    ):
                        raise ProviderError("MODEL_CONFIG_MISMATCH")
                    endpoint_host = urlsplit(settings.endpoint or "").hostname or ""
                    if not endpoint_host:
                        raise ProviderError("MODEL_CONFIG_MISMATCH")
                    pricing = self._pricing_attestation
                    expected_pricing_source = (
                        "TEST_ONLY" if isinstance(provider, MockResponsesProvider) else "REVIEWED"
                    )
                    if pricing is None or pricing.source != expected_pricing_source:
                        raise ProviderError("PRICING_ATTESTATION_REQUIRED")
                    policy = EvaluationProviderPolicy(
                        provider=provider.provider_name,
                        endpoint_host=endpoint_host,
                        configured_model=provider.model_name,
                        allowed_response_models=[provider.model_name],
                        prompt_version=PROMPT_VERSION,
                        pricing_hash=pricing.rate_card_hash,
                    )
                    if policy.sha256 != self._binding.model_config_hash:
                        raise ProviderError("MODEL_CONFIG_MISMATCH")
                    if (
                        pricing.provider != policy.provider
                        or pricing.model != policy.configured_model
                        or pricing.repository_commit != self._binding.repository.commit
                        or pricing.model_config_hash != self._binding.model_config_hash
                    ):
                        raise ProviderError("PRICING_ATTESTATION_REQUIRED")
                    self.store.put(
                        run.id,
                        "agent/provider-policy.json",
                        policy.content(),
                        self.store.visibility,
                    )
                    self.store.put(
                        run.id,
                        "agent/pricing-attestation.json",
                        pricing.model_dump_json().encode(),
                        self.store.visibility,
                    )
                gate = provider.gate
                vector = '#include "vector_api.h"' in text
                hashes = {"kernel.cu": ref.sha256}
                source_refs = [ref]
                if vector:
                    for name, path in {
                        "vector_io.cpp": BENCHMARK_ROOT / "harness/vector_io.cpp",
                        "vector_api.h": BENCHMARK_ROOT / "harness/vector_api.h",
                        "json.hpp": BENCHMARK_ROOT / "harness/vendor/json.hpp",
                    }.items():
                        content = read_regular(path, 4 * 1024 * 1024)
                        IsolatedGPUBackend._write_snapshot(root / name, content)
                        source_ref = self.store.put(
                            run.id, f"sources/{name}", content, self.store.visibility
                        )
                        source_refs.append(source_ref)
                        hashes[name] = source_ref.sha256
                _evidence(self.store).save(
                    run.id, EvidenceBundle(source_snapshot=source_refs, public_task=public_task)
                )
                snapshot = snapshot.model_copy(update={"hashes": hashes})
                backend = self._backend_factory(self.store, root, root / "tasks")
                handle = backend.prepare(
                    WorkspaceRequest(run_id=run.id, source_manifest=hashes, trust_level="UNTRUSTED")
                )
                if public_task is not None:
                    prepared = _evidence(self.store).view(run.id)
                    _evidence(self.store).save(
                        run.id, prepared.model_copy(update={"public_task": public_task})
                    )
                self.store.transition(run.id, "RUNNING", "COMPILING")
                build = backend.build(
                    BuildRequest(workspace_id=handle.id, timeout_seconds=gate.timeout(120))
                )
                if not build.success:
                    raise ProviderError(build.tool_result.tool_error or "BUILD_FAILED")
                stdin = public_input if vector else b""
                stdin_ref = self.store.put(
                    run.id, "public-input.json", stdin, self.store.visibility
                )
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
                orchestrator = AgentOrchestrator(
                    self.store,
                    provider,
                    backend,
                    handle,
                    stdin_ref,
                    self.knowledge,
                    self.knowledge_version,
                )
                result = orchestrator.investigate(run.id, mode=mode, required_tools=required_tools)
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
                    if repair_policy is not None:
                        from gpu_agent.repair_coordinator import RepairCoordinator

                        coordinator = None
                        if repair_policy.version == "public-repair-v3":
                            assert public_task is not None
                            coordinator = RepairCoordinator(
                                self.store,
                                orchestrator,
                                self._backend_factory,
                                public_task,
                                public_source,
                                result,
                                max_reinvestigations=repair_policy.max_reinvestigations,
                                mode=mode,
                            )
                        candidate = repair_candidates(
                            self.store,
                            snapshot,
                            candidate,
                            public_source,
                            result,
                            stdin,
                            provider,
                            self._backend_factory,
                            repair_policy,
                            public_task,
                            coordinator=coordinator,
                        )
                    register_candidate(self.store, candidate)
            except (ProviderError, BackendInfrastructureError) as exc:
                code = (
                    exc.code
                    if isinstance(exc, ProviderError)
                    else "EXECUTION_INFRASTRUCTURE_UNAVAILABLE"
                )
                if result.diagnostic_outcome == "DIAGNOSED":
                    result = result.model_copy(update={"limitations": [*result.limitations, code]})
                else:
                    result = DiagnosisResult.inconclusive(code)
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
                self.store.visibility,
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
            run.id,
            "agent/final-budget.json",
            final_budget.model_dump_json().encode(),
            self.store.visibility,
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
            self.store.visibility,
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
        return self.verify_exact(run_id, candidate_id, strict)[0]

    def verify_exact(
        self, run_id: str, candidate_id: str | None = None, strict: bool = False
    ) -> tuple[VerificationResult, str]:
        """Verify and return the exact persisted verification run without discovery."""
        mode: Literal["standard", "full"] = "full" if strict else "standard"
        self.store.load(run_id)
        if candidate_id is None:
            ids = self.candidates(run_id)
            if len(ids) != 1:
                result = self._inconclusive_verification(run_id, "CANDIDATE_UNAVAILABLE", mode=mode)
                return result, verification_run_id(run_id, result.candidate_hash, mode)
            candidate_id = ids[0]
            candidate_ref = next(
                r for r in self.store.load(candidate_id).artifact_refs if r.name == "candidate.json"
            )
            generated = PatchCandidate.model_validate_json(self.store.read(candidate_ref))
            if generated.generated_by != "agent":
                result = self._inconclusive_verification(run_id, "CANDIDATE_UNAVAILABLE", mode=mode)
                return result, verification_run_id(run_id, result.candidate_hash, mode)
        registration = self.store.load(candidate_id)
        if registration.kind != "candidate" or registration.parent_run_id != run_id:
            raise ValueError("candidate does not belong to original run")
        bundle = _evidence(self.store).view(run_id)
        if len(bundle.source_snapshot) != 4:
            result = self._inconclusive_verification(
                run_id, "ORACLE_UNAVAILABLE", candidate_id, mode=mode
            )
            return result, verification_run_id(run_id, result.candidate_hash, mode)
        result = VerificationEngine(self.store, self.evaluator_store).verify(
            run_id, candidate_id, mode
        )
        return result, verification_run_id(run_id, result.candidate_hash, mode)

    def _inconclusive_verification(
        self,
        run_id: str,
        code: str,
        candidate_id: str | None = None,
        *,
        mode: Literal["standard", "full"] = "standard",
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
            required_checks={"oracle": "NOT_RUN"},
            candidate_hash=candidate_hash,
            limitations=[code],
        )
        return VerificationEngine(self.store, self.evaluator_store)._persist_public(
            run_id, result, mode
        )

    def report(self, run_id: str) -> str:
        from gpu_agent.reporting import render_report

        _evidence(self.store).public_view(run_id)
        return render_report(
            self.store,
            run_id,
            self.evaluator_store,
        )
