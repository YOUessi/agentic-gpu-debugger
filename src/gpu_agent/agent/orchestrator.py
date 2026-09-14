"""Controller-owned typed registry: the model proposes, policy authorizes, tools execute."""

from collections.abc import Callable
from pathlib import PurePosixPath

from pydantic import ValidationError

from gpu_agent.agent.models import (
    ACTION_ADAPTER,
    AgentAction,
    AgentBudget,
    DiagnosisResult,
    EvidenceClaim,
    PublicEvidence,
    PublicFinding,
    PublicSource,
)
from gpu_agent.agent.policy import decide_action, validate_diagnosis
from gpu_agent.agent.provider import LLMProvider, ProviderError
from gpu_agent.contracts import ArtifactRef, CurrentPhase
from gpu_agent.evidence.repository import EvidenceRepository
from gpu_agent.execution.backend import ExecutionBackend
from gpu_agent.execution.models import SanitizerRequest, SourceLocation, WorkspaceHandle
from gpu_agent.knowledge.models import DocumentChunk, KnowledgeError
from gpu_agent.knowledge.retrieve import KnowledgeIndex
from gpu_agent.store import RunStore


def public_evidence(store: RunStore, run_id: str) -> PublicEvidence:
    bundle = EvidenceRepository(store).public_view(run_id)
    sources = [
        PublicSource(source_id=ref.id, content=store.read(ref).decode("utf-8"))
        for ref in bundle.source_snapshot
        if PurePosixPath(ref.name).name == "kernel.cu"
    ]
    facts: list[EvidenceClaim] = []
    if bundle.build_result is not None:
        build = bundle.build_result
        facts.append(
            EvidenceClaim(
                text=f"Build success: {build.success}",
                citation_ids=[build.tool_result.stdout_artifact.id],
            )
        )
    if bundle.execution_result is not None:
        run = bundle.execution_result
        facts.append(
            EvidenceClaim(
                text=f"Runtime status: {run.runtime_status}", citation_ids=[run.output_ref.id]
            )
        )
    findings: list[PublicFinding] = []
    for result in bundle.sanitizer_results:
        for finding in result.findings:
            if finding.raw_ref is None:
                continue
            location = finding.source_location
            public_location = None
            if location is not None and PurePosixPath(location.path).name == "kernel.cu":
                public_location = SourceLocation(path="kernel.cu", line=location.line)
            findings.append(
                PublicFinding(
                    artifact_id=finding.raw_ref.id,
                    category=finding.category,
                    source_location=public_location,
                )
            )
    docs = [DocumentChunk.model_validate_json(store.read(ref)) for ref in bundle.retrieved_chunks]
    return PublicEvidence(
        sources=sources, observed_facts=facts, tool_findings=findings, documentation=docs
    )


class AgentOrchestrator:
    def __init__(
        self,
        store: RunStore,
        provider: LLMProvider,
        backend: ExecutionBackend,
        handle: WorkspaceHandle,
        stdin_ref: ArtifactRef,
        knowledge: KnowledgeIndex | None,
        knowledge_version: str,
        budget: AgentBudget | None = None,
    ) -> None:
        self.store, self.provider, self.backend = store, provider, backend
        self.handle, self.stdin_ref = handle, stdin_ref
        self.knowledge, self.knowledge_version = knowledge, knowledge_version
        self.budget = budget or AgentBudget()
        self.seen: set[str] = set()
        self.registry: dict[str, Callable[[AgentAction], None]] = {
            "run_memcheck": self._memcheck,
            "retrieve_official_docs": self._docs,
            "inspect_source": self._source,
        }

    def _memcheck(self, action: AgentAction) -> None:
        if action.action_type != "run_memcheck":
            raise ValueError("typed registry mismatch")
        self.budget = self.budget.model_copy(
            update={"sanitizer_calls": self.budget.sanitizer_calls + 1}
        )
        result = self.backend.run_sanitizer(
            SanitizerRequest(
                workspace_id=self.handle.id,
                stdin_ref=self.stdin_ref,
                timeout_seconds=self.provider.gate.timeout(120),
            )
        )
        if not result.completed or result.check_outcome in {"TOOL_ERROR", "UNSUPPORTED"}:
            raise ProviderError("SANITIZER_EVIDENCE_UNAVAILABLE")

    def _docs(self, action: AgentAction) -> None:
        if action.action_type != "retrieve_official_docs":
            raise ValueError("typed registry mismatch")
        self.budget = self.budget.model_copy(update={"rag_calls": self.budget.rag_calls + 1})
        if self.knowledge is None:
            raise ProviderError("KNOWLEDGE_UNAVAILABLE")
        try:
            result = self.knowledge.retrieve(
                action.typed_arguments.query, self.knowledge_version, action.typed_arguments.k
            )
        except KnowledgeError:
            raise ProviderError("KNOWLEDGE_UNAVAILABLE") from None
        if not result.chunks:
            raise ProviderError("NO_INFORMATION_GAIN")
        repo = EvidenceRepository(self.store)
        bundle = repo.public_view(self.handle.run_id)
        known = {
            DocumentChunk.model_validate_json(self.store.read(r)).chunk_id
            for r in bundle.retrieved_chunks
        }
        new = [c for c in result.chunks if c.chunk_id not in known]
        if not new:
            raise ProviderError("NO_INFORMATION_GAIN")
        refs = [
            self.store.put(
                self.handle.run_id,
                f"docs/{c.chunk_id}.json",
                c.model_dump_json().encode(),
                "public",
            )
            for c in new
        ]
        repo.save(
            self.handle.run_id,
            bundle.model_copy(update={"retrieved_chunks": [*bundle.retrieved_chunks, *refs]}),
        )

    def _source(self, action: AgentAction) -> None:
        if action.action_type != "inspect_source":
            raise ValueError("typed registry mismatch")
        self.budget = self.budget.model_copy(update={"source_reads": self.budget.source_reads + 1})
        # Source is already supplied in the public observation. Validate the registered range,
        # then record exactly which immutable lines were inspected; never open a model path.
        source = next(
            s
            for s in public_evidence(self.store, self.handle.run_id).sources
            if s.source_id == action.typed_arguments.source_id
        )
        args = action.typed_arguments
        lines = source.content.splitlines()[args.start_line - 1 : args.end_line]
        if not lines:
            raise ProviderError("NO_INFORMATION_GAIN")
        self.store.put(
            self.handle.run_id,
            f"source-reads/{action.action_id}.json",
            args.model_dump_json().encode(),
            "public",
        )

    def investigate(self, run_id: str) -> DiagnosisResult:
        if run_id != self.handle.run_id:
            raise ValueError("workspace belongs to another run")
        self.store.transition(run_id, "RUNNING", CurrentPhase.DIAGNOSING)
        try:
            while True:
                evidence = public_evidence(self.store, run_id)
                self.budget = self.budget.model_copy(
                    update={
                        "llm_calls": self.provider.gate.snapshot().llm_calls,
                        "remaining_seconds": self.provider.gate.remaining(),
                    }
                )
                if (
                    self.budget.agent_steps >= self.budget.max_agent_steps
                    or self.budget.remaining_seconds <= 0
                ):
                    raise ProviderError("AGENT_BUDGET_EXHAUSTED")
                proposed = self.provider.plan(evidence, self.budget)
                try:
                    action = ACTION_ADAPTER.validate_python(proposed.model_dump())
                except ValidationError:
                    raise ProviderError("ACTION_INVALID") from None
                decision = decide_action(
                    action, evidence, self.budget, CurrentPhase.DIAGNOSING, self.seen
                )
                self.store.put(
                    run_id,
                    f"actions/{self.budget.agent_steps}/decision.json",
                    decision.model_dump_json().encode(),
                    "public",
                )
                # Do not persist model budget snapshots or arbitrary rationale as controller state.
                if not decision.allowed:
                    raise ProviderError(decision.reason_codes[0])
                self.seen.add(action.action_type + action.typed_arguments.model_dump_json())
                self.budget = self.budget.model_copy(
                    update={"agent_steps": self.budget.agent_steps + 1}
                )
                if action.action_type == "declare_inconclusive":
                    return DiagnosisResult.inconclusive("MODEL_DECLARED_INCONCLUSIVE")
                if action.action_type == "finish_diagnosis":
                    result = self.provider.diagnose(evidence)
                    if not validate_diagnosis(result, evidence):
                        return DiagnosisResult.inconclusive("INVALID_DIAGNOSIS_EVIDENCE")
                    return result
                self.registry[action.action_type](action)
        except ProviderError as exc:
            return DiagnosisResult.inconclusive(exc.code)
        finally:
            snapshot = self.budget.model_copy(
                update={
                    "llm_calls": self.provider.gate.snapshot().llm_calls,
                    "remaining_seconds": self.provider.gate.remaining(),
                }
            )
            self.store.put(
                run_id, "agent/budget.json", snapshot.model_dump_json().encode(), "public"
            )
