"""Opt-in LangChain Core (LCEL + BaseRetriever) component adapters.

This module is not a second model transport. It deliberately retains the project's
existing killable provider worker, paid-call ledger, strict output validation,
private-evaluator boundary and controller-owned tools. LangChain runnables must
never introduce their own API retries, networked retrievers or callbacks.

Use only on explicitly authorized public development runs, not frozen evaluation.
"""

import os
from collections.abc import Callable

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ConfigDict, Field

from gpu_agent.agent.models import (
    ACTION_ADAPTER,
    AgentAction,
    AgentBudget,
    DiagnosisResult,
    PlannerState,
    PublicEvidence,
    PublicSource,
)
from gpu_agent.agent.provider import LLMProvider
from gpu_agent.knowledge.models import (
    KnowledgeIntegrityError,
    RetrievalResult,
)
from gpu_agent.knowledge.retrieve import KnowledgeIndex


def component_layer() -> str:
    """Fail closed if a typo would silently bypass the requested execution path."""
    layer = os.environ.get("GPU_AGENT_COMPONENT_LAYER", "native")
    if layer not in {"native", "langchain"}:
        raise ValueError("GPU_AGENT_COMPONENT_LAYER must be native or langchain")
    return layer


def require_no_external_tracing() -> None:
    """Never allow an ambient LangSmith trace to transmit source/evidence data."""
    for key in ("LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
        if os.environ.get(key, "").strip().lower() in {"true", "1", "yes", "on"}:
            raise ValueError("LangChain external tracing is forbidden for GPU Agent evidence")


class PinnedNvidiaRetriever(BaseRetriever):
    """LangChain retriever backed exclusively by the frozen, versioned local index."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: KnowledgeIndex = Field(exclude=True)
    version: str
    k: int = Field(default=5, ge=1, le=5)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        result = self.index.retrieve(query, self.version, k=self.k)
        return [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "content_hash": chunk.content_hash,
                    "source_id": chunk.source_id,
                    "document_version": chunk.document_version,
                    "source_url": chunk.source_url,
                    "corpus_hash": result.corpus_hash,
                },
            )
            for chunk in result.chunks
        ]


def retrieve_pinned(index: KnowledgeIndex, query: str, version: str, k: int) -> RetrievalResult:
    """Use the LangChain Retriever API without weakening corpus/citation validation."""
    require_no_external_tracing()
    documents = PinnedNvidiaRetriever(index=index, version=version, k=k).invoke(query)
    # The original domain retriever defines the canonical, version-gated answer.
    # Never trust arbitrary LangChain Document metadata as a citation authority.
    canonical = index.retrieve(query, version, k=k)
    actual = [
        (doc.metadata.get("chunk_id"), doc.metadata.get("content_hash"), doc.page_content)
        for doc in documents
    ]
    expected = [(chunk.chunk_id, chunk.content_hash, chunk.text) for chunk in canonical.chunks]
    if actual != expected:
        raise KnowledgeIntegrityError("LangChain retrieval diverged from the frozen CUDA index")
    return canonical


class LangChainModelComponents:
    """LCEL adapters over the *existing* typed, budgeted model provider.

    .invoke() actually uses LangChain RunnableLambda, but no second ChatOpenAI
    client is constructed. The original provider still validates response schemas,
    records physical usage, and owns cancellation/uncertain-delivery behavior.
    """

    def __init__(self, provider: LLMProvider) -> None:
        require_no_external_tracing()
        self.provider = provider

    def plan(
        self,
        evidence: PublicEvidence,
        budget: AgentBudget,
        feedback: list[str] | None,
        state: PlannerState | None,
    ) -> AgentAction:
        require_no_external_tracing()

        def call(_: PublicEvidence) -> AgentAction:
            return self.provider.plan(evidence, budget, feedback, state=state)

        runnable = RunnableLambda(call) | RunnableLambda(
            lambda action: ACTION_ADAPTER.validate_python(action.model_dump())
        )
        return runnable.invoke(evidence)

    def diagnose(
        self,
        evidence: PublicEvidence,
        diagnose_fn: Callable[[PublicEvidence], DiagnosisResult],
    ) -> DiagnosisResult:
        require_no_external_tracing()
        runnable = RunnableLambda(diagnose_fn) | RunnableLambda(
            lambda result: DiagnosisResult.model_validate(result.model_dump(mode="json"))
        )
        return runnable.invoke(evidence)

    def propose_patch(
        self,
        source: PublicSource,
        diagnosis: DiagnosisResult,
        experience_hints: list[dict[str, str]] | None = None,
    ) -> str:
        """LCEL composition retaining the provider's schema, safety and patch gates."""
        require_no_external_tracing()

        def call(_: PublicSource) -> str:
            return (
                self.provider.propose_patch(source, diagnosis, experience_hints=experience_hints)
                if experience_hints
                else self.provider.propose_patch(source, diagnosis)
            )

        runnable = RunnableLambda(call) | RunnableLambda(_require_patch_string)
        return runnable.invoke(source)

    def revise_patch(
        self,
        source: PublicSource,
        diagnosis: DiagnosisResult,
        feedback: dict[str, object],
    ) -> str:
        """Reuse the provider's bounded public-feedback revision and diff checks."""
        require_no_external_tracing()

        def call(_: PublicSource) -> str:
            return self.provider.revise_patch(source, diagnosis, feedback)

        runnable: Runnable[PublicSource, str] = RunnableLambda(call) | RunnableLambda(
            _require_patch_string
        )
        return runnable.invoke(source)


def _require_patch_string(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("patch result must be a typed diff string")
    return value
