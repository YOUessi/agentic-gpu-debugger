"""LangChain Core adapters: parity, source provenance, isolation and fail-closed tests."""

import json

import pytest


@pytest.mark.parametrize("layer", ["native", "langchain"])
def test_dev_agent_has_identical_tool_trace_and_physical_budget(
    oob_service, monkeypatch, layer
):
    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_ORCHESTRATOR", "langgraph")
    monkeypatch.setenv("GPU_AGENT_COMPONENT_LAYER", layer)

    run = service.diagnose(source, mode="E")
    diagnosis = service.diagnosis(run.id)
    assert diagnosis.diagnostic_outcome == "DIAGNOSED"
    assert provider.kinds == ["plan", "plan", "plan", "diagnose", "patch"]
    assert len(service.candidates(run.id)) == 1

    refs = {r.name: r for r in run.artifact_refs}
    assert json.loads(service.store.read(refs["agent/final-budget.json"]))["llm_calls"] == 5
    assert json.loads(service.store.read(refs["agent/usage-summary.json"]))["physical_calls"] == 5
    assert {f"actions/{n}/decision.json" for n in range(3)} <= set(refs)
    assert "agent/component-layer.json" in refs if layer == "langchain" else (
        "agent/component-layer.json" not in refs
    )
    assert "secret-canary" not in json.dumps(provider.inputs)


def test_langchain_retriever_uses_frozen_documents_and_canonical_ids(oob_service):
    from gpu_agent.agent.langchain_components import PinnedNvidiaRetriever, retrieve_pinned

    service, _, _ = oob_service
    index = service.knowledge
    assert index is not None
    query = "global memory bounds"
    version = service.knowledge_version
    canonical = index.retrieve(query, version, k=3)
    via_lc = retrieve_pinned(index, query, version, k=3)
    assert via_lc == canonical
    docs = PinnedNvidiaRetriever(index=index, version=version, k=3).invoke(query)
    assert [d.metadata["chunk_id"] for d in docs] == [c.chunk_id for c in canonical.chunks]
    assert all(d.metadata["corpus_hash"] == index.corpus_hash for d in docs)
    assert all(d.page_content == c.text for d, c in zip(docs, canonical.chunks, strict=True))


def test_langchain_retriever_never_guesses_toolchain_version(oob_service):
    from gpu_agent.agent.langchain_components import retrieve_pinned
    from gpu_agent.knowledge.models import KnowledgeVersionUnavailableError

    service, _, _ = oob_service
    assert service.knowledge is not None
    with pytest.raises(KnowledgeVersionUnavailableError):
        retrieve_pinned(service.knowledge, "memory", "latest", 3)


def test_langchain_tracing_is_rejected_before_a_model_call(oob_service, monkeypatch):
    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_COMPONENT_LAYER", "langchain")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with pytest.raises(ValueError, match="external tracing"):
        service.diagnose(source)
    assert provider.kinds == []


def test_unknown_component_setting_fails_closed(oob_service, monkeypatch):
    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_COMPONENT_LAYER", "not-a-framework")
    with pytest.raises(ValueError, match="GPU_AGENT_COMPONENT_LAYER"):
        service.diagnose(source)
    assert provider.kinds == []


def test_langchain_does_not_run_under_legacy_loop(oob_service, monkeypatch):
    service, provider, source = oob_service
    monkeypatch.setenv("GPU_AGENT_ORCHESTRATOR", "legacy")
    monkeypatch.setenv("GPU_AGENT_COMPONENT_LAYER", "langchain")
    with pytest.raises(ValueError, match="require LangGraph"):
        service.diagnose(source)
    assert provider.kinds == []


def test_langchain_retrieval_citation_mismatch_fails_closed(
    oob_service, monkeypatch
):
    from langchain_core.documents import Document

    from gpu_agent.agent.langchain_components import retrieve_pinned
    from gpu_agent.knowledge.models import KnowledgeIntegrityError

    service, _, _ = oob_service
    assert service.knowledge is not None
    index = service.knowledge

    def forged(self, query):
        return [Document(page_content="untrusted text", metadata={"chunk_id": "fabricated"})]

    monkeypatch.setattr(
        "gpu_agent.agent.langchain_components.PinnedNvidiaRetriever._get_relevant_documents",
        forged,
    )
    with pytest.raises(KnowledgeIntegrityError):
        retrieve_pinned(index, "memory bounds", service.knowledge_version, k=3)


def test_lcel_revision_reuses_provider_feedback_contract(oob_service):
    from gpu_agent.agent.langchain_components import LangChainModelComponents

    service, provider, _ = oob_service
    evidence = service.knowledge
    assert evidence is not None
    source = None
    from gpu_agent.agent.models import DiagnosisResult, PublicSource

    source = PublicSource(source_id="a" * 32, content="int index;\n")
    result = DiagnosisResult.inconclusive("TEST")
    hints = {"round": 1, "public_self_check": {"status": "FAILED"}}
    adapted = LangChainModelComponents(provider)
    assert adapted.revise_patch(source, result, hints) == provider.diff
    assert provider.kinds == ["patch"]
    assert provider.inputs[-1]["public_repair_feedback"] == hints
