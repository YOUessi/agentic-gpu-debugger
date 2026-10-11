"""Synthetic documents exercise ingestion, general ranking and the Agent evidence path."""

import hashlib
from pathlib import Path

import pytest

from gpu_agent.knowledge.ingest import extract_chunks, load_manifest
from gpu_agent.knowledge.models import KnowledgeVersionUnavailableError, make_chunk
from gpu_agent.knowledge.retrieve import KnowledgeIndex, tokenize

ROOT = Path(__file__).resolve().parents[2]
STAMP = "2026-09-29T00:00:00Z"
VERSION = "cuda=12.8.1;compute-sanitizer=2025.1.0.0"


def source(identity):
    return next(
        s for s in load_manifest(ROOT / "knowledge/sources.json").sources if s.source_id == identity
    )


def release_chunks():
    body = (
        b"<title>Release Notes</title><section id='updates-in-2025-1'>"
        b"<h2>Updates in 2025.1</h2><p>Synthetic release capability.</p></section>"
        b"<section id='other'><p>Unselected future capability.</p></section>"
    )
    return extract_chunks(
        source("sanitizer-release-evidence"), body, STAMP, verified_version="13.4"
    )


def test_evidence_opt_in_retains_actual_selected_body():
    chunks = release_chunks()
    assert len(chunks) == 1
    assert chunks[0].text == "Synthetic release capability."
    assert chunks[0].source_url.endswith("#updates-in-2025-1")
    assert chunks[0].limitation
    probe = source("sanitizer-version-probe")
    body = b"const options = {VERSION: '13.4'};"
    assert (
        extract_chunks(probe.model_copy(update={"retrievable_evidence": False}), body, STAMP) == []
    )
    result = extract_chunks(probe, body, STAMP)
    assert result[0].text == body.decode()
    assert result[0].source_url == probe.canonical_url
    with pytest.raises(KnowledgeVersionUnavailableError):
        extract_chunks(probe, b"VERSION: '99.1'", STAMP)


def test_attached_comment_only_and_hash_still_checked():
    body = b"/* unrelated */\nint x;\n/* Explain the operation. */\n__global__ void vectorAdd() {}"
    sample = source("cuda-samples-vector-add-v12.8").model_copy(
        update={"body_sha256": hashlib.sha256(body).hexdigest()}
    )
    result = extract_chunks(sample, body, STAMP)[0]
    assert "Explain the operation" in result.text
    assert "unrelated" not in result.text
    old = extract_chunks(sample.model_copy(update={"include_attached_comment": False}), body, STAMP)
    assert "Explain" not in old[0].text


@pytest.mark.parametrize(
    "query,term",
    [("sorts", "sort"), ("buffers", "buffer"), ("reads", "read"), ("segments", "segment")],
)
def test_general_inflection_companions_preserve_original(query, term):
    assert {query, term} <= set(tokenize(query, "cuda-lex-v6"))
    assert term not in tokenize(query, "cuda-lex-v5")


def test_source_diversity_and_backfill_do_not_invent_matches(tmp_path):
    chunks = [
        make_chunk(
            source_id=src,
            document_title="Fixture",
            document_version="1",
            section_title="Fixture",
            source_url="https://example.org/" + src,
            retrieved_at=STAMP,
            text="needle " * (10 if src == "large" else 1) + str(i),
            block_ordinal=i,
            compatibility={"cuda": ">=12,<13"},
        )
        for i, src in enumerate(["large"] * 8 + ["small", "other"])
    ]
    index = KnowledgeIndex(chunks, tokenizer_version="cuda-lex-v6")
    result = index.retrieve("needle", VERSION, 5)
    assert {c.source_id for c in result.chunks} == {"large", "small", "other"}
    assert len(result.chunks) == 5  # remaining slots backfilled, no lost context
    assert not index.retrieve("unicorn", VERSION).chunks
    with pytest.raises(KnowledgeVersionUnavailableError):
        index.retrieve("needle", "cuda=99;compute-sanitizer=99")
    path = tmp_path / "index.json"
    index.save(path)
    assert KnowledgeIndex.load(path).retrieve("needle", VERSION) == result


def test_agent_retrieves_persists_and_cites_new_release_content(oob_service):
    from gpu_agent.agent.models import FinishAction, MemcheckAction, RetrieveDocsAction

    service, provider, input_path = oob_service
    chunks = release_chunks()
    service.knowledge = KnowledgeIndex(chunks, tokenizer_version="cuda-lex-v6")
    service.knowledge_version = VERSION
    provider.actions = [
        MemcheckAction(),
        RetrieveDocsAction(typed_arguments={"query": "release capability", "k": 3}),
        FinishAction(),
    ]
    run = service.diagnose(input_path)
    diagnosis = service.diagnosis(run.id)
    assert diagnosis.diagnostic_outcome == "DIAGNOSED"
    assert chunks[0].chunk_id in {
        citation for claim in diagnosis.documentation_evidence for citation in claim.citation_ids
    }
    stored = next(ref for ref in run.artifact_refs if ref.name == f"docs/{chunks[0].chunk_id}.json")
    assert chunks[0].text.encode() in service.store.read(stored)
