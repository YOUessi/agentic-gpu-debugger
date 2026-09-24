"""Candidate-anchor review is offline-testable and mirrors ingest's block hashing."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _source():
    from gpu_agent.knowledge.ingest import load_manifest

    return next(
        s
        for s in load_manifest(ROOT / "knowledge/sources.json").sources
        if s.source_id == "sanitizer-approved-memcheck"
    )


def test_review_lists_blocks_hashes_and_missing_anchors():
    from gpu_agent.knowledge.models import normalize_text, sha256
    from gpu_agent.knowledge.review import review_blocks

    source = _source()
    html = (
        f"<html><head><title>{source.document_title}</title></head><body>"
        '<section id="what-is-racecheck"><h2>What is Racecheck?</h2>'
        "<p>Racecheck reports shared memory hazards.</p>"
        "<ul><li>RAW hazard</li></ul></section></body></html>"
    ).encode()
    blocks, missing = review_blocks(source, html, ["what-is-racecheck", "what-is-synccheck"])
    assert missing == ["what-is-synccheck"]
    paragraph = next(b for b in blocks if b.tag == "p")
    assert paragraph.sha256 == sha256(normalize_text("Racecheck reports shared memory hazards."))
    assert paragraph.pinnable
    # The sanitizer manual admits only reviewed <p> paragraphs.
    assert [b.pinnable for b in blocks if b.tag == "li"] == [False]


def test_candidates_reference_manifest_sources_and_leave_the_corpus_unchanged():
    import json

    from gpu_agent.knowledge.ingest import load_manifest
    from gpu_agent.knowledge.review import Candidates

    candidates = Candidates.model_validate_json(
        (ROOT / "docs/knowledge-expansion/candidates.json").read_text()
    )
    manifest = load_manifest(ROOT / "knowledge/sources.json")
    ids = {s.source_id for s in manifest.sources}
    assert {c.source_id for c in candidates.anchors} <= ids
    assert all(set(q.relevant_source_ids) <= ids for q in candidates.retrieval_queries)
    # Candidate sections and relevance queries have now been explicitly promoted.
    assert candidates.target_corpus_version == manifest.corpus_version
    sources = {source.source_id: source for source in manifest.sources}
    assert all(
        set(c.anchors) <= set(sources[c.source_id].include_anchors) for c in candidates.anchors
    )
    tracked = json.loads((ROOT / "knowledge/retrieval-eval.json").read_text())
    assert {q.query_id for q in candidates.retrieval_queries} <= {
        q["query_id"] for q in tracked["queries"]
    }
