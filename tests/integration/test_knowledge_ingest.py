"""Live means actual statically allowlisted official HTTP responses, never fixtures."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.release
def test_live_official_oob_retrieval(request, tmp_path):
    from gpu_agent.knowledge.ingest import ingest, load_manifest
    from gpu_agent.knowledge.models import KnowledgeError, validate_citations
    from gpu_agent.knowledge.retrieve import KnowledgeIndex

    manifest = load_manifest(ROOT / "knowledge/sources.json")
    try:
        index, receipts = ingest(manifest)
    except KnowledgeError as exc:
        if request.config.getoption("--require-live"):
            pytest.fail(f"Required official-source ingest failed: {exc}")
        pytest.skip(f"Official-source ingest unavailable: {exc}")
    assert len(receipts) == len(manifest.sources)
    for receipt in receipts:
        print("OFFICIAL_FETCH", receipt.model_dump_json())
        assert receipt.byte_count > 0 and len(receipt.body_sha256) == 64
    result = index.retrieve(
        "global memory out of bounds OOB", "cuda=12.8.1;compute-sanitizer=2025.1.0.0", k=3
    )
    approved = {
        "c99fe0f86d47e42df35628766432962544d7935a45e4edeec9ef9e781dbbc004",
        "5e9c378e0c32929013661506ff81ed4c237dcdd49e06ac4bca24ce6986928bb9",
        "c36ea100900980d1c74a72368264d4b61f7acf6ee0f642b02d971a35105ad175",
    }
    relevant = [c for c in result.chunks if c.content_hash in approved]
    assert relevant
    best = relevant[0]
    # Existence is separate from these narrow content/approved-human-review checks.
    validate_citations([best.chunk_id], result)
    assert best.document_version == "13.4" and best.limitation
    assert best.source_url in {
        "https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#padding",
        "https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#what-is-memcheck",
    }
    text = best.text.lower()
    assert "memcheck" in text and "global memory" in text
    assert "out of bounds" in text or "out-of-bounds" in text
    # Round-trip the local index; query performs no HTTP calls.
    path = tmp_path / "index.json"
    index.save(path)
    assert (
        KnowledgeIndex.load(path).retrieve(
            result.query, "cuda=12.8.1;compute-sanitizer=2025.1.0.0", 3
        )
        == result
    )
