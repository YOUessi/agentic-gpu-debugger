"""Operator review of candidate documentation anchors before they enter the corpus.

Nothing here changes the knowledge index. It fetches an already-approved manifest source
through the same fetch policy and HTML selection as ingest, and prints every semantic block
under the candidate anchors with its content hash, so a human can read the text, judge
version applicability, and pin hashes in knowledge/sources.json deliberately.

Usage (on a machine allowed to reach docs.nvidia.com):
    python -m gpu_agent.knowledge.review knowledge/sources.json \
        docs/knowledge-expansion/candidates.json > review.json
"""

import argparse
import json
import sys
from pathlib import Path

from bs4 import Tag
from pydantic import Field

from gpu_agent.knowledge.ingest import (
    Source,
    _heading,
    _section,
    _soup,
    fetch,
    load_manifest,
)
from gpu_agent.knowledge.models import (
    KnowledgeSourceError,
    StrictModel,
    normalize_text,
    sha256,
)


class CandidateAnchors(StrictModel):
    source_id: str
    anchors: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class CandidateQuery(StrictModel):
    query_id: str
    query: str
    relevant_source_ids: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class Candidates(StrictModel):
    schema_version: int = Field(ge=1, le=1)
    target_corpus_version: str
    anchors: list[CandidateAnchors]
    retrieval_queries: list[CandidateQuery] = Field(default_factory=list)


class ReviewBlock(StrictModel):
    source_id: str
    anchor: str
    tag: str
    heading: str
    sha256: str
    chars: int
    pinnable: bool  # approved_paragraphs sources can pin only <p> blocks
    text: str


def review_blocks(
    source: Source, body: bytes, anchors: list[str]
) -> tuple[list[ReviewBlock], list[str]]:
    """Every block ingest would consider under `anchors`, with the hash ingest computes.

    Returns the blocks and the anchors absent from the page (candidate ids are proposals
    until a fetch confirms them).
    """
    soup = _soup(source, body.decode("utf-8", errors="strict"))
    blocks: list[ReviewBlock] = []
    missing: list[str] = []
    for anchor in anchors:
        try:
            section = _section(soup, anchor)
        except KnowledgeSourceError:
            missing.append(anchor)
            continue
        for tag in section.find_all(["p", "pre", "li", "tr"]):
            if not isinstance(tag, Tag):
                continue
            if any(p.name in {"pre", "li", "tr"} for p in tag.parents if p is not section):
                continue
            is_code = tag.name == "pre"
            content = normalize_text(
                tag.get_text("" if is_code else " ", strip=not is_code), code=is_code
            )
            if not content:
                continue
            heading, _ = _heading(tag, source.document_title)
            blocks.append(
                ReviewBlock(
                    source_id=source.source_id,
                    anchor=anchor,
                    tag=tag.name,
                    heading=heading,
                    sha256=sha256(content),
                    chars=len(content),
                    pinnable=source.chunk_strategy != "approved_paragraphs" or tag.name == "p",
                    text=content,
                )
            )
    return blocks, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("candidates", type=Path)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    candidates = Candidates.model_validate_json(args.candidates.read_text(encoding="utf-8"))
    sources = {source.source_id: source for source in manifest.sources}
    report: list[dict[str, object]] = []
    for candidate in candidates.anchors:
        source = sources[candidate.source_id]
        fetched = fetch(source, manifest.fetch_policy)
        blocks, missing = review_blocks(source, fetched.body, candidate.anchors)
        report.append(
            {
                "source_id": source.source_id,
                "effective_url": fetched.effective_url,
                "retrieved_at": fetched.retrieved_at,
                "body_sha256": sha256(fetched.body.decode("utf-8")),
                "missing_anchors": missing,
                "blocks": [block.model_dump() for block in blocks],
                "oversized_blocks": [b.sha256 for b in blocks if b.chars > 1400],
            }
        )
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
