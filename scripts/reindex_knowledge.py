"""Reindex an explicitly pinned, previously verified cache without fetching new text.

This does not ingest newly added sources. It preserves every existing chunk and
records both index identities. Evaluation labels are read only by this report,
never by retrieval or the model. Output directory must not already exist.
"""

import argparse
import hashlib
import json
from pathlib import Path

from gpu_agent.knowledge.ingest import load_manifest
from gpu_agent.knowledge.retrieve import KnowledgeIndex
from gpu_agent.knowledge.semantic import compare_retrieval_methods, load_evaluation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--expected-corpus-hash", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-version", choices=[f"cuda-lex-v{i}" for i in range(1, 7)])
    args = parser.parse_args()
    prior = KnowledgeIndex.load(args.cache)
    if prior.corpus_hash != args.expected_corpus_hash:
        raise ValueError("cache identity differs from reviewed corpus")
    manifest = load_manifest(args.repository / "knowledge/sources.json")
    if not {c.source_id for c in prior.chunks} <= {s.source_id for s in manifest.sources}:
        raise ValueError("cache contains unlisted sources")
    current = KnowledgeIndex(
        prior.chunks,
        corpus_version=manifest.corpus_version,
        normalizer_version=manifest.normalizer_version,
        tokenizer_version=args.tokenizer_version or manifest.tokenizer_version,
    )
    suite = load_evaluation_suite(args.repository / "knowledge/retrieval-eval.json")
    report = compare_retrieval_methods(current, suite)
    args.output.mkdir(parents=True, exist_ok=False)
    current.save(args.output / "index.json")
    (args.output / "retrieval-report.json").write_text(report.model_dump_json(indent=2))
    receipt = dict(
        prior_corpus_hash=prior.corpus_hash,
        corpus_hash=current.corpus_hash,
        input_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        corpus_version=current.corpus_version,
        tokenizer_version=current.tokenizer_version,
        chunks=len(current.chunks),
        operation="reindex-existing-verified-chunks-no-new-ingestion",
    )
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    for method in report.methods:
        selected = [
            o
            for o in method.outcomes
            if o.query_id.startswith(("dev-sync-", "dev-race-", "dev-init-"))
        ]
        print(
            method.method,
            f"{method.hits}/{method.queries}",
            "new content",
            f"{sum(o.hit for o in selected)}/{len(selected)}",
        )


if __name__ == "__main__":
    main()
