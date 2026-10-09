"""Build a new versioned official corpus; never overwrite an existing output directory."""

import argparse
import json
from pathlib import Path

from gpu_agent.knowledge.ingest import ingest, load_manifest
from gpu_agent.knowledge.semantic import compare_retrieval_methods, load_evaluation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; keep historical corpora immutable")
    manifest_path = args.repository / "knowledge/sources.json"
    manifest = load_manifest(manifest_path)
    index, receipts = ingest(manifest)
    report = compare_retrieval_methods(
        index, load_evaluation_suite(args.repository / "knowledge/retrieval-eval.json")
    )
    args.output.mkdir(parents=True, exist_ok=False)
    index.save(args.output / "index.json")
    (args.output / "sources.json").write_bytes(manifest_path.read_bytes())
    (args.output / "receipts.json").write_text(
        json.dumps([r.model_dump() for r in receipts], indent=2), encoding="utf-8"
    )
    (args.output / "retrieval-report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    print(json.dumps(dict(corpus_hash=index.corpus_hash, chunks=len(index.chunks))))
    for method in report.methods:
        print(method.method, f"{method.hits}/{method.queries}")
        print("misses:", [o.query_id for o in method.outcomes if not o.hit])


if __name__ == "__main__":
    main()
