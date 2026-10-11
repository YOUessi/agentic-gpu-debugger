"""Explain development retrieval misses without changing ranking or the denominator."""

import argparse
import json
from collections import Counter
from pathlib import Path

from gpu_agent.knowledge.models import KnowledgeVersionUnavailableError
from gpu_agent.knowledge.retrieve import KnowledgeIndex
from gpu_agent.knowledge.semantic import RetrievalEvaluationSuite, load_evaluation_suite


def retrieval_coverage(index: KnowledgeIndex, suite: RetrievalEvaluationSuite) -> dict[str, object]:
    outcomes = []
    for query in suite.queries:
        expected = {
            i
            for i, chunk in enumerate(index.chunks)
            if (
                chunk.chunk_id in query.relevant_chunk_ids
                if query.relevant_chunk_ids
                else chunk.source_id in query.relevant_source_ids
            )
        }
        try:
            eligible = index._eligible(query.version)
            result = index.retrieve(query.query, query.version, suite.k)
            retrieved = {c.chunk_id for c in result.chunks}
        except KnowledgeVersionUnavailableError:
            eligible, retrieved = set(), set()
        hit = any(index.chunks[i].chunk_id in retrieved for i in expected)
        status = (
            "HIT"
            if hit
            else "EXPECTED_CONTENT_ABSENT"
            if not expected
            else "EXPECTED_CONTENT_VERSION_EXCLUDED"
            if not expected.intersection(eligible)
            else "RANKING_MISS"
        )
        outcomes.append({"query_id": query.query_id, "status": status})
    return {
        "corpus_hash": index.corpus_hash,
        "queries": len(outcomes),
        "k": suite.k,
        "method": "lexical",
        "counts": dict(Counter(row["status"] for row in outcomes)),
        "outcomes": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            retrieval_coverage(KnowledgeIndex.load(args.index), load_evaluation_suite(args.suite)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
