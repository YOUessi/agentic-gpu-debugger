from gpu_agent.knowledge.coverage import retrieval_coverage
from gpu_agent.knowledge.models import make_chunk
from gpu_agent.knowledge.retrieve import KnowledgeIndex
from gpu_agent.knowledge.semantic import RetrievalEvaluationSuite, RetrievalJudgment


def test_missing_version_excluded_and_ranking_are_distinguished():
    chunk = make_chunk(
        source_id="fixture",
        document_title="Fixture",
        document_version="12.8",
        section_title="Alpha",
        source_url="https://example.org/guide",
        retrieved_at="2026-09-28T00:00:00Z",
        text="alpha beta",
        block_ordinal=0,
        compatibility={"cuda": "==12.8"},
    )
    index = KnowledgeIndex([chunk])
    queries = []
    for i in range(20):
        queries.append(
            RetrievalJudgment(
                query_id=str(i),
                query="alpha" if i % 4 != 3 else "unfindable",
                version=f"cuda={'12.7' if i % 4 == 2 else '12.8'};compute-sanitizer=2025.1",
                relevant_source_ids=["absent" if i % 4 == 1 else "fixture"],
                rationale="Synthetic fixture",
            )
        )
    suite = RetrievalEvaluationSuite(schema_version=1, split="development", k=5, queries=queries)
    result = retrieval_coverage(index, suite)
    assert result["queries"] == 20
    assert result["counts"] == {
        "HIT": 5,
        "EXPECTED_CONTENT_ABSENT": 5,
        "EXPECTED_CONTENT_VERSION_EXCLUDED": 5,
        "RANKING_MISS": 5,
    }
