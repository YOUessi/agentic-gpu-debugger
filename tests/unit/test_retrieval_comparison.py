"""Development-only retrieval comparison with self-authored fixture text."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERSION = "cuda=12.8.1;compute-sanitizer=2025.1.0.0"
URL = "https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html"


def _chunk(source_id: str, title: str, text: str, ordinal: int):
    from gpu_agent.knowledge.models import make_chunk

    return make_chunk(
        source_id=source_id,
        document_title="Self-authored retrieval comparison fixture",
        document_version="12.8.1",
        section_title=title,
        source_url=f"{URL}#{source_id}",
        retrieved_at="2026-09-20T00:00:00+00:00",
        text=text,
        block_ordinal=ordinal,
        compatibility={"cuda": ">=12.8,<12.9", "compute_sanitizer": "==2025.1.0.0"},
    )


@pytest.fixture
def comparison_index():
    from gpu_agent.knowledge.retrieve import KnowledgeIndex

    return KnowledgeIndex(
        [
            _chunk(
                "race-doc",
                "Shared-memory hazard",
                "A write-after-write hazard can occur between unsynchronised CUDA threads.",
                0,
            ),
            _chunk(
                "memory-doc",
                "Device access",
                "An invalid device access can result from an out-of-range global memory index.",
                1,
            ),
            _chunk(
                "init-doc",
                "Undefined values",
                "Reading bytes before initialization propagates an undefined value.",
                2,
            ),
            _chunk(
                "sync-doc",
                "Barrier discipline",
                "A divergent __syncthreads() barrier creates a synchronization error.",
                3,
            ),
        ]
    )


def test_semantic_is_vector_similarity_not_a_bm25_alias(comparison_index):
    from gpu_agent.knowledge.semantic import SemanticVectorizer, cosine_similarity

    lexical = comparison_index.retrieve("concurrent ordering bug", VERSION, method="lexical")
    semantic = comparison_index.retrieve("concurrent ordering bug", VERSION, method="semantic")

    assert lexical.chunks == []
    assert semantic.chunks[0].source_id == "race-doc"
    query = SemanticVectorizer().encode("concurrent ordering bug")
    document = SemanticVectorizer().encode("write-after-write hazard")
    assert query
    assert cosine_similarity(query, document) > 0


@pytest.mark.parametrize("method", ["lexical", "semantic", "hybrid"])
def test_all_methods_are_repeatable_and_version_filtered(comparison_index, method):
    from gpu_agent.knowledge.models import KnowledgeVersionUnavailableError

    first = comparison_index.retrieve("illegal address from OOB", VERSION, k=3, method=method)
    second = comparison_index.retrieve("illegal address from OOB", VERSION, k=3, method=method)
    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]
    assert first.chunks[0].source_id == "memory-doc"
    with pytest.raises(KnowledgeVersionUnavailableError):
        comparison_index.retrieve(
            "illegal address", "cuda=11.0;compute-sanitizer=2025.1.0.0", method=method
        )


def test_hybrid_uses_stable_reciprocal_rank_fusion(comparison_index):
    result = comparison_index.retrieve(
        "undefined uninitialized memory value", VERSION, k=4, method="hybrid"
    )
    assert result.chunks[0].source_id == "init-doc"
    assert len({chunk.chunk_id for chunk in result.chunks}) == len(result.chunks)


def test_development_judgments_disclose_mixed_review_and_are_public_only():
    from gpu_agent.knowledge.semantic import load_evaluation_suite

    suite = load_evaluation_suite(ROOT / "knowledge/retrieval-eval.json")
    assert suite.split == "development"
    assert suite.annotation_method == "mixed_development"
    assert len(suite.queries) >= 20
    assert len({query.query_id for query in suite.queries}) == len(suite.queries)
    assert all(query.relevant_source_ids for query in suite.queries)
    assert all(
        "private" not in source.lower()
        for query in suite.queries
        for source in query.relevant_source_ids
    )


def test_chunk_judgments_do_not_accept_an_unrelated_paragraph_from_same_source(comparison_index):
    from gpu_agent.knowledge.semantic import (
        RetrievalEvaluationSuite,
        RetrievalJudgment,
        compare_retrieval_methods,
    )

    judgments = [
        RetrievalJudgment(
            query_id=f"content-{i}",
            query="write-after-write hazard",
            version=VERSION,
            relevant_source_ids=["race-doc"],
            relevant_chunk_ids=["different-paragraph"],
            rationale="Source identity alone cannot prove paragraph relevance.",
        )
        for i in range(20)
    ]
    suite = RetrievalEvaluationSuite(schema_version=1, split="development", k=4, queries=judgments)
    report = compare_retrieval_methods(comparison_index, suite)
    assert all(method.hits == 0 for method in report.methods)


def test_comparison_reports_hit_at_k_latency_and_dev_only_selection(comparison_index, tmp_path):
    from gpu_agent.knowledge.semantic import (
        RetrievalEvaluationSuite,
        RetrievalJudgment,
        compare_retrieval_methods,
        write_comparison_report,
    )

    labels = [
        ("concurrent ordering bug", "race-doc"),
        ("illegal address caused by OOB", "memory-doc"),
        ("value was never initialized", "init-doc"),
        ("threads disagree at a barrier", "sync-doc"),
    ]
    queries = [
        RetrievalJudgment(
            query_id=f"dev-{index:02d}",
            query=query,
            version=VERSION,
            relevant_source_ids=[source_id],
            rationale="Self-authored development label for deterministic unit testing.",
        )
        for index, (query, source_id) in enumerate(labels * 5, start=1)
    ]
    suite = RetrievalEvaluationSuite(schema_version=1, split="development", k=2, queries=queries)
    report = compare_retrieval_methods(comparison_index, suite)

    assert report.selection_basis == "development_hit_at_k_only"
    assert report.lexical_version == "cuda-lex-v1"
    assert report.semantic_version == "cuda-concept-hash-v1-d2048"
    assert report.hybrid_version == "rrf-k60-v1"
    assert len(report.evaluation_hash) == 64
    assert report.default_method in {"lexical", "semantic", "hybrid"}
    assert {result.method for result in report.methods} == {"lexical", "semantic", "hybrid"}
    assert all(result.queries == 20 for result in report.methods)
    assert all(0 <= result.hit_at_k <= 1 for result in report.methods)
    assert all(result.total_latency_ms >= 0 for result in report.methods)
    assert all(len(result.outcomes) == 20 for result in report.methods)
    assert (
        report.default_method
        == max(
            report.methods,
            key=lambda item: (item.hits, {"lexical": 0, "semantic": 1, "hybrid": 2}[item.method]),
        ).method
    )
    report_path = tmp_path / "retrieval-comparison.json"
    write_comparison_report(report, report_path)
    persisted = report.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert persisted == report


def test_evaluation_rejects_non_development_split():
    from pydantic import ValidationError

    from gpu_agent.knowledge.semantic import RetrievalEvaluationSuite

    with pytest.raises(ValidationError):
        RetrievalEvaluationSuite.model_validate(
            {"schema_version": 1, "split": "private", "k": 5, "queries": []}
        )
