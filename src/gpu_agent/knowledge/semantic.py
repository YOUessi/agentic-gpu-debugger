"""Deterministic local vector retrieval and development-label comparison.

This module deliberately has no model, network, or optional dependency.  It maps
self-contained CUDA concepts and token n-grams into a stable sparse vector and
uses cosine similarity.  It is therefore a real vector-space retriever, while
remaining reproducible and auditable rather than pretending BM25 scores are
"embeddings".
"""

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from gpu_agent.knowledge.models import RetrievalResult, StrictModel

RetrievalMethod = Literal["lexical", "semantic", "hybrid"]
SparseVector = dict[int, float]
SEMANTIC_ENCODER_VERSION: Literal["cuda-concept-hash-v1-d2048"] = "cuda-concept-hash-v1-d2048"
HYBRID_FUSION_VERSION: Literal["rrf-k60-v1"] = "rrf-k60-v1"

_WORD = re.compile(r"[a-z0-9_]+(?:[.-][a-z0-9_]+)*")
_PHRASE_CONCEPTS: tuple[tuple[str, str], ...] = (
    (r"\b(?:out[- ]of[- ]bounds|out[- ]of[- ]range|oob)\b", "out_of_bounds"),
    (r"\b(?:illegal address|invalid device access|bad device pointer)\b", "device_memory_fault"),
    (r"\b(?:write[- ]after[- ]write|read[- ]after[- ]write|write[- ]after[- ]read)\b", "data_race"),
    (r"\b(?:race condition|concurrent ordering bug|thread hazard)\b", "data_race"),
    (
        r"\b(?:uninitialized|uninitialised|undefined value|never initialized)\b",
        "uninitialized_value",
    ),
    (r"\b(?:divergent barrier|barrier mismatch|threads disagree at a barrier)\b", "sync_error"),
    (r"\b(?:global memory|device memory)\b", "device_memory"),
    (r"\b(?:vector addition|vector add)\b", "vector_add"),
)
_TOKEN_CONCEPTS: dict[str, str] = {
    "concurrent": "data_race",
    "hazard": "data_race",
    "ordering": "data_race",
    "racecheck": "data_race",
    "memcheck": "device_memory_fault",
    "initcheck": "uninitialized_value",
    "synccheck": "sync_error",
    "barrier": "sync_error",
    "synchronization": "sync_error",
    "synchronisation": "sync_error",
    "undefined": "uninitialized_value",
}


class SemanticVectorizer:
    """Feature-hashed CUDA concept vectors with deterministic L2 normalization."""

    def __init__(self, dimensions: int = 2048) -> None:
        if dimensions < 128:
            raise ValueError("dimensions must be at least 128")
        self.dimensions = dimensions

    def encode(self, text: str) -> SparseVector:
        normalized = text.casefold()
        concepts: list[str] = []
        for pattern, concept in _PHRASE_CONCEPTS:
            if re.search(pattern, normalized):
                concepts.append(concept)
                normalized = re.sub(pattern, f" {concept} ", normalized)
        tokens = _WORD.findall(normalized)
        concepts.extend(_TOKEN_CONCEPTS[token] for token in tokens if token in _TOKEN_CONCEPTS)

        weighted: list[tuple[str, float]] = [(f"token:{token}", 1.0) for token in tokens]
        weighted.extend((f"concept:{concept}", 2.0) for concept in concepts)
        weighted.extend(
            (f"bigram:{left}:{right}", 0.5) for left, right in zip(tokens, tokens[1:], strict=False)
        )
        vector: defaultdict[int, float] = defaultdict(float)
        for feature, weight in weighted:
            bucket = int.from_bytes(hashlib.sha256(feature.encode("utf-8")).digest()[:8], "big")
            vector[bucket % self.dimensions] += weight
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {bucket: value / norm for bucket, value in sorted(vector.items())}


def cosine_similarity(left: SparseVector, right: SparseVector) -> float:
    """Cosine for normalized sparse vectors returned by :class:`SemanticVectorizer`."""
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(bucket, 0.0) for bucket, value in left.items())


class RetrievalJudgment(StrictModel):
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    version: str = Field(min_length=1)
    relevant_source_ids: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class RetrievalEvaluationSuite(StrictModel):
    schema_version: Literal[1]
    split: Literal["development"]
    annotation_method: Literal["human_development"] = "human_development"
    k: int = Field(ge=1, le=100)
    queries: list[RetrievalJudgment] = Field(min_length=20)

    @model_validator(mode="after")
    def unique_queries(self) -> "RetrievalEvaluationSuite":
        identities = [query.query_id for query in self.queries]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate retrieval query_id")
        return self


class RetrievalQueryOutcome(StrictModel):
    query_id: str
    relevant_source_ids: list[str]
    retrieved_source_ids: list[str]
    hit: bool
    latency_ms: float = Field(ge=0)


class RetrievalMethodEvaluation(StrictModel):
    method: RetrievalMethod
    queries: int = Field(ge=1)
    hits: int = Field(ge=0)
    hit_at_k: float = Field(ge=0, le=1)
    total_latency_ms: float = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)
    outcomes: list[RetrievalQueryOutcome]


class RetrievalComparisonReport(StrictModel):
    schema_version: Literal[1] = 1
    split: Literal["development"] = "development"
    k: int
    corpus_hash: str
    evaluation_hash: str
    lexical_version: str
    semantic_version: Literal["cuda-concept-hash-v1-d2048"] = SEMANTIC_ENCODER_VERSION
    hybrid_version: Literal["rrf-k60-v1"] = HYBRID_FUSION_VERSION
    methods: list[RetrievalMethodEvaluation]
    default_method: RetrievalMethod
    selection_basis: Literal["development_hit_at_k_only"] = "development_hit_at_k_only"


class _Retriever(Protocol):
    corpus_hash: str
    tokenizer_version: str

    def retrieve(
        self,
        query: str,
        version: str,
        k: int = 5,
        *,
        method: RetrievalMethod = "lexical",
    ) -> RetrievalResult: ...


def load_evaluation_suite(path: Path) -> RetrievalEvaluationSuite:
    """Load strict, development-only human judgments from a tracked JSON file."""
    return RetrievalEvaluationSuite.model_validate_json(path.read_text(encoding="utf-8"))


def compare_retrieval_methods(
    index: _Retriever, suite: RetrievalEvaluationSuite
) -> RetrievalComparisonReport:
    """Compare rankings; choose the default only by development hit@k.

    Latency is measured and retained for inspection, but is intentionally excluded
    from method selection so machine load cannot change the chosen default.
    """
    methods: tuple[RetrievalMethod, ...] = ("lexical", "semantic", "hybrid")
    evaluations: list[RetrievalMethodEvaluation] = []
    for method in methods:
        outcomes: list[RetrievalQueryOutcome] = []
        for judgment in suite.queries:
            started = time.perf_counter_ns()
            result = index.retrieve(judgment.query, judgment.version, k=suite.k, method=method)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            retrieved = [chunk.source_id for chunk in result.chunks]
            outcomes.append(
                RetrievalQueryOutcome(
                    query_id=judgment.query_id,
                    relevant_source_ids=judgment.relevant_source_ids,
                    retrieved_source_ids=retrieved,
                    hit=bool(set(judgment.relevant_source_ids).intersection(retrieved)),
                    latency_ms=elapsed_ms,
                )
            )
        hits = sum(outcome.hit for outcome in outcomes)
        total_latency_ms = sum(outcome.latency_ms for outcome in outcomes)
        evaluations.append(
            RetrievalMethodEvaluation(
                method=method,
                queries=len(outcomes),
                hits=hits,
                hit_at_k=hits / len(outcomes),
                total_latency_ms=total_latency_ms,
                mean_latency_ms=total_latency_ms / len(outcomes),
                outcomes=outcomes,
            )
        )
    preference = {"lexical": 0, "semantic": 1, "hybrid": 2}
    selected = max(evaluations, key=lambda item: (item.hits, preference[item.method])).method
    evaluation_hash = hashlib.sha256(
        json.dumps(
            suite.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return RetrievalComparisonReport(
        k=suite.k,
        corpus_hash=index.corpus_hash,
        evaluation_hash=evaluation_hash,
        lexical_version=index.tokenizer_version,
        methods=evaluations,
        default_method=selected,
    )


def write_comparison_report(report: RetrievalComparisonReport, path: Path) -> None:
    """Persist a stable-shape audit artifact selected by the caller."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
