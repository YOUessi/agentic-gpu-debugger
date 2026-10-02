"""Local lexical, semantic-vector and hybrid retrieval; never fetches URLs."""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import ValidationError

from gpu_agent.knowledge.models import (
    DocumentChunk,
    KnowledgeCorruptError,
    KnowledgeOfflineError,
    KnowledgeVersionUnavailableError,
    RetrievalResult,
    StrictModel,
    corpus_digest,
)
from gpu_agent.knowledge.semantic import RetrievalMethod, SemanticVectorizer, cosine_similarity

ATOM = re.compile(
    r"[A-Za-z]+(?:-[A-Za-z]+)+|(?:__)?[A-Za-z_][A-Za-z0-9_]*"
    r"(?:(?:::|\.)[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?|[0-9]+(?:\.[0-9]+){1,3}"
)

# Query-only function-word filtering. No CUDA API, failure label, case identifier,
# or expected document is promoted. Keep negations (not/no/without) meaningful.
_QUERY_STOPWORDS = frozenset(
    "a an the is are was were be been being am do does did to of for from in on at "
    "by with as it its this that these those i you we they he she can could would "
    "should may might must will shall how what which who when where why "
    "and or if then than".split()
)


def tokenize(text: str, version: str = "cuda-lex-v1") -> list[str]:
    """Preserve display atoms plus case-folded lookup companions; explicit OOB alias."""
    tokens: list[str] = []
    for match in ATOM.finditer(text):
        atom = match.group()
        tokens.append(atom)
        if version in {
            "cuda-lex-v2",
            "cuda-lex-v3",
            "cuda-lex-v4",
            "cuda-lex-v5",
            "cuda-lex-v6",
        } and atom.endswith("()"):
            tokens.append(atom[:-2])
            if atom[:-2] != atom[:-2].lower():
                tokens.append(atom[:-2].lower())
        if atom != atom.lower():
            tokens.append(atom.lower())
        if "-" in atom:
            tokens.extend([atom.lower().replace("-", "_"), *atom.lower().split("-")])
        if atom.lower() == "oob":
            tokens.append("out_of_bounds")
        if version in {"cuda-lex-v5", "cuda-lex-v6"}:
            # Keep exact symbols, additionally index identifier components for prose queries.
            spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", atom)
            parts = re.findall(r"[A-Za-z]+", spaced)
            if len(parts) > 1:
                tokens.extend(part.lower() for part in parts)
    if re.search(r"\bout of bounds\b", text, re.I):
        tokens.append("out_of_bounds")
    if version == "cuda-lex-v6":
        # Conservative English plural/third-person inflection companion; preserve originals.
        tokens.extend(
            t[:-1]
            for t in list(tokens)
            if re.fullmatch(r"[a-z]{3,}s", t) and not t.endswith(("ss", "us", "is"))
        )
    return tokens


def parse_version(version: str) -> dict[str, Version]:
    try:
        parts = [part.split("=", 1) for part in version.split(";")]
        values = dict(parts)
        if len(parts) != 2 or set(values) != {"cuda", "compute-sanitizer"}:
            raise ValueError("Specify cuda=VERSION;compute-sanitizer=VERSION")
        return {
            "cuda": Version(values["cuda"]),
            "compute_sanitizer": Version(values["compute-sanitizer"]),
        }
    except (ValueError, InvalidVersion) as exc:
        raise KnowledgeVersionUnavailableError(f"Invalid toolchain version: {version}") from exc


class SavedIndex(StrictModel):
    schema_version: int
    corpus_version: str
    normalizer_version: str
    tokenizer_version: str
    corpus_hash: str
    chunks: list[DocumentChunk]


class KnowledgeIndex:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        *,
        corpus_version: str = "1",
        normalizer_version: str = "nvidia-html-heading-v1",
        tokenizer_version: str = "cuda-lex-v1",
    ) -> None:
        # Validate again at the boundary, including objects made with model_copy.
        try:
            self.chunks = [DocumentChunk.model_validate(c.model_dump()) for c in chunks]
        except (ValueError, ValidationError) as exc:
            raise KnowledgeCorruptError("Invalid corpus chunk") from exc
        if len({c.chunk_id for c in chunks}) != len(chunks) or not chunks:
            raise KnowledgeCorruptError("Empty corpus or duplicate chunk identities")
        self.corpus_version = corpus_version
        self.normalizer_version = normalizer_version
        self.tokenizer_version = tokenizer_version
        if tokenizer_version not in {
            "cuda-lex-v1",
            "cuda-lex-v2",
            "cuda-lex-v3",
            "cuda-lex-v4",
            "cuda-lex-v5",
            "cuda-lex-v6",
        }:
            raise KnowledgeCorruptError("Unsupported tokenizer version")
        self.corpus_hash = corpus_digest(
            chunks, corpus_version, normalizer_version, tokenizer_version
        )
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._lengths: list[int] = []
        vectorizer = SemanticVectorizer()
        self._semantic_vectors = []
        for ordinal, chunk in enumerate(chunks):
            counts = Counter(
                tokenize(chunk.text, self.tokenizer_version)
                + tokenize(chunk.section_title, self.tokenizer_version) * 2
                + (
                    tokenize(chunk.document_title, self.tokenizer_version)
                    if self.tokenizer_version in {"cuda-lex-v4", "cuda-lex-v5", "cuda-lex-v6"}
                    else []
                )
            )
            self._lengths.append(counts.total())
            self._semantic_vectors.append(
                vectorizer.encode(f"{chunk.section_title} {chunk.section_title} {chunk.text}")
            )
            for token, count in counts.items():
                self._postings[token][ordinal] = count

    def _eligible(self, version: str) -> set[int]:
        versions = parse_version(version)
        eligible = {
            i
            for i, chunk in enumerate(self.chunks)
            if all(
                versions[name] in SpecifierSet(spec) for name, spec in chunk.compatibility.items()
            )
        }
        if not eligible:
            raise KnowledgeVersionUnavailableError(f"No knowledge for {version}")
        return eligible

    def _lexical_scores(self, query: str, eligible: set[int]) -> dict[int, float]:
        average = sum(self._lengths[i] for i in eligible) / len(eligible)
        scores: dict[int, float] = defaultdict(float)
        query_tokens = set(tokenize(query, self.tokenizer_version))
        if self.tokenizer_version in {"cuda-lex-v3", "cuda-lex-v4", "cuda-lex-v5", "cuda-lex-v6"}:
            query_tokens = {t for t in query_tokens if t.casefold() not in _QUERY_STOPWORDS}
        for token in query_tokens:
            postings = self._postings.get(token, {})
            candidates = eligible.intersection(postings)
            idf = math.log(1 + (len(eligible) - len(candidates) + 0.5) / (len(candidates) + 0.5))
            for i in candidates:
                tf = postings[i]
                scores[i] += (
                    idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self._lengths[i] / average))
                )
        if "out_of_bounds" in query_tokens:
            for i in scores:
                if "out_of_bounds" in tokenize(self.chunks[i].text):
                    scores[i] += 1.0
        if self.tokenizer_version in {"cuda-lex-v3", "cuda-lex-v4", "cuda-lex-v5", "cuda-lex-v6"}:
            # Exact, explicitly named APIs take precedence over incidental prose.
            # Fall back to ordinary lexical matching if the corpus has no anchor.
            apis = {
                t.removesuffix("()")
                for t in query_tokens
                if re.fullmatch(r"__[A-Za-z_]\w*(?:\(\))?|cuda[A-Z]\w*(?:\(\))?", t)
            }
            anchored = {i for api in apis for i in self._postings.get(api, {}) if i in scores}
            if anchored:
                scores = {i: score for i, score in scores.items() if i in anchored}
        return scores

    def _semantic_scores(self, query: str, eligible: set[int]) -> dict[int, float]:
        query_vector = SemanticVectorizer().encode(query)
        return {
            ordinal: score
            for ordinal in eligible
            if (score := cosine_similarity(query_vector, self._semantic_vectors[ordinal])) > 0
        }

    def _ordered(self, scores: dict[int, float]) -> list[int]:
        return sorted(scores, key=lambda i: (-scores[i], self.chunks[i].chunk_id))

    def _select_lexical(self, scores: dict[int, float], k: int) -> list[int]:
        ordered = self._ordered(scores)
        if self.tokenizer_version != "cuda-lex-v6" or k < 3:
            return ordered[:k]
        # Avoid filling a small context with fragments from one document. Backfill
        # deferred fragments when fewer sources match; never insert a nonmatch.
        counts: Counter[str] = Counter()
        primary, deferred = [], []
        for ordinal in ordered:
            source = self.chunks[ordinal].source_id
            if counts[source] < 2:
                primary.append(ordinal)
                counts[source] += 1
            else:
                deferred.append(ordinal)
        return (primary + deferred)[:k]

    def retrieve(
        self,
        query: str,
        version: str,
        k: int = 5,
        *,
        method: RetrievalMethod = "lexical",
    ) -> RetrievalResult:
        if not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")
        eligible = self._eligible(version)
        lexical = self._lexical_scores(query, eligible)
        if method == "lexical":
            selected = self._select_lexical(lexical, k)
        elif method == "semantic":
            selected = self._ordered(self._semantic_scores(query, eligible))[:k]
        elif method == "hybrid":
            semantic = self._semantic_scores(query, eligible)
            combined: dict[int, float] = defaultdict(float)
            for ranking in (self._ordered(lexical), self._ordered(semantic)):
                for rank, ordinal in enumerate(ranking, start=1):
                    combined[ordinal] += 1 / (60 + rank)
            selected = self._ordered(combined)[:k]
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        return RetrievalResult(
            chunks=[self.chunks[i] for i in selected], corpus_hash=self.corpus_hash, query=query
        )

    def save(self, path: Path) -> None:
        """Persist to caller-selected local cache, never to source-controlled data by default."""
        payload = SavedIndex(
            schema_version=1,
            corpus_version=self.corpus_version,
            normalizer_version=self.normalizer_version,
            tokenizer_version=self.tokenizer_version,
            corpus_hash=self.corpus_hash,
            chunks=self.chunks,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload.model_dump_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeIndex":
        try:
            payload = SavedIndex.model_validate(json.loads(path.read_text(encoding="utf-8")))
            if (
                payload.schema_version != 1
                or payload.normalizer_version != "nvidia-html-heading-v1"
                or payload.tokenizer_version
                not in {
                    "cuda-lex-v1",
                    "cuda-lex-v2",
                    "cuda-lex-v3",
                    "cuda-lex-v4",
                    "cuda-lex-v5",
                    "cuda-lex-v6",
                }
            ):
                raise ValueError("Unsupported index format")
            index = cls(
                payload.chunks,
                corpus_version=payload.corpus_version,
                normalizer_version=payload.normalizer_version,
                tokenizer_version=payload.tokenizer_version,
            )
            if index.corpus_hash != payload.corpus_hash:
                raise ValueError("Corpus checksum mismatch")
            return index
        except OSError as exc:
            raise KnowledgeOfflineError(f"Local corpus unavailable: {path}") from exc
        except (ValueError, UnicodeError) as exc:
            raise KnowledgeCorruptError(f"Corrupted local corpus: {path}") from exc
