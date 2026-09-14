"""Small local inverted BM25 index; no URL fetching and no embedding dependency."""

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

ATOM = re.compile(
    r"[A-Za-z]+(?:-[A-Za-z]+)+|(?:__)?[A-Za-z_][A-Za-z0-9_]*"
    r"(?:(?:::|\.)[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?|[0-9]+(?:\.[0-9]+){1,3}"
)


def tokenize(text: str) -> list[str]:
    """Preserve display atoms plus case-folded lookup companions; explicit OOB alias."""
    tokens: list[str] = []
    for match in ATOM.finditer(text):
        atom = match.group()
        tokens.append(atom)
        if atom != atom.lower():
            tokens.append(atom.lower())
        if "-" in atom:
            tokens.extend([atom.lower().replace("-", "_"), *atom.lower().split("-")])
        if atom.lower() == "oob":
            tokens.append("out_of_bounds")
    if re.search(r"\bout of bounds\b", text, re.I):
        tokens.append("out_of_bounds")
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
        self.corpus_hash = corpus_digest(
            chunks, corpus_version, normalizer_version, tokenizer_version
        )
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._lengths: list[int] = []
        for ordinal, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text) + tokenize(chunk.section_title) * 2)
            self._lengths.append(counts.total())
            for token, count in counts.items():
                self._postings[token][ordinal] = count

    def retrieve(self, query: str, version: str, k: int = 5) -> RetrievalResult:
        if not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")
        versions = parse_version(version)
        eligible = {
            i
            for i, c in enumerate(self.chunks)
            if all(versions[name] in SpecifierSet(spec) for name, spec in c.compatibility.items())
        }
        if not eligible:
            raise KnowledgeVersionUnavailableError(f"No knowledge for {version}")
        average = sum(self._lengths[i] for i in eligible) / len(eligible)
        scores: dict[int, float] = defaultdict(float)
        for token in set(tokenize(query)):
            postings = self._postings.get(token, {})
            candidates = eligible.intersection(postings)
            idf = math.log(1 + (len(eligible) - len(candidates) + 0.5) / (len(candidates) + 0.5))
            for i in candidates:
                tf = postings[i]
                scores[i] += (
                    idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self._lengths[i] / average))
                )
        if "out_of_bounds" in tokenize(query):
            for i in scores:
                if "out_of_bounds" in tokenize(self.chunks[i].text):
                    scores[i] += 1.0
        selected = sorted(scores, key=lambda i: (-scores[i], self.chunks[i].chunk_id))[:k]
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
                or payload.tokenizer_version != "cuda-lex-v1"
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
