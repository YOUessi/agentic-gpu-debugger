"""Knowledge contracts and citation existence checks (not entailment)."""

import hashlib
import json
import unicodedata
from datetime import datetime
from typing import Annotated

from packaging.specifiers import SpecifierSet
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class KnowledgeError(Exception):
    """Explicit knowledge limitation; never synthesize substitute evidence."""


class KnowledgeSourceError(KnowledgeError):
    """Unapproved URL, invalid source, response or extraction."""


class KnowledgeOfflineError(KnowledgeError):
    """Local corpus missing or network unavailable."""


class KnowledgeTimeoutError(KnowledgeOfflineError):
    """Connect, read or total wall deadline exceeded."""


class KnowledgeIntegrityError(KnowledgeError):
    """Approved source content hash changed."""


class KnowledgeCorruptError(KnowledgeError):
    """Local corpus failed schema/integrity validation."""


class KnowledgeVersionUnavailableError(KnowledgeError):
    """Requested version is invalid, missing or incompatible."""


class InvalidCitationError(KnowledgeError):
    """A citation was not included in this retrieval result."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def normalize_text(text: str, *, code: bool = False) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    if code:
        return "\n".join(line.rstrip() for line in text.splitlines()).strip("\n")
    return " ".join(text.split())


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identity(
    source_id: str, document_version: str, source_url: str, block_ordinal: int, content_hash: str
) -> str:
    return (
        "nvcuda-"
        + sha256(f"{source_id}\n{document_version}\n{source_url}\n{block_ordinal}\n{content_hash}")[
            :24
        ]
    )


Nonempty = Annotated[str, Field(min_length=1)]


class DocumentChunk(StrictModel):
    chunk_id: Nonempty
    document_title: Nonempty
    document_version: Nonempty
    section_title: Nonempty
    source_url: Nonempty
    retrieved_at: Nonempty
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(min_length=1, max_length=1400)
    source_id: Nonempty
    block_ordinal: int = Field(ge=0)
    compatibility: dict[str, str]
    archive_release: str | None = None
    limitation: str | None = None

    @field_validator("retrieved_at")
    @classmethod
    def timestamp(cls, value: str) -> str:
        if datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("retrieved_at requires timezone")
        return value

    @field_validator("compatibility")
    @classmethod
    def versions(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or set(value) - {"cuda", "compute_sanitizer"}:
            raise ValueError("Explicit CUDA/Compute Sanitizer compatibility required")
        for specifier in value.values():
            if not specifier:
                raise ValueError("Empty compatibility range")
            SpecifierSet(specifier)
        return value

    @model_validator(mode="after")
    def integrity(self) -> "DocumentChunk":
        if sha256(self.text) != self.content_hash or self.chunk_id != identity(
            self.source_id,
            self.document_version,
            self.source_url,
            self.block_ordinal,
            self.content_hash,
        ):
            raise ValueError("Chunk hash/identity mismatch")
        return self


def make_chunk(
    *,
    source_id: str,
    document_title: str,
    document_version: str,
    section_title: str,
    source_url: str,
    retrieved_at: str,
    text: str,
    block_ordinal: int,
    compatibility: dict[str, str],
    archive_release: str | None = None,
    limitation: str | None = None,
) -> DocumentChunk:
    """Text must already use the prose/code normalizer chosen by extraction."""
    digest = sha256(text)
    return DocumentChunk(
        chunk_id=identity(source_id, document_version, source_url, block_ordinal, digest),
        source_id=source_id,
        document_title=document_title,
        document_version=document_version,
        section_title=section_title,
        source_url=source_url,
        retrieved_at=retrieved_at,
        content_hash=digest,
        text=text,
        block_ordinal=block_ordinal,
        compatibility=compatibility,
        archive_release=archive_release,
        limitation=limitation,
    )


class RetrievalResult(StrictModel):
    chunks: list[DocumentChunk]
    corpus_hash: str
    query: str


def validate_citations(ids: list[str], result: RetrievalResult) -> None:
    """Check membership only. Relevance and claim support require separate assessment."""
    missing = set(ids) - {chunk.chunk_id for chunk in result.chunks}
    if missing:
        raise InvalidCitationError(f"Citations absent from this result: {sorted(missing)}")


def corpus_digest(
    chunks: list[DocumentChunk],
    corpus_version: str,
    normalizer_version: str,
    tokenizer_version: str,
) -> str:
    payload = [
        corpus_version,
        normalizer_version,
        tokenizer_version,
        sorted((c.chunk_id, c.content_hash) for c in chunks),
        [
            c.model_dump(exclude={"retrieved_at"})
            for c in sorted(chunks, key=lambda chunk: chunk.chunk_id)
        ],
    ]
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
