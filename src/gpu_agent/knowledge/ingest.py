"""Explicit preparation step: fetch only static manifest sources, then build a local index.

No crawler, link expansion or user-URL entry point. Raw official content is never saved.
Each network operation runs in a disposable process to bound DNS/connect/slow-drip wall time.
"""

import hashlib
import json
import multiprocessing
import posixpath
import re
import time
import zlib
from collections.abc import Callable
from datetime import UTC, datetime
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, Tag
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic import Field, ValidationError, model_validator

from gpu_agent.knowledge.models import (
    DocumentChunk,
    KnowledgeError,
    KnowledgeIntegrityError,
    KnowledgeOfflineError,
    KnowledgeSourceError,
    KnowledgeTimeoutError,
    KnowledgeVersionUnavailableError,
    StrictModel,
    make_chunk,
    normalize_text,
    sha256,
)
from gpu_agent.knowledge.retrieve import KnowledgeIndex


class FetchPolicy(StrictModel):
    max_redirects: int = Field(ge=0, le=3)
    connect_timeout_seconds: float = Field(gt=0, le=5)
    read_timeout_seconds: float = Field(gt=0, le=10)
    wall_timeout_seconds: float = Field(gt=0, le=20)
    max_total_bytes: int = Field(gt=0, le=16777216)
    user_agent: str


class License(StrictModel):
    id: str
    redistribution_review_required: bool
    license_url: str | None = None
    copyright_notice: str | None = None
    notice_required: bool = False


class Toolchain(StrictModel):
    cuda_toolkit: str
    compute_sanitizer: str


class Source(StrictModel):
    source_id: str
    kind: Literal["html", "text"]
    document_title: str
    document_version: str
    archive_release: str | None = None
    canonical_url: str
    allowed_host: Literal["docs.nvidia.com", "raw.githubusercontent.com"]
    allowed_path_regex: str
    expected_media_types: list[str]
    max_bytes: int = Field(gt=0, le=6291456)
    include_anchors: list[str]
    chunk_strategy: Literal[
        "heading_blocks", "cuda_error", "approved_paragraphs", "evidence", "vector_add"
    ]
    version_regex: str | None = None
    version_selector: str | None = None
    version_probe_source: str | None = None
    release_evidence_source: str | None = None
    approved_chunk_sha256: dict[str, list[str]] = Field(default_factory=dict)
    body_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    git_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    compatibility: dict[str, str]
    limitation: str | None = None
    license: License

    @model_validator(mode="after")
    def source_contract(self) -> "Source":
        if not self.allowed_path_regex.startswith("^") or not self.allowed_path_regex.endswith("$"):
            raise ValueError("Path allowlist must be anchored")
        if re.search(r"[\[\]()*+?{}|]", self.allowed_path_regex) or re.search(
            r"(?<!\\)\.", self.allowed_path_regex
        ):
            raise ValueError("Path allowlist must describe one exact literal path")
        re.compile(self.allowed_path_regex)
        if validate_url(self.canonical_url, self) != self.canonical_url:
            raise ValueError("Manifest URL must already be canonical")
        if not self.version_regex and not self.version_probe_source and not self.git_revision:
            raise ValueError("Version evidence required")
        if self.version_regex:
            if re.compile(self.version_regex).groups != 1:
                raise ValueError("Version regex requires one capture")
        if not self.compatibility or set(self.compatibility) - {"cuda", "compute_sanitizer"}:
            raise ValueError("Invalid compatibility dimensions")
        for value in self.compatibility.values():
            if not value:
                raise ValueError("Empty compatibility")
            SpecifierSet(value)
        if self.allowed_host == "raw.githubusercontent.com":
            prefix = f"/NVIDIA/cuda-samples/{self.git_revision}/"
            if not self.git_revision or not urlsplit(self.canonical_url).path.startswith(prefix):
                raise ValueError("Samples require exact official commit")
            if not self.body_sha256:
                raise ValueError("Pinned source body hash required")
        if self.chunk_strategy == "approved_paragraphs":
            if (
                set(self.approved_chunk_sha256) != set(self.include_anchors)
                or not self.limitation
                or not self.release_evidence_source
            ):
                raise ValueError("Rolling manual requires approved hashes and version limitation")
        if (
            urlsplit(self.canonical_url).path == "/compute-sanitizer/ComputeSanitizer/index.html"
            and self.chunk_strategy != "approved_paragraphs"
        ):
            raise ValueError("Rolling manual cannot bypass paragraph approval")
        for hashes in self.approved_chunk_sha256.values():
            if not hashes or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
                raise ValueError("Invalid approved paragraph hashes")
        return self


class Manifest(StrictModel):
    schema_version: Literal[1]
    corpus_id: str
    corpus_version: str
    normalizer_version: Literal["nvidia-html-heading-v1"]
    tokenizer_version: Literal["cuda-lex-v1"]
    target_toolchain: Toolchain
    fetch_policy: FetchPolicy
    sources: list[Source]

    @model_validator(mode="after")
    def references(self) -> "Manifest":
        seen: dict[str, Source] = {}
        for source in self.sources:
            if source.source_id in seen:
                raise ValueError("Duplicate source_id")
            for reference in (source.version_probe_source, source.release_evidence_source):
                if reference is not None and reference not in seen:
                    raise ValueError("Source evidence must precede its consumer")
                if (
                    reference is not None
                    and seen[reference].document_version != source.document_version
                ):
                    raise ValueError("Source version must agree with its external version evidence")
            if source.release_evidence_source:
                release = seen[source.release_evidence_source]
                target = Version(self.target_toolchain.compute_sanitizer).release
                if len(target) < 2:
                    raise ValueError("Sanitizer target must identify a release line")
                required_anchor = f"updates-in-{target[0]}-{target[1]}"
                if (
                    release.canonical_url
                    != "https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html"
                    or release.kind != "html"
                    or release.chunk_strategy != "evidence"
                    or required_anchor not in release.include_anchors
                ):
                    raise ValueError("Official target release-notes anchor evidence required")
            seen[source.source_id] = source
        if not seen:
            raise ValueError("Empty manifest")
        return self


def load_manifest(path: Path) -> Manifest:
    """path is operator configuration, not agent/user URL input; unknown fields fail closed."""
    try:
        return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, UnicodeError) as exc:
        raise KnowledgeSourceError(f"Invalid static manifest: {path}") from exc


def _path(path: str) -> str:
    if re.search(r"%(?![0-9A-Fa-f]{2})", path):
        raise KnowledgeSourceError("Malformed percent encoding")
    try:
        decoded = unquote(path, errors="strict")
    except UnicodeError as exc:
        raise KnowledgeSourceError("Invalid path encoding") from exc
    if (
        "%" in decoded
        or "\\" in decoded
        or ".." in decoded.split("/")
        or any(ord(c) < 32 or ord(c) == 127 for c in decoded)
    ):
        raise KnowledgeSourceError("Unsafe path")
    return posixpath.normpath(decoded)


def validate_url(url: str, source: Source) -> str:
    if any(ord(c) < 33 or ord(c) >= 127 for c in url) or "\\" in url:
        raise KnowledgeSourceError("URL must be printable ASCII without backslashes")
    try:
        parts = urlsplit(url)
        if (
            parts.scheme != "https"
            or parts.netloc not in {source.allowed_host, source.allowed_host + ":443"}
            or parts.query
            or "?" in url.split("#", 1)[0]
            or parts.username
            or parts.password
        ):
            raise KnowledgeSourceError("URL authority/scheme/query outside static allowlist")
        path = _path(parts.path)
        canonical_path = _path(urlsplit(source.canonical_url).path)
        if path != canonical_path or not re.fullmatch(source.allowed_path_regex, path):
            raise KnowledgeSourceError("URL full path outside static allowlist")
        return urlunsplit(("https", source.allowed_host, path, "", ""))
    except ValueError as exc:
        raise KnowledgeSourceError("Malformed URL") from exc


class FetchReceipt(StrictModel):
    source_id: str
    effective_url: str
    retrieved_at: str
    body_sha256: str
    byte_count: int
    wire_bytes: int
    document_version: str


class Fetched(StrictModel):
    body: bytes
    effective_url: str
    retrieved_at: str
    wire_bytes: int


def fetch_stream(
    source: Source,
    policy: FetchPolicy,
    *,
    client: httpx.Client,
    clock: Callable[[], float] | None = None,
) -> Fetched:
    """Bound each stream and redirect; production also enforces an outer process deadline."""
    now = clock or time.monotonic
    deadline = now() + policy.wall_timeout_seconds
    current = validate_url(source.canonical_url, source)
    wire_bytes = 0

    def check_time() -> None:
        if now() >= deadline:
            raise KnowledgeTimeoutError("Official fetch wall timeout")

    try:
        for hop in range(policy.max_redirects + 1):
            check_time()
            with client.stream("GET", current, follow_redirects=False) as response:
                check_time()
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location", "")
                    if not location or hop == policy.max_redirects:
                        raise KnowledgeSourceError("Missing redirect location or redirect limit")
                    if any(ord(c) < 33 or ord(c) >= 127 for c in location) or "\\" in location:
                        raise KnowledgeSourceError("Malformed redirect location")
                    # Validate before urljoin, which otherwise erases traversal components.
                    _path(urlsplit(location).path)
                    current = validate_url(urljoin(current, location), source)
                    continue
                validate_url(str(response.url), source)
                if response.status_code != 200:
                    raise KnowledgeOfflineError(f"Official source HTTP {response.status_code}")
                media = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if media not in source.expected_media_types:
                    raise KnowledgeSourceError(f"Unexpected source media type: {media}")
                length = response.headers.get("content-length")
                if length is not None and (not length.isdigit() or int(length) > source.max_bytes):
                    raise KnowledgeSourceError("Invalid/oversized Content-Length")
                encoding = response.headers.get("content-encoding", "identity").lower()
                if encoding not in {"identity", "gzip"}:
                    raise KnowledgeSourceError("Unsupported content encoding")
                decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else None
                body = bytearray()
                for raw in response.iter_raw():
                    check_time()
                    wire_bytes += len(raw)
                    if wire_bytes > source.max_bytes:
                        raise KnowledgeSourceError("Wire byte limit exceeded")
                    remaining = source.max_bytes - len(body)
                    decoded = decoder.decompress(raw, remaining + 1) if decoder else raw
                    body.extend(decoded)
                    if len(body) > source.max_bytes:
                        raise KnowledgeSourceError("Decompressed byte limit exceeded")
                check_time()
                if decoder and (not decoder.eof or decoder.unused_data):
                    raise KnowledgeSourceError("Truncated or concatenated gzip stream")
                if not body:
                    raise KnowledgeSourceError("Empty source body")
                body.decode("utf-8", errors="strict")
                return Fetched(
                    body=bytes(body),
                    effective_url=current,
                    wire_bytes=wire_bytes,
                    retrieved_at=datetime.now(UTC).isoformat(),
                )
    except httpx.TimeoutException as exc:
        raise KnowledgeTimeoutError("Official source connect/read timeout") from exc
    except (httpx.InvalidURL, httpx.RemoteProtocolError) as exc:
        raise KnowledgeSourceError("Malformed redirect/source response") from exc
    except httpx.HTTPError as exc:
        raise KnowledgeOfflineError(f"Official source unavailable: {type(exc).__name__}") from exc
    except (UnicodeError, zlib.error, ValueError) as exc:
        raise KnowledgeSourceError("Malformed source response") from exc
    raise KnowledgeSourceError("Redirect limit")


def _fetch_worker(connection: Connection, source: Source, policy: FetchPolicy) -> None:
    try:
        timeout = httpx.Timeout(policy.read_timeout_seconds, connect=policy.connect_timeout_seconds)
        with httpx.Client(
            timeout=timeout,
            trust_env=False,
            headers={
                "User-Agent": policy.user_agent,
                "Accept-Encoding": "gzip, identity",
            },
        ) as client:
            result = fetch_stream(source, policy, client=client)
        connection.send((None, result.model_dump()))
    except KnowledgeError as exc:
        connection.send((type(exc).__name__, str(exc)))
    finally:
        connection.close()


def fetch(source: Source, policy: FetchPolicy) -> Fetched:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_fetch_worker, args=(child, source, policy), daemon=True)
    process.start()
    child.close()
    try:
        if not parent.poll(policy.wall_timeout_seconds):
            raise KnowledgeTimeoutError("Official fetch process wall deadline exceeded")
        error, payload = parent.recv()
        errors: dict[str, type[KnowledgeError]] = {
            "KnowledgeSourceError": KnowledgeSourceError,
            "KnowledgeOfflineError": KnowledgeOfflineError,
            "KnowledgeTimeoutError": KnowledgeTimeoutError,
        }
        if error:
            raise errors.get(error, KnowledgeSourceError)(payload)
        return Fetched.model_validate(payload)
    except (EOFError, OSError) as exc:
        raise KnowledgeOfflineError("Official fetch worker failed") from exc
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join()


def _soup(source: Source, text: str) -> BeautifulSoup:
    soup = BeautifulSoup(text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if source.document_title not in title or soup.find("input", attrs={"type": "password"}):
        raise KnowledgeSourceError("Source title mismatch/login page")
    for element in soup.select("nav, header, footer, script, style"):
        element.decompose()
    return soup


def _section(soup: BeautifulSoup, anchor: str) -> Tag:
    node = soup.find(id=anchor)
    if not isinstance(node, Tag):
        raise KnowledgeSourceError(f"Missing required anchor: {anchor}")
    # Older NVIDIA docs attach anchors to headings nested in a section div.
    if node.name not in {"section", "div"} and isinstance(node.parent, Tag):
        node = node.parent
    return node


def _heading(node: Tag, document_title: str) -> tuple[str, str]:
    sections = list(reversed(list(node.parents))) + [node]
    titles = [document_title]
    anchor = ""
    for parent in sections:
        if not isinstance(parent, Tag) or parent.name not in {"section", "div"}:
            continue
        heading = parent.find(re.compile(r"^h[1-4]$"), recursive=False)
        if isinstance(heading, Tag):
            title = normalize_text(heading.get_text(" ", strip=True)).rstrip("#¶ ")
            if title and title not in titles:
                titles.append(title)
            raw_anchor = parent.get("id") or heading.get("id")
            if isinstance(raw_anchor, str):
                anchor = raw_anchor
    return " > ".join(titles), anchor


def extract_chunks(
    source: Source, body: bytes, retrieved_at: str, *, verified_version: str | None = None
) -> list[DocumentChunk]:
    if source.body_sha256 and hashlib.sha256(body).hexdigest() != source.body_sha256:
        raise KnowledgeIntegrityError(f"Pinned source body changed: {source.source_id}")
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise KnowledgeSourceError("Source is not UTF-8") from exc
    if source.version_regex:
        version_text = text
        if source.version_selector:
            markers = BeautifulSoup(text, "html.parser").select(source.version_selector)
            if len(markers) != 1:
                raise KnowledgeVersionUnavailableError("Missing/ambiguous page version marker")
            version_text = markers[0].get_text(" ", strip=True)
        match = re.search(source.version_regex, version_text)
        verified_version = match.group(1) if match else None
    if source.git_revision and source.body_sha256:
        verified_version = source.document_version
    if verified_version != source.document_version:
        raise KnowledgeVersionUnavailableError(f"Source version mismatch: {source.source_id}")
    soup = _soup(source, text) if source.kind == "html" else None
    sections = {anchor: _section(soup, anchor) for anchor in source.include_anchors} if soup else {}
    if source.chunk_strategy == "evidence":
        return []
    blocks: list[tuple[str, str, str]] = []
    if source.chunk_strategy == "vector_add":
        match = re.search(r"__global__\s+void\s+vectorAdd\b[^\{]*\{", text)
        if not match:
            raise KnowledgeSourceError("Missing vectorAdd kernel")
        start, end, depth = match.start(), match.end(), 1
        while depth and end < len(text):
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        if depth:
            raise KnowledgeSourceError("Incomplete vectorAdd kernel")
        blocks.append(("vectorAdd", "vectorAdd", normalize_text(text[start:end], code=True)))
    elif source.chunk_strategy == "cuda_error" and soup:
        definition = next(
            (dt for dt in soup.find_all("dt") if "cudaErrorIllegalAddress" in dt.get_text()), None
        )
        description = definition.find_next_sibling("dd") if definition else None
        if not definition or not description:
            raise KnowledgeSourceError("Missing illegal-address definition")
        blocks.append(
            (
                source.include_anchors[0],
                "cudaErrorIllegalAddress",
                normalize_text(
                    definition.get_text(" ", strip=True)
                    + " "
                    + description.get_text(" ", strip=True)
                ),
            )
        )
    else:
        for selected_anchor, section in sections.items():
            expected = source.approved_chunk_sha256.get(selected_anchor)
            found_hashes: set[str] = set()
            tags = section.find_all("p" if expected else ["p", "pre", "li", "tr"])
            for tag in tags:
                if not isinstance(tag, Tag):
                    continue
                if not expected and any(
                    p.name in {"pre", "li", "tr"} for p in tag.parents if p is not section
                ):
                    continue
                is_code = tag.name == "pre"
                content = normalize_text(
                    tag.get_text("" if is_code else " ", strip=not is_code), code=is_code
                )
                digest = sha256(content)
                if expected and digest not in expected:
                    continue
                if not content:
                    continue
                if len(content) > 1400:
                    raise KnowledgeSourceError("Semantic block exceeds 1400 characters")
                heading, anchor = _heading(tag, source.document_title)
                blocks.append((anchor or selected_anchor, heading, content))
                found_hashes.add(digest)
            if expected and set(expected) != found_hashes:
                raise KnowledgeIntegrityError(
                    f"Approved paragraph changed/missing: {selected_anchor}"
                )
    chunks: list[DocumentChunk] = []
    ordinals: dict[str, int] = {}
    for anchor, title, content in blocks:
        ordinal = ordinals.get(anchor, 0)
        ordinals[anchor] = ordinal + 1
        try:
            chunks.append(
                make_chunk(
                    source_id=source.source_id,
                    document_title=source.document_title,
                    document_version=source.document_version,
                    section_title=title,
                    source_url=source.canonical_url + "#" + anchor,
                    retrieved_at=retrieved_at,
                    text=content,
                    block_ordinal=ordinal,
                    compatibility=source.compatibility,
                    archive_release=source.archive_release,
                    limitation=source.limitation,
                )
            )
        except ValidationError as exc:
            raise KnowledgeSourceError("Invalid extracted semantic block") from exc
    if not chunks:
        raise KnowledgeSourceError(f"No selected source content: {source.source_id}")
    return chunks


def ingest(manifest: Manifest) -> tuple[KnowledgeIndex, list[FetchReceipt]]:
    try:
        manifest = Manifest.model_validate(manifest.model_dump())
    except ValueError as exc:
        raise KnowledgeSourceError("Invalid manifest at ingest boundary") from exc
    chunks: list[DocumentChunk] = []
    receipts: list[FetchReceipt] = []
    verified: dict[str, str] = {}
    total = 0
    for source in manifest.sources:
        remaining = manifest.fetch_policy.max_total_bytes - total
        if remaining <= 0:
            raise KnowledgeSourceError("Cumulative ingest byte budget exhausted")
        fetched = fetch(
            source.model_copy(update={"max_bytes": min(source.max_bytes, remaining)}),
            manifest.fetch_policy,
        )
        total += max(len(fetched.body), fetched.wire_bytes)
        version = verified.get(source.version_probe_source or "")
        if source.release_evidence_source and source.release_evidence_source not in verified:
            raise KnowledgeVersionUnavailableError("Target sanitizer release evidence missing")
        chunks.extend(
            extract_chunks(source, fetched.body, fetched.retrieved_at, verified_version=version)
        )
        verified[source.source_id] = source.document_version
        receipts.append(
            FetchReceipt(
                source_id=source.source_id,
                effective_url=fetched.effective_url,
                retrieved_at=fetched.retrieved_at,
                body_sha256=hashlib.sha256(fetched.body).hexdigest(),
                byte_count=len(fetched.body),
                wire_bytes=fetched.wire_bytes,
                document_version=source.document_version,
            )
        )
    return KnowledgeIndex(
        chunks,
        corpus_version=manifest.corpus_version,
        normalizer_version=manifest.normalizer_version,
        tokenizer_version=manifest.tokenizer_version,
    ), receipts
