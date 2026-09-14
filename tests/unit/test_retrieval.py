"""Self-authored fixtures: no redistributed NVIDIA document text."""

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERSION = "cuda=12.8.1;compute-sanitizer=2025.1.0.0"
URL = "https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html"


def source():
    from gpu_agent.knowledge.ingest import load_manifest

    return load_manifest(ROOT / "knowledge/sources.json").sources[0]


@pytest.fixture
def index():
    from gpu_agent.knowledge.models import make_chunk
    from gpu_agent.knowledge.retrieve import KnowledgeIndex

    # Three original paraphrases, not NVIDIA excerpts or adapted sample code.
    texts = [
        ("Bounds", "Guard global memory indices to prevent out-of-bounds writes.", "12.8.1"),
        ("Errors", "cudaErrorIllegalAddress identifies an invalid device access.", "12.8.1"),
        ("Old", "global memory out-of-bounds legacy advice", "12.7.0"),
    ]
    chunks = [
        make_chunk(
            source_id="self-authored",
            document_title="Independent test paraphrases",
            document_version=version,
            section_title=heading,
            source_url=URL + "#device-memory",
            retrieved_at="2026-09-15T00:00:00+00:00",
            text=text,
            block_ordinal=i,
            compatibility={"cuda": "==" + version},
        )
        for i, (heading, text, version) in enumerate(texts)
    ]
    return KnowledgeIndex(chunks)


def test_stable_identity_and_content_changes(index):
    from gpu_agent.knowledge.models import make_chunk
    from gpu_agent.knowledge.retrieve import KnowledgeIndex

    original = index.chunks[0]
    fields = original.model_dump(exclude={"chunk_id", "content_hash"})
    fields["retrieved_at"] = "2026-09-16T00:00:00+00:00"
    repeated = make_chunk(**fields)
    assert repeated.chunk_id == original.chunk_id
    assert KnowledgeIndex([repeated]).corpus_hash == KnowledgeIndex([original]).corpus_hash
    fields["text"] += " Additional original detail."
    changed = make_chunk(**fields)
    assert changed.chunk_id != original.chunk_id
    assert changed.content_hash != original.content_hash
    assert KnowledgeIndex([changed]).corpus_hash != KnowledgeIndex([original]).corpus_hash


def test_cuda_atoms():
    from gpu_agent.knowledge.retrieve import tokenize

    atoms = [
        "__syncthreads()",
        "threadIdx.x",
        "blockIdx.x",
        "cudaErrorIllegalAddress",
        "cudaMalloc()",
        "cp.async.mbarrier.arrive",
        "sm_90",
        "12.8.1",
    ]
    tokens = tokenize(" ".join(atoms) + " out-of-bounds OOB")
    assert all(atom in tokens for atom in atoms)
    assert "cudaerrorillegaladdress" in tokens
    assert "threadidx.x" in tokens
    assert {"out_of_bounds", "out", "of", "bounds"}.issubset(tokens)
    assert "threadidx" not in tokens


def test_version_filter_bm25_and_no_unrelated_results(index):
    from gpu_agent.knowledge.models import KnowledgeVersionUnavailableError

    result = index.retrieve("global memory OOB", VERSION, k=1)
    assert len(result.chunks) == 1
    assert result.chunks[0].section_title == "Bounds"
    assert result.query == "global memory OOB"
    assert index.retrieve("absentlexeme", VERSION).chunks == []
    with pytest.raises(KnowledgeVersionUnavailableError):
        index.retrieve("global memory", "cuda=11.0;compute-sanitizer=2025.1")
    with pytest.raises(KnowledgeVersionUnavailableError):
        index.retrieve("global memory", "latest")


def test_both_version_dimensions_required(index):
    from gpu_agent.knowledge.models import KnowledgeVersionUnavailableError
    from gpu_agent.knowledge.retrieve import KnowledgeIndex

    chunk = index.chunks[0].model_copy(
        update={"compatibility": {"cuda": ">=12.8,<12.9", "compute_sanitizer": "==2025.4"}}
    )
    other = KnowledgeIndex([chunk])
    with pytest.raises(KnowledgeVersionUnavailableError):
        other.retrieve("global memory", VERSION)


def test_citation_existence_does_not_assert_relevance(index):
    from gpu_agent.knowledge.models import InvalidCitationError, validate_citations

    result = index.retrieve("cudaErrorIllegalAddress", VERSION)
    validate_citations([result.chunks[0].chunk_id], result)
    assert "global memory" not in result.chunks[0].text
    with pytest.raises(InvalidCitationError):
        validate_citations(["invented-id"], result)
    with pytest.raises(InvalidCitationError):
        validate_citations([index.chunks[0].chunk_id], result)


def test_cache_missing_corrupt_and_tampered(index, tmp_path):
    from gpu_agent.knowledge.models import KnowledgeCorruptError, KnowledgeOfflineError
    from gpu_agent.knowledge.retrieve import KnowledgeIndex

    path = tmp_path / "index.json"
    with pytest.raises(KnowledgeOfflineError):
        KnowledgeIndex.load(path)
    index.save(path)
    assert KnowledgeIndex.load(path).corpus_hash == index.corpus_hash
    data = json.loads(path.read_text())
    data["chunks"][0]["text"] += " tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(KnowledgeCorruptError):
        KnowledgeIndex.load(path)
    path.write_text("not json")
    with pytest.raises(KnowledgeCorruptError):
        KnowledgeIndex.load(path)


@pytest.mark.parametrize(
    "bad",
    [
        URL.replace("https:", "http:"),
        URL.replace("docs.nvidia.com", "docs.nvidia.com.evil.test"),
        URL.replace("docs.nvidia.com", "developer.nvidia.com"),
        URL.replace("docs.nvidia.com", "user@docs.nvidia.com"),
        URL.replace("docs.nvidia.com", "docs.nvidia.com:444"),
        URL + "?q=1",
        URL.replace("index.html", "../index.html"),
        URL.replace("index.html", "%2e%2e/index.html"),
        URL.replace("index.html", "%252e%252e/index.html"),
        URL.replace("index.html", "%00index.html"),
        URL.replace("index.html", "\\index.html"),
        URL.replace("index.html", "%zz"),
        URL.replace("index.html", "other.html"),
        URL.replace("docs.nvidia.com", "DOCS.NVIDIA.COM"),
        URL + "\n",
    ],
)
def test_exact_url_allowlist(bad):
    from gpu_agent.knowledge.ingest import validate_url
    from gpu_agent.knowledge.models import KnowledgeSourceError

    with pytest.raises(KnowledgeSourceError):
        validate_url(bad, source())


def test_url_normalization():
    from gpu_agent.knowledge.ingest import validate_url

    assert validate_url(URL + "#device-memory", source()) == URL
    assert validate_url(URL.replace("index.html", "%69ndex.html"), source()) == URL
    assert validate_url(URL.replace("docs.nvidia.com", "docs.nvidia.com:443"), source()) == URL


@pytest.mark.parametrize("mutation", ["unknown", "duplicate", "missing", "unsafe"])
def test_manifest_validation(mutation, tmp_path):
    from gpu_agent.knowledge.ingest import load_manifest
    from gpu_agent.knowledge.models import KnowledgeSourceError

    data = json.loads((ROOT / "knowledge/sources.json").read_text())
    if mutation == "unknown":
        data["sources"][0]["surprise"] = True
    elif mutation == "duplicate":
        data["sources"].append(data["sources"][0])
    elif mutation == "missing":
        del data["schema_version"]
    else:
        data["sources"][0]["canonical_url"] = "http://evil.test/x"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(KnowledgeSourceError):
        load_manifest(path)


def fixture_html(body="<p>Independent CUDA test text.</p>", version="12.8"):
    return (
        f"<html><head><title>CUDA C++ Programming Guide</title></head><body>"
        f'<li class="wy-breadcrumbs-aside"><span>v{version}</span></li>'
        f'<main><h1>Guide</h1><section id="device-memory">'
        f"<h2>Device Memory</h2>{body}</section></main></body></html>"
    ).encode()


def test_heading_chunks_normalization_and_code():
    from gpu_agent.knowledge.ingest import extract_chunks

    chunks = extract_chunks(
        source(),
        fixture_html(
            '<p>Independent   text &amp; details.</p><section id="nested"><h3>Indexing</h3>'
            "<pre>threadIdx.x;  \n__syncthreads();</pre></section>"
        ),
        "2026-09-15T00:00:00Z",
    )
    assert chunks[0].text == "Independent text & details."
    assert chunks[1].text == "threadIdx.x;\n__syncthreads();"
    assert chunks[1].source_url.endswith("#nested")
    assert "Device Memory" in chunks[1].section_title and "Indexing" in chunks[1].section_title
    assert all(len(c.text) <= 1400 for c in chunks)
    assert all(c.document_version == "12.8" and c.archive_release == "12.8.1" for c in chunks)


def test_oversize_semantic_block_is_rejected():
    from gpu_agent.knowledge.ingest import extract_chunks
    from gpu_agent.knowledge.models import KnowledgeSourceError

    with pytest.raises(KnowledgeSourceError):
        extract_chunks(
            source(), fixture_html("<pre>" + "x" * 1401 + "</pre>"), "2026-09-15T00:00:00Z"
        )


def test_page_version_and_approved_hash_fail_closed():
    from gpu_agent.knowledge.ingest import extract_chunks
    from gpu_agent.knowledge.models import KnowledgeIntegrityError, KnowledgeVersionUnavailableError

    with pytest.raises(KnowledgeVersionUnavailableError):
        extract_chunks(source(), fixture_html(version="12.0.0"), "2026-09-15T00:00:00Z")
    restricted = source().model_copy(
        update={
            "approved_chunk_sha256": {"device-memory": ["0" * 64]},
            "limitation": "Pinned cross-version paragraph only",
        }
    )
    with pytest.raises(KnowledgeIntegrityError):
        extract_chunks(restricted, fixture_html(), "2026-09-15T00:00:00Z")


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_revalidates_each_hop(status):
    import httpx

    from gpu_agent.knowledge.ingest import fetch_stream, load_manifest
    from gpu_agent.knowledge.models import KnowledgeSourceError

    def handler(request):
        return httpx.Response(status, headers={"location": "https://developer.nvidia.com/x"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(KnowledgeSourceError):
            fetch_stream(
                source(), load_manifest(ROOT / "knowledge/sources.json").fetch_policy, client=client
            )


def test_redirect_success_and_loop_limit():
    import httpx

    from gpu_agent.knowledge.ingest import fetch_stream, load_manifest
    from gpu_agent.knowledge.models import KnowledgeSourceError

    policy = load_manifest(ROOT / "knowledge/sources.json").fetch_policy
    responses = iter(
        [
            httpx.Response(302, headers={"location": "index.html#anchor"}),
            httpx.Response(
                200, headers={"content-type": "text/html"}, stream=httpx.ByteStream(fixture_html())
            ),
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(lambda _: next(responses))) as client:
        assert fetch_stream(source(), policy, client=client).effective_url == URL
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(302, headers={"location": "index.html"})
        )
    ) as client:
        with pytest.raises(KnowledgeSourceError):
            fetch_stream(source(), policy, client=client)


@pytest.mark.parametrize(
    "mode", ["chunked", "false-length", "gzip", "mime", "empty", "utf8", "connect", "read", "wall"]
)
def test_stream_media_size_and_time_bounds(mode):
    import httpx

    from gpu_agent.knowledge.ingest import fetch_stream, load_manifest
    from gpu_agent.knowledge.models import KnowledgeError

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield body[:20]
            yield body[20:]

    body = fixture_html() + b"x" * 512
    headers = {"content-type": "text/html"}
    if mode == "gzip":
        headers["content-encoding"] = "gzip"
        body = gzip.compress(body)
    elif mode == "false-length":
        headers["content-length"] = "1"
    elif mode == "mime":
        headers["content-type"] = "application/pdf"
    elif mode == "empty":
        body = b""
    elif mode == "utf8":
        body = b"\xff"

    def handler(request):
        if mode == "connect":
            raise httpx.ConnectTimeout("test")
        if mode == "read":
            raise httpx.ReadTimeout("test")
        return httpx.Response(200, headers=headers, stream=Stream())

    policy = load_manifest(ROOT / "knowledge/sources.json").fetch_policy
    ticks = iter([0.0, 100.0, 100.0, 100.0]) if mode == "wall" else None
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(KnowledgeError):
            fetch_stream(
                source().model_copy(update={"max_bytes": 256}),
                policy,
                client=client,
                clock=(lambda: next(ticks)) if ticks else None,
            )


def test_tracked_corpus_boundary():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    assert not any(p.startswith(".cache/gpu-agent/knowledge/") for p in tracked)
    assert not any(p.startswith("knowledge/") and p != "knowledge/sources.json" for p in tracked)
    ignored = subprocess.run(
        ["git", "check-ignore", ".cache/gpu-agent/knowledge/index.json"],
        cwd=ROOT,
        capture_output=True,
    )
    assert ignored.returncode == 0


def test_content_hash_normalization():
    from gpu_agent.knowledge.models import normalize_text

    assert normalize_text("e\u0301\r\n a  b") == "é a b"
    assert hashlib.sha256(normalize_text("A  B").encode()).hexdigest() == (
        "fea4c5ce720c1d6a1cbc47c1607cc4ea172a69de8948e76d67910120597950fc"
    )


def test_cache_version_metadata_tampering_is_corrupt(index, tmp_path):
    from gpu_agent.knowledge.models import KnowledgeCorruptError
    from gpu_agent.knowledge.retrieve import KnowledgeIndex

    path = tmp_path / "index.json"
    index.save(path)
    data = json.loads(path.read_text())
    data["chunks"][0]["compatibility"]["cuda"] = "==11.0"
    path.write_text(json.dumps(data))
    with pytest.raises(KnowledgeCorruptError):
        KnowledgeIndex.load(path)


@pytest.mark.parametrize(
    "location",
    [
        "../cuda-c-programming-guide/index.html",
        "\nindex.html",
        "index.html\t",
        "%2e%2e/cuda-c-programming-guide/index.html",
        "/cuda/archive/12.8.1/other/index.html",
    ],
)
def test_relative_redirect_is_validated_before_join(location):
    import httpx

    from gpu_agent.knowledge.ingest import fetch_stream, load_manifest
    from gpu_agent.knowledge.models import KnowledgeSourceError

    responses = iter(
        [
            httpx.Response(302, headers={"location": location}),
            httpx.Response(
                200, headers={"content-type": "text/html"}, stream=httpx.ByteStream(fixture_html())
            ),
        ]
    )
    policy = load_manifest(ROOT / "knowledge/sources.json").fetch_policy
    with httpx.Client(transport=httpx.MockTransport(lambda _: next(responses))) as client:
        with pytest.raises(KnowledgeSourceError):
            fetch_stream(source(), policy, client=client)


def test_second_redirect_hop_is_rejected():
    import httpx

    from gpu_agent.knowledge.ingest import fetch_stream, load_manifest
    from gpu_agent.knowledge.models import KnowledgeSourceError

    responses = iter(
        [
            httpx.Response(302, headers={"location": "index.html"}),
            httpx.Response(307, headers={"location": "https://developer.nvidia.com/index.html"}),
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(lambda _: next(responses))) as client:
        with pytest.raises(KnowledgeSourceError):
            fetch_stream(
                source(), load_manifest(ROOT / "knowledge/sources.json").fetch_policy, client=client
            )


@pytest.mark.parametrize("mutation", ["unpin", "relabel", "broad-path"])
def test_rolling_manual_cannot_bypass_approval(mutation, tmp_path):
    from gpu_agent.knowledge.ingest import load_manifest
    from gpu_agent.knowledge.models import KnowledgeSourceError

    data = json.loads((ROOT / "knowledge/sources.json").read_text())
    manual = next(s for s in data["sources"] if s["source_id"] == "sanitizer-approved-memcheck")
    if mutation == "unpin":
        manual["chunk_strategy"] = "heading_blocks"
        manual["approved_chunk_sha256"] = {}
    elif mutation == "relabel":
        manual["document_version"] = "2025.1"
    else:
        data["sources"][0]["allowed_path_regex"] = "^/.*$"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(KnowledgeSourceError):
        load_manifest(path)


def stalled_fetch_worker(connection, source, policy):
    import time

    time.sleep(10)


def test_process_wall_deadline_cancels_stalled_worker(monkeypatch):
    import time

    from gpu_agent.knowledge import ingest
    from gpu_agent.knowledge.models import KnowledgeTimeoutError

    monkeypatch.setattr(ingest, "_fetch_worker", stalled_fetch_worker)
    policy = ingest.load_manifest(ROOT / "knowledge/sources.json").fetch_policy.model_copy(
        update={"wall_timeout_seconds": 0.2}
    )
    started = time.monotonic()
    with pytest.raises(KnowledgeTimeoutError):
        ingest.fetch(source(), policy)
    assert time.monotonic() - started < 2


def test_ingest_cumulative_size_limit(monkeypatch):
    from gpu_agent.knowledge import ingest
    from gpu_agent.knowledge.models import KnowledgeSourceError

    manifest = ingest.load_manifest(ROOT / "knowledge/sources.json")
    size = len(fixture_html())
    manifest = manifest.model_copy(
        update={
            "sources": [source(), source().model_copy(update={"source_id": "second"})],
            "fetch_policy": manifest.fetch_policy.model_copy(update={"max_total_bytes": size}),
        }
    )
    monkeypatch.setattr(
        ingest,
        "fetch",
        lambda s, p: ingest.Fetched(
            body=fixture_html(),
            effective_url=URL,
            retrieved_at="2026-09-15T00:00:00Z",
            wire_bytes=size,
        ),
    )
    with pytest.raises(KnowledgeSourceError):
        ingest.ingest(manifest)


def test_pinned_source_hash_and_release_anchor():
    from gpu_agent.knowledge.ingest import extract_chunks, load_manifest
    from gpu_agent.knowledge.models import KnowledgeIntegrityError, KnowledgeSourceError

    manifest = load_manifest(ROOT / "knowledge/sources.json")
    sample = next(s for s in manifest.sources if s.chunk_strategy == "vector_add")
    with pytest.raises(KnowledgeIntegrityError):
        extract_chunks(sample, b"changed source", "2026-09-15T00:00:00Z")
    release = next(s for s in manifest.sources if s.source_id == "sanitizer-release-evidence")
    with pytest.raises(KnowledgeSourceError):
        extract_chunks(
            release,
            b"<title>Release Notes</title><main>No release anchor</main>",
            "2026-09-15T00:00:00Z",
            verified_version="13.4",
        )


def test_bm25_term_frequency_and_length_normalization(index):
    from gpu_agent.knowledge.models import make_chunk
    from gpu_agent.knowledge.retrieve import KnowledgeIndex

    base = index.chunks[0].model_dump(exclude={"chunk_id", "content_hash"})
    base["section_title"] = "Common"
    documents = []
    for i, text in enumerate(["needle filler", "needle needle filler", "needle " + "filler " * 80]):
        documents.append(make_chunk(**{**base, "text": text, "block_ordinal": i}))
    result = KnowledgeIndex(documents).retrieve("needle", VERSION)
    assert [c.block_ordinal for c in result.chunks] == [1, 0, 2]


def test_version_mentions_in_body_are_not_version_evidence():
    from gpu_agent.knowledge.ingest import extract_chunks
    from gpu_agent.knowledge.models import KnowledgeVersionUnavailableError

    with pytest.raises(KnowledgeVersionUnavailableError):
        extract_chunks(
            source(),
            fixture_html("<p>Old v12.8 reference.</p>", version="12.0.0"),
            "2026-09-15T00:00:00Z",
        )


def test_lock_covers_declared_dependencies_and_runtime_closure():
    import importlib.metadata
    import tomllib

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    lines = [
        line
        for line in (ROOT / "requirements.lock").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    locked = {}
    for line in lines:
        requirement = Requirement(line)
        assert requirement.url is None and str(requirement.specifier).startswith("==")
        locked[canonicalize_name(requirement.name)] = next(iter(requirement.specifier)).version
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    pending = [Requirement(value) for value in project["dependencies"]]
    visited = set()
    while pending:
        requirement = pending.pop()
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(requirement.name)
        assert name in locked, f"Missing dependency from install lock: {name}"
        assert locked[name] in requirement.specifier, f"Incompatible locked version: {name}"
        if name not in visited:
            visited.add(name)
            metadata = importlib.metadata.requires(requirement.name) or []
            pending.extend(Requirement(value) for value in metadata)


@pytest.mark.parametrize("escape, replacement", [(r"\d", "2"), (r"\w", "a"), (r"\w", "2")])
def test_regex_escape_cannot_expand_exact_manifest_path(escape, replacement):
    from gpu_agent.knowledge.ingest import validate_url
    from gpu_agent.knowledge.models import KnowledgeSourceError

    original = source()
    modified = original.model_copy(
        update={
            "allowed_path_regex": original.allowed_path_regex.replace(
                r"12\.8\.1", r"12\.8\." + escape
            )
        }
    )
    assert validate_url(URL, modified) == URL
    with pytest.raises(KnowledgeSourceError):
        validate_url(URL.replace("12.8.1", "12.8." + replacement), modified)


@pytest.mark.parametrize(
    "mutation",
    ["substitute-probe", "missing-anchor", "wrong-anchor", "wrong-strategy", "wrong-url"],
)
def test_release_evidence_is_bound_to_official_target_notes(mutation, tmp_path):
    from gpu_agent.knowledge.ingest import load_manifest
    from gpu_agent.knowledge.models import KnowledgeSourceError

    data = json.loads((ROOT / "knowledge/sources.json").read_text())
    manual = next(s for s in data["sources"] if s["source_id"] == "sanitizer-approved-memcheck")
    release = next(s for s in data["sources"] if s["source_id"] == "sanitizer-release-evidence")
    if mutation == "substitute-probe":
        data["sources"].remove(release)
        manual["release_evidence_source"] = "sanitizer-version-probe"
    elif mutation == "missing-anchor":
        release["include_anchors"] = []
    elif mutation == "wrong-anchor":
        release["include_anchors"] = ["updates-in-2025-4"]
    elif mutation == "wrong-strategy":
        release["chunk_strategy"] = "heading_blocks"
    else:
        release["canonical_url"] = release["canonical_url"].replace("ReleaseNotes", "OtherNotes")
        release["allowed_path_regex"] = release["allowed_path_regex"].replace(
            "ReleaseNotes", "OtherNotes"
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(KnowledgeSourceError):
        load_manifest(path)


def test_ingest_revalidates_mutated_release_reference_before_fetch(monkeypatch):
    from gpu_agent.knowledge import ingest
    from gpu_agent.knowledge.models import KnowledgeSourceError

    manifest = ingest.load_manifest(ROOT / "knowledge/sources.json")
    manual = next(s for s in manifest.sources if s.source_id == "sanitizer-approved-memcheck")
    manifest.sources[manifest.sources.index(manual)] = manual.model_copy(
        update={"release_evidence_source": "sanitizer-version-probe"}
    )

    def unexpected_fetch(*args):
        pytest.fail("Invalid manifest must be rejected before network access")

    monkeypatch.setattr(ingest, "fetch", unexpected_fetch)
    with pytest.raises(KnowledgeSourceError):
        ingest.ingest(manifest)
