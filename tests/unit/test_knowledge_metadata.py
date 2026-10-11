import hashlib
import json
from pathlib import Path

import pytest

from gpu_agent.knowledge.ingest import load_manifest
from gpu_agent.knowledge.metadata import lookup_metadata

MANIFEST = Path(__file__).resolve().parents[2] / "knowledge/sources.json"


def receipt_file(tmp_path, **updates):
    source = load_manifest(MANIFEST).sources[0]
    entry = dict(
        source_id=source.source_id,
        effective_url=source.canonical_url,
        retrieved_at="2026-09-29T00:00:00Z",
        body_sha256="a" * 64,
        byte_count=123,
        wire_bytes=100,
        document_version=source.document_version,
    )
    entry.update(updates)
    data = json.dumps([entry]).encode()
    path = tmp_path / "receipts.json"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest(), source.source_id


def test_lookup_is_read_only_and_missing_receipt_stays_unknown(tmp_path):
    path, digest, sid = receipt_file(tmp_path)
    original = path.read_bytes()
    result = lookup_metadata(MANIFEST, path, digest)
    entries = result["entries"]
    assert entries[0]["source_id"] == sid
    assert entries[0]["status"] == "RECORDED_FETCH"
    assert entries[0]["citation"].startswith("sha256:" + digest)
    assert all(e["status"] == "NO_FETCH_RECEIPT" for e in entries[1:])
    assert path.read_bytes() == original
    assert lookup_metadata(MANIFEST, path, digest, sid)["entries"] == entries[:1]


@pytest.mark.parametrize(
    "updates",
    [
        {"document_version": "999"},
        {"effective_url": "https://example.org/other"},
        {"source_id": "unlisted"},
        {"retrieved_at": "2026-09-29"},
        {"body_sha256": "invalid"},
        {"byte_count": 0},
    ],
)
def test_invalid_receipt_rejected(tmp_path, updates):
    path, digest, _ = receipt_file(tmp_path, **updates)
    with pytest.raises(ValueError):
        lookup_metadata(MANIFEST, path, digest)


def test_pin_and_duplicates_and_unknown_query_rejected(tmp_path):
    path, digest, _ = receipt_file(tmp_path)
    with pytest.raises(ValueError, match="hash"):
        lookup_metadata(MANIFEST, path, "0" * 64)
    with pytest.raises(ValueError, match="unknown source"):
        lookup_metadata(MANIFEST, path, digest, "unknown")
    path.write_text(json.dumps(json.loads(path.read_text()) * 2))
    with pytest.raises(ValueError, match="duplicate"):
        lookup_metadata(MANIFEST, path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_cli_metadata_uses_real_lookup_and_refuses_bad_pin(tmp_path):
    from typer.testing import CliRunner

    from gpu_agent.cli import app

    path, digest, sid = receipt_file(tmp_path)
    command = [
        "knowledge",
        "metadata",
        "--manifest",
        str(MANIFEST),
        "--receipts",
        str(path),
        "--source-id",
        sid,
        "--expected-receipts-sha256",
    ]
    ok = CliRunner().invoke(app, command + [digest])
    assert ok.exit_code == 0, ok.output
    assert json.loads(ok.stdout)["entries"][0]["status"] == "RECORDED_FETCH"
    bad = CliRunner().invoke(app, command + ["0" * 64])
    assert bad.exit_code == 2
    assert "KNOWLEDGE_METADATA_INVALID" in bad.output
