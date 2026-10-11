"""Offline source/version lookup. Receipts prove recorded fetches, not release contents."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from gpu_agent.knowledge.ingest import FetchReceipt, load_manifest


def lookup_metadata(
    manifest_path: Path,
    receipts_path: Path,
    expected_receipts_sha256: str,
    source_id: str | None = None,
) -> dict[str, object]:
    raw = receipts_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_receipts_sha256:
        raise ValueError("receipt file hash mismatch")
    manifest = load_manifest(manifest_path)
    sources = {s.source_id: s for s in manifest.sources}
    if source_id is not None and source_id not in sources:
        raise ValueError("unknown source ID")
    decoded = json.loads(raw)
    if not isinstance(decoded, list):
        raise ValueError("receipts must be a list")
    receipts = [FetchReceipt.model_validate(r) for r in decoded]
    if len({r.source_id for r in receipts}) != len(receipts):
        raise ValueError("duplicate source receipt")
    records = {}
    for receipt in receipts:
        source = sources.get(receipt.source_id)
        if source is None or receipt.document_version != source.document_version:
            raise ValueError("receipt source/version mismatch")
        if receipt.effective_url != source.canonical_url:
            raise ValueError("receipt URL differs from approved canonical URL")
        if (
            len(receipt.body_sha256) != 64
            or any(c not in "0123456789abcdef" for c in receipt.body_sha256)
            or receipt.byte_count <= 0
            or receipt.wire_bytes <= 0
            or datetime.fromisoformat(receipt.retrieved_at.replace("Z", "+00:00")).tzinfo is None
        ):
            raise ValueError("invalid fetch provenance")
        if source.body_sha256 and source.body_sha256 != receipt.body_sha256:
            raise ValueError("receipt differs from pinned source body")
        records[receipt.source_id] = receipt
    entries = []
    for sid, source in sources.items():
        if source_id is not None and source_id != sid:
            continue
        matched = records.get(sid)
        entries.append(
            {
                "source_id": sid,
                "document_title": source.document_title,
                "declared_document_version": source.document_version,
                "source_url": source.canonical_url,
                "chunk_strategy": source.chunk_strategy,
                "declared_compatibility": source.compatibility,
                "status": "RECORDED_FETCH" if matched else "NO_FETCH_RECEIPT",
                "receipt": matched.model_dump(mode="json") if matched else None,
                "citation": f"sha256:{digest}#source_id={sid}" if matched else None,
            }
        )
    return {
        "schema_version": 1,
        "receipts_sha256": digest,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "entries": entries,
        "limitations": [
            "A caller-pinned local receipt is not a fresh network or body revalidation.",
            "Document version is not the installed Compute Sanitizer version.",
            "Release-note contents cannot be inferred from a receipt; query stored body text.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("--expected-receipts-sha256", required=True)
    parser.add_argument("--source-id")
    args = parser.parse_args()
    print(
        json.dumps(
            lookup_metadata(
                args.manifest, args.receipts, args.expected_receipts_sha256, args.source_id
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
