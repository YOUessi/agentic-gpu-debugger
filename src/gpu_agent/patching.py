"""Fail-closed M1 unified diffs over immutable, hash-checked source snapshots."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from gpu_agent.contracts import new_id, now
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.store import read_regular, reject_symlinks


class SourceSnapshot(ExecutionModel):
    parent_run_id: str
    root: Path
    hashes: dict[str, str]


class PatchCandidate(ExecutionModel):
    patch_id: str = Field(default_factory=new_id, pattern=r"^[a-f0-9]{32}$")
    parent_run_id: str
    base_source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    patched_source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    unified_diff: str
    generated_by: Literal["human", "agent"] = "human"
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    created_at: datetime = Field(default_factory=now)
    allowed_paths: list[str]
    scope_validation: Literal["VALID"] = "VALID"


def source_hash(sources: dict[str, bytes]) -> str:
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def read_snapshot(snapshot: SourceSnapshot) -> dict[str, bytes]:
    reject_symlinks(snapshot.root)
    if "kernel.cu" not in snapshot.hashes:
        raise ValueError("missing kernel snapshot")
    sources = {}
    for name, expected in snapshot.hashes.items():
        if name not in {"kernel.cu", "vector_io.cpp", "vector_api.h", "json.hpp"}:
            raise ValueError("unsupported snapshot path")
        data = read_regular(snapshot.root / name, 4 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("base hash mismatch")
        sources[name] = data
    return sources


def _apply(source: bytes, diff: str) -> tuple[bytes, dict[int, int | None]]:
    if len(diff.encode()) > 4 * 1024 * 1024 or "\x00" in diff or "\r" in diff:
        raise ValueError("unsupported patch encoding or size")
    lines = diff.splitlines(keepends=True)
    blob_hashes: tuple[str, str] | None = None
    if lines[:1] == ["diff --git a/kernel.cu b/kernel.cu\n"]:
        lines = lines[1:]
        if lines:
            index_header = re.fullmatch(
                r"index ([a-f0-9]{7,40})\.\.([a-f0-9]{7,40})(?: 100644)?\n", lines[0]
            )
            if index_header:
                blob_hashes = index_header[1], index_header[2]
                lines = lines[1:]
    if lines[:2] != ["--- a/kernel.cu\n", "+++ b/kernel.cu\n"]:
        raise ValueError("only existing kernel.cu unified diffs are supported")
    original = source.decode("utf-8").splitlines(keepends=True)
    if not original or any(not line.endswith("\n") for line in original):
        raise ValueError("source must use complete LF lines")
    output: list[str] = []
    mapping: dict[int, int | None] = {}
    cursor, index, hunks = 0, 2, 0
    while index < len(lines):
        match = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n", lines[index])
        if not match:
            raise ValueError("unsupported hunk header")
        old_start, old_count, new_start, new_count = (
            int(match[1]),
            int(match[2] or 1),
            int(match[3]),
            int(match[4] or 1),
        )
        start = old_start - 1 if old_count else old_start
        if start < cursor or start > len(original):
            raise ValueError("overlapping or out of range hunk")
        for pos in range(cursor, start):
            mapping[pos + 1] = len(output) + 1
            output.append(original[pos])
        cursor = start
        if new_start != len(output) + (1 if new_count else 0):
            raise ValueError("new hunk offset mismatch")
        index += 1
        removed = added = 0
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if not line.endswith("\n") or line[:1] not in {" ", "+", "-"}:
                raise ValueError("unsupported hunk content")
            prefix, text = line[0], line[1:]
            if prefix in {" ", "-"}:
                if cursor >= len(original) or original[cursor] != text:
                    raise ValueError("hunk context mismatch")
                mapping[cursor + 1] = len(output) + 1 if prefix == " " else None
                cursor += 1
                removed += 1
            if prefix in {" ", "+"}:
                output.append(text)
                added += 1
            index += 1
        if (removed, added) != (old_count, new_count):
            raise ValueError("hunk line count mismatch")
        hunks += 1
    for pos in range(cursor, len(original)):
        mapping[pos + 1] = len(output) + 1
        output.append(original[pos])
    patched = "".join(output)
    if not hunks or patched.encode() == source:
        raise ValueError("empty candidate")
    if blob_hashes:
        for data, expected in zip((source, patched.encode()), blob_hashes, strict=True):
            blob = b"blob " + str(len(data)).encode() + b"\0" + data
            if not hashlib.sha1(blob, usedforsecurity=False).hexdigest().startswith(expected):
                raise ValueError("git blob hash mismatch")

    # Include directives are controller-owned, including macro includes and line splices.
    def directives(text: str) -> list[str]:
        text = text.replace("\\\n", "")
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        return re.findall(r"^\s*(?:#|%:)\s*(?:include\w*|import)\b[^\n]*", text, re.MULTILINE)

    if directives(patched) != directives(source.decode()):
        raise ValueError("include directives cannot change")
    return patched.encode(), mapping


def apply_candidate(
    source_snapshot: SourceSnapshot, diff: str, allowed_paths: list[str]
) -> PatchCandidate:
    if allowed_paths != ["kernel.cu"]:
        raise ValueError("M1 allows only kernel.cu")
    sources = read_snapshot(source_snapshot)
    base_hash = source_hash(sources)
    sources["kernel.cu"], _ = _apply(sources["kernel.cu"], diff)
    return PatchCandidate(
        parent_run_id=source_snapshot.parent_run_id,
        base_source_hash=base_hash,
        patched_source_hash=source_hash(sources),
        unified_diff=diff,
        allowed_paths=allowed_paths,
    )


def materialize_candidate(snapshot: SourceSnapshot, candidate: PatchCandidate) -> dict[str, bytes]:
    checked = apply_candidate(snapshot, candidate.unified_diff, candidate.allowed_paths)
    if (
        checked.base_source_hash != candidate.base_source_hash
        or checked.patched_source_hash != candidate.patched_source_hash
        or candidate.parent_run_id != snapshot.parent_run_id
    ):
        raise ValueError("candidate provenance mismatch")
    sources = read_snapshot(snapshot)
    sources["kernel.cu"], _ = _apply(sources["kernel.cu"], candidate.unified_diff)
    return sources


def candidate_line_map(
    snapshot: SourceSnapshot, candidate: PatchCandidate
) -> dict[int, int | None]:
    materialize_candidate(snapshot, candidate)
    return _apply(read_snapshot(snapshot)["kernel.cu"], candidate.unified_diff)[1]
