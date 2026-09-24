"""Offline calibration of the participation check against recorded GPU verdicts.

Reads only a public run store: candidate runs (candidate.json), their parent run's
sources/kernel.cu, and public verification results. Prints per-candidate check statuses
next to the recorded verdict and synccheck outcome, plus a cross-tab. It prints no source
or diff text and changes nothing. Run it before relying on the check:

    python -m gpu_agent.sync_participation_audit /path/to/public-store > audit.json

A candidate the check would reject (patched MISMATCH) whose recorded synccheck was CLEAN is
a disagreement with the sanitizer and must be reviewed before the check is frozen.
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gpu_agent.patching import _apply
from gpu_agent.sync_participation import analyze


def _artifact(root: Path, manifest: dict[str, Any], name: str) -> bytes | None:
    for ref in manifest.get("artifact_refs", []):
        if ref.get("name") == name:
            relative_path = ref["relative_path"]
            if not isinstance(relative_path, str):
                raise ValueError("artifact path must be a string")
            path: Path = root / relative_path
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("artifact escapes public store")
            return path.read_bytes()
    return None


def _summary(source: str) -> str:
    statuses = {result.status for result in analyze(source)}
    for status in ("MISMATCH", "UNANALYZABLE", "CONSISTENT"):
        if status in statuses:
            return status
    return "NO_SITES"


def audit(root: Path) -> dict[str, Any]:
    manifests: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/manifest.json")):
        manifests[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    verdicts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for manifest in manifests.values():
        if manifest.get("kind") != "verification":
            continue
        raw = _artifact(root, manifest, "verification/result.json")
        if raw is not None:
            result = json.loads(raw)
            key = (manifest.get("parent_run_id") or "", result.get("candidate_hash", ""))
            verdicts.setdefault(key, []).append(result)
    rows: list[dict[str, Any]] = []
    for run_id, manifest in manifests.items():
        if manifest.get("kind") != "candidate":
            continue
        raw = _artifact(root, manifest, "candidate.json")
        parent = manifests.get(manifest.get("parent_run_id") or "")
        source = _artifact(root, parent, "sources/kernel.cu") if parent else None
        if raw is None or source is None:
            continue
        candidate = json.loads(raw)
        patched: str | None
        try:
            patched = _apply(source, candidate["unified_diff"])[0].decode("utf-8")
        except (ValueError, KeyError, UnicodeDecodeError):
            patched = None
        results: list[dict[str, Any] | None] = [
            *verdicts.get(
                (manifest.get("parent_run_id") or "", candidate.get("patched_source_hash", "")), []
            )
        ] or [None]
        for result in results:
            rows.append(
                {
                    "candidate_run_id": run_id,
                    "parent_run_id": manifest.get("parent_run_id"),
                    "original": _summary(source.decode("utf-8", errors="replace")),
                    "patched": _summary(patched) if patched is not None else "NOT_APPLICABLE",
                    "sites": [asdict(site) for site in analyze(patched)] if patched else [],
                    "verdict": result.get("verdict") if result else None,
                    "synccheck": (result.get("check_outcomes") or {}).get("synccheck")
                    if result
                    else None,
                }
            )
    crosstab = Counter(
        f"{row['original']}->{row['patched']} | {row['verdict']} | synccheck={row['synccheck']}"
        for row in rows
    )
    return {
        "advisory_only": True,
        "synccheck_clean_comparisons": sum(row["synccheck"] == "CLEAN" for row in rows),
        "missing_synccheck_outcome": sum(row["synccheck"] is None for row in rows),
        "candidates": rows,
        "crosstab": dict(sorted(crosstab.items())),
        "rejected_but_synccheck_clean": [
            row["candidate_run_id"]
            for row in rows
            if row["patched"] == "MISMATCH" and row["synccheck"] == "CLEAN"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_store", type=Path)
    args = parser.parse_args(argv)
    json.dump(audit(args.public_store), sys.stdout, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
