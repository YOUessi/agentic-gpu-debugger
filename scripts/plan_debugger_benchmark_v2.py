"""Plan a source-pinned, group-disjoint CUDABeaver benchmark candidate set.

This tool only reads upstream *metadata*. It neither fetches nor executes
untrusted CUDA code, build commands or testbench scripts. Its output is a
candidate list, not validated cases, an independent holdout, or a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

CATEGORIES = frozenset({"compile_error", "logic_error", "memory_crash", "perf_broken", "timeout"})
DIFFICULTIES = frozenset({"easy", "medium", "hard"})
_ID = re.compile(
    r"^[a-z0-9_]+__(?:compile_error|logic_error|memory_crash|perf_broken|timeout)__[0-9a-f]{8}$"
)
_HASH = re.compile(r"^[0-9a-f]{16}$")


def git_blob_sha1(payload: bytes) -> str:
    """Git SHA-1 object identity (NOT a security signature)."""
    header = b"blob " + str(len(payload)).encode("ascii") + b"\x00"
    return hashlib.sha1(header + payload).hexdigest()


def _read_file(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError("input file absent or symlinked")
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("metadata input exceeds 4 MiB")
    return path.read_bytes()


def parse_catalog(raw: bytes, policy: dict[str, Any]) -> list[dict[str, Any]]:
    pin = policy["upstream"]
    if git_blob_sha1(raw) != pin["manifest_git_blob_sha1"]:
        raise ValueError("upstream metadata does not match the pinned Git blob")
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid manifest JSON") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("instances"), list):
        raise ValueError("manifest instances must be a list")
    items = manifest["instances"]
    if manifest.get("schema_version") != pin.get("expected_manifest_schema"):
        raise ValueError("upstream manifest schema changed")
    if len(items) != pin["expected_instances"] or manifest.get("n_instances") != len(items):
        raise ValueError("manifest instance count changed")
    quarantine_list = pin.get("quarantined_instance_ids", [])
    if (
        not isinstance(quarantine_list, list)
        or any(not isinstance(id_, str) for id_ in quarantine_list)
        or len(set(quarantine_list)) != len(quarantine_list)
    ):
        raise ValueError("invalid quarantine policy")
    quarantined = set(quarantine_list)
    observed_anomalies: set[str] = set()
    ids: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    by_category: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("invalid instance record")
        identifier = item.get("id")
        category = item.get("error_category")
        family = item.get("task_family")
        difficulty = item.get("difficulty")
        digest = item.get("broken_start_hash")
        lines = item.get("broken_start_lines")
        if (
            not isinstance(identifier, str)
            or _ID.fullmatch(identifier) is None
            or identifier in ids
            or not isinstance(category, str)
            or category not in CATEGORIES
            or not isinstance(family, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", family) is None
            or difficulty not in DIFFICULTIES
            or not isinstance(digest, str)
            or not isinstance(lines, int)
            or isinstance(lines, bool)
            or lines <= 0
        ):
            raise ValueError("invalid or duplicate instance metadata")
        base, id_category, id_hash = identifier.rsplit("__", 2)
        if id_category != category or item.get("path") != f"data/{identifier}/":
            raise ValueError("instance metadata and identity disagree")
        # A few pinned upstream records have an ID suffix differing from the
        # declared broken source hash. Never silently admit or erase them:
        # quarantine exactly the pre-registered anomalies before selecting.
        anomalous = _HASH.fullmatch(digest) is None or not digest.startswith(id_hash)
        if anomalous:
            if identifier not in quarantined:
                raise ValueError("unexpected upstream source-hash inconsistency")
            observed_anomalies.add(identifier)
        ids.add(identifier)
        by_category[category] += 1
        by_family[family] += 1
        if not anomalous:
            cleaned.append(
                {
                    "id": identifier,
                    "base_stem": base,
                    "task_family": family,
                    "error_category": category,
                    "difficulty_label": difficulty,
                    "broken_start_lines": lines,
                }
            )
    if dict(by_category) != manifest.get("error_category_distribution"):
        raise ValueError("declared failure category distribution disagrees")
    if dict(by_family) != manifest.get("task_families"):
        raise ValueError("declared task family distribution disagrees")
    if observed_anomalies != quarantined:
        raise ValueError("quarantine allowlist differs from observed upstream anomalies")
    return cleaned


def _quotas(value: Any) -> dict[str, int]:
    if (
        not isinstance(value, dict)
        or set(value) != CATEGORIES
        or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in value.values())
    ):
        raise ValueError("category quotas must cover every failure family")
    return {str(key): int(n) for key, n in value.items()}


def plan_candidates(
    items: list[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Prefer diverse task families, then upstream hard cases, then a seeded tie."""
    config = policy["selection"]
    seed = config["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("selection seed must be an integer")
    evaluation_quota = _quotas(config["frozen_public_eval_candidate_quotas"])
    development_quota = _quotas(config["development_quotas"])
    used_bases: set[str] = set()
    family_counts: Counter[str] = Counter()

    def take(name: str, quotas: dict[str, int]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        # Rare categories are allocated first. This order is frozen pre-execution.
        counts = Counter(e["error_category"] for e in items)
        for category in sorted(CATEGORIES, key=lambda c: (counts[c], c)):
            for _ in range(quotas[category]):
                available = [
                    e
                    for e in items
                    if e["error_category"] == category and e["base_stem"] not in used_bases
                ]
                if not available:
                    raise ValueError("insufficient unique task stems for requested quotas")
                chosen = min(
                    available,
                    key=lambda e: (
                        family_counts[e["task_family"]],
                        0 if e["difficulty_label"] == "hard" else 1,
                        hashlib.sha256(f"{seed}:{name}:{e['id']}".encode()).hexdigest(),
                    ),
                )
                used_bases.add(chosen["base_stem"])
                family_counts[chosen["task_family"]] += 1
                result.append(
                    {
                        **chosen,
                        "execution_eligibility": "PENDING_NATIVE_PREFLIGHT",
                        "candidate_only": True,
                    }
                )
        # Deterministic but opaque ordering; it is NOT an LLM action schedule.
        return sorted(
            result,
            key=lambda e: hashlib.sha256(f"order:{seed}:{name}:{e['id']}".encode()).hexdigest(),
        )

    # Freeze the evaluation *candidates* before selecting development examples.
    evaluation = take("frozen_public_eval_candidates", evaluation_quota)
    development = take("development", development_quota)
    if len({e["base_stem"] for e in evaluation + development}) != len(evaluation + development):
        raise ValueError("base stem leakage across development/evaluation")
    return {"development": development, "frozen_public_eval_candidates": evaluation}


def build_selection(raw: bytes, policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("schema_version") != "debugger-benchmark-v2-plan-1":
        raise ValueError("unsupported candidate-selection schema")
    if policy.get("selection", {}).get("group_disjoint_by") != (
        "base_stem_before_double_underscore_category"
    ):
        raise ValueError("group split contract changed")
    items = parse_catalog(raw, policy)
    parts = plan_candidates(items, policy)
    return {
        "schema_version": "debugger-benchmark-v2-selection-1",
        "status": "CANDIDATE_METADATA_ONLY_NOT_GPU_VALIDATED",
        "upstream_repo": policy["upstream"]["repository"],
        "upstream_commit": policy["upstream"]["commit_sha"],
        "upstream_manifest_git_blob_sha1": git_blob_sha1(raw),
        "upstream_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "universe_count": policy["upstream"]["expected_instances"],
        "metadata_eligible_count": len(items),
        "quarantined_instance_ids": sorted(policy["upstream"].get("quarantined_instance_ids", [])),
        "seed": policy["selection"]["seed"],
        "counts": {name: len(cases) for name, cases in parts.items()},
        "splits": parts,
        "caveats": [
            "No candidate is certified to run on RTX 4090/Ada or the current sandbox.",
            "Public benchmark content is not an independent private holdout.",
            "Labels, reference implementations and evaluator code are never model context.",
            "No case exclusion or replacement after observing model outcome.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "benchmarks/v2/cudabeaver_selection_policy.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(_read_file(args.policy))
    if not isinstance(policy, dict):
        parser.error("policy must be an object")
    result = build_selection(_read_file(args.manifest), policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Fail closed rather than silently overwrite an already frozen selection.
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n")
    print(
        f"CANDIDATE_SELECTION_ONLY: {result['counts']}; "
        f"manifest_sha256={result['upstream_manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
