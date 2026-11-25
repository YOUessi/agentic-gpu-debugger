#!/usr/bin/env python3
"""Read-only analysis of complete public same-task, same-replicate comparisons.

No private evaluator data, network, paid calls, GPU or trial replay.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

PAIRS = (
    ("native-react", "legacy-react"),
    ("native-dag", "legacy-planner"),
    ("native-dag", "native-react"),
)
CASES = ("h001", "h002", "h003")


def mcnemar_exact(first_only: int, second_only: int) -> float | None:
    n = first_only + second_only
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(first_only, second_only) + 1))
    return min(1.0, 2 * tail / (2**n))


def analyze(root: Path) -> dict:
    manifest_file = root / "manifest.json"
    manifest = json.loads(manifest_file.read_bytes())
    cases = tuple(manifest["cases"])
    if set(cases) != set(CASES) or len(manifest["trials"]) != 60:
        raise ValueError("expected frozen 60-trial development comparison")
    rows = {}
    for trial in manifest["trials"]:
        key = (trial["replicate"], trial["case"], trial["arm"])
        path = root / "trials" / trial["id"] / "result.json"
        if path.is_symlink() or not path.is_file() or key in rows:
            raise ValueError("missing, duplicated or unsafe trial result")
        row = json.loads(path.read_bytes())
        if row["trial"] != trial:
            raise ValueError("trial metadata does not match frozen manifest")
        rows[key] = row
    paired = {}
    for a, b in PAIRS:
        counts = collections.Counter()
        by_case = {}
        for case in cases:
            local = collections.Counter()
            for rep in range(1, manifest["repetitions"] + 1):
                success_a = bool(rows[(rep, case, a)]["strict_fix_within_budget"])
                success_b = bool(rows[(rep, case, b)]["strict_fix_within_budget"])
                key = (success_a, success_b)
                local[key] += 1
                counts[key] += 1
            by_case[case] = {
                "both_success": local[(True, True)],
                "only_first": local[(True, False)],
                "only_second": local[(False, True)],
                "both_failed": local[(False, False)],
            }
        first_only, second_only = counts[(True, False)], counts[(False, True)]
        paired[f"{a}:vs:{b}"] = {
            "both_success": counts[(True, True)],
            "only_first": first_only,
            "only_second": second_only,
            "both_failed": counts[(False, False)],
            "discordant": first_only + second_only,
            "exact_two_sided_mcnemar_p_exploratory": mcnemar_exact(first_only, second_only),
            "by_case": by_case,
        }
    arms = {}
    for arm in ("legacy-react", "native-react", "legacy-planner", "native-dag"):
        group = [row for key, row in rows.items() if key[2] == arm]
        arms[arm] = {
            "started": len(group),
            "strict_success": sum(bool(row["strict_fix_within_budget"]) for row in group),
            "tokens": sum(row["usage"].get("tokens_lower_bound", 0) for row in group),
            "model_calls": sum(row["usage"].get("physical_calls", 0) for row in group),
            "status_counts": dict(collections.Counter(row["status"] for row in group)),
        }
    dag_failures = {}
    for trial in manifest["trials"]:
        if trial["arm"] != "native-dag":
            continue
        row = rows[(trial["replicate"], trial["case"], trial["arm"])]
        if row["strict_fix_within_budget"]:
            continue
        reason = (row.get("public_report") or {}).get("reason")
        if reason == "GRAPH_TASK_FRONTIER_BLOCKED":
            classification, role = "RUNTIME_DAG_SCHEDULER_FAILURE", "scheduler"
        elif reason == "GRAPH_PLAN_INVALID_SCOPE":
            classification, role = "PLAN_SCHEMA_SCOPE_FAILURE", "planner"
        elif row["status"] == "MODEL_FAILURE" and not row.get("public_report"):
            classification = "MODEL_OUTPUT_VALIDATION_AND_TERMINALIZATION_FAILURE"
            # Historical role is attested separately in the provider audit;
            # there is no complete outcome to infer from this JSON alone.
            role = "inspect_provider_invocations"
        else:
            classification, role = "OTHER_UNRESOLVED", "unknown"
        dag_failures[trial["id"]] = {
            "classification": classification,
            "role": role,
            "original_reason": reason,
            "original_status": row["status"],
            "original_result_present": row.get("public_report") is not None,
        }
    return {
        "schema": "gpu-agent-paired-architecture-analysis-v1",
        "frozen_manifest_file_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "frozen_source_commit": manifest["code_sha"],
        "population": "three development CUDA defects, 5 repetitions each; not holdout",
        "success": "independent strict verdict per original fixed trial; no reruns",
        "paired": paired,
        "arms": arms,
        "original_native_dag_failed_trials": dag_failures,
        "limitations": [
            "The sample contains only 15 observations per arm and three unique task defects.",
            "Pairing controls task/replicate identity but does not guarantee identical model RNG.",
            "A small exploratory p-value is not evidence of broad runtime superiority.",
            "Runtime, context, and role allocations may all affect token use; no causal DAG claim.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.comparison_root)
    if args.output.exists():
        raise ValueError("paired output must be new and immutable")
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "paired": result["paired"],
                "failure_count": len(result["original_native_dag_failed_trials"]),
            },
            ensure_ascii=False,
        )
    )
