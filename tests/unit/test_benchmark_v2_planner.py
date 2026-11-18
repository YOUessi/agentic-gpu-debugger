"""Offline, synthetic fixtures for the candidate-selection gate (no CUDA/LLM)."""

import copy
import json
from collections import Counter

import pytest

from scripts.plan_debugger_benchmark_v2 import (
    build_selection,
    git_blob_sha1,
    parse_catalog,
    plan_candidates,
)

CATEGORIES = ("compile_error", "logic_error", "memory_crash", "perf_broken", "timeout")


def catalog_and_policy():
    items = []
    for i, category in enumerate(CATEGORIES):
        for j in range(5):
            stem = f"cuda_{i:02d}{j:02d}"
            suffix = f"{i * 5 + j + 1:08x}"
            name = f"{stem}__{category}__{suffix}"
            items.append(
                {
                    "id": name,
                    "path": f"data/{name}/",
                    "task_family": "cuda" if j < 3 else "kernelbench",
                    "error_category": category,
                    "difficulty": "hard" if j == 4 else "medium",
                    "broken_start_lines": 40 + i * 10 + j,
                    "broken_start_hash": suffix + "0" * 8,
                }
            )
    data = {
        "schema_version": "neurips2026_v1",
        "n_instances": len(items),
        "instances": items,
        "task_families": dict(Counter(e["task_family"] for e in items)),
        "error_category_distribution": dict(Counter(e["error_category"] for e in items)),
    }
    raw = json.dumps(data, sort_keys=True).encode()
    policy = {
        "schema_version": "debugger-benchmark-v2-plan-1",
        "upstream": {
            "repository": "UMN-APEX-Lab/CUDABeaver",
            "commit_sha": "a" * 40,
            "manifest_git_blob_sha1": git_blob_sha1(raw),
            "expected_instances": len(items),
            "expected_manifest_schema": "neurips2026_v1",
            "quarantined_instance_ids": [],
        },
        "selection": {
            "seed": 20261119,
            "development_quotas": {category: 1 for category in CATEGORIES},
            "frozen_public_eval_candidate_quotas": {category: 2 for category in CATEGORIES},
            "group_disjoint_by": "base_stem_before_double_underscore_category",
        },
    }
    return data, raw, policy


def test_pinned_catalog_and_group_disjoint_splits_are_reproducible():
    _, raw, policy = catalog_and_policy()
    result = build_selection(raw, policy)
    again = build_selection(raw, policy)
    assert again == result
    assert result["counts"] == {
        "development": 5,
        "frozen_public_eval_candidates": 10,
    }
    assert result["status"] == "CANDIDATE_METADATA_ONLY_NOT_GPU_VALIDATED"
    selected = [c for group in result["splits"].values() for c in group]
    assert len({c["base_stem"] for c in selected}) == len(selected)
    assert all(c["candidate_only"] for c in selected)
    assert all(c["execution_eligibility"] == "PENDING_NATIVE_PREFLIGHT" for c in selected)
    assert all(
        Counter(e["error_category"] for e in group) == quotas
        for group, quotas in (
            (
                result["splits"]["development"],
                policy["selection"]["development_quotas"],
            ),
            (
                result["splits"]["frozen_public_eval_candidates"],
                policy["selection"]["frozen_public_eval_candidate_quotas"],
            ),
        )
    )
    serialized = json.dumps(result).lower()
    for forbidden in ("reference.cu", "ground_truth", "answer_key", "build_command"):
        assert forbidden not in serialized


def test_order_of_metadata_rows_does_not_change_selection():
    data, raw, policy = catalog_and_policy()
    before = build_selection(raw, policy)
    shuffled = copy.deepcopy(data)
    shuffled["instances"].reverse()
    new_raw = json.dumps(shuffled, sort_keys=True).encode()
    policy["upstream"]["manifest_git_blob_sha1"] = git_blob_sha1(new_raw)
    after = build_selection(new_raw, policy)
    assert before["splits"] == after["splits"]


@pytest.mark.parametrize("mutation", ["count", "path", "identity", "duplicated", "family"])
def test_tampered_manifest_fails_closed(mutation):
    data, _, policy = catalog_and_policy()
    if mutation == "count":
        data["n_instances"] -= 1
    elif mutation == "path":
        data["instances"][0]["path"] = "../exfiltrate.json"
    elif mutation == "identity":
        data["instances"][0]["broken_start_hash"] = "f" * 16
    elif mutation == "duplicated":
        data["instances"][1] = copy.deepcopy(data["instances"][0])
        data["error_category_distribution"] = dict(
            Counter(e["error_category"] for e in data["instances"])
        )
        data["task_families"] = dict(Counter(e["task_family"] for e in data["instances"]))
    else:
        data["task_families"]["cuda"] += 1
    raw = json.dumps(data, sort_keys=True).encode()
    policy["upstream"]["manifest_git_blob_sha1"] = git_blob_sha1(raw)
    with pytest.raises(ValueError):
        parse_catalog(raw, policy)


def test_mutated_blob_is_rejected_even_if_metadata_looks_valid():
    data, raw, policy = catalog_and_policy()
    mutated = raw + b"\n"
    assert mutated != raw
    with pytest.raises(ValueError, match="pinned Git blob"):
        parse_catalog(mutated, policy)


def test_group_duplicates_cannot_leak_between_splits():
    _, raw, policy = catalog_and_policy()
    items = parse_catalog(raw, policy)
    dupe = dict(items[1])
    dupe["id"] = "cuda_0000__logic_error__ffffffff"
    dupe["base_stem"] = items[0]["base_stem"]
    items.append(dupe)
    selected = plan_candidates(items, policy)
    all_cases = [x for group in selected.values() for x in group]
    stems = [x["base_stem"] for x in all_cases]
    assert len(stems) == len(set(stems))


def test_invalid_quotas_or_seed_are_rejected():
    _, raw, policy = catalog_and_policy()
    items = parse_catalog(raw, policy)
    policy["selection"]["seed"] = "not-an-integer"
    with pytest.raises(ValueError, match="seed"):
        plan_candidates(items, policy)
    policy["selection"]["seed"] = 123
    policy["selection"]["development_quotas"] = {"compile_error": 5}
    with pytest.raises(ValueError, match="quotas"):
        plan_candidates(items, policy)


def test_exact_quarantine_is_reported_without_falsifying_universe_count():
    data, _, policy = catalog_and_policy()
    bad_id = data["instances"][0]["id"]
    data["instances"][0]["broken_start_hash"] = "manual_v1"
    raw = json.dumps(data, sort_keys=True).encode()
    policy["upstream"]["manifest_git_blob_sha1"] = git_blob_sha1(raw)
    policy["upstream"]["quarantined_instance_ids"] = [bad_id]
    selected = build_selection(raw, policy)
    assert selected["universe_count"] == 25
    assert selected["metadata_eligible_count"] == 24
    assert selected["quarantined_instance_ids"] == [bad_id]
    assert bad_id not in [c["id"] for part in selected["splits"].values() for c in part]
    policy["upstream"]["quarantined_instance_ids"] = []
    with pytest.raises(ValueError, match="unexpected upstream source-hash"):
        parse_catalog(raw, policy)


def test_quarantine_must_match_actual_snapshot_anomalies():
    data, raw, policy = catalog_and_policy()
    policy["upstream"]["quarantined_instance_ids"] = [data["instances"][0]["id"]]
    with pytest.raises(ValueError, match="quarantine allowlist"):
        parse_catalog(raw, policy)
