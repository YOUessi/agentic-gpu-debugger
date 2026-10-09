"""The trajectory audit is scoped, read-only and does not emit source/model text."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def audit_module(monkeypatch):
    tools = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "investigation_audit", tools / "dev_investigation_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step(kind="inspect_source", start=1, end=2, allowed=True):
    evidence = {
        "sanitizer_outcomes": {"memcheck": "FINDING"},
        "tool_findings": [{}],
        "documentation": [{}],
        "sources": [{"source_id": "s", "content": "SECRET SOURCE"}],
    }
    action = {
        "action_type": kind,
        "typed_arguments": {"source_id": "s", "start_line": start, "end_line": end},
        "rationale": "SECRET MODEL RATIONALE",
    }
    return {"evidence": evidence, "action": action}, {
        "action_type": kind,
        "allowed": allowed,
        "reason_codes": [] if allowed else ["DUPLICATE_NO_BENEFIT"],
    }


def test_ranges_and_eligibility_do_not_claim_causal_waste(audit_module):
    pairs = [(i, *step(start=s, end=e)) for i, (s, e) in enumerate([(1, 4), (3, 6), (2, 5)])]
    report = audit_module.summarize_steps(pairs)
    assert report["counts"]["fully_repeated_source_reads"] == 1
    assert report["counts"]["source_reads_with_content_already_supplied"] == 3
    assert report["counts"]["source_reads_with_next_snapshot"] == 2
    assert report["counts"]["source_reads_with_unchanged_next_snapshot"] == 2
    assert report["counts"]["nonterminal_allowed_when_finish_eligible"] == 3
    assert "SECRET" not in json.dumps(report)


def test_denied_action_is_not_counted_as_allowed(audit_module):
    report = audit_module.summarize_steps([(0, *step(allowed=False))])
    assert report["counts"]["denied"] == 1
    assert not report["counts"].get("source_reads_with_content_already_supplied")
    assert report["rejections"] == {"DUPLICATE_NO_BENEFIT": 1}


def test_missing_evidence_matches_frozen_contract(audit_module):
    evidence = step()[0]["evidence"]
    assert audit_module.missing(evidence) == []
    evidence["documentation"] = []
    assert audit_module.missing(evidence) == ["documentation_for_finding"]
    evidence["sanitizer_outcomes"] = {}
    evidence["tool_findings"] = []
    assert audit_module.missing(evidence) == ["memcheck_outcome", "tool_finding"]


@pytest.fixture
def store(tmp_path, audit_module):
    counter = 10000

    def write_run(rid, kind, artifacts, **extra):
        nonlocal counter
        refs = []
        for name, value in artifacts.items():
            counter += 1
            aid = f"{counter:032x}"
            content = json.dumps(value).encode()
            relative = f"{rid}/artifacts/{aid}"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            refs.append(
                {
                    "id": aid,
                    "run_id": rid,
                    "name": name,
                    "relative_path": relative,
                    "byte_count": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        manifest = {
            "id": rid,
            "kind": kind,
            "status": "COMPLETED",
            "artifact_refs": refs,
            "binding": {"repository": {"commit": audit_module.COMMIT}},
            **extra,
        }
        (tmp_path / rid / "manifest.json").write_text(json.dumps(manifest))

    records = []
    for mode in "ABCDE":
        for case in range(1, 17):
            for repeat in range(3):
                rid = f"{len(records) + 1:032x}"
                records.append(
                    {
                        "mode": mode,
                        "case_id": f"case_{case:04d}",
                        "repeat": repeat,
                        "lineage": {"diagnosis_run_id": rid},
                        "usage": {"physical_calls": 0, "sanitizer_calls": 0, "retrieval_calls": 0},
                        "verdict": "VERIFIED_FIXED",
                        "failure_reason": None,
                        "patch_hash": "h",
                        "latency_ms": 1,
                        "diagnosis": {"diagnostic_outcome": "DIAGNOSED"},
                    }
                )
                if mode in "DE":
                    write_run(rid, "diagnosis", {"unused.json": {"text": "SECRET"}})
    eid = "f" * 32
    summary = {
        "run_id": eid,
        "split": "development",
        "commit": audit_module.COMMIT,
        "expected_units": 240,
        "executed_units": 240,
        "stopped_reason": None,
        "records": records,
    }
    schedule = {"split": "development", "bindings": {"commit": audit_module.COMMIT}}
    write_run(
        eid,
        "evaluation",
        {"evaluation/manifest.json": summary, "evaluation/schedule.json": schedule},
    )
    return tmp_path, eid, summary, schedule, write_run


def test_complete_scope_and_no_writes(audit_module, store):
    root, eid, *_ = store
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = audit_module.audit(root, eid)
    assert len(result["units"]) == 96
    assert result["totals"]["D"]["fixed"] == 48
    assert "SECRET" not in json.dumps(result)
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", ["holdout", "commit", "duplicate"])
def test_refuses_wrong_scope(audit_module, store, change):
    root, eid, summary, schedule, write_run = store
    if change == "holdout":
        summary["split"] = "holdout"
    elif change == "commit":
        summary["commit"] = "a" * 40
    else:
        summary["records"][-1] = summary["records"][-2]
    write_run(
        eid,
        "evaluation",
        {"evaluation/manifest.json": summary, "evaluation/schedule.json": schedule},
    )
    with pytest.raises(audit_module.InventoryError):
        audit_module.audit(root, eid)


@pytest.mark.parametrize("damage", ["hash", "symlink"])
def test_refuses_damaged_artifact(audit_module, store, damage):
    root, eid, *_ = store
    path = next((root / eid / "artifacts").iterdir())
    if damage == "hash":
        path.write_bytes(b"{}")
    else:
        dest = root / "linked.json"
        dest.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(dest)
    with pytest.raises(audit_module.InventoryError):
        audit_module.audit(root, eid)
