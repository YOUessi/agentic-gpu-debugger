"""The development failure inventory is read-only, scoped, hash-checked and text-free."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "dev_failure_inventory.py"
_SPEC = importlib.util.spec_from_file_location("dev_failure_inventory", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
inv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inv)

COMMIT = "c" * 40
SECRET_TEXT = "MODEL-ROOT-CAUSE-TEXT"


def _rid(n: int) -> str:
    return f"{n:032x}"


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.counter = 0

    def run(self, run_id: str, kind: str, artifacts: dict[str, bytes], **extra) -> None:
        refs = []
        for name, data in artifacts.items():
            self.counter += 1
            art = f"{self.counter:032x}"
            path = self.root / run_id / "artifacts" / art
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            refs.append(
                {
                    "id": art,
                    "run_id": run_id,
                    "name": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "relative_path": f"{run_id}/artifacts/{art}",
                    "byte_count": len(data),
                }
            )
        manifest = {
            "id": run_id,
            "kind": kind,
            "status": "COMPLETED",
            "binding": {"repository": {"commit": COMMIT}, "runtime_code_hash": "r" * 64},
            "artifact_refs": refs,
            **extra,
        }
        (self.root / run_id / "manifest.json").write_text(json.dumps(manifest))


def _call(run_id, n, kind, state, error=None, usage=True, retry_of=None, diag=None, ms=5000.0):
    return {
        "invocation_id": f"inv{n}",
        "run_id": run_id,
        "kind": kind,
        "attempt": 1 if retry_of else 0,
        "state": state,
        "started_at": "2026-09-24T00:00:00Z",
        "elapsed_ms": ms if state != "STARTED" else None,
        "usage": {"input_tokens": 10, "output_tokens": 5} if usage else None,
        "error_code": error,
        "http_status": None,
        "retryable": False,
        "format_retry_of": retry_of,
        "output_diagnostics": diag,
    }


def _record(n, mode, status, run_id, calls, verdict=None, patch=None, reason=None, cost=0.01):
    return {
        "record_id": _rid(n),
        "case_id": f"case_{n:04d}",
        "mode": mode,
        "status": status,
        "verdict": verdict,
        "patch_hash": patch,
        "failure_reason": reason,
        "cost_usd": cost,
        "executed_checks": {"verification/build": "CLEAN", "verification/racecheck": "FINDING"}
        if verdict == "NOT_FIXED"
        else {},
        "diagnosis": {"root_cause": SECRET_TEXT},
        "usage": {"physical_calls": calls},
        "lineage": {
            "diagnosis_run_id": run_id,
            "provider_invocation_hashes": ["h"] * calls,
        },
    }


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    d1, d2, d3, d4 = _rid(101), _rid(102), _rid(103), _rid(104)
    started = _call(d1, 1, "diagnose", "STARTED", usage=False)
    s.run(
        d1,
        "diagnosis",
        {
            "provider/inv1/STARTED.json": json.dumps(started).encode(),
            "provider/inv1/COMPLETED.json": json.dumps(
                _call(d1, 1, "diagnose", "COMPLETED")
            ).encode(),
            "provider/inv2/COMPLETED.json": json.dumps(_call(d1, 2, "patch", "COMPLETED")).encode(),
        },
    )
    s.run(
        d2,
        "diagnosis",
        {
            "provider/inv1/UNCERTAIN.json": json.dumps(
                _call(d2, 1, "diagnose", "UNCERTAIN", "LLM_TIMEOUT", usage=False, ms=60010.0)
            ).encode()
        },
    )
    rejected = {"failure_class": "SCHEMA_INVALID", "issues": [{"loc": "$schema", "type": "x"}]}
    s.run(
        d3,
        "diagnosis",
        {
            "provider/inv1/FAILED.json": json.dumps(
                _call(d3, 1, "diagnose", "FAILED", "LLM_INVALID_OUTPUT", diag=rejected)
            ).encode(),
            "provider/inv2/FAILED.json": json.dumps(
                _call(
                    d3,
                    2,
                    "diagnose",
                    "FAILED",
                    "LLM_INVALID_OUTPUT",
                    retry_of="inv1",
                    diag=rejected,
                )
            ).encode(),
        },
    )
    s.run(
        d4,
        "diagnosis",
        {
            "provider/inv1/STARTED.json": json.dumps(
                _call(d4, 1, "plan", "STARTED", usage=False)
            ).encode()
        },
    )
    records = [
        _record(1, "C", "COMPLETED", d1, 2, verdict="NOT_FIXED", patch="p"),
        _record(2, "C", "TIMEOUT", d2, 1, reason="LLM_TIMEOUT", cost=None),
        _record(3, "B", "FAILED", d3, 2, reason="LLM_INVALID_OUTPUT"),
        _record(4, "E", "TIMEOUT", d4, 1, reason="LLM_TIMEOUT", cost=None),
    ]
    evaluation = _rid(1)
    schedule = {"split": "development", "bindings": {"commit": COMMIT}}
    summary = {
        "run_id": evaluation,
        "split": "development",
        "commit": COMMIT,
        "expected_units": 4,
        "executed_units": 4,
        "stopped_reason": None,
        "records": records,
    }
    s.run(
        evaluation,
        "evaluation",
        {
            "evaluation/schedule.json": json.dumps(schedule).encode(),
            "evaluation/manifest.json": json.dumps(summary).encode(),
        },
    )
    return s, evaluation


def test_units_are_mutually_exclusive_and_calls_counted_once(store):
    s, evaluation = store
    report = inv.inventory(s.root, evaluation, "0" * 64)
    units = report["units"]["by_category"]
    assert sum(units.values()) == 4
    assert units == {
        "status:FAILED/LLM_INVALID_OUTPUT": 1,
        "status:TIMEOUT/LLM_TIMEOUT": 2,
        "verdict:NOT_FIXED": 1,
    }
    calls = report["calls"]["by_mode_kind_outcome"]
    # STARTED then COMPLETED for the same invocation is one call, not two.
    assert calls["C|diagnose|COMPLETED"] == 1
    assert calls["C|diagnose|LLM_TIMEOUT"] == 1
    assert calls["E|plan|STARTED_ONLY"] == 1
    assert report["calls"]["format_retries_by_kind_origin_outcome"] == {
        "diagnose|after:SCHEMA_INVALID|LLM_INVALID_OUTPUT": 1
    }
    assert report["units"]["provider_outcome_unknown_by_mode_kind"] == {
        "C|diagnose": 1,
        "E|plan": 1,
    }
    assert report["units"]["verification_first_failing_check"] == {
        "NOT_FIXED|verification/racecheck=FINDING": 1
    }


def test_unknown_cost_is_not_zero(store):
    s, evaluation = store
    report = inv.inventory(s.root, evaluation, "0" * 64)
    assert report["units"]["cost_unknown_units_by_mode"] == {"C": 1, "E": 1}
    assert report["calls"]["usage_missing_by_kind_outcome"] == {
        "diagnose|LLM_TIMEOUT": 1,
        "plan|STARTED_ONLY": 1,
    }


def test_output_has_no_model_text(store):
    s, evaluation = store
    report = json.dumps(inv.inventory(s.root, evaluation, "0" * 64))
    assert SECRET_TEXT not in report
    assert "root_cause" not in report


def test_holdout_split_is_refused(store):
    s, evaluation = store
    summary_path = next(
        p for p in (s.root / evaluation / "artifacts").iterdir() if b'"records"' in p.read_bytes()
    )
    data = json.loads(summary_path.read_text())
    data["split"] = "holdout"
    # Rebuild honestly so the refusal comes from scope, not from the hash check.
    s.run(
        evaluation,
        "evaluation",
        {
            "evaluation/schedule.json": json.dumps({"split": "holdout"}).encode(),
            "evaluation/manifest.json": json.dumps(data).encode(),
        },
    )
    with pytest.raises(inv.InventoryError, match="development"):
        inv.inventory(s.root, evaluation, "0" * 64)


def test_tampered_artifact_is_refused(store):
    s, evaluation = store
    target = next((s.root / _rid(102) / "artifacts").iterdir())
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(inv.InventoryError, match="hash"):
        inv.inventory(s.root, evaluation, "0" * 64)


def test_symlinked_artifact_is_refused(store, tmp_path_factory):
    s, evaluation = store
    target = next((s.root / _rid(102) / "artifacts").iterdir())
    outside = tmp_path_factory.mktemp("outside") / "copy"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    os.symlink(outside, target)
    with pytest.raises(inv.InventoryError, match="symlink"):
        inv.inventory(s.root, evaluation, "0" * 64)


def test_inputs_are_not_modified(store):
    s, evaluation = store
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in s.root.rglob("*") if p.is_file()}
    inv.inventory(s.root, evaluation, "0" * 64)
    after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in s.root.rglob("*") if p.is_file()}
    assert before == after


def test_cli_refuses_with_exit_code_two(store, capsys):
    s, _ = store
    assert inv.main([str(s.root), "not-a-run-id"]) == 2
    assert "refused" in capsys.readouterr().err


def test_symlinked_root_is_refused(store, tmp_path_factory):
    s, evaluation = store
    link = tmp_path_factory.mktemp("links") / "public"
    link.symlink_to(s.root, target_is_directory=True)
    with pytest.raises(inv.InventoryError, match="symlink"):
        inv.inventory(link, evaluation, "0" * 64)


def test_diagnosis_commit_mismatch_is_refused(store):
    s, evaluation = store
    path = s.root / _rid(101) / "manifest.json"
    data = json.loads(path.read_bytes())
    data["binding"]["repository"]["commit"] = "d" * 40
    path.write_text(json.dumps(data))
    with pytest.raises(inv.InventoryError, match="commit"):
        inv.inventory(s.root, evaluation, "0" * 64)


def test_incomplete_record_set_is_refused(store):
    s, evaluation = store
    run = inv._Run(s.root, evaluation)
    _, content = run.only("evaluation/manifest.json")
    summary = json.loads(content)
    summary["records"].pop()
    s.run(
        evaluation,
        "evaluation",
        {
            "evaluation/schedule.json": run.only("evaluation/schedule.json")[1],
            "evaluation/manifest.json": json.dumps(summary).encode(),
        },
    )
    with pytest.raises(inv.InventoryError, match="incomplete"):
        inv.inventory(s.root, evaluation, "0" * 64)


def test_conflicting_terminal_calls_are_refused(store):
    s, evaluation = store
    run = inv._Run(s.root, _rid(101))
    artifacts = {ref["name"]: run.read(ref) for ref in run.refs}
    artifacts["provider/inv1/FAILED.json"] = json.dumps(
        _call(_rid(101), 1, "diagnose", "FAILED", "LLM_INVALID_OUTPUT")
    ).encode()
    s.run(_rid(101), "diagnosis", artifacts)
    with pytest.raises(inv.InventoryError, match="conflicting"):
        inv.inventory(s.root, evaluation, "0" * 64)
