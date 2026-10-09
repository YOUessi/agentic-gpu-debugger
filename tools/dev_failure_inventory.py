#!/usr/bin/env python3
"""Read-only failure inventory for ONE development evaluation run.

Scope (enforced, fail closed):
  * reads the evaluation run's manifest, its evaluation/schedule.json and
    evaluation/manifest.json, and the manifests plus provider/* invocation records of the
    diagnosis runs named in its records' lineage — nothing else;
  * refuses any run whose schedule or evaluation manifest is not split == "development";
  * every artifact byte is checked against the manifest's byte_count and sha256;
    symlinks, absolute paths and paths outside the public root are refused.

It opens files read-only, writes nothing, sends no request, imports nothing from
gpu_agent, and prints only codes, counts, hashes and latencies: no diagnosis text, no
model output, no source, no credential.

Semantics that must not be over-read:
  * A STARTED record proves only that the controller recorded the intent to call.
    It does not prove the server received or billed the request.
  * UNCERTAIN (timeout, connection or worker error, interrupted) means the outcome and
    billing are unknown. Missing usage is reported as unknown cost, never as zero.
  * The client timeout of a call is min(configured timeout, remaining wall budget) and is
    not stored per call, so an elapsed time near the timeout is evidence, not proof, of
    where the delay was.

Usage:
    python tools/dev_failure_inventory.py PUBLIC_ROOT EVALUATION_RUN_ID > inventory.json
"""

import argparse
import hashlib
import json
import os
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
UNCERTAIN_CODES = frozenset(
    {"LLM_TIMEOUT", "LLM_CONNECTION_ERROR", "LLM_WORKER_ERROR", "INTERRUPTED_INVOCATION"}
)
VERIFICATION_ORDER = ("build", "runtime", "public_oracle", "memcheck")
_ID_CHARS = frozenset("0123456789abcdef")


class InventoryError(Exception):
    """Input is outside scope or fails an integrity check; nothing is reported."""


# ------------------------------------------------------------------------ safe reads


def _run_id(value: object) -> str:
    if not isinstance(value, str) or len(value) != 32 or not set(value) <= _ID_CHARS:
        raise InventoryError("malformed run id")
    return value


def _read_regular(root: Path, relative: str, limit: int = MAX_ARTIFACT_BYTES) -> bytes:
    parts = Path(relative).parts
    if Path(relative).is_absolute() or not parts or any(p in {"", ".", ".."} for p in parts):
        raise InventoryError("artifact path escapes the public root")
    directory = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in (*root.parts[1:], *parts[:-1]):
            try:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
            except OSError:
                raise InventoryError("unavailable directory or symlink in input path") from None
            os.close(directory)
            directory = child
        try:
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        except OSError:
            raise InventoryError("unavailable file or symlink in input path") from None
    finally:
        os.close(directory)
    with os.fdopen(fd, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise InventoryError("input must be a single-link regular file")
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise InventoryError("artifact exceeds size bound")
    return data


class _Run:
    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root
        self.run_id = _run_id(run_id)
        self.manifest_bytes = _read_regular(root, f"{self.run_id}/manifest.json")
        self.manifest = json.loads(self.manifest_bytes)
        if self.manifest.get("id") != self.run_id:
            raise InventoryError("manifest id differs from its directory")
        self.refs: list[dict[str, Any]] = list(self.manifest.get("artifact_refs") or [])

    @property
    def kind(self) -> str:
        return str(self.manifest.get("kind"))

    def read(self, ref: dict[str, Any]) -> bytes:
        relative = str(ref.get("relative_path", ""))
        if (
            ref.get("run_id") != self.run_id
            or relative != f"{self.run_id}/artifacts/{_run_id(ref.get('id'))}"
        ):
            raise InventoryError("artifact outside its run")
        data = _read_regular(self.root, relative)
        if len(data) != ref.get("byte_count") or hashlib.sha256(data).hexdigest() != ref.get(
            "sha256"
        ):
            raise InventoryError("artifact hash or size mismatch")
        return data

    def only(self, name: str) -> tuple[dict[str, Any], bytes]:
        refs = [ref for ref in self.refs if ref.get("name") == name]
        if len(refs) != 1:
            raise InventoryError(f"expected exactly one {name}")
        return refs[0], self.read(refs[0])


# -------------------------------------------------------------------- classification


def unit_category(record: dict[str, Any]) -> str:
    """Mutually exclusive final class from recorded fields only; no interpretation."""
    if record.get("verdict"):
        return f"verdict:{record['verdict']}"
    if record.get("patch_hash"):
        return "patched:no_verdict"
    return f"status:{record.get('status')}/{record.get('failure_reason') or 'none'}"


def unit_group(record: dict[str, Any]) -> str:
    """Coarse, documented mapping of unit_category; still no judgement of cause."""
    verdict = record.get("verdict")
    reason = record.get("failure_reason")
    if verdict == "VERIFIED_FIXED":
        return "fixed"
    if verdict in {"NOT_FIXED", "REGRESSION_DETECTED"}:
        return "verification_negative"
    if verdict or record.get("patch_hash"):
        return "verification_inconclusive"
    if reason == "MODEL_DECLARED_INCONCLUSIVE":
        return "model_declared_inconclusive"
    if reason in UNCERTAIN_CODES:
        return "provider_outcome_unknown"
    if reason == "LLM_INVALID_OUTPUT":
        return "output_contract_rejected"
    return "other"


def first_failing_check(record: dict[str, Any]) -> str | None:
    checks = record.get("executed_checks") or {}
    ordered = [f"verification/{name}" for name in VERIFICATION_ORDER]
    ordered += sorted(k for k in checks if k.startswith("verification/") and k not in ordered)
    for key in ordered:
        if key in checks and checks[key] != "CLEAN":
            return f"{key}={checks[key]}"
    return None


def _latest_invocations(run: _Run) -> list[dict[str, Any]]:
    """Latest persisted state per invocation (terminal states win over STARTED)."""
    rank = {"STARTED": 0, "UNCERTAIN": 1, "FAILED": 1, "COMPLETED": 1}
    latest: dict[str, dict[str, Any]] = {}
    for ref in run.refs:
        name = str(ref.get("name", ""))
        if not name.startswith("provider/"):
            continue
        record = json.loads(run.read(ref))
        if record.get("run_id") != run.run_id:
            raise InventoryError("invocation belongs to another run")
        key = str(record.get("invocation_id"))
        state = str(record.get("state"))
        if state not in rank or name != f"provider/{key}/{state}.json":
            raise InventoryError("invalid invocation state or locator")
        previous = latest.get(key)
        if previous is not None and previous.get("state") != "STARTED":
            if state != "STARTED" and previous != record:
                raise InventoryError("conflicting terminal invocation records")
        if previous is None or rank[state] >= rank[str(previous.get("state"))]:
            latest[key] = record
    return list(latest.values())


def _call_outcome(call: dict[str, Any]) -> str:
    if call.get("state") == "STARTED":
        return "STARTED_ONLY"  # intent recorded; receipt and billing unknown
    return str(call.get("error_code") or call.get("state"))


def _percentiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pick(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))], 1)

    return {"n": len(ordered), "p50": pick(0.5), "p90": pick(0.9), "max": round(ordered[-1], 1)}


# ------------------------------------------------------------------------------ main


def inventory(root: Path, evaluation_run_id: str, script_sha256: str) -> dict[str, Any]:
    # Keep original components: resolve() would silently accept a symlinked root.
    root = root.absolute()
    evaluation = _Run(root, evaluation_run_id)
    if evaluation.kind != "evaluation" or evaluation.manifest.get("status") != "COMPLETED":
        raise InventoryError("not a completed evaluation run")
    schedule_ref, schedule_raw = evaluation.only("evaluation/schedule.json")
    manifest_ref, manifest_raw = evaluation.only("evaluation/manifest.json")
    schedule = json.loads(schedule_raw)
    summary = json.loads(manifest_raw)
    if schedule.get("split") != "development" or summary.get("split") != "development":
        raise InventoryError("only a development split is in scope")
    if summary.get("run_id") != evaluation.run_id:
        raise InventoryError("evaluation manifest belongs to another run")
    binding = evaluation.manifest.get("binding") or {}
    commit = (binding.get("repository") or {}).get("commit")
    if summary.get("commit") != commit or (schedule.get("bindings") or {}).get("commit") != commit:
        raise InventoryError("commit differs between run binding, schedule and manifest")
    records: list[dict[str, Any]] = list(summary.get("records") or [])
    if (
        len(records) != summary.get("executed_units")
        or len(records) != summary.get("expected_units")
        or summary.get("stopped_reason") is not None
        or len({_run_id(record.get("record_id")) for record in records}) != len(records)
    ):
        raise InventoryError("incomplete or duplicate evaluation records")

    units: Counter[str] = Counter()
    unit_by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    failing_checks: Counter[str] = Counter()
    non_fixed_by_case: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_cost_units: Counter[str] = Counter()
    known_cost: dict[str, float] = defaultdict(float)
    calls: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    normalizations: Counter[str] = Counter()
    retries: Counter[str] = Counter()
    usage_missing: Counter[str] = Counter()
    latency: dict[str, list[float]] = defaultdict(list)
    uncertain_units_by_kind: Counter[str] = Counter()
    counterexamples = 0
    consistency: Counter[str] = Counter()
    diagnosis_manifest_hashes: list[str] = []
    seen_runs: set[str] = set()

    for record in records:
        mode = str(record.get("mode"))
        category = unit_category(record)
        group = unit_group(record)
        units[category] += 1
        unit_by_mode[mode][category] += 1
        groups[mode][group] += 1
        if group != "fixed":
            non_fixed_by_case[str(record.get("case_id"))][category] += 1
        failing = first_failing_check(record) if record.get("verdict") != "VERIFIED_FIXED" else None
        if failing and record.get("verdict"):
            failing_checks[f"{record['verdict']}|{failing}"] += 1
        cost = record.get("cost_usd")
        if cost is None:
            unknown_cost_units[mode] += 1
        else:
            known_cost[mode] += float(cost)

        lineage = record.get("lineage") or {}
        run_id = _run_id(lineage.get("diagnosis_run_id"))
        if run_id in seen_runs:
            raise InventoryError("two records share one diagnosis run")
        seen_runs.add(run_id)
        diagnosis = _Run(root, run_id)
        if diagnosis.kind != "diagnosis":
            raise InventoryError("lineage points at a non-diagnosis run")
        if ((diagnosis.manifest.get("binding") or {}).get("repository") or {}).get(
            "commit"
        ) != commit:
            raise InventoryError("diagnosis commit differs from evaluation")
        diagnosis_manifest_hashes.append(hashlib.sha256(diagnosis.manifest_bytes).hexdigest())
        invocations = _latest_invocations(diagnosis)
        by_id = {str(call.get("invocation_id")): call for call in invocations}
        if len(invocations) != int((record.get("usage") or {}).get("physical_calls") or 0):
            consistency["invocations_differ_from_usage_physical_calls"] += 1
        if len(invocations) != len(lineage.get("provider_invocation_hashes") or []):
            consistency["invocations_differ_from_lineage_hashes"] += 1
        uncertain_kinds = set()
        for call in invocations:
            kind = str(call.get("kind"))
            outcome = _call_outcome(call)
            calls[f"{mode}|{kind}|{outcome}"] += 1
            if call.get("usage") is None:
                usage_missing[f"{kind}|{outcome}"] += 1
            if isinstance(call.get("elapsed_ms"), int | float):
                latency[f"{kind}|{outcome}"].append(float(call["elapsed_ms"]))
            if outcome in UNCERTAIN_CODES or outcome == "STARTED_ONLY":
                uncertain_kinds.add(kind)
            diagnostics = call.get("output_diagnostics") or {}
            for applied in diagnostics.get("normalizations") or []:
                normalizations[f"{kind}|{applied}"] += 1
            if diagnostics.get("failure_class"):
                issue_types = sorted(
                    {f"{i.get('loc')}:{i.get('type')}" for i in diagnostics.get("issues") or []}
                ) or ["-"]
                for issue in issue_types:
                    rejections[f"{kind}|{diagnostics['failure_class']}|{issue}"] += 1
            if diagnostics.get("sync_counterexample") is not None:
                counterexamples += 1
            parent = call.get("format_retry_of")
            if parent:
                origin = by_id.get(str(parent), {})
                origin_class = (origin.get("output_diagnostics") or {}).get("failure_class")
                origin_label = origin_class or (_call_outcome(origin) if origin else "?")
                retries[f"{kind}|after:{origin_label}|{outcome}"] += 1
        if group == "provider_outcome_unknown":
            for kind in sorted(uncertain_kinds) or ["none_recorded"]:
                uncertain_units_by_kind[f"{mode}|{kind}"] += 1

    diagnosis_hashes = hashlib.sha256("\n".join(sorted(diagnosis_manifest_hashes)).encode())
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "script_sha256": script_sha256,
            "public_root": str(root),
            "evaluation_run_id": evaluation.run_id,
            "evaluation_run_manifest_sha256": hashlib.sha256(evaluation.manifest_bytes).hexdigest(),
            "evaluation_manifest_artifact_sha256": manifest_ref.get("sha256"),
            "schedule_artifact_sha256": schedule_ref.get("sha256"),
            "commit": commit,
            "tracked_tree_hash": (binding.get("repository") or {}).get("tracked_tree_hash"),
            "runtime_code_hash": binding.get("runtime_code_hash"),
            "prompt_version": binding.get("prompt_version"),
            "model_config_hash": binding.get("model_config_hash"),
            "diagnosis_runs_read": len(diagnosis_manifest_hashes),
            "diagnosis_manifests_sha256": diagnosis_hashes.hexdigest(),
        },
        "notes": [
            "units: one mutually exclusive final category per record, from recorded fields",
            "calls: every provider invocation once, keyed mode|kind|outcome; not unit counts",
            "STARTED_ONLY/UNCERTAIN: server receipt and billing unknown; usage absent means "
            "cost unknown, not zero",
            "client timeout per call = min(configured, remaining wall budget); not recorded",
        ],
        "units": {
            "expected": summary.get("expected_units"),
            "executed": summary.get("executed_units"),
            "records": len(records),
            "stopped_reason": summary.get("stopped_reason"),
            "by_category": dict(sorted(units.items())),
            "by_mode_category": {
                m: dict(sorted(c.items())) for m, c in sorted(unit_by_mode.items())
            },
            "by_mode_group": {m: dict(sorted(c.items())) for m, c in sorted(groups.items())},
            "verification_first_failing_check": dict(sorted(failing_checks.items())),
            "non_fixed_by_case": {
                case: dict(sorted(c.items())) for case, c in sorted(non_fixed_by_case.items())
            },
            "provider_outcome_unknown_by_mode_kind": dict(sorted(uncertain_units_by_kind.items())),
            "cost_usd_known_by_mode": {m: round(v, 6) for m, v in sorted(known_cost.items())},
            "cost_unknown_units_by_mode": dict(sorted(unknown_cost_units.items())),
        },
        "calls": {
            "by_mode_kind_outcome": dict(sorted(calls.items())),
            "usage_missing_by_kind_outcome": dict(sorted(usage_missing.items())),
            "elapsed_ms_by_kind_outcome": {k: _percentiles(v) for k, v in sorted(latency.items())},
            "output_rejections_by_kind_class_issue": dict(sorted(rejections.items())),
            "format_retries_by_kind_origin_outcome": dict(sorted(retries.items())),
            "normalizations_by_kind": dict(sorted(normalizations.items())),
            "sync_counterexamples_recorded": counterexamples,
        },
        "consistency": dict(sorted(consistency.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_root", type=Path)
    parser.add_argument("evaluation_run_id")
    args = parser.parse_args(argv)
    script = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        report = inventory(args.public_root, args.evaluation_run_id, script)
    except (InventoryError, OSError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    json.dump(report, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
