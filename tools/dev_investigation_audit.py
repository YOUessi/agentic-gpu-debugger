"""Read-only D/E trajectory audit of the frozen e80ce75 public development run.

No provider, CUDA, evaluator, candidate-code or knowledge calls. Outputs controller
codes/counts/hashes only. An allowed proposal is not proof its dispatch completed;
minimum finish eligibility is not proof a root cause is sufficiently understood.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from .dev_failure_inventory import InventoryError, _latest_invocations, _Run
except ImportError:
    from dev_failure_inventory import InventoryError, _latest_invocations, _Run

COMMIT = "e80ce75dde90e7624a89a63e6de1c68ef18bb7d5"


def missing(evidence):
    """Frozen policy, not an import of the changing application implementation."""
    result = []
    if "memcheck" not in evidence["sanitizer_outcomes"]:
        result.append("memcheck_outcome")
    if not evidence["tool_findings"]:
        result.append("tool_finding")
    elif not evidence["documentation"]:
        result.append("documentation_for_finding")
    return result


def summarize_steps(pairs):
    counts, denied, trace = Counter(), Counter(), []
    previous_ranges = {}
    for position, (index, step, decision) in enumerate(pairs):
        action, evidence = step["action"], step["evidence"]
        kind, args = action["action_type"], action["typed_arguments"]
        if kind != decision["action_type"]:
            raise InventoryError("step and decision disagree")
        if not re.fullmatch(r"[a-z_]+", kind) or type(decision["allowed"]) is not bool:
            raise InventoryError("invalid action or decision")
        eligible = not missing(evidence)
        allowed = decision["allowed"]
        counts["proposals"] += 1
        counts["allowed" if allowed else "denied"] += 1
        counts[f"proposed:{kind}"] += 1
        if allowed:
            counts[f"allowed:{kind}"] += 1
        for reason in decision["reason_codes"]:
            if not re.fullmatch(r"[A-Z_]+", reason):
                raise InventoryError("invalid reason code")
            denied[reason] += 1
        if allowed and eligible and kind not in {"finish_diagnosis", "declare_inconclusive"}:
            counts["nonterminal_allowed_when_finish_eligible"] += 1
        row = {
            "step": index,
            "action": kind,
            "allowed": allowed,
            "missing_evidence": missing(evidence),
            "reasons": decision["reason_codes"],
        }
        if kind == "inspect_source" and allowed:
            source = next(s for s in evidence["sources"] if s["source_id"] == args["source_id"])
            start, end = args["start_line"], args["end_line"]
            if not (type(start) is int and type(end) is int and 1 <= start <= end <= 100000):
                raise InventoryError("invalid source range")
            already = previous_ranges.setdefault(args["source_id"], set())
            lines = set(range(start, end + 1))
            row["fully_repeated_range"] = lines <= already
            row["source_already_in_snapshot"] = bool(source.get("content"))
            counts["fully_repeated_source_reads"] += int(lines <= already)
            counts["source_reads_with_content_already_supplied"] += int(bool(source.get("content")))
            if position + 1 < len(pairs):
                unchanged = evidence == pairs[position + 1][1]["evidence"]
                row["next_public_snapshot_unchanged"] = unchanged
                counts["source_reads_with_next_snapshot"] += 1
                counts["source_reads_with_unchanged_next_snapshot"] += int(unchanged)
            already.update(lines)
        trace.append(row)
    return {"counts": dict(counts), "rejections": dict(denied), "trace": trace}


def audit(root, evaluation_id):
    root = root.absolute()
    evaluation = _Run(root, evaluation_id)
    if evaluation.kind != "evaluation" or evaluation.manifest["status"] != "COMPLETED":
        raise InventoryError("not a completed evaluation")
    schedule_ref, schedule_raw = evaluation.only("evaluation/schedule.json")
    summary_ref, summary_raw = evaluation.only("evaluation/manifest.json")
    schedule, summary = json.loads(schedule_raw), json.loads(summary_raw)
    if summary["split"] != "development" or schedule["split"] != "development":
        raise InventoryError("development only")
    if summary["run_id"] != evaluation_id:
        raise InventoryError("wrong evaluation identity")
    binding = evaluation.manifest["binding"]
    if {summary["commit"], schedule["bindings"]["commit"], binding["repository"]["commit"]} != {
        COMMIT
    }:
        raise InventoryError("this audit only implements the frozen e80ce75 policy")
    records = summary["records"]
    if not (len(records) == summary["expected_units"] == summary["executed_units"] == 240):
        raise InventoryError("incomplete development batch")
    if summary["stopped_reason"] is not None:
        raise InventoryError("batch stopped")
    units, keys, runs = [], set(), set()
    for record in records:
        if record["mode"] not in {"D", "E"}:
            continue
        key = (record["case_id"], record["mode"], record["repeat"])
        if (
            key in keys
            or not re.fullmatch(r"case_00(0[1-9]|1[0-6])", key[0])
            or key[2] not in (0, 1, 2)
        ):
            raise InventoryError("duplicate or unexpected development unit")
        keys.add(key)
        run = _Run(root, record["lineage"]["diagnosis_run_id"])
        if run.run_id in runs or run.kind != "diagnosis":
            raise InventoryError("duplicate or invalid diagnosis")
        runs.add(run.run_id)
        if run.manifest["binding"]["repository"]["commit"] != COMMIT:
            raise InventoryError("diagnosis commit differs")
        indices = []
        for ref in run.refs:
            match = re.fullmatch(r"actions/(\d+)/step.json", ref["name"])
            if match:
                indices.append(int(match[1]))
        if sorted(indices) != list(range(len(indices))):
            raise InventoryError("non-contiguous or duplicate action steps")
        decisions = [r for r in run.refs if re.fullmatch(r"actions/\d+/decision.json", r["name"])]
        if len(decisions) != len(indices):
            raise InventoryError("missing or orphan decision")
        pairs = []
        for i in sorted(indices):
            step = json.loads(run.only(f"actions/{i}/step.json")[1])
            decision = json.loads(run.only(f"actions/{i}/decision.json")[1])
            evidence_ref = step["evidence_ref"]
            if evidence_ref not in run.refs:
                raise InventoryError("evidence snapshot is not registered")
            run.read(evidence_ref)  # Integrity only; embedded public projection differs by design.
            pairs.append((i, step, decision))
        item = summarize_steps(pairs)
        calls = _latest_invocations(run)
        if len(calls) != record["usage"]["physical_calls"]:
            raise InventoryError("call count differs from evaluation usage")
        call_counts = Counter(c["kind"] for c in calls)
        failures = Counter(
            f"{c['kind']}:{c.get('error_code') or c['state']}"
            for c in calls
            if c["state"] != "COMPLETED"
        )
        item.update(
            case_id=key[0],
            mode=key[1],
            repeat=key[2],
            run_id=run.run_id,
            manifest_sha256=hashlib.sha256(run.manifest_bytes).hexdigest(),
            verdict=record["verdict"],
            failure_reason=record["failure_reason"],
            diagnosis_outcome=record["diagnosis"]["diagnostic_outcome"],
            patch_generated=record["patch_hash"] is not None,
            calls=dict(call_counts),
            call_failures=dict(failures),
            latency_ms=record["latency_ms"],
            sanitizer_calls=record["usage"]["sanitizer_calls"],
            retrieval_calls=record["usage"]["retrieval_calls"],
        )
        units.append(item)
    if len(units) != 96:
        raise InventoryError("expected 48 units per mode")
    totals = {}
    for mode in ("D", "E"):
        rows = [r for r in units if r["mode"] == mode]
        counts, calls, failures, rejections = Counter(), Counter(), Counter(), Counter()
        for row in rows:
            counts.update(row["counts"])
            calls.update(row["calls"])
            failures.update(row["call_failures"])
            rejections.update(row["rejections"])
        totals[mode] = {
            "units": len(rows),
            "fixed": sum(r["verdict"] == "VERIFIED_FIXED" for r in rows),
            "diagnosed": sum(r["diagnosis_outcome"] == "DIAGNOSED" for r in rows),
            "patched": sum(r["patch_generated"] for r in rows),
            "outcomes": dict(
                Counter(r["verdict"] or r["failure_reason"] or "unknown" for r in rows)
            ),
            "counts": dict(counts),
            "calls": dict(calls),
            "call_failures": dict(failures),
            "rejections": dict(rejections),
            "sanitizer_calls": sum(r["sanitizer_calls"] for r in rows),
            "retrieval_calls": sum(r["retrieval_calls"] for r in rows),
            "latency_ms_sum": sum(r["latency_ms"] for r in rows),
        }
    return {
        "evaluation_run_id": evaluation_id,
        "commit": COMMIT,
        "schedule_sha256": schedule_ref["sha256"],
        "summary_sha256": summary_ref["sha256"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "notes": [
            "Allowed proposals are not completed dispatches.",
            "Finish eligibility is the controller minimum, not diagnostic sufficiency.",
            "No causal benefit or population significance follows from these counts.",
        ],
        "totals": totals,
        "units": units,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_root", type=Path)
    parser.add_argument("evaluation_run_id")
    args = parser.parse_args()
    try:
        report = audit(args.public_root, args.evaluation_run_id)
    except (InventoryError, OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
