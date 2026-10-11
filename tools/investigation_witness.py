"""Read-only witnesses for explicitly selected public development E runs.

Checks native bundle/result links, not merely allowed proposals. Does not assert
counterfactual model causality, semantic diagnosis correctness, or superiority.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from .dev_failure_inventory import InventoryError, _latest_invocations, _Run
except ImportError:
    from dev_failure_inventory import InventoryError, _latest_invocations, _Run


def outcomes(bundle):
    return {
        r["tool_result"]["typed_payload"]["tool"]: r["check_outcome"]
        for r in bundle["sanitizer_results"]
        if r.get("tool_result")
    }


def transition(action, before, after):
    kind = action["action_type"]
    if kind.startswith("run_") and kind.endswith("check"):
        tool = kind.removeprefix("run_")
        previous, current = before["sanitizer_results"], after["sanitizer_results"]
        if len(current) != len(previous) + 1 or current[:-1] != previous:
            raise InventoryError("tool dispatch is missing or has extra results")
        result = current[-1]
        if (
            result["tool_result"]["typed_payload"]["tool"] != tool
            or not result["completed"]
            or result["check_outcome"] not in {"CLEAN", "FINDING"}
        ):
            raise InventoryError("selected tool was not successfully executed")
        return {"action": kind, "actual_outcome": result["check_outcome"]}
    if kind == "retrieve_official_docs":
        if not after["retrieved_chunks"]:
            raise InventoryError("retrieval did not produce documentation")
        return {"action": kind, "chunks": len(after["retrieved_chunks"])}
    raise InventoryError("unsupported witness transition")


def witness(root, run_id):
    run = _Run(root.absolute(), run_id)
    if (
        run.kind != "diagnosis"
        or run.manifest["status"] != "COMPLETED"
        or run.manifest.get("binding") is not None
        or any(r.get("visibility") != "public" for r in run.refs)
    ):
        raise InventoryError("only unbound public development diagnoses are accepted")
    policy = json.loads(run.only("agent/acquisition-policy.json")[1])
    if policy["mode"] != "E":
        raise InventoryError("witness requires E")
    run.only("agent/development-mode.json")
    indices = sorted(
        int(m[1]) for r in run.refs if (m := re.fullmatch(r"actions/(\d+)/step.json", r["name"]))
    )
    if indices != list(range(len(indices))) or not indices:
        raise InventoryError("invalid step sequence")
    steps, bundles = [], []
    for i in indices:
        step = json.loads(run.only(f"actions/{i}/step.json")[1])
        decision = json.loads(run.only(f"actions/{i}/decision.json")[1])
        if not decision["allowed"] or decision["action_type"] != step["action"]["action_type"]:
            raise InventoryError("this witness requires an all-allowed trace")
        ref = step["evidence_ref"]
        if ref not in run.refs:
            raise InventoryError("unregistered evidence")
        bundle = json.loads(run.read(ref))
        if outcomes(bundle) != step["evidence"]["sanitizer_outcomes"]:
            raise InventoryError("public outcomes differ from native evidence")
        for result in bundle["sanitizer_results"]:
            tool_result = result["tool_result"]
            for field in ("stdout_artifact", "stderr_artifact"):
                run.read(tool_result[field])
            found = any(
                json.loads(run.read(r)) == tool_result
                for r in run.refs
                if r["name"].startswith("sanitizer/") and r["name"].endswith("/result.json")
            )
            if not found:
                raise InventoryError("native tool result is missing")
        for ref in bundle["retrieved_chunks"]:
            run.read(ref)
        steps.append(step)
        bundles.append(bundle)
    trace = [
        transition(steps[i]["action"], bundles[i], bundles[i + 1]) for i in range(len(steps) - 1)
    ]
    terminal = steps[-1]["action"]["action_type"]
    if terminal not in {"finish_diagnosis", "declare_inconclusive"}:
        raise InventoryError("trace is not terminal")
    diagnosis = json.loads(run.only("diagnosis.json")[1])
    calls = _latest_invocations(run)
    plans = [c for c in calls if c["kind"] == "plan" and c["state"] == "COMPLETED"]
    if len(plans) != len(steps):
        raise InventoryError("completed planner count differs from proposals")
    return {
        "run_id": run_id,
        "manifest_sha256": hashlib.sha256(run.manifest_bytes).hexdigest(),
        "trace": trace,
        "terminal": terminal,
        "diagnostic_outcome": diagnosis["diagnostic_outcome"],
        "completed_planner_calls": len(plans),
        "limitations": [
            "No counterfactual LLM test; CLEAN does not prove absence of defects.",
            "Initial memcheck and minimum finish evidence are controller constraints.",
            "Artifact consistency is not independent hardware attestation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_root", type=Path)
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()
    reports = [witness(args.public_root, rid) for rid in args.run_ids]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
