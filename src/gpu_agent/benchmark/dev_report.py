"""Per-mode summary of one development evaluation run (diagnostic, not release evidence).

Usage:
    python -m gpu_agent.benchmark.dev_report RUN_ROOT EVALUATION_RUN_ID [--json]

RUN_ROOT is the public RunStore root; the evaluation run's terminal manifest is read through
the store (hash-checked). Family accuracy uses the frozen labels in
evaluation/development-labels.json; location and root cause are not scored until adjudicated.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from gpu_agent._resources import runtime_resource
from gpu_agent.benchmark.evaluation import EvaluationManifest, PublicEvaluationRecord
from gpu_agent.store import RunStore, read_regular

LABELS = runtime_resource("evaluation/development-labels.json")


def load_labels(path: Path | None = None) -> dict[str, dict[str, str]]:
    document = json.loads(read_regular(path or LABELS, 1024 * 1024))
    cases = document.get("cases")
    if document.get("schema_version") != 1 or not isinstance(cases, dict):
        raise ValueError("development labels are malformed")
    return cases


def summarize(
    records: list[PublicEvaluationRecord], labels: dict[str, dict[str, str]]
) -> dict[str, object]:
    by_mode: dict[str, list[PublicEvaluationRecord]] = defaultdict(list)
    for record in records:
        by_mode[record.mode].append(record)
    modes: dict[str, object] = {}
    for mode in sorted(by_mode):
        items = by_mode[mode]
        diagnosed = [r for r in items if r.diagnosis.get("diagnostic_outcome") == "DIAGNOSED"]
        family_hits = [
            r.diagnosis.get("failure_family") == labels.get(r.case_id, {}).get("failure_family")
            for r in diagnosed
            if r.case_id in labels
        ]
        costs = [r.cost_usd for r in items if r.cost_usd is not None]
        modes[mode] = {
            "units": len(items),
            "status": dict(Counter(r.status for r in items)),
            "failure_reasons": dict(Counter(r.failure_reason for r in items if r.failure_reason)),
            "diagnosed": len(diagnosed),
            "family_correct": sum(family_hits),
            "family_scored": len(family_hits),
            "patched": sum(r.patch_hash is not None for r in items),
            "verdicts": dict(Counter(r.verdict for r in items if r.verdict)),
            "verified_fixed": sum(r.verdict == "VERIFIED_FIXED" for r in items),
            "physical_llm_calls": sum(int(r.usage.get("physical_calls") or 0) for r in items),
            "cost_usd_known": round(sum(costs), 6),
            "cost_unknown_units": len(items) - len(costs),
        }
    return {"records": len(records), "modes": modes}


def _render(summary: dict[str, object]) -> str:
    lines = [
        "mode units  COMPLETED INCONCL FAILED TIMEOUT  diagnosed family  patched fixed calls  cost$"
    ]
    modes = summary["modes"]
    assert isinstance(modes, dict)
    for mode, row in modes.items():
        status = row["status"]
        lines.append(
            f"{mode:>4} {row['units']:>5}  {status.get('COMPLETED', 0):>9} "
            f"{status.get('INCONCLUSIVE', 0):>7} {status.get('FAILED', 0):>6} "
            f"{status.get('TIMEOUT', 0):>7}  {row['diagnosed']:>9} "
            f"{row['family_correct']:>3}/{row['family_scored']:<3} {row['patched']:>7} "
            f"{row['verified_fixed']:>5} {row['physical_llm_calls']:>5} "
            f"{row['cost_usd_known']:>7}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("evaluation_run_id")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    store = RunStore(args.run_root)
    run = store.load(args.evaluation_run_id)
    refs = [ref for ref in run.artifact_refs if ref.name == "evaluation/manifest.json"]
    if run.kind != "evaluation" or len(refs) != 1:
        raise SystemExit("not a terminal evaluation run")
    manifest = EvaluationManifest.model_validate_json(store.read(refs[0]))
    summary = summarize(list(manifest.records), load_labels())
    summary["stopped_reason"] = manifest.stopped_reason
    summary["expected_units"] = manifest.expected_units
    print(json.dumps(summary, indent=2) if args.json else _render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
