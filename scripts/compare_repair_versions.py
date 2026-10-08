"""A predeclared four-unit public Repair V2/V3 *exploratory* comparison.

Two CUDA algorithm families, each under V2 and V3. No adaptive case selection,
retry-until-success, resuming uncertain calls, or private evaluator data export.
"""

import argparse
import hashlib
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path

from compare_repair_modes import preflight, save

from gpu_agent.agent.provider import DevelopmentCallPolicy, OpenAIProviderSettings
from gpu_agent.knowledge.retrieve import KnowledgeIndex
from gpu_agent.provenance import capture_repository_snapshot, runtime_code_fingerprint
from gpu_agent.repair import RepairPolicy
from gpu_agent.service import ApplicationService

CASES = ("case_0021", "case_0022")
VERSIONS = ("V2", "V3")
SEED = 20261009


def schedule() -> list[tuple[str, str]]:
    units = [(case, version) for case in CASES for version in VERSIONS]
    random.Random(SEED).shuffle(units)
    return units


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--knowledge-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-paid-calls", action="store_true")
    args = parser.parse_args()

    if not args.allow_paid_calls:
        raise ValueError("real development provider calls require explicit opt-in")
    repo = args.repository.resolve()
    output_root = args.output.absolute()
    snapshot = capture_repository_snapshot(repo)
    code_hash = runtime_code_fingerprint(repo)
    index_bytes = args.knowledge_index.read_bytes()
    index = KnowledgeIndex.load(args.knowledge_index)
    version = os.environ["GPU_AGENT_KNOWLEDGE_VERSION"]
    units = schedule()
    hashes = preflight(repo, index, version, CASES)
    settings = OpenAIProviderSettings.from_environment()
    if not settings.model or not settings.api_key:
        raise ValueError("real provider configuration unavailable")

    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "knowledge-index.json").write_bytes(index_bytes)
    (output_root / "knowledge-index.json").chmod(0o400)
    os.environ["GPU_AGENT_KNOWLEDGE_INDEX"] = str(output_root / "knowledge-index.json")
    save(
        output_root / "predeclared.json",
        {
            "kind": "exploratory_public_v2_v3_pilot",
            "date_utc": datetime.now(UTC).isoformat(),
            "code_commit": snapshot.commit,
            "runtime_code_hash": code_hash,
            "knowledge_hash": index.corpus_hash,
            "knowledge_file_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "knowledge_version": version,
            "model": settings.model,
            "endpoint_host": settings.endpoint,
            "input_hashes": hashes,
            "seed": SEED,
            "units": units,
            "max_candidates": 3,
            "max_reinvestigations": 1,
            "max_llm_calls_per_unit": 40,
            "max_wall_time_seconds_per_unit": 600,
            "V2": {"reinvestigate": False, "max_sanitizer_calls": 4},
            "V3": {"reinvestigate": True, "max_sanitizer_calls": None},
            "comparison_limitation": (
                "V3 also changes sanitizer acquisition policy; this is a bundled "
                "operational comparison, not a single-factor causal test."
            ),
            "repeat_count": 1,
            "zero_automatic_retries": True,
        },
    )
    with (output_root / "results.jsonl").open("x") as result_stream:
        for ordinal, (case, repair_version) in enumerate(units):
            capture_repository_snapshot(repo, expected_commit=snapshot.commit)
            if runtime_code_fingerprint(repo) != code_hash:
                raise ValueError("runtime code changed during comparison")
            if (output_root / "knowledge-index.json").read_bytes() != index_bytes:
                raise ValueError("knowledge index changed during comparison")
            unit_dir = output_root / f"unit-{ordinal:02d}"
            unit_dir.mkdir()
            save(
                unit_dir / "attempt.json",
                {
                    "case": case,
                    "mode": repair_version,
                    "started_utc": datetime.now(UTC).isoformat(),
                },
            )
            os.environ["GPU_AGENT_RUN_ROOT"] = str(unit_dir / "public")
            os.environ["GPU_AGENT_EVALUATOR_ROOT"] = str(unit_dir / "evaluator")
            service = ApplicationService.configured()
            service.allow_development_paid_calls(DevelopmentCallPolicy(max_llm_calls=40))
            policy = RepairPolicy(
                version="public-repair-v3" if repair_version == "V3" else "public-repair-v2",
                max_candidates=3,
                max_reinvestigations=1,
                unbounded_sanitizer_calls=repair_version == "V3",
            )
            print(
                f"START {ordinal + 1}/{len(units)} {case} {repair_version}",
                flush=True,
            )
            t0 = time.monotonic()
            try:
                run, final = service.repair(
                    repo / "benchmarks" / "public" / case / "public_input",
                    mode="E",
                    policy=policy,
                )
            except Exception as exc:
                save(
                    unit_dir / "interrupted.json",
                    {
                        "exception_type": type(exc).__name__,
                        "elapsed_seconds": time.monotonic() - t0,
                        "automatic_retry": False,
                    },
                )
                raise
            elapsed = time.monotonic() - t0
            invocations = service.provider_invocations(run.id)
            artifact_names = {"repair/summary.json", "agent/final-budget.json"}
            exported = {
                ref.name: json.loads(service.store.read(ref))
                for ref in run.artifact_refs
                if ref.name in artifact_names
            }
            row = {
                "ordinal": ordinal,
                "case": case,
                "version": repair_version,
                "run_id": run.id,
                "verdict": final.verdict if final is not None else "NOT_VERIFIED",
                "reason_code": final.reason_code if final is not None else None,
                "repair_summary": exported.get("repair/summary.json"),
                "final_budget": exported.get("agent/final-budget.json"),
                "elapsed_seconds": elapsed,
                "physical_calls": len(invocations),
                "llm_tokens": sum(
                    item.usage.total_tokens or 0 for item in invocations if item.usage
                ),
                "calls_with_unknown_usage": sum(
                    item.usage is None or item.usage.total_tokens is None for item in invocations
                ),
                "api_cost_usd": None,
                "attempts": 1,
            }
            save(unit_dir / "result.json", row)
            result_stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            result_stream.flush()
            os.fsync(result_stream.fileno())
            print(
                f"DONE {ordinal + 1}/{len(units)} {row['verdict']} "
                f"calls={row['physical_calls']} elapsed={round(elapsed, 1)}s",
                flush=True,
            )
    print("COMPLETE " + str(output_root), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
