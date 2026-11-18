"""Predeclared public real-GPU framework pilot; no retries or private data in reports.

This experiment compares the SAME source commit under five explicitly different
orchestration/policy environments. It is not a signed holdout or a claim that
adaptive planning improves repair. Never run without a real GPU, approved
model/paid calls, a controller-owned lock, and a public CUDA knowledge index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

ARMS = {
    "D_RULE": ("D", "legacy", "native", "baseline"),
    "E_LEGACY": ("E", "legacy", "native", "baseline"),
    "E_GRAPH": ("E", "langgraph", "native", "baseline"),
    "E_LANGCHAIN": ("E", "langgraph", "langchain", "baseline"),
    "E_ADAPTIVE": ("E", "langgraph", "langchain", "adaptive-v1"),
}
SEED = 20261114


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def schedule(cases: list[str], repeats: int, seed: int = SEED) -> list[dict[str, object]]:
    if not cases or len(set(cases)) != len(cases) or not 1 <= repeats <= 3:
        raise ValueError("cases must be unique and repeats between 1 and 3")
    trials = [
        {"case": case, "arm": arm, "repeat": repeat}
        for case in cases
        for arm in ARMS
        for repeat in range(repeats)
    ]
    random.Random(seed).shuffle(trials)
    return trials


def preflight(repo: Path, knowledge: Path, cases: list[str], commit: str) -> dict[str, object]:
    from gpu_agent.knowledge.retrieve import KnowledgeIndex, parse_version
    from gpu_agent.public_task import load_public_task

    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != commit:
        raise ValueError("repository SHA differs from frozen expected commit")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if dirty.strip():
        raise ValueError("repository is not clean")
    version = os.environ.get("GPU_AGENT_KNOWLEDGE_VERSION")
    if not version:
        raise ValueError("GPU_AGENT_KNOWLEDGE_VERSION required")
    parse_version(version)
    index = KnowledgeIndex.load(knowledge)
    for query in ("memcheck", "racecheck", "initcheck", "synccheck"):
        if not index.retrieve(query, version, 1).chunks:
            raise ValueError("frozen CUDA documentation lacks an expected sanitizer")
    files: dict[str, dict[str, str]] = {}
    for case in cases:
        if not case.startswith("case_") or len(case) != 9 or not case[-4:].isdigit():
            raise ValueError("invalid public case ID")
        root = repo / "benchmarks" / "public" / case / "public_input"
        required = ("kernel.cu", "input.json", "task.json")
        files[case] = {name: sha256(root / name) for name in required}
        if load_public_task(root / "kernel.cu", (root / "kernel.cu").read_bytes()) is None:
            raise ValueError("a public functional requirement is required")
    return {
        "toolchain_version": version,
        "knowledge_corpus_hash": index.corpus_hash,
        "knowledge_file_sha256": sha256(knowledge),
        "public_input_hashes": files,
    }


def run(args: argparse.Namespace) -> int:
    from gpu_agent.agent.prompts import PROMPT_VERSION
    from gpu_agent.agent.provider import DevelopmentCallPolicy, OpenAIProviderSettings
    from gpu_agent.repair import RepairPolicy
    from gpu_agent.service import ApplicationService

    if not args.allow_paid_calls and not args.preflight_only:
        raise ValueError("--allow-paid-calls is required for real-model execution")
    if os.environ.get("GPU_AGENT_REPAIR_MEMORY_INDEX"):
        raise ValueError("Repair Memory must be disabled for this matched pilot")
    settings = OpenAIProviderSettings.from_environment()
    if not args.preflight_only and (
        settings.api_key is None
        or not settings.model
        or not settings.endpoint
        or (
            urlsplit(settings.endpoint).hostname != "api.openai.com"
            and not settings.supports_store_false
        )
    ):
        raise ValueError("explicit configured provider and store=false support required")
    root = args.output.resolve()
    repo = args.repository.resolve()
    knowledge = args.knowledge_index.resolve()
    preflight_report = preflight(repo, knowledge, args.case, args.expected_commit)
    units = schedule(args.case, args.repeats, args.seed)

    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    frozen = root / "knowledge-index.json"
    frozen.write_bytes(knowledge.read_bytes())
    frozen.chmod(0o400)
    selection = {
        "kind": "public_framework_orchestration_pilot",
        "date_utc": datetime.now(UTC).isoformat(),
        "source_sha": args.expected_commit,
        "provider": settings.model,
        "endpoint_host": urlsplit(settings.endpoint or "").hostname,
        "prompt_version": PROMPT_VERSION,
        "arms": ARMS,
        "units": units,
        "seed": args.seed,
        "max_calls_per_unit": args.max_calls_per_unit,
        "max_total_calls": args.max_total_calls,
        "max_candidates": args.max_candidates,
        "max_reinvestigations": 1,
        "memory_enabled": False,
        "public_only": True,
        "preflight": preflight_report,
        "preflight_only": args.preflight_only,
        "unknown_cost_usd": None,
    }
    write_once(root / "predeclared.json", selection)
    if args.preflight_only:
        print("PREFLIGHT_PASSED", len(units), args.expected_commit, flush=True)
        return 0

    os.environ["GPU_AGENT_KNOWLEDGE_INDEX"] = str(frozen)
    total_calls = 0
    finished = 0
    with (root / "results.jsonl").open("x", encoding="utf-8") as report:
        for ordinal, unit in enumerate(units):
            if total_calls >= args.max_total_calls:
                write_once(
                    root / "budget-stopped.json", {"completed": finished, "calls": total_calls}
                )
                break
            assert sha256(frozen) == preflight_report["knowledge_file_sha256"]
            case, arm = str(unit["case"]), str(unit["arm"])
            mode, engine, component, planner_policy = ARMS[arm]
            os.environ.update(
                {
                    "GPU_AGENT_ORCHESTRATOR": engine,
                    "GPU_AGENT_COMPONENT_LAYER": component,
                    "GPU_AGENT_PLANNER_POLICY": planner_policy,
                }
            )
            unit_root = root / f"unit-{ordinal:02d}"
            unit_root.mkdir(mode=0o700)
            os.environ["GPU_AGENT_RUN_ROOT"] = str(unit_root / "public")
            os.environ["GPU_AGENT_EVALUATOR_ROOT"] = str(unit_root / "verification")
            write_once(unit_root / "started.json", {"ordinal": ordinal, **unit})
            print(f"START {ordinal + 1}/{len(units)} {case} {arm}", flush=True)
            started = time.monotonic()
            try:
                service = ApplicationService.configured()
                service.allow_development_paid_calls(
                    DevelopmentCallPolicy(max_llm_calls=args.max_calls_per_unit)
                )
                manifest, verified = service.repair(
                    repo / "benchmarks/public" / case / "public_input",
                    mode=mode,
                    policy=RepairPolicy(
                        version="public-repair-v3",
                        max_candidates=args.max_candidates,
                        max_reinvestigations=1,
                        unbounded_sanitizer_calls=False,
                    ),
                )
                calls = service.provider_invocations(manifest.id)
                artifacts = {ref.name: ref for ref in service.store.load(manifest.id).artifact_refs}
                reason = (
                    json.loads(service.store.read(artifacts["repair/summary.json"]))["stop_reason"]
                    if "repair/summary.json" in artifacts
                    else "NO_REPAIR_SUMMARY"
                )
                row = {
                    **unit,
                    "ordinal": ordinal,
                    "run_id": manifest.id,
                    "verdict": verified.verdict.value if verified else "NOT_VERIFIED",
                    "repair_stop_reason": reason,
                    "diagnostic_outcome": service.diagnosis(manifest.id).diagnostic_outcome,
                    "physical_calls": len(calls),
                    "known_tokens": sum((c.usage.total_tokens or 0) for c in calls if c.usage),
                    "unknown_usage_calls": sum(
                        c.usage is None or c.usage.total_tokens is None for c in calls
                    ),
                    "call_kinds": [c.kind for c in calls],
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "api_cost_usd": None,
                    "source_sha": args.expected_commit,
                }
            except Exception as exc:
                row = {
                    **unit,
                    "ordinal": ordinal,
                    "verdict": "INTERRUPTED",
                    "error_class": type(exc).__name__,
                    "physical_calls": None,
                    "known_tokens": None,
                    "api_cost_usd": None,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "source_sha": args.expected_commit,
                }
            write_once(unit_root / "result.json", row)
            report.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            report.flush()
            os.fsync(report.fileno())
            finished += 1
            total_calls += row.get("physical_calls") or 0
            print(
                f"DONE {ordinal + 1}/{len(units)} {row['verdict']} "
                f"calls={row.get('physical_calls')}",
                flush=True,
            )
    write_once(
        root / "completed.json",
        {
            "completed": finished,
            "expected": len(units),
            "complete": finished == len(units),
            "known_total_calls": total_calls,
            "source_sha": args.expected_commit,
        },
    )
    return 0 if finished == len(units) else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--knowledge-index", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-calls-per-unit", type=int, default=16)
    parser.add_argument("--max-total-calls", type=int, default=60)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if (
        not 1 <= args.max_candidates <= 3
        or not 1 <= args.max_calls_per_unit <= 40
        or not 1 <= args.max_total_calls <= 200
    ):
        parser.error("invalid candidate/call ceiling")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
