"""Fixed 2-case, 3-arm public GPU repair pilot; never a private holdout benchmark.

Runs separate pinned source checkouts in separate Python interpreters. Supports
only one attempt per unit; stores every outcome including failures. No automatic
retry, no Repair Memory and no invisible case selection.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

CASES = ("case_0019", "case_0020")
ARMS = ("V2", "V3_OLD", "V3_NEW")
SEED = 20261011
LEGACY_SHA = "e4e7a8d4d069cecb24355e9c521bf7bd121e0870"
CURRENT_SHA = "1eaac1fdf456a46118ea3aecc16338a9142c29bb"
VERSION = "cuda=12.8.1;compute-sanitizer=2025.1.0.0"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_once(path: Path, obj: object) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)
        file.write("\n")


def schedule() -> list[tuple[str, str]]:
    units = [(case, arm) for case in CASES for arm in ARMS]
    random.Random(SEED).shuffle(units)
    return units


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def require_checkout(root: Path, expected: str) -> None:
    if git_head(root) != expected:
        raise ValueError(f"frozen checkout mismatch: {root}")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip():
        raise ValueError("refuse dirty worktree in frozen pilot")


def hash_case_files(root: Path, case: str) -> dict[str, str]:
    folder = root / "benchmarks" / "public" / case / "public_input"
    result = {
        name: digest((folder / name).read_bytes())
        for name in ("kernel.cu", "task.json", "input.json")
    }
    task = json.loads((folder / "task.json").read_text())
    if task["source_sha256"] != result["kernel.cu"]:
        raise ValueError("case task not bound to original kernel")
    return result


def run_unit(args: argparse.Namespace) -> int:
    """Called in its own isolated interpreter after the frozen preflight."""
    root = args.checkout.resolve(strict=True)
    require_checkout(root, args.expected_commit)
    sys.path.insert(0, str(root / "src"))
    from gpu_agent.agent.provider import DevelopmentCallPolicy, OpenAIProviderSettings
    from gpu_agent.repair import RepairPolicy
    from gpu_agent.service import ApplicationService

    arm = args.arm
    settings = OpenAIProviderSettings.from_environment()
    if (
        settings.model != "deepseek-v4-pro"
        or not settings.api_key
        or not settings.endpoint
        or not settings.endpoint.startswith("https://api.deepseek.com")
    ):
        raise ValueError("unexpected provider/model configuration")
    if os.environ.get("GPU_AGENT_REPAIR_MEMORY_INDEX"):
        raise ValueError("memory unexpectedly configured for comparison")
    if os.environ.get("GPU_AGENT_KNOWLEDGE_VERSION") != VERSION:
        raise ValueError("incompatible official knowledge toolchain version")
    if (
        args.expected_commit == LEGACY_SHA and arm != "V3_OLD"
    ) or (args.expected_commit == CURRENT_SHA and arm == "V3_OLD"):
        raise ValueError("arm bound to wrong code checkout")

    os.environ["GPU_AGENT_RUN_ROOT"] = str(args.unit_dir / "public")
    os.environ["GPU_AGENT_EVALUATOR_ROOT"] = str(args.unit_dir / "evaluator")
    os.environ["GPU_AGENT_KNOWLEDGE_INDEX"] = str(args.knowledge_index)
    service = ApplicationService.configured()
    service.allow_development_paid_calls(DevelopmentCallPolicy(max_llm_calls=40))
    policy = RepairPolicy(
        version="public-repair-v2" if arm == "V2" else "public-repair-v3",
        max_candidates=3,
        max_reinvestigations=1,
        # Matched 4-call investigation cap across all three arms.
        unbounded_sanitizer_calls=False,
    )
    source = root / "benchmarks" / "public" / args.case / "public_input"
    started = time.monotonic()
    try:
        run, verified = service.repair(source, policy=policy, mode="E")
        refs = {r.name: r for r in run.artifact_refs}
        raw_summary = (
            json.loads(service.store.read(refs["repair/summary.json"]))
            if "repair/summary.json" in refs
            else None
        )
        usage = service.provider_invocations(run.id)
        output = {
            "case": args.case,
            "arm": arm,
            "run_id": run.id,
            "code_commit": args.expected_commit,
            "verdict": verified.verdict if verified is not None else "NOT_VERIFIED",
            "reason_code": verified.reason_code if verified is not None else None,
            "repair_stop_reason": raw_summary.get("stop_reason") if raw_summary else None,
            "candidate_round_count": len(raw_summary.get("rounds", [])) if raw_summary else 0,
            "reinvestigations": raw_summary.get("reinvestigations", 0) if raw_summary else 0,
            "sanitizer_acquisition_usage": (
                raw_summary.get("acquisition_usage") if raw_summary else None
            ),
            "public_checks": [
                {"round": r["round"], "status": r["check"]["status"],
                 "checks": r["check"].get("checks", {})}
                for r in (raw_summary.get("rounds", []) if raw_summary else [])
            ],
            "physical_llm_calls": len(usage),
            "token_count": sum((i.usage.total_tokens or 0) for i in usage if i.usage),
            "calls_with_missing_token_count": sum(
                i.usage is None or i.usage.total_tokens is None for i in usage
            ),
            "model_call_states": sorted({i.state for i in usage}),
            "model_prompt_versions": sorted({i.prompt_version for i in usage}),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "api_cost_usd": None,
            "attempts": 1,
            "no_retries": True,
            "public_run_store": str(args.unit_dir / "public"),
            "private_evaluator_artifacts_exported": False,
        }
        write_once(args.unit_dir / "result.json", output)
        print(
            "UNIT_RESULT=" + json.dumps({
                k: output[k] for k in (
                    "case", "arm", "verdict", "repair_stop_reason",
                    "candidate_round_count", "reinvestigations",
                    "physical_llm_calls", "token_count", "elapsed_seconds",
                )
            }),
            flush=True,
        )
        return 0
    except Exception as exc:
        write_once(
            args.unit_dir / "interrupted.json",
            {"error_class": type(exc).__name__,
             "elapsed_seconds": round(time.monotonic() - started, 3),
             "attempts": 1, "no_retry": True},
        )
        print("UNIT_INTERRUPTED=" + type(exc).__name__, flush=True)
        return 2


def run_pilot(args: argparse.Namespace) -> int:
    from urllib.parse import urlsplit

    if not args.allow_paid_calls:
        raise ValueError("explicit --allow-paid-calls required")
    if os.environ.get("GPU_AGENT_REPAIR_MEMORY_INDEX"):
        raise ValueError("memory must be disabled for frozen pilot")
    model = os.environ.get("OPENAI_MODEL")
    endpoint = os.environ.get("OPENAI_BASE_URL")
    if not os.environ.get("OPENAI_API_KEY") or model != "deepseek-v4-pro":
        raise ValueError("configured real model unavailable")
    if urlsplit(endpoint or "").hostname != "api.deepseek.com":
        raise ValueError("unexpected provider endpoint")
    if os.environ.get("GPU_AGENT_KNOWLEDGE_VERSION") != VERSION:
        raise ValueError("knowledge version not frozen")

    current = args.repository.resolve(strict=True)
    legacy = args.legacy_repository.resolve(strict=True)
    require_checkout(current, CURRENT_SHA)
    require_checkout(legacy, LEGACY_SHA)
    hashes: dict[str, dict[str, str]] = {}
    for case in CASES:
        hashes[case] = hash_case_files(current, case)
        if hashes[case] != hash_case_files(legacy, case):
            raise ValueError("case source/input changed between legacy and current")
    index_bytes = args.knowledge_index.resolve(strict=True).read_bytes()
    units = schedule()
    output = args.output.absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    frozen_index = output / "knowledge-index.json"
    frozen_index.write_bytes(index_bytes)
    frozen_index.chmod(0o400)

    write_once(
        output / "predeclared.json",
        {
            "kind": "public_repair_three_arm_pilot",
            "date_utc": datetime.now(UTC).isoformat(),
            "repository": "YOUessi/agentic-gpu-debugger",
            "current_commit": CURRENT_SHA,
            "legacy_commit": LEGACY_SHA,
            "case_input_hashes": hashes,
            "official_knowledge_index_sha256": digest(index_bytes),
            "official_knowledge_version": VERSION,
            "model": model,
            "endpoint_host": urlsplit(endpoint).hostname,
            "seed": SEED,
            "units": units,
            "arms": {
                "V2": {"checkout": "current", "reinvestigate": False, "sanitizer_cap": 4},
                "V3_OLD": {"checkout": "legacy", "reinvestigate": True, "sanitizer_cap": 4},
                "V3_NEW": {"checkout": "current", "reinvestigate": True, "sanitizer_cap": 4},
            },
            "budget_per_unit": {
                "candidate_limit": 3, "reinvestigations": 1,
                "model_calls": 40, "agent_steps": 38, "wall_seconds": 600,
            },
            "memory": "disabled",
            "repeat_count": 1,
            "automatic_retry": False,
            "development_public_only": True,
            "limitations": (
                "Public cases may have appeared in prior project evaluations. "
                "Only one repetition each; legacy and current code differ in "
                "more than the reinvestigation evidence gate. No causal claim."
            ),
        },
    )

    failures = 0
    with (output / "results.jsonl").open("x") as stream:
        for ordinal, (case, arm) in enumerate(units):
            require_checkout(current, CURRENT_SHA)
            require_checkout(legacy, LEGACY_SHA)
            if frozen_index.read_bytes() != index_bytes:
                raise ValueError("knowledge index changed after predeclaration")
            unit_dir = output / f"unit-{ordinal:02d}"
            unit_dir.mkdir(mode=0o700)
            checkout = legacy if arm == "V3_OLD" else current
            write_once(
                unit_dir / "attempt.json",
                {"ordinal": ordinal, "case": case, "arm": arm,
                 "code_commit": LEGACY_SHA if arm == "V3_OLD" else CURRENT_SHA,
                 "started_utc": datetime.now(UTC).isoformat()},
            )
            env = os.environ.copy()
            env.pop("GPU_AGENT_REPAIR_MEMORY_INDEX", None)
            env.update({
                "GPU_AGENT_KNOWLEDGE_INDEX": str(frozen_index),
                "GPU_AGENT_KNOWLEDGE_VERSION": VERSION,
                "PYTHONUNBUFFERED": "1",
            })
            cmd = [
                sys.executable, "-I", str(Path(__file__).resolve()),
                "--unit", "--checkout", str(checkout),
                "--expected-commit", LEGACY_SHA if arm == "V3_OLD" else CURRENT_SHA,
                "--case", case, "--arm", arm,
                "--unit-dir", str(unit_dir),
                "--knowledge-index", str(frozen_index),
            ]
            print(f"START {ordinal + 1}/{len(units)} {case} {arm}", flush=True)
            try:
                run = subprocess.run(
                    cmd, cwd=checkout, env=env, text=True,
                    capture_output=True, timeout=780,
                )
            except subprocess.TimeoutExpired:
                # Timeout is recorded, never silently re-run.
                write_once(unit_dir / "interrupted.json",
                           {"error_class": "CHILD_DEADLINE_EXCEEDED", "attempts": 1})
                run = None
            if run is not None:
                (unit_dir / "child.stdout").write_text(run.stdout)
                (unit_dir / "child.stderr").write_text(run.stderr)
            if (unit_dir / "result.json").exists():
                result = json.loads((unit_dir / "result.json").read_text())
                row = {"ordinal": ordinal, **result}
            else:
                failures += 1
                row = {
                    "ordinal": ordinal, "case": case, "arm": arm,
                    "verdict": "EXECUTION_INCOMPLETE", "attempts": 1,
                    "error_class": (
                        json.loads((unit_dir / "interrupted.json").read_text())
                        .get("error_class") if (unit_dir / "interrupted.json").exists()
                        else "UNKNOWN"
                    ),
                }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(
                f"DONE {ordinal + 1}/{len(units)} {case} {arm} "
                f"{row['verdict']} calls={row.get('physical_llm_calls', 'unknown')}",
                flush=True,
            )
    print("COMPLETE " + str(output), flush=True)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--legacy-repository", type=Path)
    parser.add_argument("--knowledge-index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--unit", action="store_true")
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--expected-commit", type=str)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--unit-dir", type=Path)
    args = parser.parse_args()
    if args.unit:
        if any(v is None for v in (
            args.checkout, args.expected_commit, args.case,
            args.arm, args.unit_dir,
        )):
            raise ValueError("unit requires frozen commit, case and output directory")
        return run_unit(args)
    if args.repository is None or args.legacy_repository is None or args.output is None:
        raise ValueError("pilot requires two pinned checkouts and output directory")
    return run_pilot(args)


if __name__ == "__main__":
    raise SystemExit(main())
