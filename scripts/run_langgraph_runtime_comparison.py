#!/usr/bin/env python3
"""Operator-only, frozen same-model comparison of repository Agent runtimes.

Planning/reporting never runs a model or GPU. Execution requires explicit
operator authorization, an existing shared GPU lock and isolated output.
No P8 samples, manifests, runtime or evaluation output are reused.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ARMS = {
    "legacy-react": {"runtime": "legacy", "hierarchical": False},
    "native-react": {"runtime": "langgraph", "hierarchical": False},
    "legacy-planner": {"runtime": "legacy", "hierarchical": True},
    "native-dag": {"runtime": "langgraph", "hierarchical": True},
}
CASES = ("h001", "h002", "h003")
SCHEMA = "gpu-agent-langgraph-comparison-v1"
BUDGET = {
    "steps": 16,
    "physical_calls": 24,
    "output_tokens": 8192,
    "request_seconds": 180,
    "agent_seconds": 600,
    "strict_seconds": 600,
    "compile_only": 6,
    "public_checks": 2,
    "probes": 4,
}
COMMON_FLAGS = (
    "--syntax-editing",
    "--compile-gate",
    "--allow-revert",
    "--public-diagnostics",
    "--ast-target-binding",
    "--state-scoped-actions",
    "--disable-batch-edits",
)
MODEL_FAILURES = {
    "AGENT_BUDGET_EXHAUSTED",
    "STEP_BUDGET_EXHAUSTED",
    "LLM_REPOSITORY_ROLE_BUDGET_EXHAUSTED",
    "LLM_INVALID_OUTPUT",
    "LLM_INCOMPLETE_MAX_OUTPUT_TOKENS",
    "LLM_INCOMPLETE_CONTENT_FILTER",
    "LLM_INCOMPLETE_OTHER",
    "LLM_REFUSED",
    "LLM_BUDGET_EXCEEDED",
    "LLM_CALL_BUDGET_EXHAUSTED",
    "LLM_WALL_TIME_BUDGET_EXHAUSTED",
}
CLI = "from gpu_agent.cli import app; app()"


def canonical(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(data: Any) -> str:
    return hashlib.sha256(canonical(data)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular record: {path.name}")
    return json.loads(path.read_bytes())


def write_once(path: Path, value: dict):
    """Immutable publication; concurrent overwrite is refused."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        timeout=20,
    ).strip()


def versions():
    names = (
        "langgraph",
        "langgraph-checkpoint",
        "langgraph-checkpoint-sqlite",
        "langchain-core",
        "openai",
        "pydantic",
        "tree-sitter",
        "tree-sitter-cuda",
    )
    return {name: importlib.metadata.version(name) for name in names}


def build_manifest(repo: Path, *, model: str, endpoint: str, repetitions: int) -> dict:
    from gpu_agent.agent.provider import OpenAIProviderSettings

    if not 1 <= repetitions <= 20 or not re.fullmatch(r"[A-Za-z0-9._:/-]{1,100}", model):
        raise ValueError("invalid comparison design")
    settings = OpenAIProviderSettings(
        endpoint=endpoint,
        model=model,
        allow_loopback_http=False,
        supports_store_false=True,
        timeout_seconds=BUDGET["request_seconds"],
        max_output_tokens=BUDGET["output_tokens"],
    )
    if git(repo, "status", "--porcelain"):
        raise ValueError("comparison source checkout must be clean")
    sha = git(repo, "rev-parse", "HEAD")
    trials = []
    for rep in range(1, repetitions + 1):
        for i, case in enumerate(CASES):
            order = list(ARMS)
            shift = (rep - 1 + i) % len(order)
            order = order[shift:] + order[:shift]
            for arm in order:
                trials.append(
                    {
                        "id": f"r{rep:02d}-{case}-{arm}",
                        "replicate": rep,
                        "case": case,
                        "arm": arm,
                        **ARMS[arm],
                    }
                )
    case_files = {
        case: file_sha(repo / "benchmarks" / "hardset_v2" / case / "mutant/kernel.cu")
        for case in CASES
    }
    return {
        "schema": SCHEMA,
        "code_sha": sha,
        "model": model,
        "endpoint": settings.endpoint,
        "dependencies": versions(),
        "python": ".".join(map(str, sys.version_info[:3])),
        "budget": BUDGET,
        "common_flags": list(COMMON_FLAGS),
        "cases": case_files,
        "repetitions": repetitions,
        "trials": trials,
        "memory": "off",
        "review": "off",
        "strict_verifier": "existing independent verify-repo",
        "design": "Same model/system/configuration, repeated development tasks; not holdout.",
        "sampling": "Provider defaults; no claim of paired or request-level deterministic RNG.",
        "allocation": "Equal ceilings. Planner/replan role allocations are runtime-dependent.",
    }


def validate_manifest(repo: Path, manifest: dict):
    expected = build_manifest(
        repo,
        model=manifest["model"],
        endpoint=manifest["endpoint"],
        repetitions=manifest["repetitions"],
    )
    if manifest != expected:
        raise ValueError("frozen comparison code, environment, task, budget or order changed")


def commands(repo: Path, folder: Path, m: dict, trial: dict, evaluator_secret_file: Path):
    case = trial["case"]
    goal = (
        f"Correct the registered CUDA kernel for {case}. "
        "Use public evidence and stage a publicly passing exact candidate."
    )
    agent = [
        sys.executable,
        "-c",
        CLI,
        "harness",
        "repo-run",
        "--repository",
        str(repo),
        "--output",
        str(folder / "agent"),
        "--base-sha",
        m["code_sha"],
        "--goal",
        goal,
        "--public-case",
        case,
        "--runtime",
        trial["runtime"],
        "--hierarchical" if trial["hierarchical"] else "--no-hierarchical",
        "--max-steps",
        str(BUDGET["steps"]),
        "--max-model-calls",
        str(BUDGET["physical_calls"]),
        "--max-compiles",
        str(BUDGET["compile_only"]),
        "--max-public-checks",
        str(BUDGET["public_checks"]),
        "--memory-mode",
        "off",
        *COMMON_FLAGS,
        "--allow-paid-calls",
    ]
    strict = [
        sys.executable,
        "-c",
        CLI,
        "harness",
        "verify-repo",
        "--repository",
        str(repo),
        "--session",
        str(folder / "agent"),
        "--output",
        str(folder / "strict"),
        "--evaluator-secret-file",
        str(evaluator_secret_file),
    ]
    return agent, strict


def environments(repo: Path, m: dict):
    # Credentials remain in the operator's process environment, not argv/records.
    env = dict(os.environ)
    for name in ("OPENAI_LOG", "PYTHONHOME", "PYTHONPATH", "GPU_AGENT_ALLOW_LOOPBACK_HTTP"):
        env.pop(name, None)
    env.update(
        PYTHONPATH=str(repo / "src"),
        OPENAI_BASE_URL=m["endpoint"],
        OPENAI_MODEL=m["model"],
        GPU_AGENT_LLM_TIMEOUT_SECONDS=str(BUDGET["request_seconds"]),
        GPU_AGENT_MAX_OUTPUT_TOKENS=str(BUDGET["output_tokens"]),
        GPU_AGENT_STORE_FALSE_SUPPORTED="1",
    )
    if not env.get("OPENAI_API_KEY"):
        raise ValueError("operator model credentials are not configured")
    strict_env = dict(env)
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"):
        strict_env.pop(name, None)
    return env, strict_env


def invoke(argv: list[str], env: dict, folder: Path, stage: str, seconds: int, cwd: Path) -> dict:
    started = time.monotonic()
    timed_out = False
    with (
        (folder / f"{stage}-stdout.log").open("xb") as out,
        (folder / f"{stage}-stderr.log").open("xb") as err,
    ):
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True
        )
        try:
            write_once(folder / f"{stage}-process.json", {"pid": proc.pid, "pgid": proc.pid})
            proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
            raise
    return {
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "wall_seconds": round(time.monotonic() - started, 3),
    }


def idle_proof():
    """No unrelated process is killed; refuse unsafe continuation."""
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).splitlines()
    compute = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).strip()
    containers = subprocess.check_output(["docker", "ps", "-q"], text=True, timeout=10).strip()
    if len(gpu) != 1 or int(gpu[0]) >= 1000 or compute or containers:
        raise ValueError("shared GPU/containers not idle; do not admit another trial")
    return {"gpu_memory_mib": int(gpu[0]), "compute_processes": [], "running_containers": []}


def ensure_no_live_attempt(root: Path) -> None:
    """Fail closed if a child of an interrupted controller is still alive."""
    if not Path("/proc").is_dir():
        raise ValueError("comparison requires Linux process inspection")
    trials = root / "trials"
    groups = set()
    for p in trials.glob("*/*-process.json"):
        groups.add(read(p)["pgid"])
    for path in Path("/proc").glob("[0-9]*"):
        try:
            args = (path / "cmdline").read_bytes().split(b"\0")
            fields = (path / "stat").read_text().rsplit(")", 1)[-1].split()
        except (OSError, ValueError):
            continue
        if len(fields) > 2 and fields[0] != "Z" and int(fields[2]) in groups:
            raise ValueError("reserved trial process group is still alive")
        if any(
            arg.startswith(str(trials).encode() + b"/")
            and (arg.endswith(b"/agent") or arg.endswith(b"/strict"))
            for arg in args
        ):
            raise ValueError("reserved trial still has a live CLI")


def observed_usage(folder: Path, m: dict) -> dict:
    from gpu_agent.agent.provider import provider_invocations
    from gpu_agent.harness.journal import read_verified_journal
    from gpu_agent.store import RunStore

    journal = folder / "agent/session-events.jsonl"
    if not journal.exists():
        return {
            "physical_calls": 0,
            "tokens_lower_bound": 0,
            "tokens_complete": False,
            "response_models": [],
            "errors": [],
            "journal_sha256": None,
        }
    records = read_verified_journal(journal)
    rid = records[0]["payload"]["public_run_id"]
    inv = provider_invocations(RunStore(folder / "agent/public"), rid)
    from urllib.parse import urlsplit

    if len(inv) > BUDGET["physical_calls"] or any(
        i.configured_model != m["model"]
        or (i.response_model is not None and i.response_model != m["model"])
        or i.endpoint_host != urlsplit(m["endpoint"]).hostname
        or i.configured_max_output_tokens != BUDGET["output_tokens"]
        or i.request_timeout_seconds is None
        or i.request_timeout_seconds > BUDGET["request_seconds"]
        for i in inv
    ):
        raise ValueError("model or request budget provenance mismatch")
    return {
        "physical_calls": len(inv),
        "tokens_lower_bound": sum(i.usage.total_tokens or 0 for i in inv if i.usage),
        "tokens_complete": all(
            i.usage is not None
            and i.usage.total_tokens is not None
            and i.state in {"COMPLETED", "FAILED"}
            for i in inv
        ),
        "response_models": sorted({i.response_model for i in inv if i.response_model}),
        "errors": [i.error_code for i in inv if i.error_code],
        "uncertain_calls": sum(i.state in {"STARTED", "UNCERTAIN"} for i in inv),
        "journal_sha256": file_sha(journal),
        "journal_head_sha256": records[-1]["sha256"],
    }


def strict_projection(folder: Path, trial: dict, expected: str):
    from gpu_agent.benchmark.hardset_v2_verifier import HardsetPrivateProjection
    from gpu_agent.store import RunStore

    public = folder / "strict/public"
    store = RunStore(public)
    manifests = [
        store.load(p.name)
        for p in public.iterdir()
        if p.is_dir() and re.fullmatch(r"[a-f0-9]{32}", p.name)
    ]
    matches = [m for m in manifests if m.kind == "hardset_v2_strict_projection"]
    if len(matches) != 1 or matches[0].status.value != "COMPLETED":
        raise ValueError("missing independent strict public projection")
    refs = [r for r in matches[0].artifact_refs if r.name == "hardset/strict-verdict.json"]
    if len(refs) != 1:
        raise ValueError("strict projection evidence is ambiguous")
    raw = store.read(refs[0])
    result = HardsetPrivateProjection.model_validate_json(raw)
    if result.case_id != trial["case"] or result.candidate_sha256 != expected:
        raise ValueError("strict projection is not bound to final public candidate")
    return {
        "verdict": result.verdict,
        "candidate_sha256": expected,
        "public_run_id": matches[0].id,
        "projection_sha256": hashlib.sha256(raw).hexdigest(),
    }


def recover_reserved(folder: Path, trial: dict, m: dict, *, proof):
    """No re-execution of a reserved attempt, even if its result is unknown."""
    row = {
        "trial": trial,
        "manifest_sha256": digest(m),
        "status": "INTERRUPTED_OUTCOME_UNKNOWN",
        "strict_fix_within_budget": False,
        "candidate_correctness": "UNKNOWN",
        "retry_same_replicate": False,
        "agent_execution": read(folder / "agent-execution.json")
        if (folder / "agent-execution.json").is_file()
        else None,
        "strict_execution": read(folder / "strict-execution.json")
        if (folder / "strict-execution.json").is_file()
        else None,
        "usage": observed_usage(folder, m),
        "cleanup": proof,
        "admitted": (folder / "agent-admitted.json").is_file(),
    }
    write_once(folder / "result.json", row)
    return row


def run_trial(repo, root, manifest, trial, secret, envs, *, runner=invoke, cleanup=idle_proof):
    """Execute exactly one frozen attempt, never a replacement sample."""
    folder = root / "trials" / trial["id"]
    if folder.exists():
        raise ValueError("trial directory already exists; cannot resample")
    cleanup()
    folder.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_once(folder / "reservation.json", {"trial": trial, "manifest_sha256": digest(manifest)})
    agent, strict = commands(repo, folder, manifest, trial, secret)
    write_once(
        folder / "agent-admitted.json", {"argv_sha256": digest(agent), "unix_seconds": time.time()}
    )
    execution = runner(agent, envs[0], folder, "agent", BUDGET["agent_seconds"], repo)
    write_once(folder / "agent-execution.json", execution)
    row = {
        "trial": trial,
        "manifest_sha256": digest(manifest),
        "admitted": True,
        "agent_execution": execution,
        "strict_execution": None,
        "strict_projection": None,
        "strict_fix_within_budget": False,
        "candidate_correctness": "UNKNOWN",
        "retry_same_replicate": False,
        "usage": observed_usage(folder, manifest),
    }
    if execution["timed_out"]:
        row["status"] = "AGENT_TIMEOUT"
    else:
        path = folder / "agent/result.json"
        if not path.is_file():
            errors = set(row["usage"]["errors"])
            row["status"] = (
                "MODEL_FAILURE" if errors and errors <= MODEL_FAILURES else "UNRESOLVED_INFRA"
            )
        else:
            from gpu_agent.harness.models import RepoHarnessOutcome

            report = RepoHarnessOutcome.model_validate_json(path.read_bytes())
            row["public_report"] = report.model_dump(mode="json")
            if report.model_calls > BUDGET["physical_calls"] or report.tool_calls > BUDGET["steps"]:
                raise ValueError("trial exceeded frozen model or tool budget")
            if report.status == "DIFF_STAGED_UNVERIFIED" and report.public_checks_passed:
                cleanup()
                write_once(
                    folder / "strict-admitted.json",
                    {"candidate": report.public_checked_candidate_sha256},
                )
                strict_execution = runner(
                    strict, envs[1], folder, "strict", BUDGET["strict_seconds"], repo
                )
                write_once(folder / "strict-execution.json", strict_execution)
                row["strict_execution"] = strict_execution
                if strict_execution["timed_out"]:
                    row["status"] = "STRICT_TIMEOUT"
                else:
                    projection = strict_projection(
                        folder, trial, report.public_checked_candidate_sha256
                    )
                    row["strict_projection"] = projection
                    row["strict_fix_within_budget"] = projection["verdict"] == "VERIFIED_FIXED"
                    row["candidate_correctness"] = projection["verdict"]
                    row["status"] = (
                        "STRICT_SUCCESS"
                        if row["strict_fix_within_budget"]
                        else "STRICT_FAILURE"
                        if projection["verdict"] == "NOT_VERIFIED"
                        else "UNRESOLVED_INFRA"
                    )
            elif report.reason == "GRAPH_INFRASTRUCTURE_FAILURE":
                row["status"] = "UNRESOLVED_INFRA"
            elif report.status == "PAUSED_RESUMABLE":
                row["status"] = "UNRESOLVED_INFRA"
            else:
                row["status"] = "MODEL_FAILURE"
    row["cleanup"] = cleanup()
    write_once(folder / "result.json", row)
    return row


def wilson(success: int, n: int):
    if not n:
        return None
    z = 1.959963984540054
    p = success / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0, center - radius), min(1, center + radius)]


def summarize(root: Path, manifest: dict) -> dict:
    rows = []
    for trial in manifest["trials"]:
        folder = root / "trials" / trial["id"]
        if (folder / "result.json").is_file():
            row = read(folder / "result.json")
            if row["trial"] != trial or row["manifest_sha256"] != digest(manifest):
                raise ValueError("result does not match frozen trial")
            rows.append(row)
        elif folder.exists():
            rows.append(
                {
                    "trial": trial,
                    "status": "RESERVED_UNRECONCILED",
                    "admitted": (folder / "agent-admitted.json").is_file(),
                    "strict_fix_within_budget": False,
                    "usage": {},
                    "agent_execution": None,
                    "strict_execution": None,
                }
            )
    groups = {}
    for arm in ARMS:
        for case in ("all", *CASES):
            group = [
                r
                for r in rows
                if r["trial"]["arm"] == arm and (case == "all" or r["trial"]["case"] == case)
            ]
            started = [r for r in group if r["admitted"]]
            det = [
                r
                for r in started
                if r["status"] in {"STRICT_SUCCESS", "STRICT_FAILURE", "MODEL_FAILURE"}
            ]
            ok = sum(r["strict_fix_within_budget"] for r in started)
            timeouts = sum(r["status"] in {"AGENT_TIMEOUT", "STRICT_TIMEOUT"} for r in started)
            agent_times = [(r["agent_execution"] or {}).get("wall_seconds") for r in started]
            token_lower = sum(r["usage"].get("tokens_lower_bound", 0) for r in started)
            tokens_complete = all(r["usage"].get("tokens_complete", False) for r in started)
            strict_seconds = sum(
                (r["strict_execution"] or {}).get("wall_seconds", 0) for r in started
            )
            total_seconds = sum(t for t in agent_times if t is not None) + strict_seconds
            groups[f"{arm}:{case}"] = {
                "started": len(started),
                "determinate": len(det),
                "success": ok,
                "timeouts": timeouts,
                "states": dict(collections.Counter(r["status"] for r in group)),
                "primary": ok / len(started) if started else None,
                "secondary": ok / len(det) if det else None,
                "wilson95": wilson(ok, len(started)),
                "tokens_lower_bound": token_lower,
                "tokens_total": token_lower if tokens_complete else None,
                "physical_calls_observed": sum(
                    r["usage"].get("physical_calls", 0) for r in started
                ),
                "agent_wall_seconds_observed": sum(t for t in agent_times if t is not None),
                "strict_wall_seconds_observed": strict_seconds,
                "total_wall_seconds_observed": total_seconds,
                "seconds_per_success_lower_bound": total_seconds / ok if ok else None,
                "tokens_per_success_lower_bound": token_lower / ok if ok else None,
                "actual_billed_cost": None,
            }
    return {
        "planned": len(manifest["trials"]),
        "reserved": len(rows),
        "pending": len(manifest["trials"]) - len(rows),
        "groups": groups,
        "states": dict(collections.Counter(row["status"] for row in rows)),
    }


def execute(args, repo, root, manifest):
    if not (args.allow_paid_calls and args.allow_real_gpu):
        raise ValueError("explicit operator approvals are required")
    secret = args.evaluator_secret_file.absolute()
    if secret.is_relative_to(repo) or secret.is_relative_to(root) or secret.is_symlink():
        raise ValueError("private evaluator secret must remain outside repository and result root")
    st = secret.stat()
    if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode) != 0o600:
        raise ValueError("private evaluator secret must be a 0600 file")
    lock = args.gpu_lock.resolve(strict=True)
    envs = environments(repo, manifest)
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def cleanup():
            ensure_no_live_attempt(root)
            return idle_proof()

        cleanup()
        for trial in manifest["trials"]:
            folder = root / "trials" / trial["id"]
            if folder.exists():
                reservation = read(folder / "reservation.json")
                if reservation != {"trial": trial, "manifest_sha256": digest(manifest)}:
                    raise ValueError("reservation identity mismatch")
                if not (folder / "result.json").is_file():
                    recover_reserved(folder, trial, manifest, proof=cleanup())
                    print(
                        "INTERRUPTED_ATTEMPT_PRESERVED; inspect before resuming remaining trials",
                        flush=True,
                    )
                    return 2
                if read(folder / "result.json")["status"] in {
                    "UNRESOLVED_INFRA",
                    "INTERRUPTED_OUTCOME_UNKNOWN",
                }:
                    # Never silently override infrastructure faults. Operator must
                    # review the evidence before continuing a fresh pending ID.
                    if not args.acknowledge_prior_infra:
                        print("PRIOR_INFRA_REQUIRES_OPERATOR_REVIEW", trial["id"], flush=True)
                        return 2
                continue
            print("TRIAL_START", trial["id"], flush=True)
            row = run_trial(repo, root, manifest, trial, secret, envs, cleanup=cleanup)
            print("TRIAL_END", trial["id"], row["status"], flush=True)
            if row["status"] == "UNRESOLVED_INFRA":
                return 2
        print(json.dumps(summarize(root, manifest), indent=2))
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "status", "execute"])
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--endpoint", default="https://api.deepseek.com")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--evaluator-secret-file", type=Path)
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--allow-real-gpu", action="store_true")
    parser.add_argument("--acknowledge-prior-infra", action="store_true")
    args = parser.parse_args()
    repo = args.repository.resolve(strict=True)
    root = args.output.resolve()
    if root.is_relative_to(repo) or repo.is_relative_to(root):
        raise ValueError("comparison output must be disjoint from the frozen source checkout")
    if args.mode == "plan":
        if root.exists():
            raise ValueError("use a new comparison output root")
        manifest = build_manifest(
            repo, model=args.model, endpoint=args.endpoint, repetitions=args.repetitions
        )
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        write_once(root / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "manifest_sha256": digest(manifest),
                    "code_sha": manifest["code_sha"],
                    "planned": len(manifest["trials"]),
                    "arms": ARMS,
                    "budget": BUDGET,
                    "model_calls": 0,
                    "gpu_checks": 0,
                },
                indent=2,
            )
        )
        return 0
    manifest = read(root / "manifest.json")
    if args.mode == "status":
        print(json.dumps(summarize(root, manifest), indent=2))
        return 0
    if args.gpu_lock is None or args.evaluator_secret_file is None:
        parser.error("execute requires GPU lock and separate evaluator secret file")
    validate_manifest(repo, manifest)
    return execute(args, repo, root, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
