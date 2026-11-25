#!/usr/bin/env python3
"""Publish bounded public comparison snapshots; never run or repeat a trial."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def git(repo, *args, data=None, env=None, absent=False):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], input=data, capture_output=True, env=env, timeout=90
    )
    if absent and result.returncode == 2:
        return ""
    if result.returncode:
        raise RuntimeError("git operation failed")
    return result.stdout.decode().strip()


def read(path):
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size > 262144:
        raise ValueError("invalid public record")
    return json.loads(path.read_bytes())


def process(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19]), fields[0] not in {"Z", "X"}
    except FileNotFoundError:
        return None, False


def public_row(row, trial, identity):
    if row["trial"] != trial or row["manifest_sha256"] != identity:
        raise ValueError("result identity differs")
    result = {
        key: row[key]
        for key in (
            "trial",
            "manifest_sha256",
            "status",
            "admitted",
            "strict_fix_within_budget",
            "candidate_correctness",
            "retry_same_replicate",
        )
        if key in row
    }
    for key in ("status", "candidate_correctness"):
        if key in result and not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", result[key]):
            raise ValueError("invalid public status")
    if any(
        type(result.get(key)) is not bool
        for key in ("admitted", "strict_fix_within_budget", "retry_same_replicate")
    ):
        raise ValueError("invalid public flag")
    fields = {
        "agent_execution": ("exit_code", "timed_out", "wall_seconds"),
        "strict_execution": ("exit_code", "timed_out", "wall_seconds"),
        "usage": ("physical_calls", "tokens_lower_bound", "tokens_complete", "uncertain_calls"),
    }
    for group, keys in fields.items():
        value = row.get(group)
        result[group] = None if value is None else {k: value[k] for k in keys if k in value}
        if value is not None and any(
            v is not None and (type(v) not in (bool, int, float) or not math.isfinite(v))
            for v in result[group].values()
        ):
            raise ValueError("invalid public metric")
    projection = row.get("strict_projection")
    if projection:
        result["strict_projection"] = {
            k: projection[k]
            for k in ("verdict", "candidate_sha256", "public_run_id", "projection_sha256")
            if k in projection
        }
        if any(
            not isinstance(v, str)
            or not re.fullmatch(
                r"[A-Z][A-Z0-9_]{0,79}"
                if k == "verdict"
                else r"[0-9a-f]{32}"
                if k == "public_run_id"
                else r"[0-9a-f]{64}",
                v,
            )
            for k, v in result["strict_projection"].items()
        ):
            raise ValueError("invalid public projection")
    return result


def publish(repo, parent, branch, payload):
    with tempfile.TemporaryDirectory(prefix="langgraph-export-index-") as temporary:
        env = dict(
            os.environ,
            GIT_INDEX_FILE=str(Path(temporary) / "index"),
            GIT_AUTHOR_NAME="LangGraph Task Bot",
            GIT_COMMITTER_NAME="LangGraph Task Bot",
            GIT_AUTHOR_EMAIL="langgraph-task-bot@example.invalid",
            GIT_COMMITTER_EMAIL="langgraph-task-bot@example.invalid",
        )
        git(repo, "read-tree", parent, env=env)
        for path, value in payload.items():
            raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
            blob = git(repo, "hash-object", "-w", "--stdin", data=raw, env=env)
            git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
        tree = git(repo, "write-tree", env=env)
        commit = git(
            repo,
            "commit-tree",
            tree,
            "-p",
            parent,
            data=b"Snapshot public LangGraph comparison results\n",
            env=env,
        )
        git(repo, "push", "origin", f"{commit}:refs/heads/{branch}", env=env)
        return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument("--git-repository", type=Path, required=True)
    parser.add_argument("--branch", default="results/langgraph-comparison-daded0d6")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    source, root, repo = (
        p.resolve(strict=True) for p in (args.source, args.comparison_root, args.git_repository)
    )
    sys.path.insert(0, str(source / "src"))
    spec = importlib.util.spec_from_file_location(
        "frozen_comparison", source / "scripts/run_langgraph_runtime_comparison.py"
    )
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    manifest = read(root / "manifest.json")
    comparison.validate_manifest(source, manifest)
    identity = comparison.digest(manifest)
    if args.branch != "results/langgraph-comparison-" + manifest["code_sha"][:8]:
        raise ValueError("dedicated results branch required")
    prefix = f"docs/benchmark/{args.branch.split('/')[-1]}/"
    reference = f"refs/heads/{args.branch}"

    def remote_head():
        answer = git(repo, "ls-remote", "--exit-code", "--heads", "origin", reference, absent=True)
        if answer and not re.fullmatch(r"[0-9a-f]{40}\s+" + re.escape(reference), answer):
            raise ValueError("ambiguous remote branch")
        return answer.split()[0] if answer else None

    parent = remote_head()
    previous = {}
    if parent:
        git(repo, "fetch", "--no-write-fetch-head", "--no-tags", "origin", reference)
        git(repo, "merge-base", "--is-ancestor", manifest["code_sha"], parent)
        saved = json.loads(git(repo, "show", f"{parent}:{prefix}manifest.json"))
        if saved != manifest:
            raise ValueError("remote manifest differs")
        previous = json.loads(git(repo, "show", f"{parent}:{prefix}controller-state.json"))
        if previous["controller_pid"] != args.controller_pid:
            raise ValueError("controller PID differs from published identity")
    else:
        git(repo, "fetch", "--no-write-fetch-head", "--no-tags", str(source), manifest["code_sha"])
        parent = manifest["code_sha"]
    ticks, alive = process(args.controller_pid)
    expected_ticks = previous.get("controller_start_ticks", ticks)
    if expected_ticks is None:
        raise ProcessLookupError("controller identity unavailable")
    seen = dict(previous.get("result_sha256", {}))
    while True:
        if read(root / "manifest.json") != manifest:
            raise ValueError("manifest changed")
        ticks, alive = process(args.controller_pid)
        rows, reserved = {}, []
        for trial in manifest["trials"]:
            folder = root / "trials" / trial["id"]
            if (root / "trials").is_symlink() or folder.is_symlink():
                raise ValueError("symlinked trial directory")
            if folder.exists():
                reserved.append(trial["id"])
            path = folder / "result.json"
            if path.exists() or path.is_symlink():
                rows[trial["id"]] = public_row(read(path), trial, identity)
        hashes = {key: comparison.digest(value) for key, value in rows.items()}
        if any(hashes.get(key) != value for key, value in seen.items()):
            raise ValueError("published or observed result changed")
        seen.update(hashes)
        summary = comparison.summarize(root, manifest)
        if sum(
            (root / "trials" / t["id"] / "result.json").is_file() for t in manifest["trials"]
        ) != len(rows) or summary["reserved"] != len(reserved):
            continue  # A new immutable result was published during this read.
        state = (
            "CONTROLLER_IDENTITY_CHANGED"
            if ticks not in (None, expected_ticks)
            else "COMPLETE"
            if len(rows) == len(manifest["trials"])
            else "STOPPED_INCOMPLETE"
            if not alive
            else "RUNNING"
            if reserved
            else "WAITING"
        )
        terminal = state not in {"WAITING", "RUNNING"}
        controller = dict(
            state=state,
            controller_pid=args.controller_pid,
            controller_start_ticks=expected_ticks,
            observed_start_ticks=ticks,
            process_alive=alive,
            observed_unix_seconds=time.time(),
            manifest_sha256=identity,
            completed_results=len(rows),
            reserved_without_result=[key for key in reserved if key not in rows],
            result_sha256=hashes,
        )
        changed = (
            not previous
            or len(rows) - previous["completed_results"] >= 5
            or previous["state"] == "WAITING"
            and state == "RUNNING"
            or terminal
            and (state != previous.get("state") or hashes != previous["result_sha256"])
        )
        if changed:
            expected_remote = parent if previous else None
            if remote_head() != expected_remote:
                raise RuntimeError("remote branch changed")
            payload = {
                prefix + "manifest.json": manifest,
                prefix + "summary.json": summary,
                prefix + "controller-state.json": controller,
            }
            payload.update(
                {prefix + f"trials/{key}/result.json": value for key, value in rows.items()}
            )
            parent = publish(repo, parent, args.branch, payload)
            previous = controller
            print(json.dumps({"state": state, "results": len(rows), "commit": parent}), flush=True)
        if args.once or terminal:
            return 0
        time.sleep(30)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(type(exc).__name__, file=sys.stderr, flush=True)
        raise SystemExit(1) from None
