"""Operator-only, manifest-driven repeated trials of an unchanged Harness.

The controller starts existing recovery and independent verification CLIs.
No model-supplied shell, source, or command is executed by this controller.
Preflight is read-only. Each attempt is reserved once, then retained.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import difflib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

FROZEN = "be3dafd6988972dfdb51cf2faee6984103565973"
RELEASE = "e6f31d0f7d45491e0781d80a825e3d2cf4886801"
CASES = {"h001", "h002", "h003"}
TRIAL_RE = re.compile(r"r0[1-5]-(qwen|deepseek)-(F-h00[1-3]|R-A-h001|R-B-h001|R-A-h003)")
RESULT_RE = re.compile(r"(build|run|sanitizer)/[0-9a-f]{32}/result\.json")
EDIT_SHA_RE = re.compile(r"source_sha256=([0-9a-f]{64})")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def command_output(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, timeout=30).strip()


def git(repo: Path, *args: str) -> str:
    return command_output(["git", "-C", str(repo), *args])


def require_frozen(repo: Path, manifest: dict) -> None:
    if git(repo, "rev-parse", "HEAD") != FROZEN or git(repo, "status", "--porcelain"):
        raise ValueError("frozen runtime checkout is not clean")
    for path, expected in manifest["runtime_git_objects"].items():
        if git(repo, "rev-parse", f"HEAD:{path}") != expected:
            raise ValueError("runtime Git object drift")


def provider_environment(path: Path) -> dict[str, str]:
    """Read only the existing authorized provider settings; never log values."""
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != 0o600:
        raise ValueError("existing provider configuration must be regular 0600")
    allowed = {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
               "GPU_AGENT_STORE_FALSE_SUPPORTED"}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = shlex.split(line, comments=True)
        if parts and parts[0] == "export":
            parts = parts[1:]
        if len(parts) != 1 or "=" not in parts[0]:
            continue
        name, value = parts[0].split("=", 1)
        if name in allowed:
            if name in result or "$" in value or chr(96) in value:
                raise ValueError("ambiguous provider configuration")
            result[name] = value
    if (not result.get("OPENAI_API_KEY")
            or result.get("OPENAI_BASE_URL", "").rstrip("/") != "https://api.deepseek.com"
            or result.get("OPENAI_MODEL") != "deepseek-v4-pro"
            or result.get("GPU_AGENT_STORE_FALSE_SUPPORTED") != "1"):
        raise ValueError("existing DeepSeek project configuration differs")
    return result


def validate_plan(m: dict) -> None:
    assert m["candidate_release_sha"] == RELEASE and m["frozen_runtime_sha"] == FROZEN
    b = m["budget"]
    assert [b[k] for k in ("max_tool_steps", "max_model_calls", "max_output_tokens",
                           "max_compile_only", "max_full_public_checks", "max_public_probes",
                           "agent_watchdog_wall_seconds")] == [16, 18, 8192, 6, 2, 4, 600]
    assert m["harness"]["ast_target_binding"] and m["harness"]["state_scoped_actions"]
    assert not m["harness"]["planner"] and not m["harness"]["extended_syntax"]
    assert len(m["trial_order"]) == 60 and len(m["blocks"]) == 10 and len(m["units"]) == 6
    assert [t["id"] for t in m["trial_order"]] == [
        i for block in m["blocks"] for i in block["trial_ids"]]
    assert len({t["id"] for t in m["trial_order"]}) == 60
    cells = collections.Counter((t["model"], t["unit"]) for t in m["trial_order"])
    assert set(cells.values()) == {5} and len(cells) == 12
    assert all(TRIAL_RE.fullmatch(t["id"]) for t in m["trial_order"])
    units = {u["id"]: u for u in m["units"]}
    for t in m["trial_order"]:
        assert 1 <= t["rep"] <= 5 and t["unit"] in units
        assert t["id"] == f"r{t['rep']:02d}-{t['model']}-{t['unit']}"
        assert t["server_seed"] == (42 + t["rep"] if t["model"] == "qwen" else None)
    for u in m["units"]:
        assert u["case_id"] in CASES
        assert u["failure_kind"] in {"none", "build", "numeric"}
        if u["group"] == "from_mutant":
            assert u["failure_kind"] == "none"
            assert u["compile_gate"] and u["allow_revert"] and u["public_diagnostics"]


def gpu_state() -> dict[str, Any]:
    fields = command_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                             "--format=csv,noheader,nounits"]).splitlines()
    if len(fields) != 1:
        raise ValueError("benchmark requires the pinned single GPU")
    name, total, used, driver = (v.strip() for v in fields[0].split(","))
    return {"name": name, "memory_total_mib": int(total), "memory_used_mib": int(used),
            "driver": driver}


def preflight(a: argparse.Namespace, m: dict) -> dict:
    validate_plan(m)
    require_frozen(a.repository, m)
    # Release/controller repo must still contain the identical runtime objects.
    controller = Path(__file__).resolve().parents[1]
    for path, expected in m["runtime_git_objects"].items():
        if git(controller, "rev-parse", f"HEAD:{path}") != expected:
            raise ValueError("controller checkout differs from frozen runtime")
    for case, detail in m["development_set"]["mutants"].items():
        path = a.repository / "benchmarks" / "hardset_v2" / case / "mutant/kernel.cu"
        assert file_digest(path) == detail["sha256"]
    for u in m["units"]:
        if u["failure_kind"] != "none":
            prior = (a.evidence_root / u["prior_path"]).resolve(strict=True)
            assert prior.is_relative_to(a.evidence_root.resolve())
            assert file_digest(prior) == u["prior_sha256"]
    assert platform.python_version() == m["runtime"]["python"]
    for name, version in m["runtime"]["packages"].items():
        assert importlib.metadata.version(name) == version, name
    vllm_version = command_output([str(a.vllm.parent / "python"), "-c",
        "import importlib.metadata as m; print(m.version('vllm'))"])
    assert vllm_version == "0.19.1"
    for path, expected in m["models"]["qwen"]["files_sha256"].items():
        assert file_digest(a.qwen_model / path) == expected, path
    image = command_output(["docker", "image", "inspect", m["runtime"]["cuda_image"],
                            "--format", "{{.Id}}"])
    assert image == m["runtime"]["cuda_image_id"]
    gs = gpu_state()
    assert gs["name"] == "NVIDIA GeForce RTX 4090 Laptop GPU"
    assert gs["memory_total_mib"] == 16376 and gs["driver"] == m["runtime"]["driver"]
    # Inspect only file type/mode, never evaluator-secret bytes here.
    secret_mode = a.evaluator_secret_file.lstat().st_mode
    assert stat.S_ISREG(secret_mode) and stat.S_IMODE(secret_mode) == 0o600
    assert not a.evaluator_secret_file.resolve().is_relative_to(a.repository.resolve())
    provider_environment(a.provider_config)
    assert a.vllm.is_file() and Path(sys.executable).with_name("gpu-agent").is_file()
    return {"ready": True, "release_sha": RELEASE, "runtime_sha": FROZEN,
            "manifest_sha256": file_digest(a.manifest), "runner_sha256": file_digest(Path(__file__)),
            "controller_commit": git(controller, "rev-parse", "HEAD"),
            "gpu": gs, "cuda_image_id": image, "vllm_version": vllm_version,
            "trials": 60, "model_calls_performed": 0, "gpu_checks_performed": 0,
            "provider_key_configured": True, "holdout_ready": False,
            "baseline_comparison_ready": False}


def stop_owned(proc: subprocess.Popen) -> None:
    """Only the process group created by this controller; never name-based kill."""
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def invoke(argv: list[str], env: dict[str, str], stdout: Path, stderr: Path,
           seconds: int, cwd: Path) -> dict[str, Any]:
    start = time.monotonic()
    timed_out = False
    with stdout.open("xb") as out, stderr.open("xb") as err:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err,
                                start_new_session=True)
        try:
            proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_owned(proc)
        except BaseException:
            stop_owned(proc)
            raise
    return {"exit_code": proc.returncode, "timed_out": timed_out,
            "wall_seconds": round(time.monotonic() - start, 3)}


@contextlib.contextmanager
def qwen_server(a: argparse.Namespace, seed: int, log: Path):
    assert seed in (43, 44, 45, 46, 47)
    if gpu_state()["memory_used_mib"] >= 1000:
        raise ValueError("GPU is not idle before local serving")
    env = dict(os.environ)
    for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_LOG"):
        env.pop(k, None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_LOGGING_LEVEL="WARNING")
    argv = [str(a.vllm), "serve", str(a.qwen_model), "--served-model-name", "qwen3-8b-awq",
            "--host", "127.0.0.1", "--port", "8127", "--max-model-len", "16384",
            "--gpu-memory-utilization", "0.65", "--enforce-eager", "--reasoning-parser",
            "qwen3", "--seed", str(seed)]
    with log.open("xb") as stream:
        proc = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise ValueError("local model service exited")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8127/v1/models", timeout=2) as response:
                        names = [x["id"] for x in json.load(response)["data"]]
                    if names == ["qwen3-8b-awq"]:
                        break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(2)
            else:
                raise ValueError("local model service did not become ready")
            print("LOCAL_SERVER_READY", seed, flush=True)
            yield
        finally:
            stop_owned(proc)


def session_environment(a: argparse.Namespace, model: str) -> dict[str, str]:
    env = dict(os.environ)
    for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_BASE_URL",
              "OPENAI_MODEL", "OPENAI_LOG", "GPU_AGENT_ALLOW_LOOPBACK_HTTP"):
        env.pop(k, None)
    if model == "deepseek":
        env.update(provider_environment(a.provider_config))
        env["GPU_AGENT_ALLOW_LOOPBACK_HTTP"] = "0"
    else:
        assert model == "qwen"
        env.update(OPENAI_BASE_URL="http://127.0.0.1:8127/v1",
                   OPENAI_MODEL="qwen3-8b-awq", OPENAI_API_KEY="local-development-key",
                   GPU_AGENT_ALLOW_LOOPBACK_HTTP="1", GPU_AGENT_STORE_FALSE_SUPPORTED="1")
    env.update(PYTHONPATH=str(a.repository / "src"),
               GPU_AGENT_LLM_TIMEOUT_SECONDS="180", GPU_AGENT_MAX_OUTPUT_TOKENS="8192")
    return env


def audit_session(repo: Path, session: Path, unit: dict, model_id: str) -> dict:
    """Verify public ledgers and reconstruct candidate bytes; no private data."""
    from gpu_agent.agent.provider import provider_invocations
    from gpu_agent.harness.check_metrics import public_check_lifecycle
    from gpu_agent.harness.journal import read_verified_journal, recovery_disposition
    from gpu_agent.harness.repo_audit import read_repository_tool_events
    from gpu_agent.harness.syntax_tools import rewrite_syntax_nodes, syntax_page
    from gpu_agent.store import RunStore, read_regular

    report = read_json(session / "controlled-recovery.json")
    if report.get("audit_error"):
        raise ValueError("frozen recovery CLI reported an audit error")
    config = read_json(session / "experiment-config.json")
    for k in ("case_id", "failure_kind", "compile_gate", "allow_revert", "public_diagnostics"):
        assert config[k] == unit[k]
    for k, expected in {"base_sha": FROZEN, "model": model_id, "max_steps": 16,
            "max_model_calls": 18, "max_output_tokens": 8192, "max_compiles": 6,
            "max_public_checks": 2, "ast_target_binding": True, "state_scoped_actions": True,
            "hierarchical_mode": False, "extended_syntax": False}.items():
        assert config[k] == expected, k
    if unit["failure_kind"] != "none":
        assert config["prior_result_sha256"] == unit["prior_sha256"]
        assert config["prior_candidate_sha256"] == unit["prior_candidate_sha256"]
    agent = session / "agent"
    journal = read_verified_journal(agent / "session-events.jsonl")
    first = journal[0]["payload"]
    assert first["base_git_commit"] == FROZEN
    for k in ("case_id", "compile_gate", "allow_revert", "public_diagnostics",
              "max_steps", "max_model_calls", "max_output_tokens", "max_compiles",
              "max_public_checks", "ast_target_binding", "state_scoped_actions"):
        key = "public_case_id" if k == "case_id" else k
        assert first[key] == config[k], k
    rid = first["public_run_id"]
    store = RunStore(agent / "public")
    parent = store.load(rid)
    n = sum(bool(re.fullmatch(r"repo-harness/v1/tool-[0-9]{3}\.json", ref.name))
            for ref in parent.artifact_refs)
    events = read_repository_tool_events(store, rid, expected_count=n,
                                         require_action=True, require_observation=True)
    lifecycle = public_check_lifecycle(store, rid, journal, events)
    inv = provider_invocations(store, rid)
    assert len(inv) <= 18 and n <= 16 and len(inv) == report["model_physical_calls"]
    assert all(x.configured_model == model_id and x.configured_max_output_tokens == 8192
               and x.repository_role == "executor" for x in inv)
    assert all(x.response_model in (None, model_id) for x in inv)
    assert sum(x.attempt == 1 for x in inv) <= 2
    prefix = report["operator_prefix_tool_turns"]
    assert 0 <= prefix <= n
    relative = f"benchmarks/hardset_v2/{unit['case_id']}/mutant/kernel.cu"
    original = read_regular(repo / relative, 1024 * 1024)
    current = original
    before_edit = None
    pinned: dict[str, dict] = {}
    live_edits = []
    probe_requests: set[tuple] = set()
    duplicate_probes = 0
    binding_rejections = 0
    rejections: collections.Counter = collections.Counter()
    live_checks = []
    build_results = []
    verified_sources: dict[str, bytes] = {}
    child_turns = {}
    last_diff = digest(b"")
    for turn, e in enumerate(events, 1):
        action, status, tool = e["action"], e["status"], e["tool"]
        obs, code = e["observation"], e.get("rejection_code")
        sha = digest(current)
        live = turn > prefix
        if status == "REJECTED":
            assert e["diff_sha256"] == last_diff, "rejection changed source diff"
            if live:
                rejections[code or "UNCLASSIFIED_REJECTION"] += 1
            if code == "CUDA_AST_TARGET_BINDING_MISMATCH":
                assert e["rejection_detail"]["no_source_written"] is True
                assert e["rejection_detail"]["source_sha256"] == sha
                binding_rejections += int(live)
        if tool == "repo_inspect_syntax" and status == "OK":
            page = json.loads(obs)
            calculated = syntax_page(current, symbol=action["symbol"],
                                     offset=action["offset"], limit=action["limit"])
            assert calculated == page
            for node in page["nodes"]:
                assert node["id"] not in pinned
                pinned[node["id"]] = node
        elif tool == "repo_replace_syntax" and status == "OK":
            assert action["expected_source_sha256"] == sha and action["path"] == relative
            edits = []
            for item in action["edits"]:
                node = pinned[item["node_id"]]
                assert not node["code_truncated"] and node["code"] == item["expected_old_code"]
                edits.append({"node_id": node["id"], "function": node["function"],
                              "line": node["start_line"], "old": node["code"],
                              "replacement": item["replacement_code"]})
            before_edit = current
            current = rewrite_syntax_nodes(current, expected_source_sha256=sha,
                edits=[(x["node_id"], x["replacement_code"]) for x in action["edits"]])
            assert EDIT_SHA_RE.search(obs) and EDIT_SHA_RE.search(obs)[1] == digest(current)
            pinned.clear()
            if live:
                live_edits.append({"turn": turn, "sha256": digest(current), "edits": edits})
        elif tool == "repo_revert_last_edit" and status == "OK":
            assert before_edit is not None and action["expected_source_sha256"] == sha
            current, before_edit = before_edit, None
            assert json.loads(obs)["source_sha256"] == digest(current)
            pinned.clear()
        elif tool == "repo_probe_public":
            key = (sha, action["phase"], action["input_size"])
            duplicate_probes += int(live and key in probe_requests)
            probe_requests.add(key)
            if status in ("OK", "FAILED"):
                body = json.loads(obs)
                assert body["candidate_sha256"] == sha
                child_turns[body["run_id"]] = turn
        elif tool == "repo_compile_candidate" and status in ("OK", "FAILED"):
            body = json.loads(obs)
            assert body["candidate_sha256"] == sha == action["expected_source_sha256"]
            child_turns[body["run_id"]] = turn
            build_results.append({"turn": turn, "sha256": sha, "status": body["status"],
                                  "actor": "model" if live else "operator"})
        elif tool == "repo_run_public_checks" and status in ("OK", "FAILED"):
            body = json.loads(obs)
            assert body["candidate_sha256"] == sha
            child_turns[body["public_run_id"]] = turn
            if live:
                live_checks.append({"turn": turn, "sha256": sha,
                    "dispatched": turn in lifecycle.executed, "passed": body["passed"],
                    "failure_code": body["failure_code"]})
                if turn in lifecycle.passed:
                    assert body["passed"] is True
                    verified_sources[sha] = current
        last_diff = e["diff_sha256"]
    actual = read_regular(agent / "worktree" / relative, 1024 * 1024)
    assert actual == current
    assert len(live_edits) == report["live_ast_edit_accepted_count"]
    assert bool(verified_sources) == report["public_pass"]
    assert set(verified_sources) == set(report["live_public_passed_candidate_shas"])
    # After public pass the frozen host only exposes Finish; there can be
    # at most one newly passing source per trial, and it must be final.
    assert len(verified_sources) <= 1
    assert not verified_sources or digest(current) in verified_sources
    durations = {"operator": {"build": 0.0, "run": 0.0, "sanitizer": 0.0},
                 "model": {"build": 0.0, "run": 0.0, "sanitizer": 0.0}}
    unattributed_gpu_records = 0
    for child in store.children(rid):
        for ref in child.artifact_refs:
            match = RESULT_RE.fullmatch(ref.name)
            if match is None:
                continue
            body = json.loads(store.read(ref))
            elapsed = body.get("elapsed_ms")
            assert isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0
            if child.id not in child_turns:
                unattributed_gpu_records += 1
                continue
            durations["operator" if child_turns[child.id] <= prefix else "model"][match[1]] += elapsed
    usage: dict[str, int | None] = {}
    for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "total_tokens"):
        values = [getattr(x.usage, k) if x.usage else None for x in inv]
        usage[k] = sum(values) if all(v is not None for v in values) else None
    usage["physical_calls"] = len(inv)
    usage["valid_physical_outputs"] = sum(x.state == "COMPLETED" for x in inv)
    usage["format_retries"] = sum(x.attempt == 1 for x in inv)
    usage["schema_invalid_outputs"] = sum(x.error_code == "LLM_INVALID_OUTPUT" for x in inv)
    usage["incomplete_outputs"] = sum((x.error_code or "").startswith("LLM_INCOMPLETE") for x in inv)
    first_sha = live_edits[0]["sha256"] if live_edits else None
    finish = any(e["tool"] == "finish" and e["status"] == "OK" for e in events[prefix:])
    return {"public_run_id": rid, "report_sha256": file_digest(session / "controlled-recovery.json"),
        "prefix_turns": prefix, "live_turns": n-prefix, "total_turns": n,
        "initial_failure_reproduced": report.get("first_failure_reproduced"),
        "terminal_reason": report["terminal_reason"], "journal_disposition": recovery_disposition(journal),
        "finish_accepted": finish, "step_budget_exhausted": report.get("step_budget_exhausted"),
        "final_candidate_sha256": digest(current), "source_reconstruction_verified": True,
        "first_accepted_live_edit_turn": live_edits[0]["turn"] if live_edits else None,
        "live_ast_edit_accepted_count": len(live_edits),
        "build_passed_candidate_count": report["build_passed_candidate_count"],
        "numeric_dispatched_candidate_count": report["numeric_dispatched_candidate_count"],
        "public_check_passed_candidate_count": len(verified_sources),
        "first_live_candidate_public_pass": first_sha in verified_sources if first_sha else None,
        "public_passed_candidate_sha256": next(iter(verified_sources), None),
        "revert_accepted": report.get("revert_accepted"),
        "binding_rejection_count": binding_rejections, "binding_mismatch_write_count": 0,
        "duplicate_probe_request_count": duplicate_probes, "rejection_counts": dict(rejections),
        "live_edits": live_edits, "live_checks": live_checks, "build_results": build_results,
        "usage": usage, "actual_response_models": sorted({x.response_model for x in inv if x.response_model}),
        "invocation_errors": [x.error_code for x in inv if x.error_code],
        "public_gpu_elapsed_ms": durations, "unattributed_gpu_result_records": unattributed_gpu_records,
        "actual_billed_cost": None}


def strict_check(a: argparse.Namespace, folder: Path, unit: dict, audit: dict,
                 env: dict[str, str]) -> dict:
    sha = audit["public_passed_candidate_sha256"]
    if sha is None:
        return {"status": "NOT_PUBLIC_PASSED", "projection": None}
    if audit["journal_disposition"] not in ("FINISHED", "FAILED_REVIEW_REQUIRED"):
        return {"status": "BLOCKED_UNCERTAIN_SESSION", "projection": None}
    session = folder / "session"
    agent = session / "agent"
    cli = str(Path(sys.executable).with_name("gpu-agent"))
    result_dir = folder / "strict-verification"
    if audit["finish_accepted"]:
        mode = "verify-repo"
        argv = [cli, "harness", "verify-repo", "--repository", str(a.repository),
                "--session", str(agent), "--output", str(result_dir),
                "--evaluator-secret-file", str(a.evaluator_secret_file)]
    else:
        # Do not change the Agent journal or invent a Finish. The independent
        # operator-only verifier accepts a scoped patch. Audit has proved the
        # source is the exact passing candidate and the session is closed.
        mode = "operator-exact-public-candidate"
        relative = f"benchmarks/hardset_v2/{unit['case_id']}/mutant/kernel.cu"
        old = (a.repository / relative).read_bytes()
        new = (agent / "worktree" / relative).read_bytes()
        assert digest(new) == sha and old != new
        lines = old.decode().splitlines(keepends=True)
        updated = new.decode().splitlines(keepends=True)
        assert all(x.endswith("\n") for x in lines+updated)
        patch = "".join(difflib.unified_diff(lines, updated, fromfile="a/kernel.cu",
                                              tofile="b/kernel.cu", n=3))
        patch_path = folder / "attested-public-candidate.diff"
        with patch_path.open("x", encoding="utf-8") as stream:
            stream.write(patch)
        argv = [cli, "benchmark", "verify-hardset", "--repository", str(a.repository),
                "--case", unit["case_id"], "--patch", str(patch_path),
                "--output", str(result_dir), "--evaluator-secret-file", str(a.evaluator_secret_file)]
    # The verifier does not need or receive model API credentials.
    private_env = {k: v for k, v in env.items() if k not in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY")}
    result = invoke(argv, private_env, folder/"strict-projection.json",
                    folder/"strict-stderr.log", 600, a.repository)
    if result["timed_out"]:
        return {"status": "BLOCKED_WATCHDOG", "mode": mode, **result, "projection": None}
    try:
        from gpu_agent.benchmark.hardset_v2_verifier import HardsetPrivateProjection
        from gpu_agent.store import RunStore
        projection = HardsetPrivateProjection.model_validate_json(
            (folder/"strict-projection.json").read_bytes()).model_dump(mode="json")
        assert projection["candidate_sha256"] == sha
        assert projection["case_id"] == unit["case_id"]
        public = RunStore(result_dir / "public")
        manifests = [public.load(p.name) for p in (result_dir/"public").iterdir()
                     if p.is_dir() and re.fullmatch(r"[0-9a-f]{32}", p.name)]
        assert len(manifests) == 1 and manifests[0].kind == "hardset_v2_strict_projection"
        refs = [r for r in manifests[0].artifact_refs if r.name == "hardset/strict-verdict.json"]
        assert len(refs) == 1 and json.loads(public.read(refs[0])) == projection
    except (ValueError, OSError, AssertionError):
        return {"status": "BLOCKED_VERIFIER_PROVENANCE", "mode": mode, **result, "projection": None}
    return {"status": projection["verdict"], "mode": mode, **result, "projection": projection,
            "strict_public_run_id": manifests[0].id,
            "projection_sha256": file_digest(folder/"strict-projection.json")}


def run_trial(a: argparse.Namespace, trial: dict, unit: dict, env: dict[str, str]) -> dict:
    folder = a.output / "trials" / trial["id"]
    folder.mkdir(mode=0o700)
    write_new(folder/"attempt-reserved.json", {"trial": trial, "unit": unit,
              "manifest_sha256": file_digest(a.manifest), "reserved_unix_seconds": time.time()})
    argv = [sys.executable, str(a.repository/"scripts/run_repository_failed_candidate_recovery.py"),
            "--repository", str(a.repository), "--output", str(folder/"session"),
            "--case", unit["case_id"], "--failure-kind", unit["failure_kind"],
            "--max-steps", "16", "--max-model-calls", "18", "--max-output-tokens", "8192",
            "--max-compiles", "6", "--ast-target-binding", "--state-scoped-actions",
            "--allow-paid-calls", "--allow-real-gpu"]
    for flag, enabled in (("compile-gate", unit["compile_gate"]), ("allow-revert", unit["allow_revert"]),
                          ("public-diagnostics", unit["public_diagnostics"])):
        if enabled:
            argv.append("--"+flag)
    if unit["failure_kind"] != "none":
        prior = a.evidence_root / unit["prior_path"]
        assert file_digest(prior) == unit["prior_sha256"]
        argv += ["--prior-result", str(prior), "--prior-result-sha256", unit["prior_sha256"]]
    print("TRIAL_START", trial["id"], flush=True)
    execution = invoke(argv, env, folder/"agent-stdout.log", folder/"agent-stderr.log",
                       600, a.repository)
    record: dict[str, Any] = {"trial": trial, "unit": unit, "agent_execution": execution,
                              "strict": None, "audit": None, "first_correct_edit_turn": None,
                              "success": False}
    if execution["timed_out"]:
        record.update(status="BLOCKED_WATCHDOG", halt_suite=True)
    else:
        try:
            audit = audit_session(a.repository, folder/"session", unit, env["OPENAI_MODEL"])
            record["audit"] = audit
            strict = strict_check(a, folder, unit, audit, env)
            record["strict"] = strict
            success = strict["status"] == "VERIFIED_FIXED"
            record["success"] = success
            if success:
                candidate = strict["projection"]["candidate_sha256"]
                record["first_correct_edit_turn"] = min(
                    x["turn"] for x in audit["live_edits"] if x["sha256"] == candidate)
            blocked = strict["status"].startswith("BLOCKED_") or (
                unit["failure_kind"] != "none" and audit["initial_failure_reproduced"] is not True)
            record["status"] = "SUCCESS" if success else ("BLOCKED" if blocked else "FAILURE")
            record["halt_suite"] = strict["status"] in ("BLOCKED_WATCHDOG", "BLOCKED_VERIFIER_PROVENANCE")
        except (ValueError, OSError, AssertionError, KeyError, TypeError) as exc:
            record.update(status="BLOCKED_AUDIT", audit_error=type(exc).__name__, halt_suite=True)
    write_new(folder/"trial-record.json", record)
    print("TRIAL_END", trial["id"], record["status"], "agent_wall", execution["wall_seconds"],
          "strict", (record.get("strict") or {}).get("status"), flush=True)
    return record


def wilson(success: int, n: int) -> list[float] | None:
    if not n:
        return None
    p = success/n
    z = 1.959963984540054
    centre = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [round(max(0.0, centre-radius), 6), round(min(1.0, centre+radius), 6)]


def summarize(output: Path, manifest: dict) -> dict:
    rows = []
    for t in manifest["trial_order"]:
        path = output/"trials"/t["id"]/"trial-record.json"
        if path.exists():
            row = read_json(path)
            assert row["trial"] == t
            rows.append(row)
    cells = []
    for model in ("deepseek", "qwen"):
        for u in manifest["units"]:
            group = [r for r in rows if r["trial"]["model"] == model and r["trial"]["unit"] == u["id"]]
            ok = sum(r["success"] for r in group)
            audited = [r["audit"] for r in group if r["audit"] is not None]
            calls = sum(x["usage"]["physical_calls"] for x in audited)
            wall = sum(r["agent_execution"]["wall_seconds"] for r in group)
            tokens = [x["usage"]["total_tokens"] for x in audited]
            total_tokens = sum(tokens) if len(audited)==len(group) and all(v is not None for v in tokens) else None
            shas = sorted({r["strict"]["projection"]["candidate_sha256"] for r in group if r["success"]})
            cells.append({"model": model, "unit": u["id"], "group": u["group"], "case_id": u["case_id"],
                "planned": 5, "completed": len(group), "successes": ok,
                "failures": sum(r["status"]=="FAILURE" for r in group),
                "blocked": sum(r["status"].startswith("BLOCKED") for r in group),
                "strict_success_wilson_95": wilson(ok, 5) if len(group)==5 else None,
                "unique_strict_source_shas": shas, "physical_calls_observed": calls,
                "agent_wall_seconds": round(wall,3), "total_tokens": total_tokens,
                "all_attempt_calls_per_success": calls/ok if ok and len(audited)==len(group) else None,
                "all_attempt_agent_wall_seconds_per_success": wall/ok if ok else None,
                "all_attempt_tokens_per_success": total_tokens/ok if ok and total_tokens is not None else None,
                "actual_billed_cost": None})
    return {"planned_trajectories": 60, "completed_records": len(rows), "cells": cells,
            "independent_development_tasks": 3, "holdout_tasks": 0,
            "success_counts_not_independent_tasks": True, "all_complete": len(rows)==60}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-config", type=Path, required=True)
    parser.add_argument("--evaluator-secret-file", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--allow-real-gpu", action="store_true")
    args = parser.parse_args()
    for key in ("manifest", "repository", "evidence_root", "provider_config", "evaluator_secret_file",
                "qwen_model", "vllm", "gpu_lock"):
        setattr(args, key, getattr(args,key).expanduser().resolve(strict=True))
    args.output = args.output.expanduser().absolute()
    os.umask(0o077)
    m = read_json(args.manifest)
    sys.path.insert(0, str(args.repository/"src"))
    p = preflight(args,m)
    print("PREFLIGHT", json.dumps(p), flush=True)
    if not args.execute:
        return 0
    if not args.allow_paid_calls or not args.allow_real_gpu:
        parser.error("explicit approval required for the frozen 60-trial suite")
    if args.output.exists() or args.output.is_symlink():
        parser.error("use a new output directory; never resample a reserved trial")
    if args.output.is_relative_to(args.repository):
        parser.error("output cannot be inside frozen runtime")
    with args.gpu_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if gpu_state()["memory_used_mib"] >= m["runtime"]["max_idle_gpu_memory_mib"]:
            raise ValueError("GPU is busy")
        args.output.mkdir(parents=True, mode=0o700)
        (args.output/"trials").mkdir(mode=0o700)
        write_new(args.output/"preflight.json", p)
        write_new(args.output/"frozen-manifest.json", m)
        units = {u["id"]:u for u in m["units"]}
        trials = {t["id"]:t for t in m["trial_order"]}
        halt = False
        for block in m["blocks"]:
            require_frozen(args.repository, m)
            model = block["model"]
            if model == "qwen":
                context = qwen_server(args, block["server_seed"],
                                      args.output/f"local-server-rep{block['rep']:02d}.log")
            else:
                context = contextlib.nullcontext()
            with context:
                env = session_environment(args, model)
                for tid in block["trial_ids"]:
                    t = trials[tid]
                    record = run_trial(args, t, units[t["unit"]], env)
                    if record["halt_suite"]:
                        halt = True
                        break
            progress = summarize(args.output, m)
            write_new(args.output/f"progress-after-{block['rep']:02d}-{model}.json", progress)
            if halt:
                break
            # No next backend admission until the owned model server released GPU.
            for _ in range(30):
                if gpu_state()["memory_used_mib"] < 1000:
                    break
                time.sleep(2)
            else:
                halt = True
                break
        require_frozen(args.repository,m)
        summary = summarize(args.output,m)
        summary["halted"] = halt
        write_new(args.output/"summary.json",summary)
        print("SUITE_END",json.dumps({"completed":summary["completed_records"],
            "all_complete":summary["all_complete"],"halted":halt}),flush=True)
        return 0 if summary["all_complete"] and not halt else 1


if __name__ == "__main__":
    raise SystemExit(main())
