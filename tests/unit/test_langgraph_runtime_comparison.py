"""Offline orchestration tests for the new fixed-budget comparison."""

from __future__ import annotations

import fcntl
import importlib.util
import select
import subprocess
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/run_langgraph_runtime_comparison.py"
SPEC = importlib.util.spec_from_file_location("comparison", PATH)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def test_queued_controller_waits_for_real_shared_lock_before_admission(tmp_path):
    secret = tmp_path / "secret"
    secret.write_bytes(b"fixture-only" * 3)
    secret.chmod(0o600)
    lock = tmp_path / "gpu.lock"
    marker = tmp_path / "admitted"
    child = """
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('queued_comparison', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
root = Path(sys.argv[2])
module.environments = lambda *args: ({}, {})
module.validate_manifest = lambda *args: None
def idle():
    (root / 'admitted').write_text('GPU admission occurred')
    return {'idle': True}
module.idle_proof = idle
args = SimpleNamespace(allow_paid_calls=True, allow_real_gpu=True,
    evaluator_secret_file=root/'secret', gpu_lock=root/'gpu.lock',
    wait_for_gpu_lock=True)
raise SystemExit(module.execute(args, root/'repo', root/'output', {'trials': []}))
"""
    with lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", child, str(PATH), str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            assert select.select([proc.stdout], [], [], 5)[0], "queued controller did not start"
            assert proc.stdout.readline().strip() == "WAITING_FOR_SHARED_GPU_LOCK"
            assert proc.poll() is None and not marker.exists()
            fcntl.flock(held, fcntl.LOCK_UN)
            out, err = proc.communicate(timeout=5)
            assert proc.returncode == 0, err
            assert "SHARED_GPU_LOCK_ACQUIRED" in out and marker.exists()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


def fake_manifest():
    trials = [
        {"id": f"r01-h001-{arm}", "replicate": 1, "case": "h001", "arm": arm, **config}
        for arm, config in comparison.ARMS.items()
    ]
    return {
        "trials": trials,
        "code_sha": "a" * 40,
        "model": "fixture",
        "endpoint": "https://example.test",
    }


def test_timeout_is_not_replayed_and_is_primary_failure(tmp_path, monkeypatch):
    m = fake_manifest()
    trial = m["trials"][0]
    calls = []
    monkeypatch.setattr(
        comparison,
        "observed_usage",
        lambda *_: {
            "physical_calls": 3,
            "tokens_lower_bound": 100,
            "tokens_complete": False,
            "errors": [],
        },
    )

    def fake_run(argv, env, folder, stage, seconds, cwd):
        calls.append((stage, seconds))
        return {"exit_code": -15, "timed_out": True, "wall_seconds": 600.008}

    output = tmp_path / "output"
    row = comparison.run_trial(
        tmp_path,
        output,
        m,
        trial,
        tmp_path / "fixture-only",
        ({}, {}),
        runner=fake_run,
        cleanup=lambda: {"idle": True},
    )
    assert row["status"] == "AGENT_TIMEOUT" and not row["strict_fix_within_budget"]
    assert calls == [("agent", 600)]
    with pytest.raises(ValueError, match="resample"):
        comparison.run_trial(
            tmp_path,
            output,
            m,
            trial,
            tmp_path / "fixture-only",
            ({}, {}),
            runner=fake_run,
            cleanup=lambda: {"idle": True},
        )
    assert calls == [("agent", 600)]
    stats = comparison.summarize(output, m)["groups"]["legacy-react:all"]
    assert stats["started"] == 1 and stats["primary"] == 0 and stats["secondary"] is None
    assert stats["timeouts"] == 1 and stats["tokens_total"] is None
    assert stats["tokens_lower_bound"] == 100


def test_reserved_interruption_is_preserved_without_evaluation(tmp_path, monkeypatch):
    m = fake_manifest()
    trial = m["trials"][0]
    root = tmp_path / "results"
    folder = root / "trials" / trial["id"]
    comparison.write_once(
        folder / "reservation.json", {"trial": trial, "manifest_sha256": comparison.digest(m)}
    )
    comparison.write_once(folder / "agent-admitted.json", {"test": True})
    monkeypatch.setattr(
        comparison,
        "observed_usage",
        lambda *_: {
            "physical_calls": 2,
            "tokens_lower_bound": 10,
            "tokens_complete": False,
            "errors": [],
        },
    )
    row = comparison.recover_reserved(folder, trial, m, proof={"idle": True})
    assert row["status"] == "INTERRUPTED_OUTCOME_UNKNOWN"
    assert not row["strict_fix_within_budget"] and not row["retry_same_replicate"]
    assert comparison.summarize(root, m)["pending"] == 3


def test_equal_budgets_and_no_evaluator_path_in_agent_argv(tmp_path):
    m = fake_manifest()
    secret = tmp_path / "fixture-only"
    for trial in m["trials"]:
        agent, strict = comparison.commands(tmp_path, tmp_path / "trial", m, trial, secret)
        for name, expected in (
            ("--max-steps", "16"),
            ("--max-model-calls", "24"),
            ("--max-compiles", "6"),
            ("--max-public-checks", "2"),
            ("--memory-mode", "off"),
        ):
            assert agent[agent.index(name) + 1] == expected
        assert "--resume" not in agent
        assert str(secret) not in agent and str(secret) in strict
        assert agent[agent.index("--runtime") + 1] == trial["runtime"]
        assert ("--hierarchical" in agent) == trial["hierarchical"]


def test_strict_success_requires_independent_exact_candidate_projection(tmp_path, monkeypatch):
    from gpu_agent.harness.models import RepoHarnessOutcome

    m = fake_manifest()
    trial = m["trials"][-1]
    candidate = "b" * 64
    calls = []
    monkeypatch.setattr(
        comparison,
        "observed_usage",
        lambda *_: {
            "physical_calls": 4,
            "tokens_lower_bound": 100,
            "tokens_complete": True,
            "errors": [],
        },
    )

    def fake_run(argv, env, folder, stage, seconds, cwd):
        calls.append(stage)
        if stage == "agent":
            report = RepoHarnessOutcome(
                status="DIFF_STAGED_UNVERIFIED",
                base_sha=m["code_sha"],
                diff_sha256="c" * 64,
                edited_files=("kernel.cu",),
                turns=4,
                tool_calls=4,
                model_calls=4,
                model_tokens=100,
                reason="WORKSPACE_DIFF_STAGED_UNVERIFIED",
                public_run_id="d" * 32,
                public_checks_attempted=1,
                public_checks_passed=True,
                public_checked_candidate_sha256=candidate,
            )
            comparison.write_once(folder / "agent/result.json", report.model_dump(mode="json"))
        return {"exit_code": 0, "timed_out": False, "wall_seconds": 12.0}

    def exact_projection(folder, selected_trial, expected):
        assert selected_trial == trial and expected == candidate
        return {"verdict": "VERIFIED_FIXED", "candidate_sha256": expected}

    monkeypatch.setattr(comparison, "strict_projection", exact_projection)
    row = comparison.run_trial(
        tmp_path,
        tmp_path / "results",
        m,
        trial,
        tmp_path / "fixture-only",
        ({}, {}),
        runner=fake_run,
        cleanup=lambda: {"idle": True},
    )
    assert calls == ["agent", "strict"] and row["strict_fix_within_budget"]
    assert row["status"] == "STRICT_SUCCESS"
