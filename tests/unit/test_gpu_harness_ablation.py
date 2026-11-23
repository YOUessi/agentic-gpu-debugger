"""Zero-cost paired study tests: preflight, frozen order, no cherry-picking.

The test executor is substituted; these tests make NO paid model call and
never start Docker/GPU or invoke the private evaluator.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from gpu_agent.harness.ablation import (
    Preflight,
    TrialRecord,
    create_benchmark_plan,
    preflight_paired_experiment,
    run_paired_experiment,
    summarize_sweep,
)

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=True,
        text=True,
        timeout=15,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    source = tmp_path / "repo"
    source.mkdir()
    _git(source, "init", "-q")
    for path in (
        "hardset_v2/manifest.json",
        "harness/vector_io.cpp",
        "harness/vector_api.h",
        "harness/vendor/json.hpp",
        *(
            f"hardset_v2/{case}/{role}/kernel.cu"
            for case in ("h001", "h002", "h003")
            for role in ("mutant", "clean")
        ),
    ):
        target = source / "benchmarks" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BENCH / path, target)
    _git(source, "add", ".")
    _git(
        source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "-q",
        "-m",
        "frozen hardset registry",
    )
    return source


def _model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("GPU_AGENT_STORE_FALSE_SUPPORTED", "1")


def _secret(tmp_path: Path) -> Path:
    secret = tmp_path / "outside-evaluator-secret"
    secret.write_bytes(b"unit-test-not-real-evaluator-secret-0001")
    secret.chmod(0o600)
    return secret


def test_2x2_plan_precommits_case_order_before_model_calls(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    plan = create_benchmark_plan(
        repo,
        cases=("h001", "h002", "h003"),
        repetitions=2,
        include_workflow_D=True,
    )
    assert plan.base_sha == _git(repo, "rev-parse", "HEAD")
    assert len(plan.units) == 30
    assert plan.units[0].arm == "legacy_single"
    assert plan.units[1].arm == "legacy_batch"
    assert plan.units[2].arm == "evidence_single"
    assert plan.units[3].arm == "evidence_batch"
    assert plan.units[4].arm == "workflow_D"
    assert [unit.arm for unit in plan.units[5:9]] == [
        "legacy_batch",
        "evidence_single",
        "evidence_batch",
        "legacy_single",
    ]
    assert plan.units[-1].index == 30
    assert len({(u.case_id, u.arm, u.repetition) for u in plan.units}) == 30


def test_missing_credentials_and_secret_are_blocked_before_any_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repo(tmp_path)
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "GPU_AGENT_STORE_FALSE_SUPPORTED",
    ):
        monkeypatch.delenv(key, raising=False)
    output = tmp_path / "blocked"
    preflight = preflight_paired_experiment(repository, output=output)
    assert not preflight.ready
    assert "MODEL_NOT_CONFIGURED" in preflight.blockers
    assert "EVALUATOR_SECRET_NOT_CONFIGURED" in preflight.blockers
    assert not output.exists()
    with pytest.raises(ValueError, match="explicit paid"):
        run_paired_experiment(
            repository,
            output,
            evaluator_secret_file=tmp_path / "missing",
        )
    blocked = run_paired_experiment(
        repository,
        output,
        evaluator_secret_file=tmp_path / "missing",
        allow_paid_calls=True,
        allow_real_gpu=True,
    )
    assert isinstance(blocked, Preflight)
    assert blocked.ready is False
    assert (output / "preflight.json").exists()
    assert (output / "frozen-plan.json").exists()
    assert not list(output.glob("unit-*.json"))
    assert not list(output.glob("unit-*/"))
    assert "test-only-not-a-real-key" not in (output / "frozen-plan.json").read_text()


def test_preflight_is_credential_safe_and_requires_nonpublic_0600_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    _model_env(monkeypatch)
    secret = _secret(tmp_path)
    plan = preflight_paired_experiment(
        repo,
        output=tmp_path / "not-yet-created",
        evaluator_secret_file=secret,
    )
    assert plan.ready
    assert plan.plan.endpoint_host == "api.deepseek.com"
    assert plan.plan.model_name == "test-model"
    assert "OPENAI_API_KEY" not in plan.model_dump_json()
    assert "test-only-not-a-real-key" not in plan.model_dump_json()
    secret.chmod(0o644)
    blocked = preflight_paired_experiment(repo, evaluator_secret_file=secret)
    assert blocked.blockers == ("EVALUATOR_SECRET_NOT_PRIVATE",)
    blocked = preflight_paired_experiment(
        repo, output=repo / "session", evaluator_secret_file=secret
    )
    assert "INVALID_OUTPUT_ROOT" in blocked.blockers


def test_all_planned_units_are_recorded_including_failures_without_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_agent.harness import ablation

    repo = _repo(tmp_path)
    _model_env(monkeypatch)
    secret = _secret(tmp_path)
    executed: list[int] = []

    def fake_one(repository, output, plan, spec, evaluator_secret):
        assert repository == repo
        assert evaluator_secret == secret.read_bytes()
        assert plan.base_sha == _git(repo, "rev-parse", "HEAD")
        executed.append(spec.index)
        verdict = "EXPERIMENTAL_STRICT_FIXED" if spec.index in (1, 4) else "MODEL_OR_RUNTIME_ERROR"
        return TrialRecord(
            spec=spec,
            disposition=verdict,
            public_accepted=verdict == "EXPERIMENTAL_STRICT_FIXED",
            strict_verdict="VERIFIED_FIXED" if verdict == "EXPERIMENTAL_STRICT_FIXED" else None,
            source_candidate_sha256=None,
            public_run_id=None,
            model_calls=2 if spec.index == 1 else None,
            model_tokens=420 if spec.index == 1 else None,
            failure_type=None if verdict == "EXPERIMENTAL_STRICT_FIXED" else "ProviderError",
            elapsed_seconds=0.01,
        )

    monkeypatch.setattr(ablation, "_run_one", fake_one)
    output = tmp_path / "paired-study"
    result = run_paired_experiment(
        repo,
        output,
        evaluator_secret_file=secret,
        cases=("h001",),
        allow_paid_calls=True,
        allow_real_gpu=True,
    )
    assert not isinstance(result, Preflight)
    assert result.all_units_accounted_for
    assert result.expected_units == result.completed_units == 4
    assert result.missing_units == 0
    assert executed == [1, 2, 3, 4]
    assert [item.disposition for item in result.records] == [
        "EXPERIMENTAL_STRICT_FIXED",
        "MODEL_OR_RUNTIME_ERROR",
        "MODEL_OR_RUNTIME_ERROR",
        "EXPERIMENTAL_STRICT_FIXED",
    ]
    assert result.by_arm["legacy_single"]["strict_fixed"] == 1
    assert result.by_arm["legacy_batch"]["failures"] == 1
    assert result.by_arm["legacy_batch"]["measured_model_tokens"] is None
    assert len(list(output.glob("unit-*.json"))) == 4
    assert json.loads((output / "paired-report.json").read_text())["completed_units"] == 4
    with pytest.raises(ValueError, match="already exists"):
        run_paired_experiment(
            repo,
            output,
            evaluator_secret_file=secret,
            allow_paid_calls=True,
            allow_real_gpu=True,
        )
    assert executed == [1, 2, 3, 4]


def test_missing_unit_is_not_silently_dropped_from_results(tmp_path: Path) -> None:
    plan = create_benchmark_plan(_repo(tmp_path))
    records = (
        TrialRecord(
            spec=plan.units[0],
            disposition="MODEL_OR_RUNTIME_ERROR",
            public_accepted=False,
            strict_verdict=None,
            source_candidate_sha256=None,
            public_run_id=None,
            elapsed_seconds=0,
        ),
    )
    report = summarize_sweep(plan, records)
    assert report.completed_units == 1
    assert report.expected_units == 4
    assert report.missing_units == 3
    assert not report.all_units_accounted_for
    assert report.by_arm["legacy_batch"]["measured_model_tokens"] is None
    with pytest.raises(ValueError, match="frozen unit identities"):
        summarize_sweep(plan, (records[0], records[0]))


def test_preflight_rejects_registry_drift_and_case_duplication(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="invalid registered"):
        create_benchmark_plan(repo, cases=("h001", "h001"))
    kernel = repo / "benchmarks/hardset_v2/h001/mutant/kernel.cu"
    kernel.write_text(kernel.read_text().replace("expf(x)", "expf(x - 1)", 1))
    with pytest.raises(ValueError, match="source does not match pinned registry"):
        create_benchmark_plan(repo)
