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


def test_partial_failed_repository_agent_keeps_audited_steps(tmp_path: Path) -> None:
    """Do not report zero steps merely because no final result.json exists."""
    from gpu_agent.harness.ablation import _failed_repository_public_audit
    from gpu_agent.harness.models import HarnessDecision
    from gpu_agent.harness.repo_runner import run_repository_coding_session

    repo = _repo(tmp_path)
    target = repo / "benchmarks/hardset_v2/h001/mutant/kernel.cu"
    assert target.is_file()

    class OneEditThenError:
        def __init__(self) -> None:
            self.calls = 0

        def choose_harness_v1(self, snapshot):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("injected model failure after durable first action")
            return HarnessDecision.model_validate(
                {
                    "action": {
                        "tool": "repo_replace_snippet",
                        "path": "benchmarks/hardset_v2/h001/mutant/kernel.cu",
                        "old_text": "denominator += expf(x);",
                        "new_text": "denominator += expf(x - maximum);",
                    },
                    "reason": "Test one completed edit before the model fails",
                }
            )

    trial_folder = tmp_path / "trial"
    with pytest.raises(RuntimeError, match="injected model failure"):
        run_repository_coding_session(
            repository=repo,
            output=trial_folder / "agent",
            base_sha=_git(repo, "rev-parse", "HEAD"),
            goal="Fix the registered h001 public softmax instability",
            public_case_id="h001",
            max_steps=6,
            max_model_calls=8,
            model_factory=lambda _: OneEditThenError(),
        )
    assert not (trial_folder / "agent/result.json").exists()
    recovered = _failed_repository_public_audit(trial_folder)
    assert recovered["tool_turns"] == 1
    assert recovered["first_edit_turn"] == 1
    assert recovered["model_calls"] == 0
    assert recovered["model_tokens"] == 0
    assert recovered["run_id"]


def test_loopback_local_vllm_accepted_only_with_explicit_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    secret = _secret(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "local-only-dummy-not-a-real-provider-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:18731/v1")
    monkeypatch.setenv("OPENAI_MODEL", "qwen3-8b-awq")
    monkeypatch.setenv("GPU_AGENT_STORE_FALSE_SUPPORTED", "1")
    monkeypatch.setenv("GPU_AGENT_ALLOW_LOOPBACK_HTTP", "1")
    ready = preflight_paired_experiment(
        repo,
        evaluator_secret_file=secret,
        output=tmp_path / "local-pilot",
    )
    assert ready.ready
    assert ready.plan.model_name == "qwen3-8b-awq"
    assert ready.plan.endpoint_host == "127.0.0.1"
    assert "local-only-dummy-not-a-real-provider-key" not in ready.model_dump_json()
    monkeypatch.delenv("GPU_AGENT_ALLOW_LOOPBACK_HTTP")
    with pytest.raises(ValueError, match="opt-in"):
        preflight_paired_experiment(repo, evaluator_secret_file=secret)


def test_three_arm_architecture_plan_preserves_causal_separation(tmp_path: Path) -> None:
    from gpu_agent.harness.ablation import _arm_configuration

    repo = _repo(tmp_path)
    plan = create_benchmark_plan(
        repo, cases=("h001", "h002"), repetitions=1, architecture_study=True
    )
    assert plan.architecture_study
    assert len(plan.units) == 6
    assert [u.arm for u in plan.units[:3]] == [
        "evidence_single",
        "react_syntax",
        "hierarchical_syntax",
    ]
    assert [u.arm for u in plan.units[3:]] == [
        "react_syntax",
        "hierarchical_syntax",
        "evidence_single",
    ]
    assert _arm_configuration("evidence_single") == ("evidence_first", False, False, False)
    assert _arm_configuration("react_syntax") == ("evidence_first", False, False, True)
    assert _arm_configuration("hierarchical_syntax") == ("evidence_first", False, True, True)
    report = summarize_sweep(plan, ())
    assert report.expected_units == 6
    assert report.completed_units == 0
    assert not report.all_units_accounted_for


def test_planner_only_frozen_arm_pair_holds_tools_context_and_budget_equal(
    tmp_path: Path,
) -> None:
    from gpu_agent.harness.ablation import _arm_configuration

    root = _repo(tmp_path)
    plan = create_benchmark_plan(
        root,
        cases=("h001", "h002"),
        repetitions=2,
        planner_only_study=True,
        max_steps=8,
        max_model_calls=10,
        max_public_checks=2,
    )
    assert plan.planner_only_study
    assert not plan.architecture_study
    assert len(plan.units) == 8
    assert plan.public_probe_budget == 4
    for arm in ("react_syntax", "hierarchical_syntax"):
        config = _arm_configuration(arm)
        assert config[0] == "evidence_first"
        assert config[1] is False
        assert config[3] is True
    assert _arm_configuration("react_syntax")[2] is False
    assert _arm_configuration("hierarchical_syntax")[2] is True
    for case in ("h001", "h002"):
        for rep in (1, 2):
            assert {u.arm for u in plan.units if u.case_id == case and u.repetition == rep} == {
                "react_syntax",
                "hierarchical_syntax",
            }
    assert plan.max_steps == 8 and plan.max_model_calls == 10
    assert plan.model_budget_scheme == "equal_executor_v1"
    for unit in plan.units:
        assert unit.executor_format_retry_limit == 2
        if unit.arm == "hierarchical_syntax":
            assert unit.physical_model_call_limit == 14
            assert unit.auxiliary_model_call_limit == 4
        else:
            assert unit.physical_model_call_limit == 10
            assert unit.auxiliary_model_call_limit == 0
    with pytest.raises(ValueError, match="invalid registered"):
        create_benchmark_plan(root, architecture_study=True, planner_only_study=True)
    with pytest.raises(ValueError, match="invalid registered"):
        create_benchmark_plan(root, include_workflow_D=True, planner_only_study=True)


def test_planner_only_primary_metrics_do_not_count_rejected_edits_as_valid(
    tmp_path: Path,
) -> None:
    plan = create_benchmark_plan(_repo(tmp_path), planner_only_study=True)
    first = TrialRecord(
        spec=plan.units[0],
        disposition="PUBLIC_NOT_ACCEPTED",
        public_accepted=False,
        strict_verdict=None,
        source_candidate_sha256=None,
        public_run_id=None,
        elapsed_seconds=0.1,
        first_edit_turn=2,
        effective_patch=True,
        public_check_reached=False,
    )
    second = TrialRecord(
        spec=plan.units[1],
        disposition="PUBLIC_NOT_ACCEPTED",
        public_accepted=False,
        strict_verdict=None,
        source_candidate_sha256=None,
        public_run_id=None,
        elapsed_seconds=0.1,
        effective_patch=False,
        public_check_reached=False,
    )
    report = summarize_sweep(plan, (first, second))
    assert report.by_arm[first.spec.arm]["effective_patch_rate"] == 1.0
    assert report.by_arm[first.spec.arm]["first_effective_edit_turn_mean"] == 2.0
    assert report.by_arm[second.spec.arm]["effective_patch_rate"] == 0.0
    assert report.by_arm[second.spec.arm]["public_check_reached_rate"] == 0.0
    assert report.missing_units == 0
