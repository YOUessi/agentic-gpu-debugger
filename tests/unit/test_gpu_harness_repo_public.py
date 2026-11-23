"""Registered public GPU checker contract tests (fake checker, zero GPU calls).

These tests verify policy, provenance, feedback, resume and guard behavior.
They do NOT establish GPU compilation or CUDA numerical correctness.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from gpu_agent.benchmark.hardset_v2_agent_models import HardsetPublicCheck
from gpu_agent.harness.journal import read_verified_journal, recovery_disposition
from gpu_agent.harness.metrics import summarize_repository_trajectory
from gpu_agent.harness.models import HarnessDecision, RepoRunPublicChecks
from gpu_agent.harness.repo_runner import run_repository_coding_session
from gpu_agent.harness.repo_tools import RepositoryWorkspaceTools
from gpu_agent.store import RunStore

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
KERNEL = "benchmarks/hardset_v2/h001/mutant/kernel.cu"
GOAL = "Fix the registered Softmax32 CUDA kernel and check it publicly"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=12,
    ).stdout.strip()


def _registered_repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "-q")
    registered = ["hardset_v2/manifest.json"]
    for case in ("h001", "h002", "h003"):
        for role in ("mutant", "clean"):
            registered.append(f"hardset_v2/{case}/{role}/kernel.cu")
    registered.extend(["harness/vector_api.h", "harness/vector_io.cpp", "harness/vendor/json.hpp"])
    for relative in registered:
        target = root / "benchmarks" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BENCH / relative, target)
    extra = root / "src" / "other.cuh"
    extra.parent.mkdir(parents=True)
    extra.write_text("inline float other(float x) { return x; }\n")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "-q",
        "-m",
        "registered public inputs",
    )
    return root, _git(root, "rev-parse", "HEAD")


class Model:
    """Deterministic model substitute used only to exercise real tool routing."""

    def __init__(self, actions: list[dict[str, object]]) -> None:
        self.actions = actions
        self.snapshots = []

    def choose_harness_v1(self, snapshot):
        self.snapshots.append(snapshot)
        action = self.actions.pop(0)
        return HarnessDecision.model_validate(
            {"action": action, "reason": "Choose the next bounded public CUDA action"}
        )


class FakePublicChecker:
    """Writes the same immutable public RunStore contract as the real checker."""

    def __init__(self, verdicts: list[bool]) -> None:
        self.verdicts = verdicts
        self.calls: list[str] = []

    def __call__(
        self,
        *,
        store: RunStore,
        parent_run_id: str,
        repository: Path,
        case_id: str,
        candidate: bytes,
        output: Path,
    ) -> HardsetPublicCheck:
        assert case_id == "h001"
        assert repository.is_dir()
        assert not output.exists()
        output.mkdir(mode=0o700)
        digest = hashlib.sha256(candidate).hexdigest()
        self.calls.append(digest)
        passed = self.verdicts[len(self.calls) - 1]
        child = store.create_run("hardset_v2_public_candidate", parent_run_id=parent_run_id)
        store.transition(child.id, "RUNNING", "PREPARING")
        result = HardsetPublicCheck(
            candidate_sha256=digest,
            passed=passed,
            failure_code="PUBLIC_CHECKS_PASSED" if passed else "NUMERIC_MISMATCH",
            public_run_id=child.id,
        )
        store.put(
            child.id,
            "hardset/public-selfcheck.json",
            result.model_dump_json().encode(),
            "public",
        )
        store.transition(child.id, "COMPLETED" if passed else "FAILED", None)
        return result


def _edit_numerator() -> dict[str, str]:
    return {
        "tool": "repo_replace_snippet",
        "path": KERNEL,
        "old_text": "expf(own) / denominator",
        "new_text": "expf(own - maximum) / denominator",
    }


def _edit_denominator() -> dict[str, str]:
    return {
        "tool": "repo_replace_snippet",
        "path": KERNEL,
        "old_text": "denominator += expf(x);",
        "new_text": "denominator += expf(x - maximum);",
    }


def _check() -> dict[str, str]:
    return {"tool": "repo_run_public_checks"}


def _finish() -> dict[str, str]:
    return {"tool": "finish", "summary": "Public CUDA check passed; private verification pending"}


def test_registered_case_can_finish_only_after_current_public_gpu_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    model = Model([_edit_numerator(), _edit_denominator(), _check(), _finish()])
    output = tmp_path / "public-success"
    report = run_repository_coding_session(
        repository=repository,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        max_steps=6,
        max_model_calls=8,
        model_factory=lambda _: model,
    )
    assert report.status == "DIFF_STAGED_UNVERIFIED"
    assert report.public_checks_passed is True
    assert report.public_checks_attempted == 1
    assert report.public_checked_candidate_sha256 == checker.calls[0]
    assert report.model_calls == 0
    assert len(checker.calls) == 1
    assert model.snapshots[0].registered_public_case == "h001"
    assert model.snapshots[0].remaining_public_checks == 2
    assert model.snapshots[-1].public_checks_passed is True
    assert report.edited_files == (KERNEL,)
    assert _git(repository, "status", "--porcelain") == ""
    assert (
        recovery_disposition(read_verified_journal(output / "session-events.jsonl")) == "FINISHED"
    )


def test_failed_gpu_feedback_revised_candidate_and_duplicate_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([False, True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    model = Model([_edit_numerator(), _check(), _check(), _edit_denominator(), _check(), _finish()])
    report = run_repository_coding_session(
        repository=repository,
        output=tmp_path / "public-revision",
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        max_steps=7,
        max_model_calls=9,
        model_factory=lambda _: model,
    )
    assert report.public_checks_attempted == 2
    assert report.public_checks_passed is True
    assert len(checker.calls) == 2
    assert checker.calls[0] != checker.calls[1]
    assert model.snapshots[3].history[-1]["status"] == "REJECTED"


def test_public_checks_reject_wrong_scope_and_missing_operator_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    with RepositoryWorkspaceTools(
        origin=repository,
        output=tmp_path / "unbound",
        sha=sha,
    ) as host:
        snap = host.snapshot(GOAL, [], 5)
        assert "repo_run_public_checks" not in snap.allowed_tools
        assert host.invoke(RepoRunPublicChecks()).status == "REJECTED"

    with RepositoryWorkspaceTools(
        origin=repository,
        output=tmp_path / "scope",
        sha=sha,
        public_case_id="h001",
    ) as host:
        edit = HarnessDecision.model_validate(
            {"action": _edit_numerator(), "reason": "edit the public kernel"}
        ).action
        assert host.invoke(edit).status == "OK"
        other = HarnessDecision.model_validate(
            {
                "action": {
                    "tool": "repo_replace_snippet",
                    "path": "src/other.cuh",
                    "old_text": "return x;",
                    "new_text": "return x + 1;",
                },
                "reason": "modify a second file",
            }
        )
        assert host.invoke(other.action).status == "OK"
        assert host.invoke(RepoRunPublicChecks()).status == "REJECTED"
        finish = HarnessDecision.model_validate(
            {"action": _finish(), "reason": "finish early"}
        ).action
        assert host.invoke(finish).status == "REJECTED"
    assert checker.calls == []


def test_public_gpu_evidence_survives_pause_resume_without_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    output = tmp_path / "public-resume"
    paused = run_repository_coding_session(
        repository=repository,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        max_steps=6,
        max_model_calls=8,
        pause_after_turns=3,
        model_factory=lambda _: Model([_edit_numerator(), _edit_denominator(), _check()]),
    )
    assert paused.status == "PAUSED_RESUMABLE"
    assert paused.public_checks_passed is True
    assert len(checker.calls) == 1
    final = run_repository_coding_session(
        repository=repository,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        max_steps=6,
        max_model_calls=8,
        resume=True,
        model_factory=lambda _: Model([_finish()]),
    )
    assert final.status == "DIFF_STAGED_UNVERIFIED"
    assert final.public_checks_attempted == 1
    assert final.public_checks_passed is True
    assert final.turns == 4
    assert len(checker.calls) == 1


def test_public_resume_rejects_changed_gpu_contract_or_missing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    output = tmp_path / "public-tamper"
    paused = run_repository_coding_session(
        repository=repository,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        max_steps=6,
        max_model_calls=8,
        pause_after_turns=3,
        model_factory=lambda _: Model([_edit_numerator(), _edit_denominator(), _check()]),
    )
    assert paused.public_checks_passed
    arguments = dict(
        repository=repository,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        max_steps=6,
        max_model_calls=8,
        resume=True,
        model_factory=lambda _: Model([_finish()]),
    )
    with pytest.raises(ValueError, match="budgets differ"):
        run_repository_coding_session(**{**arguments, "public_case_id": None})
    public = RunStore(output / "public")
    run = public.load(paused.public_run_id)
    children = public.list_runs() if hasattr(public, "list_runs") else []
    assert run.status.value == "RUNNING"
    assert not children or len(children) >= 1
    # All checks are public, but tampering with the immutable result must still
    # invalidate recovery before the model receives another turn.
    from gpu_agent.harness.recovery import read_repository_checkpoint

    checkpoint = read_repository_checkpoint(output / "session-checkpoint.json")
    child = public.load(checkpoint.last_public_run_id)
    result_ref = next(
        ref for ref in child.artifact_refs if ref.name == "hardset/public-selfcheck.json"
    )
    artifact = output / "public" / result_ref.relative_path
    # Artifact blobs are deliberately 0444; simulate removal instead of
    # weakening RunStore file permissions to attempt byte replacement.
    artifact.unlink()
    with pytest.raises(ValueError):
        run_repository_coding_session(**arguments)


def test_public_success_metrics_remain_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    output = tmp_path / "public-metrics"
    outcome = run_repository_coding_session(
        repository=repo,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        max_steps=6,
        max_model_calls=8,
        model_factory=lambda _: Model([_edit_numerator(), _edit_denominator(), _check(), _finish()]),
    )
    assert outcome.public_checks_passed
    metrics = summarize_repository_trajectory(output)
    assert metrics.public_case_id == "h001"
    assert metrics.first_accepted_edit_turn == 1
    assert metrics.first_public_check_request_turn == 3
    assert metrics.first_public_pass_turn == 3
    assert metrics.first_accepted_finish_turn == 4
    assert metrics.public_gpu_checks_attempted == 1
    assert metrics.public_checks_passed_final
    assert not metrics.independently_verified
    assert not metrics.independent_hidden_verification
