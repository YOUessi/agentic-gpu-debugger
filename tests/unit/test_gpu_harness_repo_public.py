"""Registered public GPU checker contract tests (fake checker, zero GPU calls).

These tests verify policy, provenance, feedback, resume and guard behavior.
They do NOT establish GPU compilation or CUDA numerical correctness.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from gpu_agent.benchmark.hardset_v2_agent_models import HardsetPublicCheck
from gpu_agent.harness.journal import read_verified_journal, recovery_disposition
from gpu_agent.harness.metrics import summarize_repository_trajectory
from gpu_agent.harness.models import (
    HarnessDecision,
    RepoInspectSyntax,
    RepoReadFile,
    RepoReplaceSyntax,
    RepoRunPublicChecks,
    RepoSyntaxEdit,
)
from gpu_agent.harness.repo_runner import run_repository_coding_session
from gpu_agent.harness.repo_tools import RepositoryWorkspaceTools
from gpu_agent.harness.repo_verification import verify_repository_hardset_candidate
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

    def __init__(
        self, verdicts: list[bool], *, case_id: str = "h001", dispatched: bool = True
    ) -> None:
        self.verdicts = verdicts
        self.case_id = case_id
        self.dispatched = dispatched
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
        assert case_id == self.case_id
        assert repository.is_dir()
        assert not output.exists()
        output.mkdir(mode=0o700)
        digest = hashlib.sha256(candidate).hexdigest()
        self.calls.append(digest)
        passed = self.verdicts[len(self.calls) - 1]
        child = store.create_run("hardset_v2_public_candidate", parent_run_id=parent_run_id)
        store.transition(child.id, "RUNNING", "PREPARING")
        store.put(
            child.id,
            "hardset/public-check-candidate.json",
            json.dumps({"candidate_sha256": digest, "case_id": case_id}).encode(),
            "public",
        )
        if self.dispatched:
            store.put(
                child.id,
                "hardset/public-numeric-dispatch.json",
                json.dumps({"candidate_sha256": digest, "input_size": 32}).encode(),
                "public",
            )
        result = HardsetPublicCheck(
            candidate_sha256=digest,
            passed=passed,
            failure_code=(
                "PUBLIC_CHECKS_PASSED"
                if passed
                else "NUMERIC_MISMATCH"
                if self.dispatched
                else "PUBLIC_EXECUTION_UNAVAILABLE"
            ),
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
        model_factory=lambda _: Model(
            [_edit_numerator(), _edit_denominator(), _check(), _finish()]
        ),
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


def test_operator_verifier_receives_exact_scoped_cuda_diff_only_after_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_agent.benchmark.hardset_v2_verifier import HardsetPrivateProjection

    repository, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    session = tmp_path / "public-verified-separate"
    report = run_repository_coding_session(
        repository=repository,
        output=session,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        max_steps=6,
        max_model_calls=8,
        model_factory=lambda _: Model(
            [_edit_numerator(), _edit_denominator(), _check(), _finish()]
        ),
    )
    assert report.public_checks_passed and report.status == "DIFF_STAGED_UNVERIFIED"
    calls = []

    def strict_stub(origin, output, *, case_id, unified_diff, evaluator_secret):
        assert origin == repository.resolve()
        assert case_id == "h001"
        assert evaluator_secret == b"operator-owned-experimental-secret-00001"
        assert not output.exists()
        assert unified_diff.startswith("--- a/kernel.cu\n+++ b/kernel.cu\n")
        assert "expf(own - maximum)" in unified_diff
        assert "denominator += expf(x - maximum)" in unified_diff
        calls.append(unified_diff)
        return HardsetPrivateProjection(
            case_id=case_id,
            candidate_sha256=report.public_checked_candidate_sha256,
            verdict="VERIFIED_FIXED",
            reason_code="ALL_REQUIRED_CHECKS_PASSED",
        )

    monkeypatch.setattr(
        "gpu_agent.harness.repo_verification.run_strict_hardset_verification", strict_stub
    )
    private_output = tmp_path / "outside-private-evaluator"
    projection = verify_repository_hardset_candidate(
        repository=repository,
        session_output=session,
        output=private_output,
        evaluator_secret=b"operator-owned-experimental-secret-00001",
    )
    assert projection.verdict == "VERIFIED_FIXED" and len(calls) == 1
    # This is a *stubbed* unit test: no real hidden GPU verification was run.
    assert not private_output.exists()
    assert report.status == "DIFF_STAGED_UNVERIFIED"
    assert RunStore(session / "public").load(report.public_run_id).status.value == "COMPLETED"
    assert all(
        ref.visibility == "public"
        for ref in RunStore(session / "public").load(report.public_run_id).artifact_refs
    )


def test_operator_verifier_rejects_external_source_drift_and_private_output_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, sha = _registered_repository(tmp_path)
    monkeypatch.setattr(
        "gpu_agent.harness.repo_tools.check_candidate_publicly", FakePublicChecker([True])
    )
    session = tmp_path / "public-drift"
    run_repository_coding_session(
        repository=repository,
        output=session,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        max_steps=6,
        max_model_calls=8,
        model_factory=lambda _: Model(
            [_edit_numerator(), _edit_denominator(), _check(), _finish()]
        ),
    )

    kwargs = dict(
        repository=repository,
        session_output=session,
        output=tmp_path / "private-eval",
        evaluator_secret=b"operator-owned-experimental-secret-00001",
    )
    with pytest.raises(ValueError, match="private secret"):
        verify_repository_hardset_candidate(**{**kwargs, "evaluator_secret": b"too-short"})
    with pytest.raises(ValueError, match="outside repository"):
        verify_repository_hardset_candidate(**{**kwargs, "output": session / "private-evaluator"})
    candidate = session / "worktree" / KERNEL
    candidate.write_text(candidate.read_text().replace("maximum", "wrongname", 1))
    with pytest.raises(ValueError, match="outside controller"):
        verify_repository_hardset_candidate(**kwargs)
    assert not (tmp_path / "private-eval").exists()


def test_operator_verifier_preflight_rejects_source_scope_even_if_public_stub_lies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, sha = _registered_repository(tmp_path)
    monkeypatch.setattr(
        "gpu_agent.harness.repo_tools.check_candidate_publicly", FakePublicChecker([True])
    )
    tampered_include = {
        "tool": "repo_replace_snippet",
        "path": KERNEL,
        "old_text": "#include <cmath>",
        "new_text": "#include <math.h>",
    }
    session = tmp_path / "scope-refusal"
    result = run_repository_coding_session(
        repository=repository,
        output=session,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        max_steps=6,
        max_model_calls=8,
        model_factory=lambda _: Model([tampered_include, _check(), _finish()]),
    )
    assert result.public_checks_passed
    called = []

    def forbidden_verifier(*args, **kwargs):
        called.append(True)
        raise AssertionError("strict verifier must never receive an invalid scoped patch")

    monkeypatch.setattr(
        "gpu_agent.harness.repo_verification.run_strict_hardset_verification",
        forbidden_verifier,
    )
    with pytest.raises(ValueError, match="include directives"):
        verify_repository_hardset_candidate(
            repository=repository,
            session_output=session,
            output=tmp_path / "invalid-eval-output",
            evaluator_secret=b"operator-owned-experimental-secret-00001",
        )
    assert not called
    assert not (tmp_path / "invalid-eval-output").exists()


def _complete_softmax_batch() -> dict[str, object]:
    return {
        "tool": "repo_replace_many",
        "path": KERNEL,
        "edits": [
            {
                "old_text": "denominator += expf(x);",
                "new_text": "denominator += expf(x - maximum);",
            },
            {
                "old_text": "expf(own) / denominator",
                "new_text": "expf(own - maximum) / denominator",
            },
        ],
    }


class EvidenceAwareScriptedModel:
    """Ablation *fixture*: not a real LLM or demonstrated model improvement."""

    def __init__(self) -> None:
        self.snapshots = []

    def choose_harness_v1(self, snapshot):
        self.snapshots.append(snapshot)
        if snapshot.public_checks_passed:
            action = _finish()
        elif snapshot.current_candidate_sha256 is not None:
            action = _check()
        elif "denominator += expf(x);" in snapshot.workspace_source:
            action = _complete_softmax_batch()
        else:
            action = {"tool": "repo_read_file", "path": KERNEL}
        assert action["tool"] in snapshot.allowed_tools
        return HarnessDecision.model_validate(
            {"action": action, "reason": "Deterministic evidence-sensitive unit test"}
        )


def test_evidence_first_source_bootstrap_and_atomic_batch_fixes_registered_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, sha = _registered_repository(tmp_path)
    monkeypatch.setattr(
        "gpu_agent.harness.repo_tools.check_candidate_publicly", FakePublicChecker([True])
    )
    with RepositoryWorkspaceTools(
        origin=repository,
        output=tmp_path / "evidence-context-only",
        sha=sha,
        public_case_id="h001",
        context_mode="evidence_first",
    ) as host:
        before = host.snapshot(GOAL, [], 4)
        assert before.context_mode == "evidence_first"
        assert "denominator += expf(x);" in before.workspace_source
        assert "Registered editable CUDA kernel:" in before.workspace_source
        assert "hardset_v2/h001/clean/kernel.cu" not in before.workspace_source
        assert "repo_replace_many" in before.allowed_tools
        edited = host.invoke(
            HarnessDecision.model_validate(
                {"action": _complete_softmax_batch(), "reason": "Apply disjoint fixes together"}
            ).action
        )
        assert edited.status == "OK"
        updated = host.snapshot(GOAL, [], 3)
        assert "denominator += expf(x - maximum);" in updated.workspace_source
        assert "expf(own - maximum) / denominator" in updated.workspace_source
        assert len(host.workspace.edited_files) == 1
        assert "maximum" in host.workspace.diff()
        assert "expf(own) / denominator" in (repository / KERNEL).read_text()


def test_evidence_first_vs_legacy_uses_one_fewer_scripted_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([True, True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    measured = {}
    for mode in ("legacy", "evidence_first"):
        model = EvidenceAwareScriptedModel()
        output = tmp_path / mode
        result = run_repository_coding_session(
            repository=repo,
            output=output,
            base_sha=sha,
            goal=GOAL,
            public_case_id="h001",
            context_mode=mode,
            max_steps=6,
            max_model_calls=8,
            model_factory=lambda _, m=model: m,
        )
        assert result.status == "DIFF_STAGED_UNVERIFIED"
        assert result.public_checks_passed is True
        assert result.public_checks_attempted == 1
        assert result.model_calls == 0
        metrics = summarize_repository_trajectory(output)
        assert metrics.context_mode == mode
        measured[mode] = (result.turns, metrics.first_accepted_edit_turn, metrics.inspected_actions)
        assert metrics.independently_verified is False
    assert measured["legacy"] == (4, 2, 1)
    assert measured["evidence_first"] == (3, 1, 0)
    # Controlled scripted-model experiment only, not a claimed DeepSeek gain.
    assert len(checker.calls) == 2
    assert checker.calls[0] == checker.calls[1]


def test_context_ablation_mode_is_bound_into_resume_identity(tmp_path: Path) -> None:
    repo, sha = _registered_repository(tmp_path)
    output = tmp_path / "context-resume"
    pause = run_repository_coding_session(
        repository=repo,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        context_mode="evidence_first",
        max_steps=6,
        max_model_calls=8,
        pause_after_turns=1,
        model_factory=lambda _: Model([_complete_softmax_batch()]),
    )
    assert pause.status == "PAUSED_RESUMABLE"
    from gpu_agent.harness.recovery import read_repository_checkpoint

    checkpoint = read_repository_checkpoint(output / "session-checkpoint.json")
    assert checkpoint.context_mode == "evidence_first"
    with pytest.raises(ValueError, match="budgets differ"):
        run_repository_coding_session(
            repository=repo,
            output=output,
            base_sha=sha,
            goal=GOAL,
            public_case_id="h001",
            context_mode="legacy",
            max_steps=6,
            max_model_calls=8,
            resume=True,
            model_factory=lambda _: Model([_check()]),
        )
    assert "QUIESCENT" in recovery_disposition(
        read_verified_journal(output / "session-events.jsonl")
    )


class FactorialDecisionFixture:
    """A deterministic A/B test policy, NOT evidence of real model quality."""

    def choose_harness_v1(self, snapshot):
        if snapshot.public_checks_passed:
            action: dict[str, object] = _finish()
        else:
            visible = snapshot.workspace_source
            denom_bad = "denominator += expf(x);" in visible
            numerator_bad = "expf(own) / denominator" in visible
            if not denom_bad and not numerator_bad and snapshot.current_candidate_sha256 is None:
                action = {"tool": "repo_read_file", "path": KERNEL}
            elif denom_bad and numerator_bad and "repo_replace_many" in snapshot.allowed_tools:
                action = _complete_softmax_batch()
            elif denom_bad:
                action = _edit_denominator()
            elif numerator_bad:
                action = _edit_numerator()
            else:
                action = _check()
        assert action["tool"] in snapshot.allowed_tools
        return HarnessDecision.model_validate(
            {"action": action, "reason": "Deterministic factorial ablation fixture"}
        )


def test_four_arm_factorial_ablation_separates_context_and_batch_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([True, True, True, True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    totals = {}
    for context in ("legacy", "evidence_first"):
        for batch in (False, True):
            run = tmp_path / f"{context}-batch-{int(batch)}"
            result = run_repository_coding_session(
                repository=repository,
                output=run,
                base_sha=sha,
                goal=GOAL,
                public_case_id="h001",
                context_mode=context,
                allow_batch_edits=batch,
                max_steps=7,
                max_model_calls=9,
                model_factory=lambda _: FactorialDecisionFixture(),
            )
            metrics = summarize_repository_trajectory(run)
            assert result.status == "DIFF_STAGED_UNVERIFIED"
            assert result.model_calls == 0
            assert result.public_checks_passed
            assert metrics.allow_batch_edits is batch
            assert result.allow_batch_edits is batch
            assert metrics.context_mode == context
            totals[(context, batch)] = result.turns
    assert totals == {
        ("legacy", False): 5,
        ("legacy", True): 4,
        ("evidence_first", False): 4,
        ("evidence_first", True): 3,
    }
    assert len(checker.calls) == 4
    assert len(set(checker.calls)) == 1


def test_batch_ablation_permission_is_enforced_by_controller_and_resume(
    tmp_path: Path,
) -> None:
    repository, sha = _registered_repository(tmp_path)
    with RepositoryWorkspaceTools(
        origin=repository,
        output=tmp_path / "no-atomic-permission",
        sha=sha,
        public_case_id="h001",
        allow_batch_edits=False,
    ) as host:
        snapshot = host.snapshot(GOAL, [], 5)
        assert "repo_replace_many" not in snapshot.allowed_tools
        denied = HarnessDecision.model_validate(
            {"action": _complete_softmax_batch(), "reason": "Try disabled batch edit"}
        )
        observation = host.invoke(denied.action)
        assert observation.status == "REJECTED"
        assert "disabled" in observation.content
        assert host.workspace.edited_files == ()

    output = tmp_path / "batch-policy-paused"
    pause = run_repository_coding_session(
        repository=repository,
        output=output,
        base_sha=sha,
        goal=GOAL,
        context_mode="evidence_first",
        allow_batch_edits=False,
        public_case_id="h001",
        max_steps=6,
        max_model_calls=8,
        pause_after_turns=1,
        model_factory=lambda _: Model([_edit_denominator()]),
    )
    assert pause.status == "PAUSED_RESUMABLE"
    with pytest.raises(ValueError, match="budgets differ"):
        run_repository_coding_session(
            repository=repository,
            output=output,
            base_sha=sha,
            goal=GOAL,
            context_mode="evidence_first",
            allow_batch_edits=True,
            public_case_id="h001",
            max_steps=6,
            max_model_calls=8,
            resume=True,
            model_factory=lambda _: Model([_edit_numerator()]),
        )


def test_registered_gpu_tool_capabilities_follow_real_candidate_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([False, True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    with RepositoryWorkspaceTools(
        origin=repository,
        output=tmp_path / "state-scoped",
        sha=sha,
        public_case_id="h001",
        context_mode="evidence_first",
    ) as host:
        initial = host.snapshot(GOAL, [], 7)
        assert "repo_list_files" not in initial.allowed_tools
        assert "repo_run_public_checks" not in initial.allowed_tools
        assert "finish" not in initial.allowed_tools
        forbidden = host.invoke(RepoRunPublicChecks())
        assert forbidden.status == "REJECTED"
        assert "CURRENT_STATE" in forbidden.content
        first_edit = HarnessDecision.model_validate(
            {"action": _edit_numerator(), "reason": "First source change"}
        ).action
        assert host.invoke(first_edit).status == "OK"
        edited = host.snapshot(GOAL, [], 6)
        assert "repo_run_public_checks" in edited.allowed_tools
        assert "finish" not in edited.allowed_tools
        assert host.invoke(RepoRunPublicChecks()).status == "FAILED"
        after_failed = host.snapshot(GOAL, [], 5)
        assert "repo_run_public_checks" not in after_failed.allowed_tools
        assert host.invoke(RepoRunPublicChecks()).status == "REJECTED"
        second_edit = HarnessDecision.model_validate(
            {"action": _edit_denominator(), "reason": "Complete the repair"}
        ).action
        assert host.invoke(second_edit).status == "OK"
        assert "repo_run_public_checks" in host.snapshot(GOAL, [], 4).allowed_tools
        assert host.invoke(RepoRunPublicChecks()).status == "OK"
        assert host.snapshot(GOAL, [], 3).allowed_tools == ("finish",)
    assert len(checker.calls) == 2


def test_syntax_mode_uses_controller_ast_ids_and_never_model_quoted_old_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    with RepositoryWorkspaceTools(
        origin=repo,
        output=tmp_path / "syntax-edits",
        sha=sha,
        public_case_id="h001",
        syntax_mode=True,
    ) as host:
        first = host.snapshot(GOAL, [], 8)
        assert "repo_inspect_syntax" in first.allowed_tools
        assert "repo_replace_syntax" not in first.allowed_tools
        from gpu_agent.harness.tool_schema import repository_decision_schema

        initial_schema = repository_decision_schema(
            first.allowed_tools, registered_kernel_path=KERNEL, bind_syntax_cursor=True
        )
        assert "repo_replace_syntax" not in str(initial_schema.model_json_schema())
        assert "repo_replace_snippet" not in first.allowed_tools
        inspected = HarnessDecision.model_validate(
            {
                "action": {
                    "tool": "repo_inspect_syntax",
                    "path": KERNEL,
                    "symbol": "softmax32",
                    "limit": 20,
                },
                "reason": "Inspect syntax nodes in the exact mutable CUDA kernel",
            }
        )
        observation = host.invoke(inspected.action)
        assert observation.status == "OK"
        mapping = json.loads(observation.content)
        assert mapping["source_sha256"] == hashlib.sha256((repo / KERNEL).read_bytes()).hexdigest()
        nodes = mapping["nodes"]
        denominator = next(node for node in nodes if node["code"] == "denominator += expf(x);")
        numerator = next(
            node for node in nodes if node["code"] == "out[i] = expf(own) / denominator;"
        )
        edits = [
            {
                "node_id": denominator["id"],
                "replacement_code": "denominator += expf(x - maximum);",
            },
            {
                "node_id": numerator["id"],
                "replacement_code": "out[i] = expf(own - maximum) / denominator;",
            },
        ]
        original = host.workspace.read(KERNEL)
        stale = HarnessDecision.model_validate(
            {
                "action": {
                    "tool": "repo_replace_syntax",
                    "path": KERNEL,
                    "expected_source_sha256": "f" * 64,
                    "edits": edits,
                },
                "reason": "Test stale hash rejection before mutating worktree",
            }
        )
        rejected = host.invoke(stale.action)
        assert rejected.status == "REJECTED"
        assert "CUDA_SOURCE_SHA_STALE" in rejected.content
        assert host.workspace.read(KERNEL) == original
        valid = HarnessDecision.model_validate(
            {
                "action": {
                    "tool": "repo_replace_syntax",
                    "path": KERNEL,
                    "expected_source_sha256": mapping["source_sha256"],
                    "edits": edits,
                },
                "reason": "Apply two stable-node CUDA fixes atomically",
            }
        )
        accepted = host.invoke(valid.action)
        assert accepted.status == "OK"
        source = (host.workspace.root / KERNEL).read_text()
        assert "denominator += expf(x - maximum);" in source
        assert "out[i] = expf(own - maximum) / denominator;" in source
        assert host.workspace.edited_files == (KERNEL,)
        assert "repo_run_public_checks" in host.snapshot(GOAL, [], 5).allowed_tools
        assert host.invoke(RepoRunPublicChecks()).status == "OK"
        assert host.snapshot(GOAL, [], 4).allowed_tools == ("finish",)
    assert len(checker.calls) == 1
    assert "denominator += expf(x);" in (repo / KERNEL).read_text()


def test_hierarchical_syntax_mode_persists_planner_reflection_and_verified_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_agent.harness.models import TaskPlan, TaskReflection
    from gpu_agent.harness.recovery import read_repository_checkpoint

    repo, sha = _registered_repository(tmp_path)
    fake = FakePublicChecker([True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", fake)

    class ModelWithPlan:
        def __init__(self) -> None:
            self.plans = 0
            self.reflections = 0

        def choose_repository_plan(self, snapshot):
            self.plans += 1
            return TaskPlan.model_validate(
                {
                    "milestones": [
                        {
                            "gate": gate,
                            "objective": f"Perform the registered {gate} task with evidence",
                            "success_evidence": f"Controller confirmed the {gate} gate passed",
                        }
                        for gate in ("inspect", "edit", "public_check", "finish")
                    ],
                    "failure_risk": "Avoid stale AST node IDs and mismatched source revisions",
                }
            )

        def reflect_repository_task(self, snapshot):
            self.reflections += 1
            assert snapshot.task_plan is not None
            return TaskReflection(
                observed_failure="No public error in the scripted CPU fixture",
                avoid_next="Avoid repeating already accepted CUDA source reads",
                next_priority="Finish after the registered public GPU pass",
            )

        def choose_harness_v1(self, snapshot):
            assert snapshot.task_plan is not None
            if snapshot.public_checks_passed:
                action = _finish()
            elif snapshot.current_candidate_sha256 is not None:
                action = _check()
            elif snapshot.syntax_state is not None:
                mapping = snapshot.syntax_state.model_dump(mode="json")
                nodes = mapping["observed_nodes"]
                d = next(n for n in nodes if n["code"] == "denominator += expf(x);")
                n = next(n for n in nodes if n["code"] == "out[i] = expf(own) / denominator;")
                action = {
                    "tool": "repo_replace_syntax",
                    "path": KERNEL,
                    "expected_source_sha256": mapping["source_sha256"],
                    "edits": [
                        {
                            "node_id": d["id"],
                            "replacement_code": "denominator += expf(x - maximum);",
                        },
                        {
                            "node_id": n["id"],
                            "replacement_code": "out[i] = expf(own - maximum) / denominator;",
                        },
                    ],
                }
            else:
                action = {
                    "tool": "repo_inspect_syntax",
                    "path": KERNEL,
                    "symbol": "softmax32",
                    "limit": 20,
                }
            assert action["tool"] in snapshot.allowed_tools
            return HarnessDecision.model_validate(
                {"action": action, "reason": "Follow evidence-backed CUDA plan milestones"}
            )

    output = tmp_path / "hierarchical-syntax-agent"
    model = ModelWithPlan()
    result = run_repository_coding_session(
        repository=repo,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        hierarchical_mode=True,
        syntax_mode=True,
        context_mode="evidence_first",
        max_steps=6,
        max_model_calls=8,
        model_factory=lambda _: model,
    )
    assert result.status == "DIFF_STAGED_UNVERIFIED"
    assert result.tool_calls == 4
    assert result.model_calls == 0
    assert result.public_checks_attempted == 1
    assert model.plans == 1
    assert model.reflections == 1
    assert result.hierarchical_mode
    assert result.syntax_mode
    checkpoint = read_repository_checkpoint(output / "session-checkpoint.json")
    assert checkpoint.hierarchical_mode
    assert checkpoint.syntax_mode
    assert checkpoint.state["task_reflection"]["next_priority"].startswith("Finish")
    assert checkpoint.state["task_progress"]["finished"]
    metrics = summarize_repository_trajectory(output)
    assert metrics.hierarchical_mode
    assert metrics.syntax_mode
    assert metrics.first_accepted_edit_turn == 2
    assert metrics.first_public_pass_turn == 3
    assert metrics.first_accepted_finish_turn == 4
    assert metrics.independently_verified is False


class FakeBoundedPublicProbe:
    """Controller-only stub; tests never run CUDA or expose CPU expected values."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    def __call__(
        self,
        *,
        store,
        parent_run_id,
        repository,
        case_id,
        phase,
        input_size,
        candidate,
        output,
    ):
        from gpu_agent.benchmark.hardset_v2_public_probe import PublicProbeResult

        assert case_id == "h001"
        assert input_size in {32, 33, 65, 129, 257}
        assert repository.is_dir()
        output.mkdir(mode=0o700)
        child = store.create_run("hardset_v2_public_functional_probe", parent_run_id=parent_run_id)
        store.transition(child.id, "RUNNING", "PREPARING")
        digest = hashlib.sha256(candidate).hexdigest()
        outcome = "NUMERIC_MISMATCH" if phase == "before" else "PASS"
        result = PublicProbeResult(
            case_id="h001",
            phase=phase,
            input_size=input_size,
            input_sha256="a" * 64,
            candidate_sha256=digest,
            run_id=child.id,
            outcome=outcome,
            diagnostic_region="cross_warp_tail",
        )
        store.put(
            child.id, "probe/public-summary.json", result.model_dump_json().encode(), "public"
        )
        store.transition(child.id, "COMPLETED", None)
        self.calls.append((phase, input_size, digest))
        return result


def test_registered_toolhost_before_after_probes_and_read_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_agent.harness.models import RepoProbePublic, RepoReadFile

    repo, sha = _registered_repository(tmp_path)
    fake = FakeBoundedPublicProbe()
    monkeypatch.setattr("gpu_agent.harness.repo_tools.probe_candidate_publicly", fake)
    with RepositoryWorkspaceTools(
        origin=repo,
        output=tmp_path / "public-before-after",
        sha=sha,
        public_case_id="h001",
        syntax_mode=False,
    ) as host:
        start = host.snapshot(GOAL, [], 7)
        assert start.public_probe_phase == "before"
        assert start.remaining_public_probes == 4
        assert "repo_probe_public" in start.allowed_tools
        before = host.invoke(RepoProbePublic(phase="before", input_size=33))
        assert before.status == "OK"
        public = json.loads(before.content)
        assert public["outcome"] == "NUMERIC_MISMATCH"
        assert public["diagnostic_region"] == "cross_warp_tail"
        assert "expected" not in before.content.lower()
        assert host.invoke(RepoProbePublic(phase="before", input_size=33)).status == "REJECTED"
        assert host.public_probes_attempted == 1
        for name in (
            "benchmarks/hardset_v2/h001/clean/kernel.cu",
            "benchmarks/harness/vector_io.cpp",
            "src/other.cuh",
        ):
            denied = host.invoke(RepoReadFile(path=name))
            assert denied.status == "REJECTED", (name, denied)
            assert "oracle" in denied.content.lower() or "scope" in denied.content.lower()
        assert "benchmarks/hardset_v2/h001/clean/kernel.cu" not in host.workspace.list_files()
        assert (
            host.invoke(
                HarnessDecision.model_validate(
                    {"action": _edit_denominator(), "reason": "Apply first verified source edit"}
                ).action
            ).status
            == "OK"
        )
        assert host.snapshot(GOAL, [], 5).public_probe_phase == "after"
        assert host.invoke(RepoProbePublic(phase="before", input_size=65)).status == "REJECTED"
        after = host.invoke(RepoProbePublic(phase="after", input_size=33))
        assert after.status == "OK"
        assert json.loads(after.content)["outcome"] == "PASS"
        assert host.public_probes_attempted == 2
        assert len(host.public_probe_run_ids) == 2
        assert len(host.public_probe_signatures) == 2
        assert len(fake.calls) == 2


def test_public_probe_evidence_survives_checkpoint_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_agent.harness.recovery import read_repository_checkpoint

    repo, sha = _registered_repository(tmp_path)
    fake = FakeBoundedPublicProbe()
    checker = FakePublicChecker([True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.probe_candidate_publicly", fake)
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    output = tmp_path / "probe-resume"
    paused = run_repository_coding_session(
        repository=repo,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        max_steps=8,
        max_model_calls=10,
        pause_after_turns=1,
        model_factory=lambda _: Model(
            [{"tool": "repo_probe_public", "phase": "before", "input_size": 33}]
        ),
    )
    assert paused.status == "PAUSED_RESUMABLE"
    checkpoint = read_repository_checkpoint(output / "session-checkpoint.json")
    assert checkpoint.public_probes_attempted == 1
    assert len(checkpoint.public_probe_run_ids) == 1
    resumed = run_repository_coding_session(
        repository=repo,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        max_steps=8,
        max_model_calls=10,
        resume=True,
        model_factory=lambda _: Model(
            [
                _edit_denominator(),
                {"tool": "repo_probe_public", "phase": "after", "input_size": 33},
                _edit_numerator(),
                _check(),
                _finish(),
            ]
        ),
    )
    assert resumed.status == "DIFF_STAGED_UNVERIFIED"
    assert resumed.public_probes_attempted == 2
    metrics = summarize_repository_trajectory(output)
    assert metrics.public_probes_attempted == 2
    assert metrics.public_gpu_checks_attempted == 1
    assert len(fake.calls) == 2


def test_pinned_ast_survives_compaction_cursor_binding_and_revision_change(tmp_path: Path) -> None:
    from gpu_agent.harness.tool_schema import repository_decision_schema

    repo, sha = _registered_repository(tmp_path)
    with RepositoryWorkspaceTools(
        origin=repo,
        output=tmp_path / "pinned-ast",
        sha=sha,
        public_case_id="h001",
        syntax_mode=True,
    ) as host:
        first = host.invoke(RepoInspectSyntax(path=KERNEL, symbol="softmax32", limit=3))
        assert first.status == "OK"
        pinned = host.syntax_state
        assert pinned is not None
        history = [{"tool": first.tool, "status": first.status, "content": first.content}]
        for line in range(1, 6):
            read = host.invoke(RepoReadFile(path=KERNEL, start_line=line, lines=1))
            assert read.status == "OK"
            history.append({"tool": read.tool, "status": read.status, "content": read.content})
        snapshot = host.snapshot(GOAL, history, 10)
        assert snapshot.syntax_state == pinned
        assert len(snapshot.history) == 5
        assert "AST page observed" in snapshot.context_summary
        assert "repo_inspect_syntax" in snapshot.allowed_tools
        grammar = repository_decision_schema(
            ("repo_inspect_syntax", "repo_replace_syntax"),
            registered_kernel_path=KERNEL,
            syntax_state=pinned,
            bind_syntax_cursor=True,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            grammar.model_validate(
                {
                    "action": {
                        "tool": "repo_inspect_syntax",
                        "path": KERNEL,
                        "symbol": pinned.symbol,
                        "offset": 0,
                        "limit": 8,
                    },
                    "reason": "Repeat an already visited page",
                }
            )
        duplicate = host.invoke(RepoInspectSyntax(path=KERNEL, symbol=pinned.symbol, limit=8))
        assert "REDUNDANT_SYNTAX_INSPECTION" in duplicate.content
        while host.syntax_state.next_offset is not None:
            state = host.syntax_state
            obs = host.invoke(
                RepoInspectSyntax(
                    path=KERNEL, symbol=state.symbol, offset=state.next_offset, limit=3
                )
            )
            assert obs.status == "OK"
        state = host.syntax_state
        assert state is not None
        assert "repo_inspect_syntax" not in host.snapshot(GOAL, history, 8).allowed_tools
        ids = [node.id for node in state.observed_nodes]
        assert ids == sorted(set(ids))
        node = next(x for x in state.observed_nodes if x.code == "denominator += expf(x);")
        base = {"path": KERNEL, "expected_source_sha256": state.source_sha256}
        noop = RepoReplaceSyntax(
            **base, edits=[RepoSyntaxEdit(node_id=node.id, replacement_code=node.code)]
        )
        noop_result = host.invoke(noop)
        assert f"CUDA_SYNTAX_NOOP: {node.id}" in noop_result.content
        assert node.code in noop_result.content
        assert "functional_requirement" in noop_result.content
        repeat_result = host.invoke(noop)
        assert "CUDA_SYNTAX_REPEATED_REJECTED_EDIT" in repeat_result.content
        missing = RepoReplaceSyntax(
            **base, edits=[RepoSyntaxEdit(node_id="n99", replacement_code="return;")]
        )
        missing_result = host.invoke(missing)
        assert "CUDA_SYNTAX_NODE_NOT_FOUND" in missing_result.content
        assert "n99" in missing_result.content
        fixed = RepoReplaceSyntax(
            **base,
            edits=[
                RepoSyntaxEdit(
                    node_id=node.id, replacement_code="denominator += expf(x - maximum);"
                )
            ],
        )
        assert host.invoke(fixed).status == "OK"
        assert host.syntax_state is None
        stale_result = host.invoke(fixed)
        assert "CUDA_SOURCE_SHA_STALE" in stale_result.content
        assert "expected_source_sha256" in stale_result.content
        assert "actual_source_sha256" in stale_result.content
        current = host.snapshot(GOAL, history, 5)
        assert current.syntax_state is None
        assert "repo_replace_syntax" not in current.allowed_tools
        assert "repo_inspect_syntax" in current.allowed_tools
        assert "repo_run_public_checks" in current.allowed_tools
        run = host.store.load(host.run_id)
        event = next(x for x in run.artifact_refs if x.name == "repo-harness/v1/tool-001.json")
        body = json.loads(host.store.read(event))
        assert body["action"]["symbol"] == "softmax32"
        assert body["action"]["limit"] == 3
        from gpu_agent.harness.repo_audit import read_repository_tool_events

        events = read_repository_tool_events(
            host.store,
            host.run_id,
            expected_count=host.tool_calls,
            require_action=True,
            require_observation=True,
        )
        for code, submitted, observed in (
            ("CUDA_SYNTAX_NOOP", noop, noop_result),
            ("CUDA_SYNTAX_REPEATED_REJECTED_EDIT", noop, repeat_result),
            ("CUDA_SYNTAX_NODE_NOT_FOUND", missing, missing_result),
            ("CUDA_SOURCE_SHA_STALE", fixed, stale_result),
        ):
            rejected = next(e for e in events if e["rejection_code"] == code)
            assert rejected["action"] == submitted.model_dump(mode="json")
            assert rejected["observation"] == observed.content
            assert (
                rejected["observation_sha256"]
                == hashlib.sha256(observed.content.encode()).hexdigest()
            )
            assert rejected["rejection_detail"]["correction"]
        noop_record = next(e for e in events if e["rejection_code"] == "CUDA_SYNTAX_NOOP")
        assert noop_record["rejection_detail"]["original_code"] == node.code
        assert noop_record["rejection_detail"]["node_id"] == node.id
        assert noop_record["rejection_detail"]["functional_requirement"]
        stale_record = next(e for e in events if e["rejection_code"] == "CUDA_SOURCE_SHA_STALE")
        assert stale_record["rejection_detail"]["expected_source_sha256"] == state.source_sha256
        assert stale_record["rejection_detail"]["actual_source_sha256"] != state.source_sha256


def test_ast_checkpoint_rehydrates_aliases_without_another_model_inspect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_agent.harness.recovery import read_repository_checkpoint

    repo, sha = _registered_repository(tmp_path)
    output = tmp_path / "ast-resume"
    checker = FakePublicChecker([True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    options = dict(
        repository=repo,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        syntax_mode=True,
        max_steps=6,
        max_model_calls=8,
    )
    paused = run_repository_coding_session(
        **options,
        pause_after_turns=1,
        model_factory=lambda _: Model(
            [{"tool": "repo_inspect_syntax", "path": KERNEL, "symbol": "softmax32", "limit": 20}]
        ),
    )
    assert paused.status == "PAUSED_RESUMABLE"
    checkpoint = read_repository_checkpoint(output / "session-checkpoint.json")
    state = checkpoint.syntax_state
    assert state is not None
    d = next(x for x in state.observed_nodes if x.code == "denominator += expf(x);")
    n = next(x for x in state.observed_nodes if x.code == "out[i] = expf(own) / denominator;")
    model = Model(
        [
            {
                "tool": "repo_replace_syntax",
                "path": KERNEL,
                "expected_source_sha256": state.source_sha256,
                "edits": [
                    {"node_id": d.id, "replacement_code": "denominator += expf(x - maximum);"},
                    {
                        "node_id": n.id,
                        "replacement_code": "out[i] = expf(own - maximum) / denominator;",
                    },
                ],
            },
            _check(),
            _finish(),
        ]
    )
    final = run_repository_coding_session(**options, resume=True, model_factory=lambda _: model)
    assert final.public_checks_passed
    assert final.tool_calls == 4
    assert model.snapshots[0].syntax_state == state
    assert "repo_inspect_syntax" not in model.snapshots[0].allowed_tools
    assert len(checker.calls) == 1


def test_single_step_microbench_calls_model_once_and_gpu_only_after_valid_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_agent.harness.ast_microbench import run_ast_single_step

    repo, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([True])
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)

    class Once:
        def __init__(self, noop: bool) -> None:
            self.noop = noop
            self.calls = 0

        def choose_harness_v1(self, snapshot):
            self.calls += 1
            assert self.calls == 1
            assert snapshot.allowed_tools == ("repo_replace_syntax",)
            assert snapshot.task_plan is None
            state = snapshot.syntax_state
            assert state is not None and state.next_offset is None
            d = next(n for n in state.observed_nodes if n.code == "denominator += expf(x);")
            n = next(
                n for n in state.observed_nodes if n.code == "out[i] = expf(own) / denominator;"
            )
            return HarnessDecision(
                reason="Single controlled edit decision",
                action=RepoReplaceSyntax(
                    path=KERNEL,
                    expected_source_sha256=state.source_sha256,
                    edits=[
                        RepoSyntaxEdit(
                            node_id=d.id,
                            replacement_code=d.code
                            if self.noop
                            else "denominator += expf(x - maximum);",
                        ),
                        RepoSyntaxEdit(
                            node_id=n.id,
                            replacement_code="out[i] = expf(own - maximum) / denominator;",
                        ),
                    ],
                ),
            )

    for noop in (True, False):
        model = Once(noop)
        report = run_ast_single_step(
            repository=repo,
            output=tmp_path / f"single-step-{noop}",
            base_sha=sha,
            case_id="h001",
            allow_real_gpu=True,
            model_factory=lambda _, m=model: m,
        )
        assert model.calls == 1
        assert report["model_physical_calls"] == 0
        assert report["legal_non_noop"] is not noop
        assert report["checker_reached"] is not noop
        assert report["public_check_passed"] is not noop
    assert len(checker.calls) == 1


def test_ast_partial_cursor_resume_and_checkpoint_tampering(tmp_path: Path) -> None:
    from gpu_agent.harness.recovery import read_repository_checkpoint
    from gpu_agent.harness.tool_schema import repository_decision_schema

    repo, sha = _registered_repository(tmp_path)
    output = tmp_path / "ast-partial-resume"
    options = dict(
        repository=repo,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        syntax_mode=True,
        max_steps=8,
        max_model_calls=10,
    )
    initial = run_repository_coding_session(
        **options,
        pause_after_turns=1,
        model_factory=lambda _: Model(
            [{"tool": "repo_inspect_syntax", "path": KERNEL, "symbol": "softmax32", "limit": 3}]
        ),
    )
    assert initial.status == "PAUSED_RESUMABLE"
    path = output / "session-checkpoint.json"
    original = path.read_bytes()
    state = read_repository_checkpoint(path).syntax_state
    assert state is not None and state.next_offset is not None
    assert len(state.observed_nodes) == 3

    # Tampering with the private checkpoint cannot supply fabricated AST evidence.
    cases = ("source_sha", "node_code", "next_offset", "missing_state", "history")
    for case in cases:
        body = json.loads(original)
        if case == "source_sha":
            body["syntax_state"]["source_sha256"] = "f" * 64
        elif case == "node_code":
            body["syntax_state"]["observed_nodes"][0]["code"] = "return;"
        elif case == "next_offset":
            body["syntax_state"]["next_offset"] += 1
        elif case == "missing_state":
            body["syntax_state"] = None
        else:
            body["state"]["history"][0]["content"] = "tampered observation"
            body["state"]["observation"]["content"] = "tampered observation"
        path.write_text(json.dumps(body))
        never_called = Model([])
        with pytest.raises(ValueError, match="recovered AST|checkpoint observation"):
            run_repository_coding_session(
                **options, resume=True, model_factory=lambda _, m=never_called: m
            )
        assert not never_called.snapshots
    path.write_bytes(original)
    model = Model(
        [
            {
                "tool": "repo_inspect_syntax",
                "path": KERNEL,
                "symbol": state.symbol,
                "offset": state.next_offset,
                "limit": 8,
            },
        ]
    )
    resumed = run_repository_coding_session(
        **options,
        resume=True,
        pause_after_turns=2,
        model_factory=lambda _: model,
    )
    assert resumed.status == "PAUSED_RESUMABLE"
    assert model.snapshots[0].syntax_state == state
    assert model.snapshots[0].syntax_mode
    grammar = repository_decision_schema(
        model.snapshots[0].allowed_tools,
        registered_kernel_path=KERNEL,
        syntax_state=state,
        bind_syntax_cursor=True,
    )
    parsed = grammar.model_validate(
        {
            "action": {
                "tool": "repo_inspect_syntax",
                "path": KERNEL,
                "symbol": state.symbol,
                "offset": state.next_offset,
                "limit": 8,
            },
            "reason": "Resume next public AST page",
        }
    )
    assert parsed.action.offset == state.next_offset
    after = read_repository_checkpoint(path).syntax_state
    assert after is not None and after.source_sha256 == state.source_sha256
    assert after.observed_nodes[: len(state.observed_nodes)] == state.observed_nodes
    assert len(after.visited_pages) == 2


def test_ast_public_audit_never_persists_raw_host_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_agent.harness.repo_audit import read_repository_tool_events

    repo, sha = _registered_repository(tmp_path)
    with RepositoryWorkspaceTools(
        origin=repo,
        output=tmp_path / "ast-public-exception",
        sha=sha,
        public_case_id="h001",
        syntax_mode=True,
    ) as host:
        forbidden = "private-host-diagnostic-do-not-publish"

        def fail_read(*args, **kwargs):
            raise OSError(f"/home/private/{forbidden}")

        monkeypatch.setattr(host.workspace, "inspect_syntax", fail_read)
        obs = host.invoke(RepoInspectSyntax(path=KERNEL))
        assert obs.status == "REJECTED"
        assert "PUBLIC_TOOL_IO_FAILED" in obs.content
        assert forbidden not in obs.content
        events = read_repository_tool_events(
            host.store,
            host.run_id,
            expected_count=1,
            require_action=True,
            require_observation=True,
        )
        assert events[0]["observation"] == obs.content
        assert forbidden not in json.dumps(events)
        assert host.syntax_state is None


def test_ast_replay_rejects_observation_hash_mismatch(tmp_path: Path) -> None:
    from gpu_agent.harness.repo_audit import read_repository_tool_events

    store = RunStore(tmp_path / "public-audit")
    run = store.create_run("gpu_repository_agent")
    store.transition(run.id, "RUNNING", "DIAGNOSING")
    action = RepoInspectSyntax(path=KERNEL)
    store.put(
        run.id,
        "repo-harness/v1/tool-001.json",
        json.dumps(
            {
                "tool": action.tool,
                "action": action.model_dump(mode="json"),
                "status": "OK",
                "observation": "changed feedback",
                "observation_sha256": hashlib.sha256(b"original feedback").hexdigest(),
            }
        ).encode(),
        "public",
    )
    with pytest.raises(ValueError, match="public observation does not match"):
        read_repository_tool_events(store, run.id, expected_count=1, require_observation=True)


def test_ast_rejected_edit_ledger_survives_resume_and_detects_tampering(tmp_path: Path) -> None:
    from gpu_agent.harness.recovery import read_repository_checkpoint

    repo, sha = _registered_repository(tmp_path)
    output = tmp_path / "ast-rejection-resume"

    class RejectOnce:
        action = None

        def choose_harness_v1(self, snapshot):
            if snapshot.syntax_state is None:
                return HarnessDecision(
                    action=RepoInspectSyntax(path=KERNEL, symbol="softmax32", limit=3),
                    reason="Collect initial AST evidence",
                )
            state = snapshot.syntax_state
            node = state.observed_nodes[0]
            self.action = RepoReplaceSyntax(
                path=KERNEL,
                expected_source_sha256=state.source_sha256,
                edits=[RepoSyntaxEdit(node_id=node.id, replacement_code=node.code)],
            )
            return HarnessDecision(action=self.action, reason="Exercise exact NOOP rejection")

    model = RejectOnce()
    options = dict(
        repository=repo,
        output=output,
        base_sha=sha,
        goal=GOAL,
        public_case_id="h001",
        syntax_mode=True,
        max_steps=8,
        max_model_calls=10,
    )
    first = run_repository_coding_session(
        **options,
        pause_after_turns=2,
        model_factory=lambda _: model,
    )
    assert first.status == "PAUSED_RESUMABLE"
    path = output / "session-checkpoint.json"
    original = path.read_bytes()
    checkpoint = read_repository_checkpoint(path)
    assert len(checkpoint.rejected_syntax_edits) == 1
    body = json.loads(original)
    body["rejected_syntax_edits"] = []
    path.write_text(json.dumps(body))
    never_called = Model([])
    with pytest.raises(ValueError, match="recovered AST rejected edits differ"):
        run_repository_coding_session(
            **options,
            resume=True,
            model_factory=lambda _: never_called,
        )
    assert not never_called.snapshots
    path.write_bytes(original)
    assert model.action is not None
    repeated = Model([model.action.model_dump(mode="json")])
    resumed = run_repository_coding_session(
        **options,
        resume=True,
        pause_after_turns=3,
        model_factory=lambda _: repeated,
    )
    assert resumed.status == "PAUSED_RESUMABLE"
    after = read_repository_checkpoint(path)
    assert after.rejected_syntax_edits == checkpoint.rejected_syntax_edits
    assert "CUDA_SYNTAX_REPEATED_REJECTED_EDIT" in after.state["history"][-1]["content"]


# The real ToolHost/AST/Git/RunStore run here; the model and GPU Checker are
# deliberately test doubles. These are not measurements of model quality.
_CASE_REPAIRS = {
    "h001": [
        ("denominator += expf(x);", "denominator += expf(x - maximum);"),
        ("out[i] = expf(own) / denominator;", "out[i] = expf(own - maximum) / denominator;"),
    ],
    "h002": [
        (
            "const float scale = rsqrtf(sum_sq / 32.0f + 1.0e-5f);",
            "const float scale = rsqrtf(sum_sq / static_cast<float>(valid) + 1.0e-5f);",
        ),
    ],
    "h003": [
        ("total += a[start + j];", "total += a[start + j] + b[start + j];"),
    ],
}


class CrossCaseASTFixture:
    def __init__(self, case_id: str, *, fail_first: bool = False) -> None:
        self.case_id = case_id
        self.kernel = f"benchmarks/hardset_v2/{case_id}/mutant/kernel.cu"
        self.fail_first = fail_first
        self.snapshots = []
        self.edits = []

    def choose_harness_v1(self, snapshot):
        self.snapshots.append(snapshot)
        state = snapshot.syntax_state
        if snapshot.public_checks_passed:
            action = _finish()
        elif "repo_run_public_checks" in snapshot.allowed_tools:
            action = _check()
        elif state is None or state.next_offset is not None:
            action = {
                "tool": "repo_inspect_syntax",
                "path": self.kernel,
                "symbol": "",
                "offset": state.next_offset if state else 0,
                "limit": 8,
            }
        else:
            assert state is not None
            by_code = {node.code: node.id for node in state.observed_nodes}
            if self.fail_first and not self.edits:
                # First candidate is syntactically valid but intentionally wrong.
                old, _ = _CASE_REPAIRS[self.case_id][0]
                if self.case_id == "h001":
                    wrong = "denominator += expf(x - 1.0f);"
                elif self.case_id == "h002":
                    wrong = "const float scale = rsqrtf(sum_sq / 16.0f + 1.0e-5f);"
                else:
                    wrong = "total += a[start + j] - b[start + j];"
                corrections = [(old, wrong)]
            else:
                corrections = list(_CASE_REPAIRS[self.case_id])
                if self.fail_first:
                    previous = self.edits[0]["edits"][0]["replacement_code"]
                    corrections[0] = (previous, corrections[0][1])
                    assert state.source_sha256 != self.edits[0]["expected_source_sha256"]
            action = {
                "tool": "repo_replace_syntax",
                "path": self.kernel,
                "expected_source_sha256": state.source_sha256,
                "edits": [
                    {"node_id": by_code[old], "replacement_code": replacement}
                    for old, replacement in corrections
                ],
            }
            self.edits.append(action)
        assert action["tool"] in snapshot.allowed_tools
        return HarnessDecision.model_validate(
            {"action": action, "reason": "Exercise the real registered AST ToolHost"}
        )


@pytest.mark.parametrize("case_id", ["h001", "h002", "h003"])
@pytest.mark.parametrize("fail_first", [False, True])
def test_cross_case_ast_toolhost_full_loop_and_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    fail_first: bool,
) -> None:
    repo, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([False, True] if fail_first else [True], case_id=case_id)
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    model = CrossCaseASTFixture(case_id, fail_first=fail_first)
    output = tmp_path / "ast-cross-case"
    result = run_repository_coding_session(
        repository=repo,
        output=output,
        base_sha=sha,
        goal="Repair the registered public CUDA kernel and check the exact candidate.",
        public_case_id=case_id,
        syntax_mode=True,
        allow_batch_edits=False,
        max_steps=18,
        max_model_calls=20,
        model_factory=lambda _: model,
    )
    assert result.status == "DIFF_STAGED_UNVERIFIED"
    assert result.public_checks_passed
    assert result.public_checks_attempted == (2 if fail_first else 1)
    assert len(set(checker.calls)) == len(checker.calls)
    metrics = summarize_repository_trajectory(output)
    assert metrics.public_check_dispatches == result.public_checks_attempted
    assert metrics.public_check_completions == result.public_checks_attempted
    assert metrics.first_public_check_admitted_turn == metrics.first_public_check_executed_turn
    assert metrics.first_public_check_completed_turn == metrics.first_public_check_executed_turn
    assert metrics.first_public_check_pass_turn is not None
    assert metrics.rejected_actions == 0
    if fail_first:
        assert len(model.edits) == 2
        assert model.edits[0]["expected_source_sha256"] != model.edits[1]["expected_source_sha256"]
        assert metrics.first_public_check_completed_turn < metrics.first_public_check_pass_turn
        assert any(s.history and s.history[-1]["status"] == "FAILED" for s in model.snapshots)
    assert any(s.syntax_state is None and s.current_candidate_sha256 for s in model.snapshots)
    if case_id == "h003":
        assert any(
            s.syntax_state
            and {"group_sum", "group_center"}.issubset(
                {n.function for n in s.syntax_state.observed_nodes}
            )
            for s in model.snapshots
        )


def test_h003_ast_resume_mid_page_then_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gpu_agent.harness.recovery import read_repository_checkpoint

    repo, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([True], case_id="h003")
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    model = CrossCaseASTFixture("h003")
    output = tmp_path / "h003-page-resume"
    options = dict(
        repository=repo,
        output=output,
        base_sha=sha,
        goal="Repair the registered group centering kernel and verify it publicly.",
        public_case_id="h003",
        syntax_mode=True,
        max_steps=12,
        max_model_calls=14,
        model_factory=lambda _: model,
    )
    paused = run_repository_coding_session(**options, pause_after_turns=2)
    assert paused.status == "PAUSED_RESUMABLE"
    checkpoint = read_repository_checkpoint(output / "session-checkpoint.json")
    pinned = checkpoint.syntax_state
    assert pinned is not None and pinned.next_offset == 16
    assert len(pinned.visited_pages) == 2
    assert {"group_sum", "group_center"}.issubset({n.function for n in pinned.observed_nodes})
    resumed = run_repository_coding_session(**options, resume=True)
    assert model.snapshots[2].syntax_state == pinned
    assert resumed.public_checks_passed
    assert len(checker.calls) == 1


def test_ast_lifecycle_rejected_request_and_unavailable_backend_are_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, sha = _registered_repository(tmp_path)
    fake = FakePublicChecker([False], case_id="h002", dispatched=False)
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", fake)
    model = CrossCaseASTFixture("h002")

    class WithRejectedRequest:
        calls = 0

        def choose_harness_v1(self, snapshot):
            self.calls += 1
            if self.calls == 1:
                return HarnessDecision(action=RepoRunPublicChecks(), reason="Test denied check")
            return model.choose_harness_v1(snapshot)

    output = tmp_path / "ast-no-backend"
    outcome = run_repository_coding_session(
        repository=repo,
        output=output,
        base_sha=sha,
        goal="Repair public RMS kernel",
        public_case_id="h002",
        syntax_mode=True,
        max_steps=6,
        max_model_calls=8,
        model_factory=lambda _: WithRejectedRequest(),
    )
    assert outcome.status == "INCONCLUSIVE"
    metrics = summarize_repository_trajectory(output)
    assert metrics.first_public_check_request_turn == 1
    assert metrics.first_public_check_admitted_turn == 6
    assert metrics.first_public_check_executed_turn is None
    assert metrics.first_public_check_completed_turn == 6
    assert metrics.first_public_check_pass_turn is None


def test_failed_ast_session_recovers_completed_check_and_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gpu_agent.harness.ablation import _failed_repository_public_audit

    repo, sha = _registered_repository(tmp_path)
    checker = FakePublicChecker([False], case_id="h002")
    monkeypatch.setattr("gpu_agent.harness.repo_tools.check_candidate_publicly", checker)
    model = CrossCaseASTFixture("h002")

    class FailAfterCheck:
        def choose_harness_v1(self, snapshot):
            if snapshot.history and snapshot.history[-1]["tool"] == "repo_run_public_checks":
                raise RuntimeError("injected model failure after checker")
            return model.choose_harness_v1(snapshot)

    folder = tmp_path / "failed-trial"
    with pytest.raises(RuntimeError, match="injected model failure"):
        run_repository_coding_session(
            repository=repo,
            output=folder / "agent",
            base_sha=sha,
            goal="Repair public RMS",
            public_case_id="h002",
            syntax_mode=True,
            max_steps=12,
            max_model_calls=14,
            model_factory=lambda _: FailAfterCheck(),
        )
    assert not (folder / "agent/result.json").exists()
    recovered = _failed_repository_public_audit(folder)
    assert recovered["first_edit_turn"] == 4
    assert recovered["first_public_check_request_turn"] == 5
    assert recovered["first_public_check_executed_turn"] == 5
    assert recovered["first_public_check_completed_turn"] == 5
    assert recovered["first_public_check_pass_turn"] is None
    assert recovered["public_checks_attempted"] == 1
    assert recovered["public_check_dispatches"] == 1


def test_extended_toolhost_full_node_upgrade_resume_and_edit(tmp_path: Path) -> None:
    from gpu_agent.harness.models import RepoInspectSyntaxNode
    from gpu_agent.harness.recovery import read_repository_checkpoint
    from gpu_agent.harness.tool_schema import repository_decision_schema

    repo, sha = _registered_repository(tmp_path)
    kernel = "benchmarks/hardset_v2/h002/mutant/kernel.cu"
    original = _CASE_REPAIRS["h002"][0][0]
    long_code = original[:-1] + " /*" + "x" * 420 + "*/;"
    output = tmp_path / "extended-full-node"

    class ModelWithNode:
        def choose_harness_v1(self, snapshot):
            state = snapshot.syntax_state
            assert snapshot.extended_syntax
            source = snapshot.workspace_source
            if state is None or state.next_offset is not None:
                action = RepoInspectSyntax(
                    path=kernel, offset=state.next_offset if state else 0, limit=8
                )
            elif long_code not in source:
                node = next(n for n in state.observed_nodes if n.code == original)
                action = RepoReplaceSyntax(
                    path=kernel,
                    expected_source_sha256=state.source_sha256,
                    edits=[RepoSyntaxEdit(node_id=node.id, replacement_code=long_code)],
                )
            else:
                node = next(
                    n
                    for n in state.observed_nodes
                    if n.kind == "declaration" and n.code.startswith(original[:-1])
                )
                if node.code_truncated:
                    action = RepoInspectSyntaxNode(
                        path=kernel, expected_source_sha256=state.source_sha256, node_id=node.id
                    )
                    grammar = repository_decision_schema(
                        snapshot.allowed_tools,
                        registered_kernel_path=kernel,
                        syntax_state=state,
                        bind_syntax_cursor=True,
                    )
                    grammar.model_validate(
                        {
                            "action": action.model_dump(mode="json"),
                            "reason": "Read full node",
                        }
                    )
                else:
                    action = RepoReplaceSyntax(
                        path=kernel,
                        expected_source_sha256=state.source_sha256,
                        edits=[
                            RepoSyntaxEdit(
                                node_id=node.id, replacement_code=_CASE_REPAIRS["h002"][0][1]
                            )
                        ],
                    )
            return HarnessDecision(action=action, reason="Exercise extended public AST tool")

    model = ModelWithNode()
    opts = dict(
        repository=repo,
        output=output,
        base_sha=sha,
        goal="Exercise long-node editing with the public RMS kernel.",
        public_case_id="h002",
        syntax_mode=True,
        extended_syntax=True,
        max_steps=24,
        max_model_calls=26,
        model_factory=lambda _: model,
    )
    # Pause after each completed action until the full-node tool succeeds.
    paused = run_repository_coding_session(**opts, pause_after_turns=1)
    for turn in range(2, 22):
        checkpoint = read_repository_checkpoint(output / "session-checkpoint.json")
        if checkpoint.syntax_state and checkpoint.syntax_state.full_node_reads:
            break
        paused = run_repository_coding_session(**opts, resume=True, pause_after_turns=turn)
        assert paused.status == "PAUSED_RESUMABLE"
    checkpoint = read_repository_checkpoint(output / "session-checkpoint.json")
    pinned = checkpoint.syntax_state
    assert pinned is not None and pinned.full_node_reads
    alias = pinned.full_node_reads[0]
    upgraded = next(n for n in pinned.observed_nodes if n.id == alias)
    assert upgraded.code == long_code and not upgraded.code_truncated
    assert alias not in pinned.readable_truncated_aliases
    with pytest.raises(ValueError, match="budgets differ"):
        run_repository_coding_session(
            **{**opts, "extended_syntax": False},
            resume=True,
            pause_after_turns=turn + 1,
        )
    next_turn = checkpoint.state["turns"] + 1
    edited = run_repository_coding_session(**opts, resume=True, pause_after_turns=next_turn)
    assert edited.status == "PAUSED_RESUMABLE"
    assert read_repository_checkpoint(output / "session-checkpoint.json").syntax_state is None
    assert _CASE_REPAIRS["h002"][0][1] in (output / "worktree" / kernel).read_text()


def test_mixed_noop_reports_all_unchanged_without_staging_subset(tmp_path: Path) -> None:
    repo, sha = _registered_repository(tmp_path)
    with RepositoryWorkspaceTools(
        origin=repo,
        output=tmp_path / "mixed-noop",
        sha=sha,
        public_case_id="h001",
        syntax_mode=True,
    ) as host:
        obs = host.invoke(RepoInspectSyntax(path=KERNEL, symbol="softmax32", limit=20))
        assert obs.status == "OK"
        state = host.syntax_state
        assert state is not None
        nodes = {n.code: n for n in state.observed_nodes}
        first = nodes["denominator += expf(x);"]
        second = nodes["out[i] = expf(own) / denominator;"]
        changed = nodes["const float own = a[i] + b[i];"]
        action = RepoReplaceSyntax(
            path=KERNEL,
            expected_source_sha256=state.source_sha256,
            edits=[
                RepoSyntaxEdit(node_id=first.id, replacement_code=first.code),
                RepoSyntaxEdit(node_id=second.id, replacement_code=second.code),
                RepoSyntaxEdit(
                    node_id=changed.id,
                    replacement_code="const float own = a[i] + b[i] - maximum;",
                ),
            ],
        )
        rejected = host.invoke(action)
        assert rejected.status == "REJECTED"
        assert "Omit every unchanged entry" in rejected.content
        assert not host.workspace.diff()
        from gpu_agent.harness.repo_audit import read_repository_tool_events

        events = read_repository_tool_events(
            host.store, host.run_id, expected_count=2, require_observation=True
        )
        detail = events[-1]["rejection_detail"]
        assert detail["unchanged_node_ids"] == [first.id, second.id]
        assert detail["changed_node_ids"] == [changed.id]
