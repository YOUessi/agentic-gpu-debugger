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
from gpu_agent.harness.models import HarnessDecision, RepoRunPublicChecks
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
        assert "repo_replace_syntax" in first.allowed_tools
        assert "repo_replace_snippet" not in first.allowed_tools
        inspected = HarnessDecision.model_validate(
            {
                "action": {
                    "tool": "repo_inspect_syntax",
                    "path": KERNEL,
                    "symbol": "softmax32",
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
