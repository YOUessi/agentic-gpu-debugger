"""Proof that a model can choose multi-file actions without a host shell."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr

from gpu_agent.agent.models import AgentBudget
from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.agent.prompts import GPU_REPOSITORY_HARNESS_PROMPT_VERSION
from gpu_agent.agent.provider import (
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
    SDKResult,
)
from gpu_agent.harness.journal import SessionJournal, read_verified_journal, recovery_disposition
from gpu_agent.harness.metrics import summarize_repository_trajectory
from gpu_agent.harness.models import HarnessDecision, RepoReadFile
from gpu_agent.harness.recovery import read_repository_checkpoint
from gpu_agent.harness.registry import ToolPermissionPolicy
from gpu_agent.harness.repo_runner import run_repository_coding_session
from gpu_agent.harness.repo_tools import RepositoryWorkspaceTools
from gpu_agent.store import RunStore


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
        timeout=12,
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    files = {
        "src/kernel.cu": "float x = 1.0f;\n",
        "src/helper.cuh": "inline float transform(float v) { return v; }\n",
        "tests/test_kernel.py": "def test_kernel(): assert True\n",
    }
    for rel, value in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value)
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "-q",
        "-m",
        "frozen",
    )
    return repo, _git(repo, "rev-parse", "HEAD")


class ScriptedLLM:
    """Only tests tool routing; not a claim of any physical model call."""

    def __init__(self) -> None:
        self.actions = [
            {"tool": "repo_list_files", "suffix": ".cu"},
            {"tool": "repo_read_file", "path": "src/kernel.cu"},
            {
                "tool": "repo_replace_snippet",
                "path": "src/kernel.cu",
                "old_text": "1.0f",
                "new_text": "2.0f",
            },
            {"tool": "repo_read_file", "path": "src/helper.cuh"},
            {
                "tool": "repo_replace_snippet",
                "path": "src/helper.cuh",
                "old_text": "return v;",
                "new_text": "return 2.0f * v;",
            },
            {"tool": "repo_git_diff"},
            {"tool": "finish", "summary": "Two files edited, operator review required"},
        ]
        self.position = 0
        self.snapshots = []

    def choose_harness_v1(self, snapshot):
        assert snapshot.base_git_commit is not None
        assert snapshot.case_id == "repository"
        assert "repo_replace_snippet" in snapshot.allowed_tools
        assert "functional_probe" not in snapshot.allowed_tools
        self.snapshots.append(snapshot)
        value = self.actions[self.position]
        self.position += 1
        return HarnessDecision.model_validate(
            {"action": value, "reason": "Proceed with the requested CUDA code change"}
        )


def test_model_directed_multi_file_worktree_and_durable_audit(tmp_path: Path) -> None:
    source, sha = _repo(tmp_path)
    model = ScriptedLLM()
    out = tmp_path / "repo-agent-result"
    report = run_repository_coding_session(
        repository=source,
        output=out,
        base_sha=sha,
        goal="Edit the CUDA kernel and helper without altering tests",
        max_steps=9,
        max_model_calls=12,
        model_factory=lambda _: model,
    )
    assert report.status == "DIFF_STAGED_UNVERIFIED"
    assert report.edited_files == ("src/helper.cuh", "src/kernel.cu")
    assert report.turns == report.tool_calls == 7
    assert report.model_calls == report.model_tokens == 0
    assert report.base_sha == sha
    assert report.reason == "WORKSPACE_DIFF_STAGED_UNVERIFIED"
    assert "1.0f" in (source / "src/kernel.cu").read_text()
    assert "return v;" in (source / "src/helper.cuh").read_text()
    assert _git(source, "status", "--porcelain") == ""
    assert "2.0f" in (out / "worktree/src/kernel.cu").read_text()
    assert "return 2.0f * v;" in (out / "worktree/src/helper.cuh").read_text()
    assert _git(out / "worktree", "rev-parse", "HEAD") == sha
    assert len(model.snapshots) == 7
    assert model.snapshots[-1].current_candidate_sha256 == report.diff_sha256

    journal = read_verified_journal(out / "session-events.jsonl")
    assert recovery_disposition(journal) == "FINISHED"
    assert journal[-1]["payload"]["status"] == "DIFF_STAGED_UNVERIFIED"
    assert (out / "result.json").exists()
    assert len(report.diff_sha256) == 64
    assert (
        report.diff_sha256
        == hashlib.sha256(
            _git(
                out / "worktree", "diff", "--no-ext-diff", "--", "src/helper.cuh", "src/kernel.cu"
            ).encode()
        ).hexdigest()
    )


def test_repo_tool_host_rejects_hardset_actions_and_edit_permission(
    tmp_path: Path,
) -> None:
    source, sha = _repo(tmp_path)
    host = RepositoryWorkspaceTools(
        origin=source,
        output=tmp_path / "only-read",
        sha=sha,
        permissions=ToolPermissionPolicy(allow_edit=False),
    )
    with host:
        snapshot = host.snapshot("Inspect source changes without editing", [], 4)
        assert "repo_replace_snippet" not in snapshot.allowed_tools
        bad = HarnessDecision.model_validate(
            {
                "action": {
                    "tool": "repo_replace_snippet",
                    "path": "src/kernel.cu",
                    "old_text": "1.0f",
                    "new_text": "2.0f",
                },
                "reason": "Try unauthorized edit",
            }
        )
        observation = host.invoke(bad.action)
        assert observation.status == "REJECTED"
        assert (source / "src/kernel.cu").read_text() == "float x = 1.0f;\n"
        invalid = HarnessDecision.model_validate(
            {
                "action": {"tool": "read_file", "path": "kernel.cu"},
                "reason": "Use wrong tool namespace",
            }
        )
        assert host.invoke(invalid.action).status == "REJECTED"


def test_repo_harness_model_wire_uses_distinct_versioned_prompt(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "audit")
    run = store.create_run("repo_model_wire")
    calls = []

    class Port:
        def call(self, request):
            calls.append(request)
            return SDKResult(
                value={
                    "action": {"tool": "repo_read_file", "path": "src/kernel.cu"},
                    "reason": "Inspect the task source before modification",
                }
            )

    provider = OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.deepseek.com",
            model="mock-only",
            api_key=SecretStr("not-a-secret"),
            supports_store_false=True,
        ),
        LLMCallGate(AgentBudget(max_llm_calls=8)),
        store,
        run.id,
        port=Port(),
    )
    source, sha = _repo(tmp_path)
    with RepositoryWorkspaceTools(origin=source, output=tmp_path / "workspace", sha=sha) as host:
        decision = provider.choose_harness_v1(host.snapshot("Fix the CUDA kernel source", [], 6))
    assert isinstance(decision.action, RepoReadFile)
    assert calls[0].payload["gpu_repo_harness"] == "v1"
    assert calls[0].kind == "plan"
    assert provider.invocations()[0].prompt_version == GPU_REPOSITORY_HARNESS_PROMPT_VERSION


def _pause_session(tmp_path: Path) -> tuple[Path, str, Path]:
    repository, sha = _repo(tmp_path)
    output = tmp_path / "resumable"
    report = run_repository_coding_session(
        repository=repository,
        output=output,
        base_sha=sha,
        goal="Edit the CUDA kernel and helper without altering tests",
        max_steps=9,
        max_model_calls=12,
        pause_after_turns=3,
        model_factory=lambda _: ScriptedLLM(),
    )
    assert report.status == "PAUSED_RESUMABLE"
    assert report.turns == report.tool_calls == 3
    assert not (output / "result.json").exists()
    assert (output / "session-checkpoint.json").exists()
    assert recovery_disposition(read_verified_journal(output / "session-events.jsonl")) == (
        "QUIESCENT_INSPECTION_ONLY"
    )
    return repository, sha, output


def test_repository_agent_pause_resume_preserves_budget_and_exactly_once_actions(
    tmp_path: Path,
) -> None:
    repository, sha, output = _pause_session(tmp_path)
    assert (
        RunStore(output / "public")
        .load(read_verified_journal(output / "session-events.jsonl")[0]["payload"]["public_run_id"])
        .status
        == "RUNNING"
    )
    model = ScriptedLLM()
    model.position = 3  # Simulate a new model that continues from saved observations.
    final = run_repository_coding_session(
        repository=repository,
        output=output,
        base_sha=sha,
        goal="Edit the CUDA kernel and helper without altering tests",
        max_steps=9,
        max_model_calls=12,
        resume=True,
        model_factory=lambda _: model,
    )
    assert final.status == "DIFF_STAGED_UNVERIFIED"
    assert final.turns == final.tool_calls == 7
    assert final.edited_files == ("src/helper.cuh", "src/kernel.cu")
    assert len(model.snapshots) == 4
    assert model.snapshots[0].total_public_actions == 3
    assert final.model_calls == 0
    events = read_verified_journal(output / "session-events.jsonl")
    assert recovery_disposition(events) == "FINISHED"
    assert sum(item["kind"] == "session_resumed" for item in events) == 1
    assert sum(item["kind"] == "tool_completed" for item in events) == 7
    assert "2.0f" in (output / "worktree/src/kernel.cu").read_text()
    assert "1.0f" in (repository / "src/kernel.cu").read_text()
    assert (output / "result.json").exists()
    assert RunStore(output / "public").load(final.public_run_id).status == "COMPLETED"


def test_resume_rejects_tampered_source_or_mismatched_budget(tmp_path: Path) -> None:
    repository, sha, output = _pause_session(tmp_path)
    kwargs = {
        "repository": repository,
        "output": output,
        "base_sha": sha,
        "goal": "Edit the CUDA kernel and helper without altering tests",
        "max_steps": 9,
        "max_model_calls": 12,
        "resume": True,
        "model_factory": lambda _: ScriptedLLM(),
    }
    with pytest.raises(ValueError, match="budgets differ"):
        run_repository_coding_session(**{**kwargs, "max_steps": 8})
    (output / "worktree/src/kernel.cu").write_text("float x = 9.0f;\n")
    with pytest.raises(ValueError, match="outside controller"):
        run_repository_coding_session(**kwargs)


def test_resume_rejects_unknown_inflight_external_action(tmp_path: Path) -> None:
    repository, sha, output = _pause_session(tmp_path)
    journal_path = output / "session-events.jsonl"
    before = read_verified_journal(journal_path)
    with SessionJournal.reopen(journal_path, expected_head=before[-1]["sha256"]) as journal:
        journal.append("tool_started", {"turn": 4, "tool": "repo_replace_snippet"})
    assert "UNCERTAIN" in recovery_disposition(read_verified_journal(journal_path))
    with pytest.raises(ValueError, match="completed tool boundary"):
        run_repository_coding_session(
            repository=repository,
            output=output,
            base_sha=sha,
            goal="Edit the CUDA kernel and helper without altering tests",
            max_steps=9,
            max_model_calls=12,
            resume=True,
            model_factory=lambda _: ScriptedLLM(),
        )


def test_resume_rejects_checkpoint_journal_mismatch(tmp_path: Path) -> None:
    repository, sha, output = _pause_session(tmp_path)
    # Even a legitimate append after the checkpoint must be reconciled.
    journal_path = output / "session-events.jsonl"
    previous = read_verified_journal(journal_path)
    with SessionJournal.reopen(journal_path, expected_head=previous[-1]["sha256"]) as journal:
        journal.append("session_resumed", {"previous_turns": 3})
    with pytest.raises(ValueError, match="completed tool boundary"):
        run_repository_coding_session(
            repository=repository,
            output=output,
            base_sha=sha,
            goal="Edit the CUDA kernel and helper without altering tests",
            max_steps=9,
            max_model_calls=12,
            resume=True,
            model_factory=lambda _: ScriptedLLM(),
        )


class SelectActions:
    def __init__(self, actions: list[dict[str, object]]) -> None:
        self.actions = actions
        self.seen = []

    def choose_harness_v1(self, snapshot):
        self.seen.append(snapshot)
        return HarnessDecision.model_validate(
            {"action": self.actions.pop(0), "reason": "Inspect or modify the bounded workspace"}
        )


def test_repo_redundant_reads_lists_and_diffs_are_rejected_per_revision(
    tmp_path: Path,
) -> None:
    source, sha = _repo(tmp_path)
    with RepositoryWorkspaceTools(origin=source, output=tmp_path / "dedup", sha=sha) as host:

        def invoke(value):
            return host.invoke(
                HarnessDecision.model_validate(
                    {"action": value, "reason": "Choose a typed repository tool"}
                ).action
            )

        files = {"tool": "repo_list_files", "suffix": ".cu"}
        read = {"tool": "repo_read_file", "path": "src/kernel.cu"}
        diff = {"tool": "repo_git_diff"}
        edit = {
            "tool": "repo_replace_snippet",
            "path": "src/kernel.cu",
            "old_text": "1.0f",
            "new_text": "2.0f",
        }
        assert invoke(files).status == "OK"
        duplicate_list = invoke(files)
        assert duplicate_list.status == "REJECTED"
        assert "REDUNDANT_LIST" in duplicate_list.content
        assert invoke(read).status == "OK"
        duplicate_read = invoke(read)
        assert duplicate_read.status == "REJECTED"
        assert "REDUNDANT_READ" in duplicate_read.content
        assert invoke(diff).status == "OK"
        duplicate_diff = invoke(diff)
        assert duplicate_diff.status == "REJECTED"
        assert "REDUNDANT_DIFF" in duplicate_diff.content
        assert invoke(edit).status == "OK"
        # Same line range and diff are useful again after the source revision changed.
        assert invoke(read).status == "OK"
        assert invoke(diff).status == "OK"
        assert len(host.observed_query_fingerprints) == 5
        refs = host.store.load(host.run_id).artifact_refs
        records = [
            __import__("json").loads(host.store.read(ref))
            for ref in refs
            if ref.name.endswith("tool-004.json")
        ]
        assert len(records) == 1
        assert records[0]["rejection_code"] == "REDUNDANT_READ"


def test_repo_inspection_keys_survive_pause_and_metrics_are_provenanced(
    tmp_path: Path,
) -> None:
    source, sha = _repo(tmp_path)
    output = tmp_path / "dedup-paused"
    read = {"tool": "repo_read_file", "path": "src/kernel.cu"}
    goal = "Inspect and repair the original CUDA kernel with bounded typed tools"
    paused = run_repository_coding_session(
        repository=source,
        output=output,
        base_sha=sha,
        goal=goal,
        max_steps=7,
        max_model_calls=9,
        pause_after_turns=1,
        model_factory=lambda _: SelectActions([read]),
    )
    assert paused.status == "PAUSED_RESUMABLE"
    checkpoint = read_repository_checkpoint(output / "session-checkpoint.json")
    assert len(checkpoint.observed_query_fingerprints) == 1

    model = SelectActions(
        [
            read,
            {
                "tool": "repo_replace_snippet",
                "path": "src/kernel.cu",
                "old_text": "1.0f",
                "new_text": "2.0f",
            },
            read,
            {"tool": "finish", "summary": "Code changes staged for public operator review"},
        ]
    )
    report = run_repository_coding_session(
        repository=source,
        output=output,
        base_sha=sha,
        goal=goal,
        max_steps=7,
        max_model_calls=9,
        resume=True,
        model_factory=lambda _: model,
    )
    assert report.status == "DIFF_STAGED_UNVERIFIED"
    assert report.turns == report.tool_calls == 5
    assert model.seen[0].history[-1]["status"] == "OK"
    assert model.seen[1].history[-1]["status"] == "REJECTED"
    metrics = summarize_repository_trajectory(output)
    assert metrics.turns == 5
    assert metrics.model_physical_calls == 0
    assert metrics.inspected_actions == 3
    assert metrics.first_accepted_edit_turn == 3
    assert metrics.first_accepted_finish_turn == 5
    assert metrics.first_public_check_request_turn is None
    assert metrics.redundant_inspection_rejections == 1
    assert metrics.rejected_actions == 1
    assert metrics.independently_verified is False
    assert metrics.public_checks_passed_final is False
    assert [step.status for step in metrics.tool_trajectory] == ["OK", "REJECTED", "OK", "OK", "OK"]
    restored_keys = read_repository_checkpoint(
        output / "session-checkpoint.json"
    ).observed_query_fingerprints
    assert len(restored_keys) == 2
    assert (source / "src/kernel.cu").read_text() == "float x = 1.0f;\n"


def test_repo_metric_reader_refuses_unfinished_or_removed_public_evidence(
    tmp_path: Path,
) -> None:
    source, sha = _repo(tmp_path)
    output = tmp_path / "metric-paused"
    run_repository_coding_session(
        repository=source,
        output=output,
        base_sha=sha,
        goal="Inspect a bounded CUDA source then stop at the safe boundary",
        max_steps=6,
        max_model_calls=8,
        pause_after_turns=1,
        model_factory=lambda _: SelectActions(
            [{"tool": "repo_read_file", "path": "src/kernel.cu"}]
        ),
    )
    with pytest.raises((ValueError, OSError)):
        summarize_repository_trajectory(output)
