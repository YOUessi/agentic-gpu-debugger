"""Proof that a model can choose multi-file actions without a host shell."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from pydantic import SecretStr

from gpu_agent.agent.models import AgentBudget
from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.agent.prompts import GPU_REPOSITORY_HARNESS_PROMPT_VERSION
from gpu_agent.agent.provider import (
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
    SDKResult,
)
from gpu_agent.harness.journal import read_verified_journal, recovery_disposition
from gpu_agent.harness.models import HarnessDecision, RepoReadFile
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
