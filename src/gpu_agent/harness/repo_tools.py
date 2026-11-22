"""Repository-level Coding Agent tool host over a pinned Git worktree.

This adapter can stage multi-file source edits without granting host shell or
access to tests/evaluator. It intentionally cannot claim CUDA correctness:
compile/test tools must be explicitly integrated into an isolated executor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gpu_agent.harness.context import compact_public_context
from gpu_agent.harness.models import (
    FinishTask,
    HarnessAction,
    RepoGitDiff,
    RepoListFiles,
    RepoReadFile,
    RepoReplaceSnippet,
    SessionSnapshot,
    ToolObservation,
)
from gpu_agent.harness.registry import ToolPermissionPolicy
from gpu_agent.harness.worktree import PinnedGitWorkspace
from gpu_agent.store import RunStore


class RepositoryWorkspaceTools:
    """Independent tool authority for a single checked-out source snapshot."""

    ALLOWED_TOOLS = (
        "repo_list_files",
        "repo_read_file",
        "repo_replace_snippet",
        "repo_git_diff",
        "finish",
    )
    completion_reason = "WORKSPACE_DIFF_STAGED_UNVERIFIED"

    def __init__(
        self,
        *,
        origin: Path,
        output: Path,
        sha: str,
        max_edited_files: int = 8,
        permissions: ToolPermissionPolicy | None = None,
    ) -> None:
        if output.exists() or output.is_symlink():
            raise ValueError("repository agent output already exists")
        output.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.output = output
        self.workspace = PinnedGitWorkspace(
            origin=origin,
            root=output / "worktree",
            sha=sha,
            max_edited_files=max_edited_files,
        )
        self._policy = permissions or ToolPermissionPolicy()
        self._store = RunStore(output / "public")
        self._run = self._store.create_run("gpu_repository_agent")
        self._tool_calls = 0
        self._finished = False
        self._last_file: str | None = None
        self._last_file_read: str = ""
        self._source_version = sha

    @property
    def run_id(self) -> str:
        return self._run.id

    @property
    def store(self) -> RunStore:
        return self._store

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    @property
    def accepted_diff(self) -> str | None:
        return self.workspace.diff() if self._finished else None

    def __enter__(self) -> RepositoryWorkspaceTools:
        self._store.transition(self._run.id, "RUNNING", "DIAGNOSING")
        return self

    def __exit__(self, typ: object, val: object, tb: object) -> None:
        try:
            self._store.transition(
                self.run_id,
                "COMPLETED" if self._finished and typ is None else "FAILED",
                None,
            )
        except ValueError:
            if typ is None:
                raise

    def snapshot(self, goal: str, history: list[dict[str, str]], steps: int) -> SessionSnapshot:
        source = (
            f"Current inspected file: {self._last_file}\n{self._last_file_read}"
            if self._last_file is not None
            else "No source file opened yet. Call repo_list_files then repo_read_file."
        )
        context = compact_public_context(history, source)
        delta = self.workspace.diff()
        return SessionSnapshot(
            goal=goal,
            case_id="repository",
            base_git_commit=self.workspace.base_sha,
            source_hash=hashlib.sha256(self.workspace.base_sha.encode()).hexdigest(),
            functional_requirement=goal,
            current_candidate_sha256=hashlib.sha256(delta.encode()).hexdigest() if delta else None,
            public_checks_passed=False,
            allowed_tools=tuple(
                name for name in self.ALLOWED_TOOLS if self._policy.check(name) == "ALLOW"
            ),
            remaining_steps=steps,
            remaining_public_checks=0,
            history=context.recent,
            context_summary=context.summary,
            workspace_source=context.source,
            total_public_actions=context.total_actions,
        )

    def invoke(self, action: HarnessAction) -> ToolObservation:
        if self._finished:
            raise ValueError("repository workspace already finalized")
        self._tool_calls += 1
        permission = self._policy.check(action.tool)
        if permission != "ALLOW" or action.tool not in self.ALLOWED_TOOLS:
            result = ToolObservation(
                tool=action.tool,
                status="REJECTED",
                content=f"TOOL_{permission}: unavailable workspace capability",
            )
        else:
            try:
                result = self._dispatch(action)
            except (ValueError, OSError) as exc:
                result = ToolObservation(
                    tool=action.tool,
                    status="REJECTED",
                    content=f"{type(exc).__name__}: {str(exc)[:500]}",
                )
        self._store.put(
            self.run_id,
            f"repo-harness/v1/tool-{self._tool_calls:03d}.json",
            json.dumps(
                {
                    "tool": action.tool,
                    "status": result.status,
                    "observation_sha256": hashlib.sha256(result.content.encode()).hexdigest(),
                    "edited_files": self.workspace.edited_files,
                    "diff_sha256": self.workspace.integrity()["diff_sha256"],
                },
                sort_keys=True,
            ).encode(),
            "public",
        )
        return result

    def _dispatch(self, action: HarnessAction) -> ToolObservation:
        if isinstance(action, RepoListFiles):
            files = self.workspace.list_files(suffix=action.suffix)
            return ToolObservation(
                tool=action.tool,
                status="OK",
                content="\n".join(files)[:4000] or "No allowed source files found.",
            )
        if isinstance(action, RepoReadFile):
            self._last_file = action.path
            text = self.workspace.read(action.path, start=action.start_line, lines=action.lines)
            self._last_file_read = text
            return ToolObservation(
                tool=action.tool,
                status="OK",
                content=text[:4000],
            )
        if isinstance(action, RepoReplaceSnippet):
            candidate_sha = self.workspace.replace_exact(
                action.path,
                action.old_text,
                action.new_text,
            )
            self._last_file = action.path
            self._last_file_read = self.workspace.read(action.path)
            return ToolObservation(
                tool=action.tool,
                status="OK",
                content=f"Edit accepted. source_sha256={candidate_sha}. "
                "Use repo_git_diff to inspect changes; staged edits are NOT GPU verified.",
            )
        if isinstance(action, RepoGitDiff):
            diff = self.workspace.diff()
            return ToolObservation(
                tool=action.tool,
                status="OK",
                content=diff[:3900] or "No changes staged.",
            )
        if isinstance(action, FinishTask):
            diff = self.workspace.diff()
            if not diff:
                raise ValueError("cannot finish without a source modification")
            self._finished = True
            return ToolObservation(
                tool=action.tool,
                status="OK",
                content="Code changes staged in isolated worktree, but NOT compiled "
                "or independently verified. Operator must inspect and run checks.",
            )
        raise ValueError("unsupported repository action")
