"""Repository-level Coding Agent tool host over a pinned Git worktree.

This adapter can stage multi-file source edits without granting host shell or
access to tests/evaluator. It intentionally cannot claim CUDA correctness:
compile/test tools must be explicitly integrated into an isolated executor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gpu_agent.benchmark.hardset_v2 import (
    HARDSET_CASES,
    PUBLIC_REQUIREMENTS,
    load_registry,
    manifest_for,
)
from gpu_agent.benchmark.hardset_v2_agent_models import HardsetPublicCheck
from gpu_agent.benchmark.hardset_v2_selfcheck import check_candidate_publicly
from gpu_agent.harness.context import compact_public_context
from gpu_agent.harness.models import (
    FinishTask,
    HarnessAction,
    RepoGitDiff,
    RepoListFiles,
    RepoReadFile,
    RepoReplaceSnippet,
    RepoRunPublicChecks,
    SessionSnapshot,
    ToolObservation,
)
from gpu_agent.harness.registry import ToolPermissionPolicy
from gpu_agent.harness.worktree import PinnedGitWorkspace
from gpu_agent.store import RunStore, read_regular


class RepositoryWorkspaceTools:
    """Independent tool authority for a single checked-out source snapshot."""

    ALLOWED_TOOLS = (
        "repo_list_files",
        "repo_read_file",
        "repo_replace_snippet",
        "repo_git_diff",
        "repo_run_public_checks",
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
        public_case_id: str | None = None,
        max_public_checks: int = 2,
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
        self._configure_public_case(public_case_id, max_public_checks)
        self._store = RunStore(output / "public")
        self._run = self._store.create_run("gpu_repository_agent")
        self._tool_calls = 0
        self._finished = False
        self._suspended = False
        self._resuming = False
        self._last_file: str | None = None
        self._last_file_read: str = ""
        self._source_version = sha
        self._public_checks_attempted = 0
        self._attempted_candidate_hashes: set[str] = set()
        self._last_public_run_id: str | None = None
        self._last_checked_diff_sha: str | None = None
        self._last_checked_candidate_sha: str | None = None
        self._last_public_passed = False

    def _configure_public_case(self, case_id: str | None, max_public_checks: int) -> None:
        if type(max_public_checks) is not int or not 1 <= max_public_checks <= 3:
            raise ValueError("public check budget must be between 1 and 3")
        if case_id is not None and case_id not in HARDSET_CASES:
            raise ValueError("unregistered public CUDA case")
        self._public_case_id = case_id
        self._max_public_checks = max_public_checks
        self._public_kernel_path: str | None = None
        if case_id is not None:
            # Only the registry's original mutant is eligible for this public
            # oracle. The agent cannot select a different problem or expected output.
            root = self.workspace.root / "benchmarks"
            registry, _ = load_registry(root)
            case = next(case for case in registry.cases if case.case_id == case_id)
            manifest = manifest_for(root, case, "mutant")
            name = f"benchmarks/hardset_v2/{case_id}/mutant/kernel.cu"
            if manifest[f"hardset_v2/{case_id}/mutant/kernel.cu"] != (
                case.mutant_source_sha256
            ):
                raise ValueError("public case mutant does not match the registered SHA")
            if name not in self.workspace.list_files(suffix=".cu"):
                raise ValueError("registered public kernel is not in frozen checkout")
            self._public_kernel_path = name

    @property
    def public_checks_attempted(self) -> int:
        return self._public_checks_attempted

    @property
    def attempted_candidate_hashes(self) -> tuple[str, ...]:
        return tuple(sorted(self._attempted_candidate_hashes))

    @property
    def last_public_run_id(self) -> str | None:
        return self._last_public_run_id

    @property
    def last_checked_diff_sha(self) -> str | None:
        return self._last_checked_diff_sha

    @property
    def last_checked_candidate_sha(self) -> str | None:
        return self._last_checked_candidate_sha

    @property
    def public_checks_passed(self) -> bool:
        return (
            self._last_public_passed
            and self._last_checked_diff_sha is not None
            and self._last_checked_diff_sha == self.workspace.integrity()["diff_sha256"]
        )

    @classmethod
    def reopen(
        cls,
        *,
        origin: Path,
        output: Path,
        sha: str,
        run_id: str,
        tool_calls: int,
        accepted_hashes: dict[str, str],
        max_edited_files: int = 8,
        permissions: ToolPermissionPolicy | None = None,
        public_case_id: str | None = None,
        max_public_checks: int = 2,
        public_checks_attempted: int = 0,
        attempted_candidate_hashes: tuple[str, ...] = (),
        last_public_run_id: str | None = None,
        last_checked_diff_sha: str | None = None,
        last_checked_candidate_sha: str | None = None,
        last_public_passed: bool = False,
    ) -> RepositoryWorkspaceTools:
        """Reattach a still-running public session with matching Git evidence."""
        if output.is_symlink() or not output.is_dir():
            raise ValueError("missing or unsafe repository session output")
        if type(tool_calls) is not int or not 0 <= tool_calls <= 40:
            raise ValueError("invalid persisted tool counter")
        host = object.__new__(cls)
        host.output = output
        host.workspace = PinnedGitWorkspace.attach_existing(
            origin=origin,
            root=output / "worktree",
            sha=sha,
            accepted_hashes=accepted_hashes,
            max_edited_files=max_edited_files,
        )
        host._store = RunStore(output / "public")
        host._run = host._store.load(run_id)
        if host._run.kind != "gpu_repository_agent" or host._run.status != "RUNNING":
            raise ValueError("resumed public RunStore must be an active repository run")
        host._policy = permissions or ToolPermissionPolicy()
        host._configure_public_case(public_case_id, max_public_checks)
        if (
            type(public_checks_attempted) is not int
            or not 0 <= public_checks_attempted <= max_public_checks
            or len(attempted_candidate_hashes) != public_checks_attempted
            or len(set(attempted_candidate_hashes)) != public_checks_attempted
        ):
            raise ValueError("recovered GPU check budget is inconsistent")
        host._public_checks_attempted = public_checks_attempted
        host._attempted_candidate_hashes = set(attempted_candidate_hashes)
        host._last_public_run_id = last_public_run_id
        host._last_checked_diff_sha = last_checked_diff_sha
        host._last_checked_candidate_sha = last_checked_candidate_sha
        host._last_public_passed = last_public_passed
        if public_checks_attempted:
            if not public_case_id or not last_public_run_id or not last_checked_candidate_sha:
                raise ValueError("missing recovered public GPU evidence")
            child = host._store.load(last_public_run_id)
            if child.parent_run_id != run_id or child.kind != "hardset_v2_public_candidate":
                raise ValueError("recovered GPU evidence belongs to another run")
            refs = [
                ref for ref in child.artifact_refs if ref.name == "hardset/public-selfcheck.json"
            ]
            if len(refs) != 1:
                raise ValueError("missing unique public GPU check attestation")
            recorded = HardsetPublicCheck.model_validate_json(host._store.read(refs[0]))
            if (
                recorded.public_run_id != last_public_run_id
                or recorded.candidate_sha256 != last_checked_candidate_sha
                or recorded.passed != last_public_passed
                or child.status.value != ("COMPLETED" if recorded.passed else "FAILED")
                or last_checked_candidate_sha not in host._attempted_candidate_hashes
            ):
                raise ValueError("recovered public GPU evidence does not match checkpoint")
        elif (
            last_public_run_id is not None
            or last_checked_diff_sha is not None
            or last_checked_candidate_sha is not None
            or last_public_passed
            or attempted_candidate_hashes
        ):
            raise ValueError("public check evidence exists without recorded GPU calls")
        host._tool_calls = tool_calls
        host._finished = False
        host._suspended = False
        host._resuming = True
        # Do not trust stale line ranges after process restart. The complete
        # model history is restored separately from the audited checkpoint.
        host._last_file = None
        host._last_file_read = ""
        host._source_version = sha
        return host

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
        if not self._resuming:
            self._store.transition(self._run.id, "RUNNING", "DIAGNOSING")
        return self

    def __exit__(self, typ: object, val: object, tb: object) -> None:
        if self._suspended and typ is None:
            # A deliberate pause preserves RUNNING authority for explicit resume.
            return
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
        gpu_allowed = (
            self._public_case_id is not None
            and self._public_checks_attempted < self._max_public_checks
        )
        allowed_tools = tuple(
            name
            for name in self.ALLOWED_TOOLS
            if self._policy.check(name) == "ALLOW"
            and (name != "repo_run_public_checks" or gpu_allowed)
        )
        requirement = goal
        if self._public_case_id is not None:
            requirement += (
                f"\nRegistered public GPU case: {self._public_case_id}. "
                + PUBLIC_REQUIREMENTS[self._public_case_id]
            )
        return SessionSnapshot(
            goal=goal,
            case_id="repository",
            base_git_commit=self.workspace.base_sha,
            source_hash=hashlib.sha256(self.workspace.base_sha.encode()).hexdigest(),
            functional_requirement=requirement,
            registered_public_case=self._public_case_id,
            current_candidate_sha256=hashlib.sha256(delta.encode()).hexdigest() if delta else None,
            public_checks_passed=self.public_checks_passed,
            allowed_tools=allowed_tools,
            remaining_steps=steps,
            remaining_public_checks=(
                self._max_public_checks - self._public_checks_attempted
                if self._public_case_id is not None
                else 0
            ),
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
        if isinstance(action, RepoRunPublicChecks):
            return self._run_registered_public_checks()
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
            if self._public_case_id is not None and not self.public_checks_passed:
                raise ValueError("current candidate has not passed registered public CUDA checks")
            self._finished = True
            return ToolObservation(
                tool=action.tool,
                status="OK",
                content=(
                    "Registered public GPU checks passed for this exact kernel and diff; "
                    "independent hidden verification has NOT run."
                    if self._public_case_id is not None
                    else "Changes staged for operator review; NO CUDA checks were run."
                ),
            )
        raise ValueError("unsupported repository action")

    def _run_registered_public_checks(self) -> ToolObservation:
        if self._public_case_id is None or self._public_kernel_path is None:
            raise ValueError("public GPU checks require an operator-registered case")
        if self._public_checks_attempted >= self._max_public_checks:
            raise ValueError("public CUDA check budget exhausted")
        delta = self.workspace.diff()
        if not delta or self.workspace.edited_files != (self._public_kernel_path,):
            raise ValueError("public HardSet checks support exactly its one kernel edit")
        candidate = read_regular(
            self.workspace._checked(self._public_kernel_path, editable=True),
            1024 * 1024,
        )
        candidate_sha = hashlib.sha256(candidate).hexdigest()
        if candidate_sha in self._attempted_candidate_hashes:
            raise ValueError("identical CUDA candidate was already checked")
        diff_sha = self.workspace.integrity()["diff_sha256"]
        self._attempted_candidate_hashes.add(candidate_sha)
        self._public_checks_attempted += 1
        result = check_candidate_publicly(
            store=self._store,
            parent_run_id=self.run_id,
            repository=self.workspace.root,
            case_id=self._public_case_id,
            candidate=candidate,
            output=self.output / f"public-candidate-{self._public_checks_attempted:02d}",
        )
        if result.candidate_sha256 != candidate_sha or result.public_run_id is None:
            raise ValueError("registered CUDA public evidence lost candidate provenance")
        self._last_checked_diff_sha = str(diff_sha)
        self._last_checked_candidate_sha = candidate_sha
        self._last_public_run_id = result.public_run_id
        self._last_public_passed = result.passed
        return ToolObservation(
            tool="repo_run_public_checks",
            status="OK" if result.passed else "FAILED",
            content=result.model_dump_json(),
        )
