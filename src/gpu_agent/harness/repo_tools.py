"""Repository-level Coding Agent tool host over a pinned Git worktree.

This adapter can stage multi-file source edits without granting host shell or
access to tests/evaluator. It intentionally cannot claim CUDA correctness:
compile/test tools must be explicitly integrated into an isolated executor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from gpu_agent.benchmark.hardset_v2 import (
    HARDSET_CASES,
    PUBLIC_REQUIREMENTS,
    load_registry,
    manifest_for,
)
from gpu_agent.benchmark.hardset_v2_agent_models import HardsetPublicCheck
from gpu_agent.benchmark.hardset_v2_public_probe import probe_candidate_publicly
from gpu_agent.benchmark.hardset_v2_selfcheck import check_candidate_publicly
from gpu_agent.harness.context import compact_public_context
from gpu_agent.harness.models import (
    FinishTask,
    HarnessAction,
    RepoGitDiff,
    RepoInspectSyntax,
    RepoListFiles,
    RepoProbePublic,
    RepoReadFile,
    RepoReplaceMany,
    RepoReplaceSnippet,
    RepoReplaceSyntax,
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
        "repo_replace_many",
        "repo_git_diff",
        "repo_inspect_syntax",
        "repo_replace_syntax",
        "repo_probe_public",
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
        context_mode: Literal["legacy", "evidence_first"] = "evidence_first",
        allow_batch_edits: bool = True,
        syntax_mode: bool = False,
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
        if context_mode not in {"legacy", "evidence_first"}:
            raise ValueError("unsupported repository context mode")
        self._context_mode = context_mode
        if type(allow_batch_edits) is not bool:
            raise ValueError("invalid batch-edit policy")
        self._batch_enabled = allow_batch_edits
        if type(syntax_mode) is not bool:
            raise ValueError("invalid syntax editing policy")
        self._syntax_mode = syntax_mode
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
        self._observed_query_fingerprints: set[str] = set()
        self._public_probe_signatures: set[str] = set()
        self._public_probe_run_ids: list[str] = []

    @property
    def observed_query_fingerprints(self) -> tuple[str, ...]:
        """Hashed read/list/diff queries; no source code is persisted here."""
        return tuple(sorted(self._observed_query_fingerprints))

    @property
    def public_probe_signatures(self) -> tuple[str, ...]:
        return tuple(sorted(self._public_probe_signatures))

    @property
    def public_probe_run_ids(self) -> tuple[str, ...]:
        return tuple(self._public_probe_run_ids)

    @property
    def public_probes_attempted(self) -> int:
        return len(self._public_probe_run_ids)

    def _query_fingerprint(self, payload: dict[str, str | int]) -> str:
        """Bind inspection requests to their exact revision, not LLM claims."""
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _require_new_observation(self, fingerprint: str, code: str) -> None:
        if fingerprint in self._observed_query_fingerprints:
            raise ValueError(f"{code}: unchanged inspection already completed; use prior evidence")

    def _configure_public_case(
        self, case_id: str | None, max_public_checks: int, *, resuming: bool = False
    ) -> None:
        if type(max_public_checks) is not int or not 1 <= max_public_checks <= 3:
            raise ValueError("public check budget must be between 1 and 3")
        if case_id is not None and case_id not in HARDSET_CASES:
            raise ValueError("unregistered public CUDA case")
        self._public_case_id = case_id
        self._max_public_checks = max_public_checks
        self._public_kernel_path: str | None = None
        if case_id is not None:
            # Initial bind proves the frozen checkout contains the registered
            # mutant and public harness. On resume its mutant is deliberately
            # modified, so rely on the verified original journal + Git HEAD.
            root = self.workspace.root / "benchmarks"
            name = f"benchmarks/hardset_v2/{case_id}/mutant/kernel.cu"
            self.workspace._checked(name, editable=True)
            if not resuming:
                registry, _ = load_registry(root)
                case = next(case for case in registry.cases if case.case_id == case_id)
                manifest = manifest_for(root, case, "mutant")
                if manifest[f"hardset_v2/{case_id}/mutant/kernel.cu"] != (
                    case.mutant_source_sha256
                ):
                    raise ValueError("public mutant does not match registered SHA")
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
        observed_query_fingerprints: tuple[str, ...] = (),
        public_probe_signatures: tuple[str, ...] = (),
        public_probe_run_ids: tuple[str, ...] = (),
        context_mode: Literal["legacy", "evidence_first"] = "evidence_first",
        allow_batch_edits: bool = True,
        syntax_mode: bool = False,
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
        if context_mode not in {"legacy", "evidence_first"}:
            raise ValueError("unsupported resumed context mode")
        host._context_mode = context_mode
        if type(allow_batch_edits) is not bool:
            raise ValueError("invalid resumed batch-edit policy")
        host._batch_enabled = allow_batch_edits
        if type(syntax_mode) is not bool:
            raise ValueError("invalid resumed syntax editing policy")
        host._syntax_mode = syntax_mode
        host._policy = permissions or ToolPermissionPolicy()
        host._configure_public_case(public_case_id, max_public_checks, resuming=True)
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
        if (
            len(observed_query_fingerprints) > tool_calls
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
                for value in observed_query_fingerprints
            )
            or len(set(observed_query_fingerprints)) != len(observed_query_fingerprints)
        ):
            raise ValueError("invalid resumed inspection query fingerprints")
        host._observed_query_fingerprints = set(observed_query_fingerprints)
        if (
            len(public_probe_signatures) != len(public_probe_run_ids)
            or len(public_probe_signatures) > 4
            or len(set(public_probe_signatures)) != len(public_probe_signatures)
            or len(set(public_probe_run_ids)) != len(public_probe_run_ids)
            or any(not isinstance(v, str) or len(v) != 64 for v in public_probe_signatures)
        ):
            raise ValueError("recovered public functional probe ledger is invalid")
        for identifier in public_probe_run_ids:
            child = host._store.load(identifier)
            if child.parent_run_id != run_id or child.kind != "hardset_v2_public_functional_probe":
                raise ValueError("recovered probe does not belong to session")
            refs = [x for x in child.artifact_refs if x.name == "probe/public-summary.json"]
            if len(refs) != 1:
                raise ValueError("missing public probe attestation")
        host._public_probe_signatures = set(public_probe_signatures)
        host._public_probe_run_ids = list(public_probe_run_ids)
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

    def _currently_allowed_tools(self) -> tuple[str, ...]:
        """Controller authority for the *current* source state.

        In registered single-kernel tasks, a repository-wide file listing is
        never needed and public GPU verification is unsafe before a fresh edit.
        Model output schemas and actual execution use this SAME capability list.
        """
        if self._public_case_id is None:
            return tuple(
                name
                for name in self.ALLOWED_TOOLS
                if self._policy.check(name) == "ALLOW"
                and (name != "repo_replace_many" or self._batch_enabled)
                and name != "repo_run_public_checks"
                and (
                    name not in {"repo_inspect_syntax", "repo_replace_syntax"} or self._syntax_mode
                )
                and (
                    name not in {"repo_replace_snippet", "repo_replace_many"}
                    or not self._syntax_mode
                )
            )
        delta = self.workspace.diff()
        dirty = bool(delta)
        passed = self.public_checks_passed
        target_only = self.workspace.edited_files == (self._public_kernel_path,)
        fresh_candidate = False
        if dirty and target_only and self._public_kernel_path is not None:
            source = read_regular(
                self.workspace._checked(self._public_kernel_path, editable=True),
                1024 * 1024,
            )
            fresh_candidate = hashlib.sha256(source).hexdigest() not in (
                self._attempted_candidate_hashes
            )
        if passed:
            return ("finish",) if self._policy.check("finish") == "ALLOW" else ()
        probe_phase = self._public_probe_phase()
        can_check = (
            dirty
            and target_only
            and fresh_candidate
            and self._public_checks_attempted < self._max_public_checks
            and not passed
        )
        diff_key = self._query_fingerprint(
            {
                "tool": "repo_git_diff",
                "source_diff_sha256": str(self.workspace.integrity()["diff_sha256"]),
            }
        )
        available: list[str] = []
        for name in self.ALLOWED_TOOLS:
            if self._policy.check(name) != "ALLOW":
                continue
            if name == "repo_list_files":
                continue
            if name == "repo_replace_many" and not self._batch_enabled:
                continue
            if name in {"repo_inspect_syntax", "repo_replace_syntax"} and not self._syntax_mode:
                continue
            if name in {"repo_replace_many", "repo_replace_snippet"} and self._syntax_mode:
                continue
            if name == "repo_inspect_syntax" and self._public_kernel_path is not None:
                original = read_regular(
                    self.workspace._checked(self._public_kernel_path, editable=False),
                    1024 * 1024,
                )
                syntax_key = self._query_fingerprint(
                    {
                        "tool": "repo_inspect_syntax",
                        "path": self._public_kernel_path,
                        "source_sha256": hashlib.sha256(original).hexdigest(),
                        "symbol": "",
                    }
                )
                if syntax_key in self._observed_query_fingerprints:
                    continue
            if name in {"repo_replace_many", "repo_replace_snippet"} and passed:
                continue
            if name == "repo_git_diff" and (
                not dirty or diff_key in self._observed_query_fingerprints
            ):
                continue
            if name == "repo_probe_public" and (
                self.public_probes_attempted >= 4 or probe_phase is None
            ):
                continue
            if name == "repo_run_public_checks" and not can_check:
                continue
            if name == "finish" and not passed:
                continue
            available.append(name)
        return tuple(available)

    def snapshot(self, goal: str, history: list[dict[str, str]], steps: int) -> SessionSnapshot:
        if self._public_kernel_path is not None and self._context_mode == "evidence_first":
            # Only the operator-pinned MUTANT source is sent as upfront evidence.
            # A candidate edit is reflected immediately on the next model turn.
            raw = read_regular(
                self.workspace._checked(self._public_kernel_path, editable=False),
                1024 * 1024,
            )
            source = (
                f"Registered editable CUDA kernel: {self._public_kernel_path}\n"
                + raw.decode("utf-8")[:10_700]
            )
        else:
            source = (
                f"Current inspected file: {self._last_file}\n{self._last_file_read}"
                if self._last_file is not None
                else (
                    f"Registered editable kernel: {self._public_kernel_path}. "
                    "Call repo_read_file on this exact path for source evidence."
                    if self._public_kernel_path is not None
                    else "No source file opened yet. Call repo_list_files then repo_read_file."
                )
            )
        context = compact_public_context(history, source)
        delta = self.workspace.diff()
        allowed_tools = self._currently_allowed_tools()
        requirement = goal
        if self._public_case_id is not None:
            requirement += (
                f"\nRegistered public GPU case: {self._public_case_id}. "
                + PUBLIC_REQUIREMENTS[self._public_case_id]
            )
        # Controller advice is non-prescriptive: the model still chooses each
        # action, but a late patch cannot consume the same turns as validation.
        guidance = ""
        if self._public_case_id is not None and steps <= 5:
            if not self.workspace.edited_files:
                guidance = (
                    "BUDGET_ALERT: no candidate source edit yet. Editing, one public "
                    "GPU check and finish require three separate remaining actions."
                )
            elif not self.public_checks_passed:
                guidance = (
                    "BUDGET_ALERT: current source is not publicly checked. "
                    "Prioritize an informative edit or registered public GPU check; "
                    "finish will be rejected until this exact diff passes."
                )
            else:
                guidance = (
                    "BUDGET_ALERT: this exact diff already passed public checks. "
                    "Call finish before losing the remaining step budget."
                )
        summary = (guidance + "\n" + context.summary)[:2800] if guidance else context.summary
        return SessionSnapshot(
            goal=goal,
            case_id="repository",
            base_git_commit=self.workspace.base_sha,
            source_hash=hashlib.sha256(self.workspace.base_sha.encode()).hexdigest(),
            functional_requirement=requirement,
            registered_public_case=self._public_case_id,
            registered_kernel_path=self._public_kernel_path,
            public_probe_phase=self._public_probe_phase(),
            remaining_public_probes=4 - self.public_probes_attempted,
            context_mode=self._context_mode,
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
            context_summary=summary,
            workspace_source=context.source,
            total_public_actions=context.total_actions,
        )

    def invoke(self, action: HarnessAction) -> ToolObservation:
        if self._finished:
            raise ValueError("repository workspace already finalized")
        self._tool_calls += 1
        permission = self._policy.check(action.tool)
        if (
            permission != "ALLOW"
            or action.tool not in self.ALLOWED_TOOLS
            or action.tool not in self._currently_allowed_tools()
        ):
            result = ToolObservation(
                tool=action.tool,
                status="REJECTED",
                content=(
                    "TOOL_DENY: atomic batch editing is disabled for this ablation"
                    if isinstance(action, RepoReplaceMany) and not self._batch_enabled
                    else "TOOL_NOT_AVAILABLE_AT_CURRENT_STATE: follow the allowed tool list"
                    if permission == "ALLOW"
                    else f"TOOL_{permission}: unavailable workspace capability"
                ),
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
                    "rejection_code": (
                        result.content.split(":", 2)[1].strip()
                        if result.status == "REJECTED"
                        and result.content.startswith("ValueError: REDUNDANT_")
                        else None
                    ),
                },
                sort_keys=True,
            ).encode(),
            "public",
        )
        return result

    def _dispatch(self, action: HarnessAction) -> ToolObservation:
        if isinstance(action, RepoListFiles):
            key = self._query_fingerprint({"tool": action.tool, "suffix": action.suffix})
            self._require_new_observation(key, "REDUNDANT_LIST")
            files = self.workspace.list_files(suffix=action.suffix)
            self._observed_query_fingerprints.add(key)
            return ToolObservation(
                tool=action.tool,
                status="OK",
                content="\n".join(files)[:4000] or "No allowed source files found.",
            )
        if isinstance(action, RepoReadFile):
            if self._public_kernel_path is not None and action.path != self._public_kernel_path:
                raise ValueError("REGISTERED_READ_SCOPE_VIOLATION")
            key = self._query_fingerprint(
                {
                    "tool": action.tool,
                    "path": action.path,
                    "start_line": action.start_line,
                    "lines": action.lines,
                    "source_diff_sha256": str(self.workspace.integrity()["diff_sha256"]),
                }
            )
            self._require_new_observation(key, "REDUNDANT_READ")
            text = self.workspace.read(action.path, start=action.start_line, lines=action.lines)
            self._observed_query_fingerprints.add(key)
            self._last_file = action.path
            self._last_file_read = text
            return ToolObservation(
                tool=action.tool,
                status="OK",
                content=text[:4000],
            )
        if isinstance(action, RepoInspectSyntax):
            if self._public_kernel_path is not None and action.path != self._public_kernel_path:
                raise ValueError("CUDA_SYNTAX_NOT_REGISTERED_KERNEL")
            target = self.workspace._checked(action.path, editable=False)
            digest = hashlib.sha256(read_regular(target, 1024 * 1024)).hexdigest()
            key = self._query_fingerprint(
                {
                    "tool": action.tool,
                    "path": action.path,
                    "source_sha256": digest,
                    "symbol": action.symbol,
                    "offset": action.offset,
                    "limit": action.limit,
                }
            )
            self._require_new_observation(key, "REDUNDANT_SYNTAX_INSPECTION")
            mapping = self.workspace.inspect_syntax(
                action.path, symbol=action.symbol, offset=action.offset, limit=action.limit
            )
            self._observed_query_fingerprints.add(key)
            # Parsing and cursor pagination are controller owned. Never trim
            # the serialized JSON: callers can use next_offset for more nodes.
            serialized = json.dumps(mapping, ensure_ascii=False)
            if len(serialized) > 3900 or not mapping["nodes"]:
                raise ValueError("CUDA_SYNTAX_PAGE_INVALID_OR_OVER_BUDGET")
            return ToolObservation(
                tool=action.tool,
                status="OK",
                content=serialized,
            )
        if isinstance(action, RepoReplaceSyntax):
            if self._public_kernel_path is not None and action.path != self._public_kernel_path:
                raise ValueError("CUDA_SYNTAX_NOT_REGISTERED_KERNEL")
            digest = self.workspace.replace_syntax(
                action.path,
                expected_source_sha256=action.expected_source_sha256,
                edits=[(e.node_id, e.replacement_code) for e in action.edits],
            )
            self._last_file = action.path
            self._last_file_read = self.workspace.read(action.path)
            return ToolObservation(
                tool=action.tool,
                status="OK",
                content=(
                    f"SHA-bound AST node edit accepted. source_sha256={digest}. "
                    "Run public CUDA checks on this new candidate."
                ),
            )
        if isinstance(action, RepoReplaceMany):
            candidate_sha = self.workspace.replace_batch(
                action.path,
                [(item.old_text, item.new_text) for item in action.edits],
            )
            self._last_file = action.path
            self._last_file_read = self.workspace.read(action.path)
            return ToolObservation(
                tool=action.tool,
                status="OK",
                content=f"Atomic batch accepted. source_sha256={candidate_sha}. "
                "Inspect changed diff or run the registered public GPU check.",
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
        if isinstance(action, RepoProbePublic):
            return self._run_registered_public_probe(action)
        if isinstance(action, RepoRunPublicChecks):
            return self._run_registered_public_checks()
        if isinstance(action, RepoGitDiff):
            diff = self.workspace.diff()
            key = self._query_fingerprint(
                {
                    "tool": action.tool,
                    "source_diff_sha256": str(self.workspace.integrity()["diff_sha256"]),
                }
            )
            self._require_new_observation(key, "REDUNDANT_DIFF")
            self._observed_query_fingerprints.add(key)
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

    def _public_probe_phase(self) -> Literal["before", "after"] | None:
        if self._public_case_id is None or self._public_kernel_path is None:
            return None
        if self.public_checks_passed or self.public_probes_attempted >= 4:
            return None
        diff = self.workspace.diff()
        if not diff and not self.workspace.edited_files:
            return "before"
        if diff and self.workspace.edited_files == (self._public_kernel_path,):
            return "after"
        return None

    def _run_registered_public_probe(self, action: RepoProbePublic) -> ToolObservation:
        if self._public_case_id is None or self._public_kernel_path is None:
            raise ValueError("PUBLIC_PROBE_UNREGISTERED_CASE")
        actual_phase = self._public_probe_phase()
        if actual_phase is None or action.phase != actual_phase:
            raise ValueError("PUBLIC_PROBE_PHASE_MISMATCH")
        source = read_regular(
            self.workspace._checked(self._public_kernel_path, editable=True),
            1024 * 1024,
        )
        digest = hashlib.sha256(source).hexdigest()
        signature = self._query_fingerprint(
            {
                "tool": action.tool,
                "phase": action.phase,
                "size": action.input_size,
                "source_sha256": digest,
            }
        )
        if signature in self._public_probe_signatures:
            raise ValueError("REDUNDANT_PUBLIC_PROBE")
        number = self.public_probes_attempted + 1
        result = probe_candidate_publicly(
            store=self._store,
            parent_run_id=self.run_id,
            repository=self.workspace.root,
            case_id=self._public_case_id,
            phase=action.phase,
            input_size=action.input_size,
            candidate=source,
            output=self.output / f"public-probe-{number:02d}",
        )
        if result.candidate_sha256 != digest or result.phase != action.phase:
            raise ValueError("PUBLIC_PROBE_SOURCE_PROVENANCE_MISMATCH")
        self._public_probe_signatures.add(signature)
        self._public_probe_run_ids.append(result.run_id)
        return ToolObservation(
            tool=action.tool,
            status="FAILED" if result.outcome == "PROBE_INCONCLUSIVE" else "OK",
            content=result.model_dump_json(),
        )

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
