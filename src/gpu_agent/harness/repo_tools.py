"""Repository-level Coding Agent tool host over a pinned Git worktree.

This adapter can stage multi-file source edits without granting host shell or
access to tests/evaluator. It intentionally cannot claim CUDA correctness:
compile/test tools must be explicitly integrated into an isolated executor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

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
    RepoInspectSyntaxNode,
    RepoListFiles,
    RepoProbePublic,
    RepoReadFile,
    RepoReplaceMany,
    RepoReplaceSnippet,
    RepoReplaceSyntax,
    RepoRunPublicChecks,
    SessionSnapshot,
    SyntaxEvidenceNode,
    SyntaxPageVisit,
    SyntaxState,
    ToolObservation,
)
from gpu_agent.harness.registry import ToolPermissionPolicy
from gpu_agent.harness.repo_audit import action_fingerprint, read_repository_tool_events
from gpu_agent.harness.syntax_feedback import SyntaxRejection
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
        "repo_inspect_syntax_node",
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
        extended_syntax: bool = False,
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
        if extended_syntax and (not syntax_mode or public_case_id is None):
            raise ValueError("extended AST requires a registered syntax session")
        self._extended_syntax = extended_syntax
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
        self._syntax_state: SyntaxState | None = None
        self._rejected_syntax_edits: set[str] = set()
        self._public_probe_signatures: set[str] = set()
        self._public_probe_run_ids: list[str] = []

    @property
    def observed_query_fingerprints(self) -> tuple[str, ...]:
        """Hashed read/list/diff queries; no source code is persisted here."""
        return tuple(sorted(self._observed_query_fingerprints))

    @property
    def syntax_state(self) -> SyntaxState | None:
        """Evidence survives prompt compaction; stale source invalidates it."""
        state = self._syntax_state
        if state is not None and self._syntax_digest(state.path) != state.source_sha256:
            self._invalidate_syntax_state()
        return self._syntax_state

    def _invalidate_syntax_state(self) -> None:
        state = self._syntax_state
        if state is not None:
            for page in state.visited_pages:
                self._observed_query_fingerprints.discard(
                    self._syntax_fingerprint(
                        state.path, state.source_sha256, state.symbol, page.offset, page.limit
                    )
                )
        # Historical actions remain immutable in RunStore. Dropping the active
        # fingerprints permits a fresh inspection if a later edit restores a SHA.
        self._syntax_state = None
        self._rejected_syntax_edits.clear()

    @property
    def rejected_syntax_edits(self) -> tuple[str, ...]:
        return tuple(sorted(self._rejected_syntax_edits))

    def _syntax_digest(self, path: str) -> str:
        target = self.workspace._checked(path, editable=False)
        return hashlib.sha256(read_regular(target, 1024 * 1024)).hexdigest()

    def _syntax_fingerprint(
        self, path: str, digest: str, symbol: str, offset: int, limit: int
    ) -> str:
        return self._query_fingerprint(
            {
                "tool": "repo_inspect_syntax",
                "path": path,
                "source_sha256": digest,
                "symbol": symbol,
                "offset": offset,
                "limit": limit,
            }
        )

    def _pin_syntax_page(self, action: RepoInspectSyntax, mapping: dict[str, Any]) -> None:
        state = self.syntax_state
        if state is not None and state.path != action.path:
            self._invalidate_syntax_state()
            state = None
        digest = self._syntax_digest(action.path)
        if mapping["source_sha256"] != digest:
            raise ValueError("CUDA_SOURCE_SHA_STALE")
        nodes = tuple(SyntaxEvidenceNode.model_validate(node) for node in mapping["nodes"])
        old_nodes = state.observed_nodes if state else ()
        old_pages = state.visited_pages if state else ()
        if set(node.id for node in old_nodes) & set(node.id for node in nodes):
            raise ValueError("REDUNDANT_SYNTAX_INSPECTION: page overlaps pinned nodes")
        self._syntax_state = SyntaxState(
            path=action.path,
            source_sha256=digest,
            symbol=action.symbol,
            next_offset=mapping["next_offset"],
            visited_pages=(*old_pages, SyntaxPageVisit(offset=action.offset, limit=action.limit)),
            observed_nodes=(*old_nodes, *nodes),
            extended_syntax=self._extended_syntax,
            full_node_reads=state.full_node_reads if state else (),
        )

    def _restore_syntax_state(
        self,
        state: SyntaxState | None,
        *,
        tool_calls: int,
        rejected_syntax_edits: tuple[str, ...],
    ) -> None:
        """Re-enumerate pinned AST pages and bind them to completed public actions."""
        active: list[dict[str, Any]] = []
        if self._syntax_mode and self._public_case_id is not None:
            records = read_repository_tool_events(
                self._store, self.run_id, expected_count=tool_calls, require_action=True
            )
            recorded_rejections: set[str] = set()
            complete_rejection_ledger = True
            for record in records:
                if record["status"] == "OK":
                    if record["tool"] == "repo_replace_syntax":
                        active.clear()
                        recorded_rejections.clear()
                        complete_rejection_ledger = True
                    elif record["tool"] in {"repo_inspect_syntax", "repo_inspect_syntax_node"}:
                        active.append(record)
                if "rejected_edit_fingerprint" not in record:
                    # Old audit records did not persist this optional field.
                    complete_rejection_ledger = False
                elif record["rejected_edit_fingerprint"] is not None:
                    recorded_rejections.add(record["rejected_edit_fingerprint"])
            if complete_rejection_ledger and recorded_rejections != set(rejected_syntax_edits):
                raise ValueError("recovered AST rejected edits differ from audited actions")
        if state is None:
            if active:
                raise ValueError("recovered AST state omitted completed inspection evidence")
            return
        if state.extended_syntax != self._extended_syntax:
            raise ValueError("recovered AST profile differs")
        if self._public_case_id is not None and len(active) != (
            len(state.visited_pages) + len(state.full_node_reads)
        ):
            raise ValueError("recovered AST page count differs from completed inspections")
        if not self._syntax_mode or (
            self._public_kernel_path is not None and state.path != self._public_kernel_path
        ):
            raise ValueError("recovered AST state is outside the authorized source")
        if self._syntax_digest(state.path) != state.source_sha256:
            raise ValueError("recovered AST state has stale source SHA")
        pages = iter(state.visited_pages)
        records_to_replay = active if self._public_case_id is not None else [
            {"tool": "repo_inspect_syntax"} for _ in state.visited_pages
        ]
        for record in records_to_replay:
            if record["tool"] == "repo_inspect_syntax_node":
                full = RepoInspectSyntaxNode.model_validate(record["action"])
                mapping = self._read_full_syntax_node(full)
                if record["observation_sha256"] != hashlib.sha256(mapping.encode()).hexdigest():
                    raise ValueError("recovered full AST node differs from audited observation")
                continue
            page = next(pages)
            previous = self._syntax_state
            if page.offset != (previous.next_offset if previous else 0):
                raise ValueError("recovered AST cursor is not contiguous")
            fingerprint = self._syntax_fingerprint(
                state.path, state.source_sha256, state.symbol, page.offset, page.limit
            )
            if fingerprint not in self._observed_query_fingerprints:
                raise ValueError("recovered AST page has no completed inspection")
            action = RepoInspectSyntax(
                path=state.path, symbol=state.symbol, offset=page.offset, limit=page.limit
            )
            mapping = self.workspace.inspect_syntax(
                state.path, symbol=state.symbol, offset=page.offset, limit=page.limit,
                extended=self._extended_syntax,
            )
            if self._public_case_id is not None:
                observation = json.dumps(mapping, ensure_ascii=False)
                if (
                    record["action"] != action.model_dump(mode="json")
                    or record["observation_sha256"]
                    != hashlib.sha256(observation.encode()).hexdigest()
                ):
                    raise ValueError("recovered AST page differs from audited public observation")
            self._pin_syntax_page(action, mapping)
        if self._syntax_state != state:
            raise ValueError("recovered AST evidence differs from current parser output")

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
        syntax_state: SyntaxState | None = None,
        rejected_syntax_edits: tuple[str, ...] = (),
        public_probe_signatures: tuple[str, ...] = (),
        public_probe_run_ids: tuple[str, ...] = (),
        context_mode: Literal["legacy", "evidence_first"] = "evidence_first",
        allow_batch_edits: bool = True,
        syntax_mode: bool = False,
        extended_syntax: bool = False,
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
        if extended_syntax and (not syntax_mode or public_case_id is None):
            raise ValueError("extended AST requires a registered syntax session")
        host._extended_syntax = extended_syntax
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
        host._syntax_state = None
        host._rejected_syntax_edits = set()
        host._restore_syntax_state(
            syntax_state, tool_calls=tool_calls, rejected_syntax_edits=rejected_syntax_edits
        )
        if (
            len(rejected_syntax_edits) > tool_calls
            or len(set(rejected_syntax_edits)) != len(rejected_syntax_edits)
            or any(
                len(v) != 64 or any(ch not in "0123456789abcdef" for ch in v)
                for v in rejected_syntax_edits
            )
            or (rejected_syntax_edits and syntax_state is None)
        ):
            raise ValueError("invalid recovered AST rejection ledger")
        host._rejected_syntax_edits = set(rejected_syntax_edits)
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
                and name not in {"repo_run_public_checks", "repo_inspect_syntax_node"}
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
                state = self.syntax_state
                if state is not None:
                    if state.next_offset is None:
                        continue
                    syntax_key = self._syntax_fingerprint(
                        state.path, state.source_sha256, state.symbol, state.next_offset, 8
                    )
                    if syntax_key in self._observed_query_fingerprints:
                        continue
            if name == "repo_inspect_syntax_node" and (
                not self._extended_syntax or self.syntax_state is None
                or not self.syntax_state.readable_truncated_aliases
            ):
                continue
            if name == "repo_replace_syntax" and (
                self.syntax_state is None
                or not any(not node.code_truncated for node in self.syntax_state.observed_nodes)
            ):
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
        context = compact_public_context(history, source, syntax_pinned=self._syntax_mode)
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
            syntax_mode=self._syntax_mode,
            extended_syntax=self._extended_syntax,
            syntax_state=self.syntax_state,
            context_summary=summary,
            workspace_source=context.source,
            total_public_actions=context.total_actions,
        )

    def _stale_syntax_sha(self, action: RepoReplaceSyntax, actual: str) -> SyntaxRejection:
        return SyntaxRejection(
            "CUDA_SOURCE_SHA_STALE",
            expected_source_sha256=action.expected_source_sha256,
            actual_source_sha256=actual,
            correction="Use the current syntax_state; inspect again if it is absent.",
        )

    def _syntax_denial(self, action: HarnessAction) -> SyntaxRejection | None:
        """Give precise corrections without accessing any unauthorized path."""
        if not self._syntax_mode or self._public_kernel_path is None:
            return None
        if not isinstance(action, (RepoInspectSyntax, RepoReplaceSyntax)):
            return None
        if action.path != self._public_kernel_path:
            return SyntaxRejection(
                "CUDA_SYNTAX_NOT_REGISTERED_KERNEL",
                correction="Use the operator-registered kernel path only.",
            )
        state = self.syntax_state
        if isinstance(action, RepoInspectSyntax):
            if (
                state is not None
                and action.symbol == state.symbol
                and any(page.offset == action.offset for page in state.visited_pages)
            ):
                return SyntaxRejection(
                    "REDUNDANT_SYNTAX_INSPECTION",
                    requested_offset=action.offset,
                    next_offset=state.next_offset,
                    correction="Use pinned observed_nodes; inspect only the next available page.",
                )
            return None
        actual = self._syntax_digest(action.path)
        if action.expected_source_sha256 != actual:
            return self._stale_syntax_sha(action, actual)
        if state is None:
            return SyntaxRejection(
                "CUDA_SYNTAX_EVIDENCE_REQUIRED",
                actual_source_sha256=actual,
                correction="Inspect the current revision before editing.",
            )
        return None

    def invoke(self, action: HarnessAction) -> ToolObservation:
        if self._finished:
            raise ValueError("repository workspace already finalized")
        self._tool_calls += 1
        permission = self._policy.check(action.tool)
        full_public_audit = self._syntax_mode and self._public_case_id is not None
        detail: dict[str, Any] | None = None
        if (
            permission != "ALLOW"
            or action.tool not in self.ALLOWED_TOOLS
            or action.tool not in self._currently_allowed_tools()
        ):
            correction = self._syntax_denial(action) if permission == "ALLOW" else None
            if correction is not None:
                detail = correction.detail
                content = f"ValueError: {correction}"
            elif isinstance(action, RepoReplaceMany) and not self._batch_enabled:
                content = "TOOL_DENY: atomic batch editing is disabled for this ablation"
            elif permission == "ALLOW":
                content = "TOOL_NOT_AVAILABLE_AT_CURRENT_STATE: follow the allowed tool list"
            else:
                content = f"TOOL_{permission}: unavailable workspace capability"
            result = ToolObservation(tool=action.tool, status="REJECTED", content=content)
        else:
            try:
                result = self._dispatch(action)
            except SyntaxRejection as exc:
                detail = exc.detail
                result = ToolObservation(
                    tool=action.tool, status="REJECTED", content=f"ValueError: {exc}"
                )
            except (ValueError, OSError) as exc:
                # Do not persist arbitrary exception text (paths, SDK messages,
                # private diagnostics) in a full public AST observation.
                message = str(exc)[:500]
                if full_public_audit:
                    code = message.split(":", 1)[0]
                    if (
                        isinstance(exc, ValueError)
                        and code.isascii()
                        and code.replace("_", "").isalnum()
                        and code.isupper()
                        and code.startswith(("CUDA_", "REDUNDANT_", "PUBLIC_", "REGISTERED_"))
                        and len(code) <= 80
                    ):
                        message = code
                    else:
                        message = (
                            "PUBLIC_TOOL_IO_FAILED"
                            if isinstance(exc, OSError)
                            else (
                                "PUBLIC_TOOL_REJECTED: controller rejected the requested operation"
                            )
                        )
                result = ToolObservation(
                    tool=action.tool,
                    status="REJECTED",
                    content=f"{type(exc).__name__}: {message}",
                )
        record: dict[str, Any] = {
            "tool": action.tool,
            "action": action.model_dump(mode="json"),
            "status": result.status,
            "observation_sha256": hashlib.sha256(result.content.encode()).hexdigest(),
            "edited_files": self.workspace.edited_files,
            "diff_sha256": self.workspace.integrity()["diff_sha256"],
            "rejection_code": (
                result.content.split(":", 2)[1].strip()
                if result.status == "REJECTED"
                and result.content.startswith(
                    (
                        "ValueError: REDUNDANT_",
                        "ValueError: CUDA_",
                        "ValueError: PUBLIC_",
                        "ValueError: REGISTERED_",
                        "OSError: PUBLIC_",
                    )
                )
                else None
            ),
        }
        if full_public_audit:
            # Only the bounded model-visible public ToolObservation, never raw
            # executor artifacts or the private verifier's observations.
            record["observation"] = result.content
            record["rejection_detail"] = detail
            fingerprint = action_fingerprint(record["action"])
            record["rejected_edit_fingerprint"] = (
                fingerprint
                if isinstance(action, RepoReplaceSyntax)
                and result.status == "REJECTED"
                and fingerprint in self._rejected_syntax_edits
                else None
            )
        self._store.put(
            self.run_id,
            f"repo-harness/v1/tool-{self._tool_calls:03d}.json",
            json.dumps(record, sort_keys=True).encode(),
            "public",
        )
        return result

    def _read_full_syntax_node(self, action: RepoInspectSyntaxNode) -> str:
        if not self._extended_syntax or action.path != self._public_kernel_path:
            raise ValueError("CUDA_SYNTAX_NOT_REGISTERED_KERNEL")
        state = self.syntax_state
        actual = self._syntax_digest(action.path)
        if actual != action.expected_source_sha256:
            raise SyntaxRejection(
                "CUDA_SOURCE_SHA_STALE", expected_source_sha256=action.expected_source_sha256,
                actual_source_sha256=actual, correction="Inspect the current revision.",
            )
        if state is None or action.node_id not in state.readable_truncated_aliases:
            raise SyntaxRejection(
                "CUDA_SYNTAX_FULL_NODE_UNAVAILABLE", node_id=action.node_id,
                correction="Read only a bounded, observed and still truncated node.",
            )
        mapping = self.workspace.inspect_syntax_node(
            action.path, expected_source_sha256=actual, node_id=action.node_id,
            extended=True,
        )
        node = SyntaxEvidenceNode.model_validate(mapping["node"])
        previous = next(n for n in state.observed_nodes if n.id == action.node_id)
        if (
            mapping["source_sha256"] != actual or node.code_truncated
            or not node.code.startswith(previous.code)
            or node.kind != previous.kind or node.function != previous.function
        ):
            raise ValueError("CUDA_SYNTAX_FULL_NODE_PROVENANCE_MISMATCH")
        content = json.dumps(mapping, ensure_ascii=False)
        if len(content) > 3900:
            raise ValueError("CUDA_SYNTAX_PAGE_INVALID_OR_OVER_BUDGET")
        self._syntax_state = state.model_copy(update={
            "observed_nodes": tuple(node if n.id == node.id else n for n in state.observed_nodes),
            "full_node_reads": (*state.full_node_reads, node.id),
        })
        return content

    def _dispatch(self, action: HarnessAction) -> ToolObservation:
        if isinstance(action, RepoInspectSyntaxNode):
            return ToolObservation(
                tool=action.tool, status="OK", content=self._read_full_syntax_node(action)
            )
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
            digest = self._syntax_digest(action.path)
            state = self.syntax_state
            if state is not None and state.path == action.path:
                if action.symbol != state.symbol:
                    raise SyntaxRejection(
                        "CUDA_SYNTAX_SYMBOL_CHANGED",
                        symbol=state.symbol,
                        next_offset=state.next_offset,
                        correction="Keep the pinned symbol and cursor.",
                    )
                if action.offset != state.next_offset:
                    code = (
                        "REDUNDANT_SYNTAX_INSPECTION"
                        if any(page.offset == action.offset for page in state.visited_pages)
                        else "CUDA_SYNTAX_CURSOR_MISMATCH"
                    )
                    raise SyntaxRejection(
                        code,
                        requested_offset=action.offset,
                        next_offset=state.next_offset,
                        correction="Use pinned observed_nodes and inspect only next_offset.",
                    )
            elif action.offset != 0:
                raise SyntaxRejection(
                    "CUDA_SYNTAX_CURSOR_MISMATCH",
                    requested_offset=action.offset,
                    next_offset=0,
                    correction="Start a fresh source revision at offset 0.",
                )
            key = self._syntax_fingerprint(
                action.path, digest, action.symbol, action.offset, action.limit
            )
            self._require_new_observation(key, "REDUNDANT_SYNTAX_INSPECTION")
            mapping = self.workspace.inspect_syntax(
                action.path, symbol=action.symbol, offset=action.offset, limit=action.limit,
                extended=self._extended_syntax,
            )
            # Parsing and cursor pagination are controller owned. Never trim
            # the serialized JSON: callers can use next_offset for more nodes.
            serialized = json.dumps(mapping, ensure_ascii=False)
            if len(serialized) > 3900 or not mapping["nodes"]:
                raise ValueError("CUDA_SYNTAX_PAGE_INVALID_OR_OVER_BUDGET")
            self._pin_syntax_page(action, mapping)
            self._observed_query_fingerprints.add(key)
            return ToolObservation(
                tool=action.tool,
                status="OK",
                content=serialized,
            )
        if isinstance(action, RepoReplaceSyntax):
            if self._public_kernel_path is not None and action.path != self._public_kernel_path:
                raise ValueError("CUDA_SYNTAX_NOT_REGISTERED_KERNEL")
            current = self._syntax_digest(action.path)
            if current != action.expected_source_sha256:
                raise self._stale_syntax_sha(action, current)
            state = self.syntax_state
            if state is None or state.path != action.path:
                raise SyntaxRejection(
                    "CUDA_SYNTAX_EVIDENCE_REQUIRED",
                    actual_source_sha256=current,
                    correction="Inspect the current revision before editing.",
                )
            fingerprint = action_fingerprint(action.model_dump(mode="json"))
            if fingerprint in self._rejected_syntax_edits:
                raise SyntaxRejection(
                    "CUDA_SYNTAX_REPEATED_REJECTED_EDIT",
                    source_sha256=current,
                    edit_fingerprint=fingerprint,
                    correction="Change the rejected edit based on the current public evidence.",
                )
            try:
                observed = {node.id: node for node in state.observed_nodes}
                for edit in action.edits:
                    node = observed.get(edit.node_id)
                    if node is None:
                        raise SyntaxRejection(
                            "CUDA_SYNTAX_NODE_NOT_FOUND",
                            node_id=edit.node_id,
                            source_sha256=current,
                            correction="Select an alias from current syntax_state.observed_nodes.",
                        )
                    if node.code_truncated:
                        raise SyntaxRejection(
                            "CUDA_SYNTAX_NODE_CODE_TRUNCATED",
                            node_id=edit.node_id,
                            correction="Only complete observed nodes are editable.",
                        )
                    if edit.replacement_code.strip() == node.code.strip():
                        requirement = PUBLIC_REQUIREMENTS.get(self._public_case_id or "", "")
                        raise SyntaxRejection(
                            "CUDA_SYNTAX_NOOP",
                            node_id=node.id,
                            source_sha256=current,
                            original_code=node.code,
                            functional_requirement=requirement,
                            correction=(
                                "replacement_code must differ from the original and address the "
                                "computation defect; do not fabricate a change."
                            ),
                        )
                digest = self.workspace.replace_syntax(
                    action.path,
                    expected_source_sha256=action.expected_source_sha256,
                    edits=[(e.node_id, e.replacement_code) for e in action.edits],
                    extended=self._extended_syntax,
                )
            except ValueError as exc:
                self._rejected_syntax_edits.add(fingerprint)
                corrections = {
                    "CUDA_SYNTAX_DUPLICATE_NODE": "Submit each observed alias only once.",
                    "CUDA_SYNTAX_OVERLAPPING_NODES": "Submit disjoint observed AST nodes.",
                    "CUDA_SYNTAX_REPLACEMENT_INVALID": "Use bounded valid UTF-8 CUDA code.",
                    "CUDA_SYNTAX_REPLACEMENT_PARSE_ERROR": "Fix the replacement CUDA syntax.",
                    "CUDA_SYNTAX_EXPRESSION_SLOT_INVALID": "Replace only one expression.",
                    "CUDA_SYNTAX_RESULT_TOO_LARGE": "Keep the source within the size limit.",
                }
                if not isinstance(exc, SyntaxRejection) and str(exc) in corrections:
                    raise SyntaxRejection(
                        str(exc), source_sha256=current, correction=corrections[str(exc)]
                    ) from None
                raise
            self._invalidate_syntax_state()
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
        # Durable admission before the backend call. A crash or rejected
        # backend request is not a completed check.
        self._store.put(
            self.run_id,
            f"repo-harness/v1/check-{self._tool_calls:03d}-admitted.json",
            json.dumps(
                {
                    "turn": self._tool_calls,
                    "candidate_sha256": candidate_sha,
                    "diff_sha256": diff_sha,
                    "case_id": self._public_case_id,
                },
                sort_keys=True,
            ).encode(),
            "public",
        )
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
