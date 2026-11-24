"""Host-owned Git worktree for multi-file CUDA source editing.

The language model never invokes git or receives a host command capability.
Only an operator-created, detached, SHA-pinned worktree can be edited. This
module does not execute or import untrusted repository code.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from gpu_agent.store import read_regular, reject_symlinks, sync_directory

SOURCE_EXTENSIONS = frozenset({".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"})
PUBLIC_READ_EXTENSIONS = SOURCE_EXTENSIONS | {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".cmake",
}
MAX_SOURCE_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 512 * 1024
MAX_TRACKED_FILES = 20_000

_READONLY_PREFIXES = (
    "tests/",
    "docs/",
    ".github/",
    "scripts/",
    "benchmarks/harness/",
    "src/gpu_agent/agent/",
    "src/gpu_agent/harness/",
    "src/gpu_agent/execution/",
)
_FORBIDDEN_NAME_PARTS = {
    ".git",
    ".env",
    ".ssh",
    ".cache",
    "__pycache__",
    "evaluator",
    "private",
    "secrets",
    "credentials",
    "clean",
}
_NAME = re.compile(r"^[a-zA-Z0-9_.-]+$")

# Model-facing source is strictly less privileged than controller/Oracle code.
# The operator may read these internally for validation; model tools cannot.
_PUBLIC_DENIED_PREFIXES = (
    "tests/",
    "test/",
    "evaluation/",
    "docs/",
    "scripts/",
    ".github/",
    "benchmarks/harness/",
    "benchmarks/development_truth/",
    "src/gpu_agent/benchmark/",
    "src/gpu_agent/verification/",
    "src/gpu_agent/execution/",
    "src/gpu_agent/harness/",
    "src/gpu_agent/agent/",
)
_PUBLIC_DENIED_COMPONENTS = frozenset(
    {
        "oracle",
        "oracles",
        "reference",
        "references",
        "gold",
        "ground_truth",
        "holdout",
        "hidden",
        "evaluation",
        "eval",
    }
)


def _model_read_allowed(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (
        not any(name.startswith(p) for p in _PUBLIC_DENIED_PREFIXES)
        and not any(p.lower() in _PUBLIC_DENIED_COMPONENTS for p in parts)
        and not (name.endswith(".test.cu") or name.endswith("_test.py"))
    )


def _run_git(repo: Path, *argv: str) -> str:
    cmd = ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", *argv]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=25,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "LC_ALL": "C",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    if completed.returncode:
        raise ValueError(f"git operation rejected: {argv[0]} failed")
    if len(completed.stdout) > 2 * 1024 * 1024:
        raise ValueError("git output exceeded controller budget")
    return completed.stdout.strip()


def _parts(path: str) -> tuple[str, ...]:
    if not path or "\\" in path or "\x00" in path:
        raise ValueError("invalid relative repository path")
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or path != parsed.as_posix()
        or path.startswith("./")
    ):
        raise ValueError("absolute or ambiguous path forbidden")
    if any(p in {".", ".."} or not _NAME.fullmatch(p) for p in parsed.parts):
        raise ValueError("path traversal or nonportable component forbidden")
    if any(p.lower() in _FORBIDDEN_NAME_PARTS for p in parsed.parts):
        raise ValueError("protected repository path forbidden")
    if len(path) > 300:
        raise ValueError("repository path too long")
    return parsed.parts


class PinnedGitWorkspace:
    """Manage a disposable detached Git worktree with a frozen base commit."""

    def __init__(
        self,
        *,
        origin: Path,
        root: Path,
        sha: str,
        max_edited_files: int = 8,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("full SHA-1 Git commit hash is required")
        if type(max_edited_files) is not int or not 1 <= max_edited_files <= 20:
            raise ValueError("invalid edit count limit")
        reject_symlinks(origin)
        reject_symlinks(root)
        if not origin.is_dir() or root.exists() or root.is_symlink():
            raise ValueError("origin must exist and isolated worktree must not exist")
        trusted_root = origin.resolve()
        out = root.absolute()
        if out.is_relative_to(trusted_root) or trusted_root.is_relative_to(out):
            raise ValueError("worktree must not overlap source repository")
        if not (trusted_root / ".git").exists():
            raise ValueError("origin is not a git worktree")
        # sha must refer to a commit in the trusted source repository.
        actual_type = _run_git(trusted_root, "cat-file", "-t", sha)
        if actual_type != "commit":
            raise ValueError("requested source SHA is not a Git commit")
        if _run_git(trusted_root, "rev-parse", "--verify", f"{sha}^{{commit}}") != sha:
            raise ValueError("source commit does not match requested SHA")

        out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _run_git(
            trusted_root,
            "worktree",
            "add",
            "--quiet",
            "--detach",
            str(out),
            sha,
        )
        self.root = out.resolve()
        self.origin = trusted_root
        self.base_sha = sha
        self.max_edited_files = max_edited_files
        if _run_git(self.root, "rev-parse", "HEAD") != sha:
            raise ValueError("materialized worktree SHA mismatch")
        tracked = _run_git(self.root, "ls-files")
        items = tuple(tracked.splitlines())
        if len(items) > MAX_TRACKED_FILES:
            raise ValueError("repository has too many tracked files for this harness")
        self._tracked = frozenset(items)
        self._edited: set[str] = set()
        self._accepted_hashes: dict[str, str] = {}

    @classmethod
    def attach_existing(
        cls,
        *,
        origin: Path,
        root: Path,
        sha: str,
        accepted_hashes: dict[str, str],
        max_edited_files: int = 8,
    ) -> PinnedGitWorkspace:
        """Reattach only a controller-owned worktree; never silently recreate it.

        The tracked checkout, Git common directory and every modified file
        must still match the frozen workspace and durable checkpoint.
        """
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("full SHA-1 Git commit hash is required")
        if type(max_edited_files) is not int or not 1 <= max_edited_files <= 20:
            raise ValueError("invalid edit count limit")
        reject_symlinks(origin)
        reject_symlinks(root)
        if not origin.is_dir() or not root.is_dir():
            raise ValueError("resume requires original Git repository and worktree")
        trusted = origin.resolve()
        checkout = root.resolve()
        if checkout.is_relative_to(trusted) or trusted.is_relative_to(checkout):
            raise ValueError("worktree must not overlap source repository")
        gitfile = checkout / ".git"
        if not gitfile.is_file() or gitfile.is_symlink():
            raise ValueError("resume requires a real Git worktree pointer")
        if _run_git(checkout, "rev-parse", "--show-toplevel") != str(checkout):
            raise ValueError("resumed worktree root mismatch")
        if _run_git(checkout, "rev-parse", "HEAD") != sha:
            raise ValueError("resumed worktree commit mismatch")

        def common_dir(path: Path) -> Path:
            value = Path(_run_git(path, "rev-parse", "--git-common-dir"))
            return (value if value.is_absolute() else path / value).resolve(strict=True)

        if common_dir(checkout) != common_dir(trusted):
            raise ValueError("resumed worktree belongs to another Git repository")
        if len(accepted_hashes) > max_edited_files:
            raise ValueError("resumed edited-file budget exceeded")
        project = object.__new__(cls)
        project.root = checkout
        project.origin = trusted
        project.base_sha = sha
        project.max_edited_files = max_edited_files
        tracked = tuple(_run_git(checkout, "ls-files").splitlines())
        if len(tracked) > MAX_TRACKED_FILES:
            raise ValueError("resumed repository exceeds tracked-file limit")
        project._tracked = frozenset(tracked)
        project._edited = set()
        project._accepted_hashes = {}
        for name, digest in accepted_hashes.items():
            if not isinstance(name, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("invalid resumed file digest")
            project._checked(name, editable=True)
            project._edited.add(name)
            project._accepted_hashes[name] = digest
        project._verify_boundary()
        return project

    @property
    def edited_files(self) -> tuple[str, ...]:
        return tuple(sorted(self._edited))

    @property
    def accepted_hashes(self) -> dict[str, str]:
        self._verify_boundary()
        return dict(self._accepted_hashes)

    def _verify_boundary(self) -> None:
        """Fail closed on ANY unstaged, staged, untracked or post-edit drift.

        A diff limited to the model's edited_files silently misses a separately
        modified verifier, test or kernel. Validate the entire checkout first.
        Git output is NUL-delimited: filenames may contain newlines or spaces.
        """
        if _run_git(self.root, "rev-parse", "HEAD") != self.base_sha:
            raise ValueError("worktree HEAD drift detected")
        if _run_git(self.root, "ls-files", "--others", "-z"):
            raise ValueError("untracked worktree content detected")
        raw = _run_git(
            self.root,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--name-status",
            "-z",
            "HEAD",
            "--",
        )
        values = raw.split("\x00") if raw else []
        if values and values[-1] == "":
            values.pop()
        if len(values) % 2:
            raise ValueError("ambiguous Git status output")
        observed: set[str] = set()
        for index in range(0, len(values), 2):
            status, name = values[index], values[index + 1]
            if status != "M" or name not in self._edited:
                # Report unsafe symlink/type substitution accurately, without
                # allowing it to be mistaken for an approved content edit.
                if name in self._tracked and _model_read_allowed(name):
                    self._checked(name, editable=False)
                raise ValueError("unauthorized worktree change detected")
            observed.add(name)
        for name in self._edited:
            expected = self._accepted_hashes.get(name)
            if expected is None:
                raise ValueError("edited file is missing controller provenance")
            actual = hashlib.sha256(
                read_regular(self._checked(name, editable=True), MAX_SOURCE_BYTES)
            ).hexdigest()
            if actual != expected:
                raise ValueError("edited source changed outside controller")

    def _checked(self, name: str, *, editable: bool) -> Path:
        parts = _parts(name)
        norm = "/".join(parts)
        if norm not in self._tracked:
            raise ValueError("file is not registered in frozen worktree")
        if editable and (
            any(norm.startswith(prefix) for prefix in _READONLY_PREFIXES)
            or norm.endswith(".test.cu")
        ):
            raise ValueError("cannot modify tests, verifier, tools or controls")
        if editable and PurePosixPath(norm).suffix not in SOURCE_EXTENSIONS:
            raise ValueError("only existing source-code files are editable")
        if not editable and PurePosixPath(norm).suffix not in PUBLIC_READ_EXTENSIONS:
            raise ValueError("file extension is not public-readable")
        if not _model_read_allowed(norm):
            raise ValueError("model access to evaluator, Oracle or test source is forbidden")
        candidate = self.root
        for part in parts:
            candidate /= part
            if candidate.is_symlink():
                raise ValueError("symlink traversal forbidden")
        if not candidate.is_file() or not candidate.resolve().is_relative_to(self.root):
            raise ValueError("file is not a regular worktree source")
        return candidate

    def list_files(self, *, suffix: str = "", max_files: int = 100) -> tuple[str, ...]:
        self._verify_boundary()
        if not 1 <= max_files <= 500 or len(suffix) > 20:
            raise ValueError("file listing out of bounds")
        return tuple(
            name
            for name in sorted(self._tracked)
            if (not suffix or name.endswith(suffix))
            and PurePosixPath(name).suffix in PUBLIC_READ_EXTENSIONS
            and not any(p.lower() in _FORBIDDEN_NAME_PARTS for p in PurePosixPath(name).parts)
            and _model_read_allowed(name)
        )[:max_files]

    def read(self, name: str, *, start: int = 1, lines: int = 120) -> str:
        self._verify_boundary()
        if not 1 <= start <= 100_000 or not 1 <= lines <= 200:
            raise ValueError("bounded source read parameters required")
        source = read_regular(self._checked(name, editable=False), MAX_SOURCE_BYTES)
        decoded = source.decode("utf-8").splitlines()
        excerpt = decoded[start - 1 : start - 1 + lines]
        return "\n".join(f"{start + index:4d} {line}" for index, line in enumerate(excerpt))[
            :20_000
        ]

    def _stage_source(self, name: str, target: Path, original: bytes, modified: bytes) -> str:
        """Host-controlled atomic write, shared by single and grouped edits."""
        if modified == original or len(modified) > MAX_SOURCE_BYTES:
            raise ValueError("no-op or overlarge source edit")
        mode = stat.S_IMODE(target.stat().st_mode)
        fd, temporary = tempfile.mkstemp(prefix=".gpu-agent-stage-", dir=target.parent)
        try:
            os.fchmod(fd, mode)
            with os.fdopen(fd, "wb") as handle:
                handle.write(modified)
                handle.flush()
                os.fsync(handle.fileno())
            if target.is_symlink():
                raise ValueError("edited path became a symlink")
            os.replace(temporary, target)
            sync_directory(target.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        digest = hashlib.sha256(modified).hexdigest()
        self._edited.add(name)
        self._accepted_hashes[name] = digest
        return digest

    def replace_exact(self, name: str, before: str, after: str) -> str:
        self._verify_boundary()
        target = self._checked(name, editable=True)
        if name not in self._edited and len(self._edited) >= self.max_edited_files:
            raise ValueError("workspace modified-file budget exhausted")
        if len(before) > 4_000 or len(after) > 4_000:
            raise ValueError("bounded patch text required")
        original = read_regular(target, MAX_SOURCE_BYTES)
        text = original.decode("utf-8")
        if not before or text.count(before) != 1:
            raise ValueError("edit source must match exactly once")
        modified = text.replace(before, after, 1).encode("utf-8")
        return self._stage_source(name, target, original, modified)

    def replace_batch(self, name: str, edits: list[tuple[str, str]]) -> str:
        """Stage several disjoint, uniquely anchored edits as ONE atomic change.

        Match all anchors against the same frozen current revision, rather than
        letting one edit alter the matching semantics of a later edit. Invalid
        or overlapping requests leave the checkout and edit ledger unchanged.
        """
        self._verify_boundary()
        target = self._checked(name, editable=True)
        if name not in self._edited and len(self._edited) >= self.max_edited_files:
            raise ValueError("workspace modified-file budget exhausted")
        if not 2 <= len(edits) <= 4:
            raise ValueError("batch edit must include 2-4 exact snippets")
        original = read_regular(target, MAX_SOURCE_BYTES)
        text = original.decode("utf-8")
        ranges: list[tuple[int, int, str]] = []
        for before, after in edits:
            if (
                not before
                or len(before) > 4_000
                or len(after) > 4_000
                or before == after
                or "\x00" in before
                or "\x00" in after
            ):
                raise ValueError("invalid bounded batch edit snippet")
            if text.count(before) != 1:
                raise ValueError("batch edit source must match exactly once")
            start = text.find(before)
            ranges.append((start, start + len(before), after))
        ranges.sort(key=lambda item: item[0])
        cursor = 0
        pieces: list[str] = []
        for start, end, replacement in ranges:
            if start < cursor:
                raise ValueError("overlapping batch edits are forbidden")
            pieces.append(text[cursor:start])
            pieces.append(replacement)
            cursor = end
        pieces.append(text[cursor:])
        return self._stage_source(name, target, original, "".join(pieces).encode("utf-8"))

    def inspect_syntax(
        self, name: str, *, symbol: str = "", offset: int = 0, limit: int = 8,
        extended: bool = False,
    ) -> dict[str, object]:
        """Parse only an existing approved CUDA source file, without executing it."""
        from gpu_agent.harness.syntax_tools import syntax_page

        self._verify_boundary()
        path = self._checked(name, editable=False)
        if path.suffix not in {".cu", ".cuh"}:
            raise ValueError("CUDA_SYNTAX_ONLY_CUDA_SOURCE")
        raw = read_regular(path, MAX_SOURCE_BYTES)
        return syntax_page(raw, symbol=symbol, offset=offset, limit=limit, extended=extended)

    def inspect_syntax_node(
        self, name: str, *, expected_source_sha256: str, node_id: str, extended: bool = False
    ) -> dict[str, object]:
        from gpu_agent.harness.syntax_tools import syntax_node_detail

        self._verify_boundary()
        path = self._checked(name, editable=False)
        if path.suffix not in {".cu", ".cuh"}:
            raise ValueError("CUDA_SYNTAX_ONLY_CUDA_SOURCE")
        return syntax_node_detail(
            read_regular(path, MAX_SOURCE_BYTES), expected_source_sha256=expected_source_sha256,
            node_id=node_id, extended=extended,
        )

    def replace_syntax(
        self,
        name: str,
        *,
        expected_source_sha256: str,
        edits: list[tuple[str, str]],
        extended: bool = False,
    ) -> str:
        """Replace exact parsed nodes bound to a SHA-pinned current source.

        Unlike model-quoted old_text, node IDs are produced by the controller
        from the real parser. It rejects stale IDs and all malformed outputs.
        """
        from gpu_agent.harness.syntax_tools import rewrite_syntax_nodes

        self._verify_boundary()
        target = self._checked(name, editable=True)
        if target.suffix not in {".cu", ".cuh"}:
            raise ValueError("CUDA_SYNTAX_ONLY_CUDA_SOURCE")
        if name not in self._edited and len(self._edited) >= self.max_edited_files:
            raise ValueError("workspace modified-file budget exhausted")
        original = read_regular(target, MAX_SOURCE_BYTES)
        changed = rewrite_syntax_nodes(
            original, expected_source_sha256=expected_source_sha256, edits=edits,
            extended=extended,
        )
        return self._stage_source(name, target, original, changed)

    def diff(self) -> str:
        self._verify_boundary()
        paths = self.edited_files
        if not paths:
            return ""
        _run_git(self.root, "diff", "--check", "--", *paths)
        value = _run_git(self.root, "diff", "--no-ext-diff", "--", *paths)
        if len(value.encode()) > MAX_OUTPUT_BYTES:
            raise ValueError("worktree diff exceeds audit limit")
        return value

    def integrity(self) -> dict[str, str | int]:
        self._verify_boundary()
        return {
            "base_sha": self.base_sha,
            "edited_files": len(self._edited),
            "diff_sha256": hashlib.sha256(self.diff().encode()).hexdigest(),
        }
