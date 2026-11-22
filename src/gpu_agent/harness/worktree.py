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

from gpu_agent.store import read_regular, reject_symlinks

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

    @property
    def edited_files(self) -> tuple[str, ...]:
        return tuple(sorted(self._edited))

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
        candidate = self.root
        for part in parts:
            candidate /= part
            if candidate.is_symlink():
                raise ValueError("symlink traversal forbidden")
        if not candidate.is_file() or not candidate.resolve().is_relative_to(self.root):
            raise ValueError("file is not a regular worktree source")
        return candidate

    def list_files(self, *, suffix: str = "", max_files: int = 100) -> tuple[str, ...]:
        if not 1 <= max_files <= 500 or len(suffix) > 20:
            raise ValueError("file listing out of bounds")
        return tuple(
            name
            for name in sorted(self._tracked)
            if (not suffix or name.endswith(suffix))
            and PurePosixPath(name).suffix in PUBLIC_READ_EXTENSIONS
            and not any(p.lower() in _FORBIDDEN_NAME_PARTS for p in PurePosixPath(name).parts)
        )[:max_files]

    def read(self, name: str, *, start: int = 1, lines: int = 120) -> str:
        if not 1 <= start <= 100_000 or not 1 <= lines <= 200:
            raise ValueError("bounded source read parameters required")
        source = read_regular(self._checked(name, editable=False), MAX_SOURCE_BYTES)
        decoded = source.decode("utf-8").splitlines()
        excerpt = decoded[start - 1 : start - 1 + lines]
        return "\n".join(f"{start + index:4d} {line}" for index, line in enumerate(excerpt))[
            :20_000
        ]

    def replace_exact(self, name: str, before: str, after: str) -> str:
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
        if modified == original or len(modified) > MAX_SOURCE_BYTES:
            raise ValueError("no-op or overlarge source edit")
        # Host-only atomic write in an isolated worktree; reject symlinks.
        mode = stat.S_IMODE(target.stat().st_mode)
        fd, temp = tempfile.mkstemp(prefix=".gpu-agent-stage-", dir=target.parent)
        try:
            os.fchmod(fd, mode)
            with os.fdopen(fd, "wb") as handle:
                handle.write(modified)
                handle.flush()
                os.fsync(handle.fileno())
            if target.is_symlink():
                raise ValueError("edited path became a symlink")
            os.replace(temp, target)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        self._edited.add(name)
        return hashlib.sha256(modified).hexdigest()

    def diff(self) -> str:
        paths = self.edited_files
        if not paths:
            return ""
        _run_git(self.root, "diff", "--check", "--", *paths)
        value = _run_git(self.root, "diff", "--no-ext-diff", "--", *paths)
        if len(value.encode()) > MAX_OUTPUT_BYTES:
            raise ValueError("worktree diff exceeds audit limit")
        return value

    def integrity(self) -> dict[str, str | int]:
        if _run_git(self.root, "rev-parse", "HEAD") != self.base_sha:
            raise ValueError("worktree HEAD drift detected")
        return {
            "base_sha": self.base_sha,
            "edited_files": len(self._edited),
            "diff_sha256": hashlib.sha256(self.diff().encode()).hexdigest(),
        }
