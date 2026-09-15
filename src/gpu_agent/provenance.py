"""Controller-owned repository snapshots captured before release-relevant work."""

import hashlib
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Protocol

from gpu_agent.contracts import RepositorySnapshot
from gpu_agent.execution.process import ProcessCapture, ProcessExecutor
from gpu_agent.store import read_regular, reject_symlinks

GIT_EXECUTABLE = "/usr/bin/git"
GIT_TIMEOUT_SECONDS = 5.0
GIT_OUTPUT_LIMIT = 1024 * 1024
TRACKED_FILE_LIMIT = 64 * 1024 * 1024
TRACKED_TOTAL_LIMIT = 256 * 1024 * 1024
_OBJECT_ID = re.compile(rb"(?:[a-f0-9]{40}|[a-f0-9]{64})\n?\Z")
_SAFE_GIT_CONFIG = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "submodule.recurse=false",
    "-c",
    "status.submoduleSummary=false",
    "-c",
    "protocol.ext.allow=never",
)


class RepositoryProcess(Protocol):
    def execute(
        self, argv: list[str], cwd: Path, timeout_seconds: float, max_log_bytes: int
    ) -> ProcessCapture: ...


class _RealRepositoryProcess:
    def __init__(self) -> None:
        self._executor = ProcessExecutor()

    def execute(
        self, argv: list[str], cwd: Path, timeout_seconds: float, max_log_bytes: int
    ) -> ProcessCapture:
        return self._executor.execute(
            argv,
            cwd,
            timeout_seconds,
            max_log_bytes,
            env={
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
            },
        )


def _git(process: RepositoryProcess, repo: Path, *arguments: str) -> bytes:
    capture = process.execute(
        [GIT_EXECUTABLE, *_SAFE_GIT_CONFIG, *arguments],
        repo,
        GIT_TIMEOUT_SECONDS,
        GIT_OUTPUT_LIMIT,
    )
    if (
        capture.exit_code != 0
        or capture.timed_out
        or capture.cancelled
        or capture.truncated
        or capture.tool_error is not None
        or len(capture.stdout) > GIT_OUTPUT_LIMIT
        or len(capture.stderr) > GIT_OUTPUT_LIMIT
    ):
        raise ValueError("repository command failed or exceeded its bound")
    return capture.stdout


def _object_id(value: bytes, label: str) -> str:
    if _OBJECT_ID.fullmatch(value) is None:
        raise ValueError(f"repository {label} is not a full object ID")
    return value.rstrip(b"\n").decode("ascii")


def _tracked_paths(raw: bytes) -> list[str]:
    if not raw or not raw.endswith(b"\0"):
        raise ValueError("repository has no canonical tracked-file listing")
    paths: list[str] = []
    for encoded in raw[:-1].split(b"\0"):
        try:
            name = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("tracked paths must be UTF-8") from exc
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or path.as_posix() != name
            or ".." in path.parts
            or ".git" in path.parts
            or "\\" in name
            or name in paths
        ):
            raise ValueError("unsafe or duplicate tracked path")
        paths.append(name)
    if paths != sorted(paths):
        raise ValueError("tracked paths are not canonical")
    return paths


def _tracked_tree_hash(repo: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    total = 0
    for name in paths:
        path = repo.joinpath(*PurePosixPath(name).parts)
        data = read_regular(path, TRACKED_FILE_LIMIT)
        total += len(data)
        if total > TRACKED_TOTAL_LIMIT:
            raise ValueError("tracked repository exceeds capture bound")
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("tracked path is not a regular file")
        executable = b"x" if info.st_mode & 0o111 else b"-"
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(executable)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _capture_state(process: RepositoryProcess, repo: Path) -> tuple[str, str, str]:
    commit = _object_id(_git(process, repo, "rev-parse", "--verify", "HEAD^{commit}"), "commit")
    head_tree = _object_id(_git(process, repo, "rev-parse", "--verify", "HEAD^{tree}"), "tree")
    if _git(process, repo, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise ValueError("repository index and worktree must be clean")
    index_tree = _object_id(_git(process, repo, "write-tree"), "index tree")
    if index_tree != head_tree:
        raise ValueError("repository HEAD and index tree disagree")
    paths = _tracked_paths(_git(process, repo, "ls-files", "-z", "--cached"))
    return commit, head_tree, _tracked_tree_hash(repo, paths)


def capture_repository_snapshot(
    repo: Path,
    *,
    expected_commit: str | None = None,
    process: RepositoryProcess | None = None,
) -> RepositorySnapshot:
    """Capture a clean repository twice so concurrent mutations fail closed."""
    root = repo.absolute()
    reject_symlinks(root)
    if not root.is_dir():
        raise ValueError("repository is unavailable")
    reject_symlinks(root / ".git")
    runner = process or _RealRepositoryProcess()
    top = _git(runner, root, "rev-parse", "--show-toplevel")
    try:
        actual_root = Path(top.rstrip(b"\n").decode("utf-8")).absolute()
    except UnicodeDecodeError as exc:
        raise ValueError("repository root is invalid") from exc
    if actual_root != root:
        raise ValueError("requested path is not the repository root")
    before = _capture_state(runner, root)
    if expected_commit is not None and before[0] != expected_commit:
        raise ValueError("expected commit differs from actual HEAD")
    after = _capture_state(runner, root)
    if before != after:
        raise ValueError("repository changed during capture")
    return RepositorySnapshot(commit=before[0], tracked_tree_hash=before[2], clean=True)
