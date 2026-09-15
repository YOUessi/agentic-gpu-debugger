import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from gpu_agent.execution.process import ProcessCapture

COMMIT = "1" * 40
TREE = "2" * 40


class FakeGit:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.calls: list[tuple[list[str], Path, float, int]] = []
        self.outputs: dict[tuple[str, ...], list[bytes]] = {
            ("rev-parse", "--show-toplevel"): [str(repo).encode() + b"\n"],
            ("rev-parse", "--verify", "HEAD^{commit}"): [COMMIT.encode() + b"\n"],
            ("rev-parse", "--verify", "HEAD^{tree}"): [TREE.encode() + b"\n"],
            ("status", "--porcelain=v1", "-z", "--untracked-files=all"): [b""],
            ("write-tree",): [TREE.encode() + b"\n"],
            ("ls-files", "-z", "--cached"): [b"kernel.cu\0"],
        }
        self.before: dict[tuple[str, ...], Callable[[], None]] = {}

    def execute(
        self, argv: list[str], cwd: Path, timeout_seconds: float, max_log_bytes: int
    ) -> ProcessCapture:
        self.calls.append((argv, cwd, timeout_seconds, max_log_bytes))
        command = tuple(argv[9:] if argv[1:2] == ["-c"] else argv[1:])
        callback = self.before.get(command)
        if callback is not None:
            callback()
            del self.before[command]
        values = self.outputs[command]
        value = values.pop(0) if len(values) > 1 else values[0]
        return ProcessCapture(0, value, b"", False)


def _repo(tmp_path: Path) -> tuple[Path, FakeGit]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "kernel.cu").write_bytes(b"int main() {}\n")
    return repo, FakeGit(repo)


def test_repository_snapshot_uses_fixed_bounded_git_commands(tmp_path):
    from gpu_agent.provenance import capture_repository_snapshot

    repo, git = _repo(tmp_path)
    snapshot = capture_repository_snapshot(repo, process=git)
    assert snapshot.commit == COMMIT
    assert snapshot.clean is True
    assert len(snapshot.tracked_tree_hash) == 64
    assert len(git.calls) > 6  # State is sampled again after the tracked bytes are read.
    assert all(argv[0] == "/usr/bin/git" for argv, _, _, _ in git.calls)
    assert all(
        argv[1:9]
        == [
            "-c",
            "core.fsmonitor=false",
            "-c",
            "submodule.recurse=false",
            "-c",
            "status.submoduleSummary=false",
            "-c",
            "protocol.ext.allow=never",
        ]
        for argv, _, _, _ in git.calls
    )
    assert all(cwd == repo for _, cwd, _, _ in git.calls)
    assert all(timeout <= 5 and limit <= 1024 * 1024 for _, _, timeout, limit in git.calls)


def test_repository_snapshot_rejects_dirty_index_or_worktree(tmp_path):
    from gpu_agent.provenance import capture_repository_snapshot

    repo, git = _repo(tmp_path)
    git.outputs[("status", "--porcelain=v1", "-z", "--untracked-files=all")] = [b" M kernel.cu\0"]
    with pytest.raises(ValueError, match="clean"):
        capture_repository_snapshot(repo, process=git)


def test_repository_snapshot_rejects_abbreviated_commit(tmp_path):
    from gpu_agent.provenance import capture_repository_snapshot

    repo, git = _repo(tmp_path)
    git.outputs[("rev-parse", "--verify", "HEAD^{commit}")] = [b"1234567\n"]
    with pytest.raises(ValueError, match="commit"):
        capture_repository_snapshot(repo, process=git)


def test_repository_snapshot_rejects_symlinked_repository(tmp_path):
    from gpu_agent.provenance import capture_repository_snapshot

    target, git = _repo(tmp_path)
    link = tmp_path / "repo-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        capture_repository_snapshot(link, process=git)
    assert git.calls == []


def test_repository_snapshot_rejects_mutable_tracked_file(tmp_path):
    from gpu_agent.provenance import capture_repository_snapshot

    repo, git = _repo(tmp_path)
    command = ("ls-files", "-z", "--cached")
    git.outputs[command] = [b"kernel.cu\0", b"kernel.cu\0"]
    git.before[command] = lambda: None
    calls = 0

    def mutate_on_second_listing() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            (repo / "kernel.cu").write_bytes(b"changed during capture\n")

    git.before[command] = mutate_on_second_listing
    # Keep the callback installed for both samples.
    original_execute = git.execute

    def execute(*args, **kwargs):
        result = original_execute(*args, **kwargs)
        actual = args[0][9:] if args[0][1:2] == ["-c"] else args[0][1:]
        if tuple(actual) == command:
            git.before[command] = mutate_on_second_listing
        return result

    git.execute = execute  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="changed"):
        capture_repository_snapshot(repo, process=git)


def test_repository_snapshot_rejects_caller_and_actual_head_disagreement(tmp_path):
    from gpu_agent.provenance import capture_repository_snapshot

    repo, git = _repo(tmp_path)
    with pytest.raises(ValueError, match="expected commit"):
        capture_repository_snapshot(repo, expected_commit="3" * 40, process=git)

    repo, git = _repo(tmp_path / "other")
    git.outputs[("rev-parse", "--verify", "HEAD^{commit}")] = [
        COMMIT.encode() + b"\n",
        ("4" * 40).encode() + b"\n",
    ]
    with pytest.raises(ValueError, match="changed"):
        capture_repository_snapshot(repo, process=git)


def test_repository_snapshot_rejects_head_and_index_tree_disagreement(tmp_path):
    from gpu_agent.provenance import capture_repository_snapshot

    repo, git = _repo(tmp_path)
    git.outputs[("write-tree",)] = [("5" * 40).encode() + b"\n"]
    with pytest.raises(ValueError, match="tree"):
        capture_repository_snapshot(repo, process=git)


def test_toolchain_lock_is_bounded_hash_checked_and_runtime_bound(tmp_path):
    from gpu_agent.environment import (
        RuntimeToolchainAttestation,
        load_toolchain_lock,
        validate_runtime_toolchain,
    )

    runner = tmp_path / "runner.py"
    dockerfile = tmp_path / "Dockerfile"
    runner.write_bytes(b"runner")
    dockerfile.write_bytes(b"dockerfile")
    raw = (
        "{"
        '"schema_version":1,'
        '"image_id":"sha256:'
        + "a" * 64
        + '","base_repo_digest":"nvidia/cuda@sha256:'
        + "b" * 64
        + '","cuda_nvcc":"12.8.93","compute_sanitizer":"2025.1.0.0",'
        '"target_arch":"sm_89","runner_sha256":"'
        + hashlib.sha256(b"runner").hexdigest()
        + '","dockerfile_sha256":"'
        + hashlib.sha256(b"dockerfile").hexdigest()
        + '"}'
    ).encode()
    lock_path = tmp_path / "toolchain.lock.json"
    lock_path.write_bytes(raw)
    lock = load_toolchain_lock(lock_path)
    assert lock.lock_hash == hashlib.sha256(raw).hexdigest()
    observed = RuntimeToolchainAttestation(
        runtime_session_id="d" * 32,
        lock_hash=lock.lock_hash,
        image_id="sha256:" + "a" * 64,
        cuda_nvcc="12.8.93",
        compute_sanitizer="2025.1.0.0",
        compute_capability="8.9",
        target_arch="sm_89",
        policy_hash="c" * 64,
    )
    validate_runtime_toolchain(
        lock,
        observed,
        expected_policy_hash="c" * 64,
        expected_runtime_session_id="d" * 32,
    )
    with pytest.raises(ValueError, match="runtime"):
        validate_runtime_toolchain(
            lock,
            observed.model_copy(update={"cuda_nvcc": "claimed-only"}),
            expected_policy_hash="c" * 64,
            expected_runtime_session_id="d" * 32,
        )


def test_toolchain_lock_rejects_symlink_or_stale_build_input(tmp_path):
    from gpu_agent.environment import load_toolchain_lock

    target = tmp_path / "actual.lock"
    target.write_bytes(b"{}")
    link = tmp_path / "toolchain.lock.json"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        load_toolchain_lock(link)

    link.unlink()
    (tmp_path / "runner.py").write_bytes(b"runner")
    (tmp_path / "Dockerfile").write_bytes(b"dockerfile")
    link.write_text(
        '{"schema_version":1,"image_id":"sha256:'
        + "a" * 64
        + '","base_repo_digest":"nvidia/cuda@sha256:'
        + "b" * 64
        + '","cuda_nvcc":"12.8.93","compute_sanitizer":"2025.1.0.0",'
        '"target_arch":"sm_89","runner_sha256":"'
        + "0" * 64
        + '","dockerfile_sha256":"'
        + hashlib.sha256(b"dockerfile").hexdigest()
        + '"}'
    )
    with pytest.raises(ValueError, match="input"):
        load_toolchain_lock(link)
