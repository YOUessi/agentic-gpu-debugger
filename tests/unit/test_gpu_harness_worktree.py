"""Multi-file Git worktree controller: isolated source changes, no model shell."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from gpu_agent.harness.worktree import PinnedGitWorkspace


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "original-repository"
    source.mkdir()
    _git(source, "init", "-q")
    files = {
        "src/kernel.cu": "__global__ void kernel() { float x = 1.0f; }\n",
        "src/other.cuh": "#pragma once\nfloat compute(float v) { return v; }\n",
        "tests/test_kernel.py": "def test_kernel(): assert True\n",
        "benchmarks/hardset_v2/h001/clean/kernel.cu": "__global__ void hidden() {}\n",
        "README.md": "# User project\n",
    }
    for name, body in files.items():
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    _git(source, "add", ".")
    _git(
        source,
        "-c",
        "user.name=GPU Agent Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "-q",
        "-m",
        "frozen fixture",
    )
    return source, _git(source, "rev-parse", "HEAD")


def test_pinned_worktree_changes_multiple_source_files_without_touching_origin(
    tmp_path: Path,
) -> None:
    source, sha = _repository(tmp_path)
    origin_head = _git(source, "rev-parse", "HEAD")
    project = PinnedGitWorkspace(
        origin=source,
        root=tmp_path / "isolated-run",
        sha=sha,
    )
    assert project.base_sha == sha
    assert _git(project.root, "rev-parse", "HEAD") == sha
    assert project.list_files(suffix=".cu") == ("src/kernel.cu",)
    assert "1.0f" in project.read("src/kernel.cu")
    first_hash = project.replace_exact("src/kernel.cu", "1.0f", "2.0f")
    second_hash = project.replace_exact("src/other.cuh", "return v;", "return v * 2;")
    assert first_hash == hashlib.sha256((project.root / "src/kernel.cu").read_bytes()).hexdigest()
    assert len(first_hash) == len(second_hash) == 64
    assert project.edited_files == ("src/kernel.cu", "src/other.cuh")
    assert "2.0f" in project.diff()
    assert "return v * 2;" in project.diff()
    assert project.integrity()["base_sha"] == sha
    assert project.integrity()["edited_files"] == 2

    # Editing in the sandbox must not modify the user's original Git tree.
    assert _git(source, "rev-parse", "HEAD") == origin_head
    assert "1.0f" in (source / "src/kernel.cu").read_text()
    assert "return v;" in (source / "src/other.cuh").read_text()
    assert _git(source, "status", "--porcelain") == ""


def test_worktree_rejects_symlink_escape_and_protected_files(tmp_path: Path) -> None:
    source, sha = _repository(tmp_path)
    project = PinnedGitWorkspace(
        origin=source,
        root=tmp_path / "worktree",
        sha=sha,
    )
    for name in (
        "../other.txt",
        "/etc/passwd",
        ".git/config",
        "tests/test_kernel.py",
        "benchmarks/hardset_v2/h001/clean/kernel.cu",
        "src/../../etc/passwd",
    ):
        with pytest.raises(ValueError):
            project.replace_exact(name, "x", "y")
    with pytest.raises(ValueError):
        project.read(".git/config")
    with pytest.raises(ValueError):
        project.replace_exact("README.md", "User", "Other")

    # Simulate an untrusted source tree replacing a tracked file with a symlink.
    (project.root / "src/kernel.cu").unlink()
    (project.root / "src/kernel.cu").symlink_to(tmp_path / "host-private.txt")
    with pytest.raises(ValueError, match="symlink"):
        project.read("src/kernel.cu")
    with pytest.raises(ValueError, match="symlink"):
        project.replace_exact("src/kernel.cu", "1.0f", "2.0f")


def test_worktree_enforces_budget_and_no_overwrite(tmp_path: Path) -> None:
    source, sha = _repository(tmp_path)
    project = PinnedGitWorkspace(
        origin=source,
        root=tmp_path / "isolated",
        sha=sha,
        max_edited_files=1,
    )
    with pytest.raises(ValueError, match="budget"):
        project.replace_exact("src/other.cuh", "return v;", "return v * 2;")
        # This alone is legal; the second, different file is over budget.
        project.replace_exact("src/kernel.cu", "1.0f", "2.0f")
    assert project.edited_files == ("src/other.cuh",)

    with pytest.raises(ValueError, match="must not exist"):
        PinnedGitWorkspace(origin=source, root=project.root, sha=sha)
    with pytest.raises(ValueError, match="overlap"):
        PinnedGitWorkspace(origin=source, root=source / "new-child", sha=sha)
    with pytest.raises(ValueError, match="SHA"):
        PinnedGitWorkspace(origin=source, root=tmp_path / "bad", sha="bad")

    with pytest.raises(ValueError, match="exactly once"):
        project.replace_exact("src/other.cuh", "return invalid;", "return v;")


def test_worktree_rejects_path_traversal_or_special_names() -> None:
    from gpu_agent.harness.worktree import _parts

    for path in (
        "../escape",
        "./kernel.cu",
        "src/../kernel.cu",
        "src//kernel.cu",
        "\\\\host\\file",
        "src/.env",
        "src/private/key.cu",
        "src/secrets/token.h",
    ):
        with pytest.raises(ValueError):
            _parts(path)


def test_integrity_detects_unrelated_tracked_and_staged_changes(tmp_path: Path) -> None:
    source, sha = _repository(tmp_path)
    project = PinnedGitWorkspace(origin=source, root=tmp_path / "run", sha=sha)
    project.replace_exact("src/kernel.cu", "1.0f", "2.0f")
    # A verifier mutation may not disappear because the diff only lists edited files.
    target = project.root / "tests/test_kernel.py"
    target.write_text("def test_kernel(): assert False\n")
    with pytest.raises(ValueError, match="unauthorized worktree change"):
        project.integrity()
    _git(project.root, "add", "tests/test_kernel.py")
    with pytest.raises(ValueError, match="unauthorized worktree change"):
        project.diff()
    assert "assert True" in (source / "tests/test_kernel.py").read_text()


def test_integrity_detects_ignored_or_untracked_new_files(tmp_path: Path) -> None:
    source, sha = _repository(tmp_path)
    project = PinnedGitWorkspace(origin=source, root=tmp_path / "run", sha=sha)
    unexpected = project.root / "src/untracked.cu"
    unexpected.write_text("void unexpected() {}\n")
    with pytest.raises(ValueError, match="untracked"):
        project.list_files()
    unexpected.unlink()
    assert "src/kernel.cu" in project.list_files()


def test_integrity_detects_external_rewrite_of_approved_source(tmp_path: Path) -> None:
    source, sha = _repository(tmp_path)
    project = PinnedGitWorkspace(origin=source, root=tmp_path / "run", sha=sha)
    project.replace_exact("src/kernel.cu", "1.0f", "2.0f")
    edited = project.root / "src/kernel.cu"
    edited.write_text(edited.read_text().replace("2.0f", "9.0f"))
    with pytest.raises(ValueError, match="outside controller"):
        project.read("src/kernel.cu")
    with pytest.raises(ValueError, match="outside controller"):
        project.replace_exact("src/other.cuh", "return v;", "return 3 * v;")


def test_atomic_batch_replacement_preserves_checkout_on_invalid_edits(tmp_path: Path) -> None:
    source, sha = _repository(tmp_path)
    workspace = PinnedGitWorkspace(
        origin=source, root=tmp_path / "atomic-batch", sha=sha, max_edited_files=1
    )
    initial = (workspace.root / "src/kernel.cu").read_bytes()
    with pytest.raises(ValueError, match="overlapping"):
        workspace.replace_batch(
            "src/kernel.cu",
            [("float x = 1.0f", "float x = 2.0f"), ("1.0f", "3.0f")],
        )
    assert (workspace.root / "src/kernel.cu").read_bytes() == initial
    assert workspace.edited_files == ()
    with pytest.raises(ValueError, match="exactly once"):
        workspace.replace_batch(
            "src/kernel.cu",
            [("1.0f", "2.0f"), ("missing();", "new();")],
        )
    assert (workspace.root / "src/kernel.cu").read_bytes() == initial
    with pytest.raises(ValueError, match="2-4"):
        workspace.replace_batch("src/kernel.cu", [("1.0f", "2.0f")])
    digest = workspace.replace_batch(
        "src/kernel.cu",
        [
            ("float x = 1.0f;", "float x = 2.0f;"),
            ("__global__ void kernel()", "__global__ void repaired()"),
        ],
    )
    assert digest == hashlib.sha256((workspace.root / "src/kernel.cu").read_bytes()).hexdigest()
    assert workspace.edited_files == ("src/kernel.cu",)
    assert "repaired" in workspace.diff()
    assert "1.0f" in (source / "src/kernel.cu").read_text()
    assert _git(source, "status", "--porcelain") == ""
    assert not tuple((workspace.root / "src").glob(".gpu-agent-stage-*"))


def test_atomic_batch_rejects_protected_paths_before_any_write(tmp_path: Path) -> None:
    source, sha = _repository(tmp_path)
    workspace = PinnedGitWorkspace(origin=source, root=tmp_path / "blocked-batch", sha=sha)
    for name in ("tests/test_kernel.py", "README.md", "../outside.cu"):
        with pytest.raises(ValueError):
            workspace.replace_batch(name, [("foo", "bar"), ("bar", "baz")])
    assert workspace.edited_files == ()
    assert workspace.diff() == ""


def test_oracle_and_evaluator_sources_are_not_readable_or_editable(
    tmp_path: Path,
) -> None:
    source, _ = _repository(tmp_path)
    extra = {
        "src/gpu_agent/benchmark/hardset_v2.py": "def expected_output(): return [1]\n",
        "src/gpu_agent/verification/oracle.py": "def reference(): return 1\n",
        "evaluation/hidden_score.json": '{"private": true}\n',
        "benchmarks/development_truth/ref/kernel.cu": "int reference() { return 1; }\n",
        "docs/expected-output.md": "protected test answer\n",
    }
    for name, raw in extra.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw)
    _git(source, "add", ".")
    _git(
        source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "-q",
        "-m",
        "audit forbidden paths",
    )
    sha = _git(source, "rev-parse", "HEAD")
    workspace = PinnedGitWorkspace(origin=source, root=tmp_path / "read-isolation", sha=sha)
    names = workspace.list_files()
    assert "src/kernel.cu" in names
    for name in (*extra, "tests/test_kernel.py", "benchmarks/hardset_v2/h001/clean/kernel.cu"):
        assert name not in names
        with pytest.raises(ValueError):
            workspace.read(name)
        with pytest.raises(ValueError):
            workspace.replace_exact(name, "return", "nope")
    assert workspace.diff() == ""
