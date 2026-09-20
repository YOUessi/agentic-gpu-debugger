import errno
import hashlib
import os
import stat
from pathlib import Path

import pytest

from gpu_agent.benchmark.controller_artifacts import (
    read_private_external,
    validate_external_artifact_path,
    write_private_atomic_new,
)


def _private_file(path: Path, content: bytes = b"{}") -> Path:
    path.write_bytes(content)
    path.chmod(0o600)
    return path


def _secure_parent(tmp_path: Path) -> Path:
    parent = tmp_path / "controller-output"
    parent.mkdir(mode=0o700)
    return parent


def _changed_stat(info: os.stat_result, index: int, value: int) -> os.stat_result:
    fields = list(info)
    fields[index] = value
    return os.stat_result(fields)


def test_external_artifact_rejects_relative_path(tmp_path):
    with pytest.raises(ValueError, match="external artifact path is unsafe"):
        validate_external_artifact_path(
            Path("labels.json"),
            repository=tmp_path / "repo",
        )


def test_external_artifact_rejects_exact_forbidden_root(tmp_path):
    forbidden = tmp_path / "public"
    forbidden.mkdir()

    with pytest.raises(ValueError, match="external artifact path is unsafe"):
        validate_external_artifact_path(
            forbidden,
            repository=tmp_path / "repo",
            forbidden_roots=(forbidden,),
        )


@pytest.mark.parametrize("store_name", ["public", "evaluator"])
def test_external_artifact_rejects_forbidden_store_descendant(tmp_path, store_name):
    forbidden = tmp_path / store_name
    forbidden.mkdir()

    with pytest.raises(ValueError, match="external artifact path is unsafe"):
        validate_external_artifact_path(
            forbidden / "labels.json",
            repository=tmp_path / "repo",
            forbidden_roots=(forbidden,),
        )


def test_external_artifact_rejects_resolved_forbidden_descendant(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    link = tmp_path / "controller-link"
    link.symlink_to(repository, target_is_directory=True)
    with pytest.raises(ValueError, match="external artifact path is unsafe"):
        validate_external_artifact_path(
            link / "labels.json",
            repository=repository,
            forbidden_roots=(),
        )


def test_external_artifact_rejects_symlink_to_safe_external_file(tmp_path):
    target = _private_file(tmp_path / "labels.json")
    link = tmp_path / "labels-link.json"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="external artifact path is unsafe"):
        validate_external_artifact_path(
            link,
            repository=tmp_path / "repo",
        )


def test_external_artifact_rejects_symlinked_parent(tmp_path):
    parent = tmp_path / "controller"
    parent.mkdir()
    link = tmp_path / "controller-link"
    link.symlink_to(parent, target_is_directory=True)

    with pytest.raises(ValueError, match="external artifact path is unsafe"):
        validate_external_artifact_path(
            link / "selection.json",
            repository=tmp_path / "repo",
        )


@pytest.mark.parametrize("mode", [0o644, 0o660, 0o606])
def test_private_reader_rejects_group_or_other_permissions(tmp_path, mode):
    path = tmp_path / "labels.json"
    path.write_bytes(b"{}")
    path.chmod(mode)
    with pytest.raises(ValueError, match="private external artifact is unsafe"):
        read_private_external(
            path,
            repository=tmp_path / "repo",
            forbidden_roots=(),
            limit=1024,
        )


def test_private_reader_rejects_non_regular_file(tmp_path):
    path = tmp_path / "labels.pipe"
    os.mkfifo(path, 0o600)

    with pytest.raises(ValueError, match="private external artifact is unsafe"):
        read_private_external(
            path,
            repository=tmp_path / "repo",
            forbidden_roots=(),
            limit=1024,
        )


def test_private_reader_rejects_wrong_owner_from_fstat(tmp_path, monkeypatch):
    from gpu_agent.benchmark import controller_artifacts

    path = _private_file(tmp_path / "labels.json")
    native_fstat = controller_artifacts.os.fstat

    def wrong_owner(fd):
        info = native_fstat(fd)
        if info.st_ino == path.stat().st_ino:
            return _changed_stat(info, 4, info.st_uid + 1)
        return info

    monkeypatch.setattr(controller_artifacts.os, "fstat", wrong_owner)

    with pytest.raises(ValueError, match="private external artifact is unsafe"):
        read_private_external(
            path,
            repository=tmp_path / "repo",
            forbidden_roots=(),
            limit=1024,
        )


def test_private_reader_rejects_multiple_hard_links(tmp_path):
    path = _private_file(tmp_path / "labels.json")
    os.link(path, tmp_path / "labels-copy.json")

    with pytest.raises(ValueError, match="private external artifact is unsafe"):
        read_private_external(
            path,
            repository=tmp_path / "repo",
            forbidden_roots=(),
            limit=1024,
        )


def test_private_reader_rejects_path_fd_identity_mismatch(tmp_path, monkeypatch):
    from gpu_agent.benchmark import controller_artifacts

    path = _private_file(tmp_path / "labels.json")
    native_fstat = controller_artifacts.os.fstat

    def changed_inode(fd):
        info = native_fstat(fd)
        if info.st_ino == path.stat().st_ino:
            return _changed_stat(info, 1, info.st_ino + 1)
        return info

    monkeypatch.setattr(controller_artifacts.os, "fstat", changed_inode)

    with pytest.raises(ValueError, match="private external artifact is unsafe"):
        read_private_external(
            path,
            repository=tmp_path / "repo",
            forbidden_roots=(),
            limit=1024,
        )


@pytest.mark.parametrize("content,limit", [(b"", 1024), (b"12345", 4)])
def test_private_reader_rejects_empty_or_oversize_content(tmp_path, content, limit):
    path = _private_file(tmp_path / "labels.json", content)

    with pytest.raises(ValueError, match="private external artifact is unsafe"):
        read_private_external(
            path,
            repository=tmp_path / "repo",
            forbidden_roots=(),
            limit=limit,
        )


def test_private_reader_reads_valid_owner_only_file(tmp_path):
    path = _private_file(tmp_path / "labels.json", b'{"private":true}')

    assert read_private_external(
        path,
        repository=tmp_path / "repo",
        forbidden_roots=(),
        limit=1024,
    ) == b'{"private":true}'


def test_atomic_writer_publishes_new_owner_only_file(tmp_path):
    output = _secure_parent(tmp_path) / "selection.json"
    content = b'{"selection":true}'

    digest = write_private_atomic_new(
        output,
        content,
        repository=tmp_path / "repo",
    )

    assert digest == hashlib.sha256(content).hexdigest()
    assert output.read_bytes() == content
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert list(output.parent.iterdir()) == [output]


def test_atomic_writer_accepts_byte_identical_retry_without_replacing_target(tmp_path):
    output = _secure_parent(tmp_path) / "selection.json"
    content = b'{"selection":true}'
    expected = write_private_atomic_new(output, content, repository=tmp_path / "repo")
    inode = output.stat().st_ino

    observed = write_private_atomic_new(output, content, repository=tmp_path / "repo")

    assert observed == expected
    assert output.stat().st_ino == inode
    assert output.read_bytes() == content


def test_atomic_writer_never_overwrites_different_content(tmp_path):
    output = _secure_parent(tmp_path) / "selection.json"
    output.write_bytes(b"first")
    output.chmod(0o600)
    with pytest.raises(ValueError, match="external artifact output conflicts"):
        write_private_atomic_new(
            output,
            b"second",
            repository=tmp_path / "repo",
        )
    assert output.read_bytes() == b"first"


def test_atomic_writer_reports_conflict_when_existing_content_is_longer(tmp_path):
    output = _secure_parent(tmp_path) / "selection.json"
    output.write_bytes(b"existing-content")
    output.chmod(0o600)

    with pytest.raises(ValueError, match="external artifact output conflicts"):
        write_private_atomic_new(
            output,
            b"new",
            repository=tmp_path / "repo",
        )

    assert output.read_bytes() == b"existing-content"


@pytest.mark.parametrize("parent_state", ["missing", "public-mode", "symlink"])
def test_atomic_writer_rejects_missing_or_unsafe_parent(tmp_path, parent_state):
    if parent_state == "missing":
        parent = tmp_path / "missing"
    elif parent_state == "public-mode":
        parent = tmp_path / "public-parent"
        parent.mkdir(mode=0o755)
    else:
        target = _secure_parent(tmp_path)
        parent = tmp_path / "linked-parent"
        parent.symlink_to(target, target_is_directory=True)
    output = parent / "selection.json"

    with pytest.raises(ValueError, match="external artifact output is unsafe"):
        write_private_atomic_new(
            output,
            b"selection",
            repository=tmp_path / "repo",
        )


def test_atomic_writer_fails_closed_when_renameat2_is_unavailable(tmp_path, monkeypatch):
    from gpu_agent.benchmark import controller_artifacts

    parent = _secure_parent(tmp_path)

    def unavailable(parent_fd, temporary, target):
        raise OSError(errno.ENOSYS, "not implemented")

    monkeypatch.setattr(controller_artifacts, "_rename_noreplace", unavailable)

    with pytest.raises(ValueError, match="external artifact output is unsafe"):
        write_private_atomic_new(
            parent / "selection.json",
            b"selection",
            repository=tmp_path / "repo",
        )
    assert list(parent.iterdir()) == []


def test_atomic_writer_removes_its_temporary_when_fchmod_fails(tmp_path, monkeypatch):
    from gpu_agent.benchmark import controller_artifacts

    parent = _secure_parent(tmp_path)

    def unavailable_mode(fd, mode):
        raise OSError(errno.EIO, "simulated fchmod failure")

    monkeypatch.setattr(controller_artifacts.os, "fchmod", unavailable_mode)

    with pytest.raises(ValueError, match="external artifact output is unsafe"):
        write_private_atomic_new(
            parent / "selection.json",
            b"selection",
            repository=tmp_path / "repo",
        )
    assert list(parent.iterdir()) == []


def test_atomic_writer_handles_unexpected_simulated_eexist_as_unsafe(tmp_path, monkeypatch):
    from gpu_agent.benchmark import controller_artifacts

    parent = _secure_parent(tmp_path)

    def unexpected_exists(parent_fd, temporary, target):
        raise FileExistsError(target)

    monkeypatch.setattr(controller_artifacts, "_rename_noreplace", unexpected_exists)

    with pytest.raises(ValueError, match="external artifact output is unsafe"):
        write_private_atomic_new(
            parent / "selection.json",
            b"selection",
            repository=tmp_path / "repo",
        )
    assert list(parent.iterdir()) == []


def test_atomic_writer_exposes_only_dot_temporary_before_rename(tmp_path, monkeypatch):
    from gpu_agent.benchmark import controller_artifacts

    parent = _secure_parent(tmp_path)
    observed: list[str] = []

    def crash_before_rename(parent_fd, temporary, target):
        observed.extend(path.name for path in parent.iterdir())
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(controller_artifacts, "_rename_noreplace", crash_before_rename)

    with pytest.raises(RuntimeError, match="simulated crash"):
        write_private_atomic_new(
            parent / "selection.json",
            b"selection",
            repository=tmp_path / "repo",
        )

    assert len(observed) == 1 and observed[0].startswith(".")
    assert list(parent.iterdir()) == []


def test_atomic_writer_rejects_target_that_becomes_unsafe_before_retry(
    tmp_path, monkeypatch
):
    from gpu_agent.benchmark import controller_artifacts

    parent = _secure_parent(tmp_path)
    safe = _private_file(tmp_path / "other.json", b"selection")
    output = parent / "selection.json"

    def adversarial_exists(parent_fd, temporary, target):
        output.symlink_to(safe)
        raise FileExistsError(target)

    monkeypatch.setattr(controller_artifacts, "_rename_noreplace", adversarial_exists)

    with pytest.raises(ValueError, match="external artifact output is unsafe"):
        write_private_atomic_new(
            output,
            b"selection",
            repository=tmp_path / "repo",
        )
    assert output.is_symlink()
    assert safe.read_bytes() == b"selection"
