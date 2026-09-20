"""Secure I/O for controller-private artifacts outside repository storage."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import secrets
import stat
from collections.abc import Sequence
from pathlib import Path

from gpu_agent.store import reject_symlinks

_PATH_ERROR = "external artifact path is unsafe"
_PRIVATE_ERROR = "private external artifact is unsafe"
_OUTPUT_ERROR = "external artifact output is unsafe"
_CONFLICT_ERROR = "external artifact output conflicts"
_RENAME_NOREPLACE = 1

_LIBC = ctypes.CDLL(None, use_errno=True)
try:
    _LIBC.renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    _LIBC.renameat2.restype = ctypes.c_int
except AttributeError:
    pass


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_external_artifact_path(
    path: Path,
    *,
    repository: Path,
    forbidden_roots: Sequence[Path] = (),
) -> Path:
    """Resolve an absolute artifact path outside every protected storage root."""
    try:
        if not path.is_absolute():
            raise ValueError(_PATH_ERROR)
        reject_symlinks(path)
        resolved = path.resolve(strict=False)
        roots = (repository, *forbidden_roots)
        if any(_within(resolved, root.resolve(strict=False)) for root in roots):
            raise ValueError(_PATH_ERROR)
        return resolved
    except (OSError, ValueError) as exc:
        raise ValueError(_PATH_ERROR) from exc


def _same_file_metadata(path_info: os.stat_result, fd_info: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
    )
    return all(getattr(path_info, field) == getattr(fd_info, field) for field in fields)


def _private_regular(info: os.stat_result) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.geteuid()
        and info.st_mode & 0o077 == 0
        and info.st_nlink == 1
    )


def _open_directory_nofollow(path: Path) -> int:
    if not path.is_absolute():
        raise ValueError(_PATH_ERROR)
    directory_fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            child_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = child_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def read_private_external(
    path: Path,
    *,
    repository: Path,
    forbidden_roots: Sequence[Path],
    limit: int,
) -> bytes:
    """Read a bounded, owner-only external file through its securely opened parent."""
    try:
        resolved = validate_external_artifact_path(
            path,
            repository=repository,
            forbidden_roots=forbidden_roots,
        )
        parent_fd = _open_directory_nofollow(resolved.parent)
        try:
            path_info = os.stat(resolved.name, dir_fd=parent_fd, follow_symlinks=False)
            fd = os.open(
                resolved.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent_fd,
            )
            try:
                fd_info = os.fstat(fd)
                if (
                    limit < 1
                    or not _same_file_metadata(path_info, fd_info)
                    or not _private_regular(fd_info)
                    or fd_info.st_size < 1
                    or fd_info.st_size > limit
                ):
                    raise ValueError(_PRIVATE_ERROR)
                chunks: list[bytes] = []
                remaining = limit + 1
                while remaining:
                    chunk = os.read(fd, remaining)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                content = b"".join(chunks)
                if not content or len(content) > limit:
                    raise ValueError(_PRIVATE_ERROR)
                return content
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)
    except (OSError, ValueError) as exc:
        raise ValueError(_PRIVATE_ERROR) from exc


def _rename_noreplace(parent_fd: int, temporary: str, target: str) -> None:
    try:
        renameat2 = _LIBC.renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS)) from exc
    result = renameat2(
        parent_fd,
        os.fsencode(temporary),
        parent_fd,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(target)
        raise OSError(error, os.strerror(error))


def _open_private_parent(path: Path) -> int:
    parent_fd = _open_directory_nofollow(path)
    try:
        fd_info = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(fd_info.st_mode)
            or fd_info.st_uid != os.geteuid()
            or fd_info.st_mode & 0o077
        ):
            raise ValueError(_OUTPUT_ERROR)
        return parent_fd
    except BaseException:
        os.close(parent_fd)
        raise


def _create_temporary(parent_fd: int) -> tuple[str, int]:
    for _ in range(16):
        name = f".gpu-agent-{secrets.token_hex(16)}.tmp"
        try:
            fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            continue
        try:
            os.fchmod(fd, 0o600)
        except BaseException:
            os.close(fd)
            try:
                os.unlink(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            raise
        return name, fd
    raise OSError(errno.EEXIST, os.strerror(errno.EEXIST))


def _write_all(fd: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(fd, view)
        if written < 1:
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        view = view[written:]


def write_private_atomic_new(
    path: Path,
    content: bytes,
    *,
    repository: Path,
    forbidden_roots: Sequence[Path] = (),
) -> str:
    """Publish private bytes once, allowing only a byte-identical retry."""
    digest = hashlib.sha256(content).hexdigest()
    try:
        resolved = validate_external_artifact_path(
            path,
            repository=repository,
            forbidden_roots=forbidden_roots,
        )
        parent_fd = _open_private_parent(resolved.parent)
    except (OSError, ValueError) as exc:
        raise ValueError(_OUTPUT_ERROR) from exc

    temporary: str | None = None
    try:
        temporary, temporary_fd = _create_temporary(parent_fd)
        try:
            _write_all(temporary_fd, content)
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)

        try:
            _rename_noreplace(parent_fd, temporary, resolved.name)
        except FileExistsError:
            try:
                target_size = os.stat(
                    resolved.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                ).st_size
                existing = read_private_external(
                    resolved,
                    repository=repository,
                    forbidden_roots=forbidden_roots,
                    limit=max(1, len(content), target_size),
                )
            except (OSError, ValueError) as exc:
                raise ValueError(_OUTPUT_ERROR) from exc
            if existing != content:
                raise ValueError(_CONFLICT_ERROR) from None
            os.fsync(parent_fd)
            return digest
        except OSError as exc:
            raise ValueError(_OUTPUT_ERROR) from exc

        os.fsync(parent_fd)
        temporary = None
        return digest
    except OSError as exc:
        raise ValueError(_OUTPUT_ERROR) from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
