"""Fd-pinned operational locks; no corpus authority creation or policy changes."""

import fcntl
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from gpu_agent.contracts import CurrentPhase, RunBinding, RunManifest, RunStatus, StateEvent
from gpu_agent.store import RunStore, read_regular, reject_symlinks


class BatchInputError(ValueError):
    """Operator configuration or integrity error; never a CUDA finding."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _private_directory(info: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.geteuid()
        and stat.S_IMODE(info.st_mode) == 0o700
    )


def _private_lock(info: os.stat_result) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.geteuid()
        and stat.S_IMODE(info.st_mode) == 0o600
        and info.st_nlink == 1
    )


class DirectoryPin:
    """Pin an existing private directory and reject pathname/object replacement."""

    def __init__(self, path: Path) -> None:
        self.path = path.absolute()
        self.fd = -1
        try:
            reject_symlinks(self.path)
            self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self.info = os.fstat(self.fd)
            self.validate()
        except (OSError, ValueError) as exc:
            self.close()
            raise BatchInputError(
                "UNSAFE_DIRECTORY", "Expected an existing owner-only directory."
            ) from exc

    def validate(self) -> None:
        if self.fd < 0:
            raise BatchInputError("LEASE_INACTIVE", "Directory lease is closed.")
        try:
            reject_symlinks(self.path)
            pinned = os.fstat(self.fd)
            named = self.path.stat(follow_symlinks=False)
            identity = (self.info.st_dev, self.info.st_ino)
            if (
                not _private_directory(pinned)
                or not _private_directory(named)
                or (pinned.st_dev, pinned.st_ino) != identity
                or (named.st_dev, named.st_ino) != identity
            ):
                raise ValueError("directory identity or permissions changed")
        except (OSError, ValueError) as exc:
            raise BatchInputError(
                "DIRECTORY_CHANGED", "Directory identity or permissions changed."
            ) from exc

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class BatchLease:
    def __init__(self, directory: DirectoryPin, lock_fd: int) -> None:
        self.directory = directory
        self.lock_fd = lock_fd
        info = os.fstat(lock_fd)
        self.identity = (info.st_dev, info.st_ino)

    def validate(self) -> None:
        self.directory.validate()
        try:
            pinned = os.fstat(self.lock_fd)
            named = os.stat(".seed-batch.lock", dir_fd=self.directory.fd, follow_symlinks=False)
            if (
                not _private_lock(pinned)
                or not _private_lock(named)
                or (pinned.st_dev, pinned.st_ino) != self.identity
                or (named.st_dev, named.st_ino) != self.identity
            ):
                raise ValueError("lock replaced")
        except (OSError, ValueError) as exc:
            raise BatchInputError(
                "LOCK_CHANGED", "Batch lock identity or permissions changed."
            ) from exc


@contextmanager
def data_lock(data: Path) -> Iterator[BatchLease]:
    """Never create/chmod the data root or silently repair unsafe existing objects."""
    directory = DirectoryPin(data)
    lock_fd = -1
    try:
        lock_fd = os.open(
            ".seed-batch.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
            dir_fd=directory.fd,
        )
        if not _private_lock(os.fstat(lock_fd)):
            raise BatchInputError(
                "UNSAFE_LOCK", "Batch lock must be a single-link owner-only file."
            )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BatchInputError("BATCH_BUSY", "Another batch holds this data root.") from exc
        os.fsync(directory.fd)
        lease = BatchLease(directory, lock_fd)
        lease.validate()
        try:
            yield lease
        finally:
            lease.validate()
    except OSError as exc:
        raise BatchInputError("LOCK_UNAVAILABLE", "Cannot acquire a safe batch lock.") from exc
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        directory.close()


def _require_parent(parent: RunManifest, binding: RunBinding) -> None:
    if (
        parent.kind != "seed_batch"
        or parent.status != RunStatus.RUNNING
        or parent.binding != binding
        or parent.binding is None
        or binding.purpose != "corpus_validation"
        or parent.external_origin is not None
    ):
        raise ValueError("parent must be a RUNNING seed_batch with the same binding")


def validate_seed_parent(store: RunStore, parent_id: str, binding: RunBinding) -> None:
    if store.visibility != "public":
        raise ValueError("seed batches cannot contain evaluator executions")
    with store.evaluation_run_lease(parent_id) as lease:
        _require_parent(lease.load(), binding)


def create_seed_child(store: RunStore, parent_id: str, binding: RunBinding) -> RunManifest:
    """Use the existing fd lease to keep parent validation and creation in one lock."""
    if store.visibility != "public":
        raise ValueError("seed batches cannot contain evaluator executions")
    with store.evaluation_run_lease(parent_id) as lease:
        _require_parent(lease.load(), binding)
        child = store.create_run("case_execution", parent_run_id=parent_id, binding=binding)
        lease.validate()
        return child


def finalize_seed_child(
    store: RunStore,
    parent_id: str,
    child_id: str,
    binding: RunBinding,
    status: RunStatus,
) -> RunManifest:
    """Terminalize a child while the exact RUNNING parent remains locked.

    The parent lease is acquired first and held through the child manifest save.  This
    prevents another controller transition from ending the batch between the final
    parent check and the child's terminal transition.
    """
    if status not in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
        raise ValueError("seed child finalization requires a terminal status")
    if store.visibility != "public":
        raise ValueError("seed batches cannot contain evaluator executions")
    with store.evaluation_run_lease(parent_id) as parent_lease:
        _require_parent(parent_lease.load(), binding)
        with store.evaluation_run_lease(child_id) as child_lease:
            child = child_lease.load()
            if (
                child.kind != "case_execution"
                or child.parent_run_id != parent_id
                or child.binding != binding
                or child.external_origin is not None
                or child.status != RunStatus.RUNNING
            ):
                raise ValueError("child cannot be finalized under this seed batch")
            if status == RunStatus.COMPLETED:
                child.last_completed_phase = child.current_phase
                child.current_phase = CurrentPhase.FINALIZING
                child.events.append(
                    StateEvent(status=RunStatus.RUNNING, phase=CurrentPhase.FINALIZING)
                )
                child_lease.save(child)
                parent_lease.validate()
            child.last_completed_phase = (
                child.current_phase if status == RunStatus.COMPLETED else child.last_completed_phase
            )
            child.status = status
            child.current_phase = None
            child.events.append(StateEvent(status=status, phase=None))
            parent_lease.validate()
            child_lease.save(child)
            parent_lease.validate()
            return child


def public_store_path(data_root: Path) -> Path:
    """Resolve the preconfigured layout; never provision a missing family or store."""
    configured = os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT")
    data = data_root.absolute()
    if not configured:
        # Read-only reports from legacy data/runs do not need a controller secret.
        return data / "runs"
    try:
        root = Path(configured).absolute()
        reject_symlinks(root)
        config = json.loads(read_regular(root / "family.json", 65536))
        path = Path(config["public_store"]).absolute()
        reject_symlinks(path)
        if path.parent != data or path.name == "workspaces":
            raise ValueError("store layout mismatch")
        return path
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BatchInputError(
            "FAMILY_CONFIG_CONFLICT", "Set --data-root to the configured public store's parent."
        ) from exc
