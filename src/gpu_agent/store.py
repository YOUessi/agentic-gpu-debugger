"""Single-store atomic manifests and immutable blobs in a controller-owned root.

State audit events are committed inside the manifest to avoid a two-file transaction.
No candidate may write this root. This is not a sandbox against the owning OS user.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from gpu_agent.contracts import (
    ArtifactRef,
    CurrentPhase,
    ExternalRunOrigin,
    RunBinding,
    RunManifest,
    RunStatus,
    StateEvent,
    Visibility,
    new_id,
)

if TYPE_CHECKING:
    from gpu_agent.benchmark.evaluation import EvaluationUnitBinding
    from gpu_agent.benchmark.ledger import CorpusLedger, EvaluationCutoffReservation
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier


@dataclass(frozen=True)
class RunStoreIdentity:
    resolved_root: str
    device: int
    inode: int
    visibility: Visibility


@dataclass(frozen=True)
class EvaluationRunIdentity:
    resolved_path: str
    root_device: int
    root_inode: int
    device: int
    inode: int
    lock_device: int
    lock_inode: int


def reject_symlinks(path: Path) -> None:
    for part in [*reversed(path.parents), path]:
        if part.is_symlink():
            raise ValueError("symlinks are not allowed")


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_regular(path: Path, limit: int) -> bytes:
    reject_symlinks(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError("file is unavailable or unsafe") from exc
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("not a bounded regular file")
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("file grew beyond limit")
        return data


def _read_regular_at(directory_fd: int, name: str, limit: int) -> bytes:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError("file is unavailable or unsafe") from exc
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("not a bounded regular file")
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("file grew beyond limit")
        return data


class EvaluationRunLease:
    """Fd-scoped authority view of one canonical evaluation run."""

    def __init__(
        self,
        store: RunStore,
        run_id: str,
        root_fd: int,
        run_fd: int,
        lock_fd: int,
        identity: EvaluationRunIdentity,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self._root_fd = root_fd
        self._run_fd = run_fd
        self._lock_fd = lock_fd
        self.identity = identity
        self._active = True
        self.__authority_finalized = False

    @property
    def _authority_commit_finalized(self) -> bool:
        return self.__authority_finalized

    def _commit_cutoff_reservation(
        self,
        ledger: CorpusLedger,
        preparation_id: str,
    ) -> EvaluationCutoffReservation:
        """Finalize only a ledger-reloaded authenticated PREPARED reservation."""
        from gpu_agent.benchmark.ledger import CorpusLedger

        if type(ledger) is not CorpusLedger:
            raise ValueError("cutoff authority commit requires native controller types")
        committed = CorpusLedger._commit_prepared_reservation(
            ledger,
            self,
            preparation_id,
        )
        # The concrete primitive returned only after the exact COMMITTED state became
        # authoritative. This assignment is intentionally unreachable independently.
        self.__authority_finalized = True
        return committed

    @staticmethod
    def _same(left: os.stat_result, right: os.stat_result) -> bool:
        return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)

    def validate(self) -> None:
        if not self._active:
            raise ValueError("evaluation run lease is closed")
        root_fd_info = os.fstat(self._root_fd)
        run_fd_info = os.fstat(self._run_fd)
        lock_fd_info = os.fstat(self._lock_fd)
        root_path_info = os.stat(self.store.root, follow_symlinks=False)
        run_path_info = os.stat(self.run_id, dir_fd=self._root_fd, follow_symlinks=False)
        lock_relative_info = os.stat(".lock", dir_fd=self._run_fd, follow_symlinks=False)
        lock_path_info = os.stat(
            f"{self.run_id}/.lock", dir_fd=self._root_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(root_fd_info.st_mode)
            or not stat.S_ISDIR(root_path_info.st_mode)
            or not self._same(root_fd_info, root_path_info)
            or root_fd_info.st_mode & 0o077
            or not stat.S_ISDIR(run_fd_info.st_mode)
            or not stat.S_ISDIR(run_path_info.st_mode)
            or not self._same(run_fd_info, run_path_info)
            or run_fd_info.st_mode & 0o077
            or not stat.S_ISREG(lock_fd_info.st_mode)
            or stat.S_IMODE(lock_fd_info.st_mode) != 0o600
            or lock_fd_info.st_nlink != 1
            or not self._same(lock_fd_info, lock_relative_info)
            or not self._same(lock_fd_info, lock_path_info)
            or self.identity.root_device != root_fd_info.st_dev
            or self.identity.root_inode != root_fd_info.st_ino
            or self.identity.device != run_fd_info.st_dev
            or self.identity.inode != run_fd_info.st_ino
            or self.identity.lock_device != lock_fd_info.st_dev
            or self.identity.lock_inode != lock_fd_info.st_ino
        ):
            raise ValueError("evaluation run lease is no longer canonical or safe")
        resolved = str((self.store.root / self.run_id).resolve(strict=True))
        if resolved != self.identity.resolved_path:
            raise ValueError("evaluation run resolved path changed")

    def load(self) -> RunManifest:
        self.validate()
        manifest = RunManifest.model_validate_json(
            _read_regular_at(self._run_fd, "manifest.json", 8 * 1024 * 1024)
        )
        if manifest.id != self.run_id:
            raise ValueError("manifest ID mismatch")
        self.validate()
        return manifest

    def save(self, manifest: RunManifest) -> None:
        if manifest.id != self.run_id:
            raise ValueError("leased manifest ID mismatch")
        self.validate()
        fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=f"/proc/self/fd/{self._run_fd}")
        name = Path(temporary).name
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(manifest.model_dump_json(indent=2).encode())
                stream.flush()
                os.fsync(stream.fileno())
            self.validate()
            os.replace(
                name,
                "manifest.json",
                src_dir_fd=self._run_fd,
                dst_dir_fd=self._run_fd,
            )
            os.fsync(self._run_fd)
            self.validate()
        finally:
            try:
                os.unlink(name, dir_fd=self._run_fd)
            except FileNotFoundError:
                pass

    def children(self) -> list[RunManifest]:
        self.validate()
        children: list[RunManifest] = []
        for name in sorted(os.listdir(self._root_fd)):
            if not re.fullmatch(r"[a-f0-9]{32}", name):
                continue
            child_fd = -1
            try:
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=self._root_fd,
                )
                child = RunManifest.model_validate_json(
                    _read_regular_at(child_fd, "manifest.json", 8 * 1024 * 1024)
                )
            except (OSError, ValueError) as exc:
                raise ValueError("evaluation child inventory is unsafe") from exc
            finally:
                if child_fd >= 0:
                    os.close(child_fd)
            if child.id != name:
                raise ValueError("child manifest ID mismatch")
            if child.parent_run_id == self.run_id:
                children.append(child)
        self.validate()
        return children

    def read(self, ref: ArtifactRef) -> bytes:
        run = self.load()
        if ref.visibility != self.store.visibility or ref not in run.artifact_refs:
            raise ValueError("unregistered artifact or wrong visibility")
        if ref.relative_path != f"{self.run_id}/artifacts/{ref.id}":
            raise ValueError("artifact path mismatch")
        artifacts_fd = os.open(
            "artifacts", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self._run_fd
        )
        try:
            data = _read_regular_at(artifacts_fd, ref.id, min(ref.byte_count, 64 * 1024 * 1024))
        finally:
            os.close(artifacts_fd)
        if len(data) != ref.byte_count or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ValueError("artifact hash mismatch")
        self.validate()
        return data


class RunStore:
    def __init__(self, root: Path, *, visibility: Visibility = "public") -> None:
        self.root = root.absolute()
        reject_symlinks(self.root)
        missing = []
        current = self.root
        while not current.exists():
            missing.append(current)
            current = current.parent
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for created in reversed(missing):
            sync_directory(created.parent)
        self.visibility = visibility
        self._evaluation_verifier: EvaluationScheduleVerifier | None = None

    @property
    def identity(self) -> RunStoreIdentity:
        reject_symlinks(self.root)
        info = self.root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("run store root must remain a directory")
        return RunStoreIdentity(
            resolved_root=str(self.root.resolve(strict=True)),
            device=info.st_dev,
            inode=info.st_ino,
            visibility=self.visibility,
        )

    def bind_evaluation_verifier(self, verifier: EvaluationScheduleVerifier) -> None:
        from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

        if type(verifier) is not EvaluationScheduleVerifier:
            raise ValueError("evaluation store requires the native schedule verifier")
        EvaluationScheduleVerifier.require_store(verifier, self)
        if self._evaluation_verifier is not None and self._evaluation_verifier is not verifier:
            raise ValueError("evaluation store authority is already bound")
        self._evaluation_verifier = verifier

    def _run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("invalid run ID")
        path = self.root / run_id
        reject_symlinks(path)
        return path

    @contextmanager
    def _lock(self, run_id: str) -> Iterator[None]:
        path = self._run_dir(run_id) / ".lock"
        reject_symlinks(path)
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("lock must be a regular file")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    @contextmanager
    def evaluation_run_lease(self, run_id: str) -> Iterator[EvaluationRunLease]:
        """Hold canonical root/run/lock fds for one complete authority operation."""
        self._run_dir(run_id)
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        run_fd = -1
        lock_fd = -1
        lease: EvaluationRunLease | None = None
        try:
            root_info = os.fstat(root_fd)
            if not stat.S_ISDIR(root_info.st_mode) or root_info.st_mode & 0o077:
                raise ValueError("evaluation store root is not a private real directory")
            run_fd = os.open(
                run_id,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=root_fd,
            )
            run_info = os.fstat(run_fd)
            if not stat.S_ISDIR(run_info.st_mode) or run_info.st_mode & 0o077:
                raise ValueError("evaluation run is not a private real directory")
            lock_fd = os.open(
                ".lock",
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
                dir_fd=run_fd,
            )
            lock_info = os.fstat(lock_fd)
            if (
                not stat.S_ISREG(lock_info.st_mode)
                or stat.S_IMODE(lock_info.st_mode) != 0o600
                or lock_info.st_nlink != 1
            ):
                raise ValueError("evaluation run lock is unsafe")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            identity = EvaluationRunIdentity(
                resolved_path=str((self.root / run_id).resolve(strict=True)),
                root_device=root_info.st_dev,
                root_inode=root_info.st_ino,
                device=run_info.st_dev,
                inode=run_info.st_ino,
                lock_device=lock_info.st_dev,
                lock_inode=lock_info.st_ino,
            )
            lease = EvaluationRunLease(self, run_id, root_fd, run_fd, lock_fd, identity)
            lease.validate()
            yield lease
            if not lease._authority_commit_finalized:
                lease.validate()
        except OSError as exc:
            raise ValueError("evaluation run lease is unavailable or unsafe") from exc
        finally:
            if lease is not None:
                lease._active = False
            for descriptor in (lock_fd, run_fd, root_fd):
                if descriptor < 0:
                    continue
                try:
                    os.close(descriptor)
                except OSError:
                    if lease is None or not lease._authority_commit_finalized:
                        raise

    def _save(self, manifest: RunManifest) -> None:
        directory = self._run_dir(manifest.id)
        fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(manifest.model_dump_json(indent=2).encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / "manifest.json")
            sync_directory(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _create_atomic_run(self, manifest: RunManifest) -> None:
        """Publish a complete controller-selected run directory in one rename."""
        target = self._run_dir(manifest.id)
        temporary = Path(tempfile.mkdtemp(prefix=".run-", dir=self.root))
        try:
            (temporary / "artifacts").mkdir(mode=0o700)
            fd, manifest_temporary = tempfile.mkstemp(prefix=".manifest-", dir=temporary)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(manifest.model_dump_json(indent=2).encode())
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(manifest_temporary, temporary / "manifest.json")
            finally:
                Path(manifest_temporary).unlink(missing_ok=True)
            sync_directory(temporary)
            os.rename(temporary, target)
            sync_directory(self.root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def create_run(
        self,
        kind: str,
        parent_run_id: str | None = None,
        *,
        binding: RunBinding | None = None,
        external_origin: ExternalRunOrigin | None = None,
        _run_id: str | None = None,
    ) -> RunManifest:
        if external_origin is not None and external_origin.visibility == self.visibility:
            raise ValueError("external origin visibility must name a different store")
        if parent_run_id is not None:
            parent = self.load(parent_run_id)
            if (
                parent.kind == "evaluation"
                and parent.binding is not None
                and parent.binding.purpose == "evaluation"
            ):
                raise ValueError("evaluation children require atomic authority validation")
            if parent.binding is None and binding is not None:
                raise ValueError("a child cannot add a missing parent release binding")
            if parent.binding is not None:
                if binding is not None and binding != parent.binding:
                    raise ValueError("external origin child binding differs from parent")
                binding = parent.binding
            if parent.external_origin is None and external_origin is not None:
                raise ValueError("a child cannot add a missing parent external origin")
            if parent.external_origin is not None:
                if external_origin is not None and external_origin != parent.external_origin:
                    raise ValueError("child external origin differs from parent")
                external_origin = parent.external_origin
        run_id = _run_id or new_id()
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("invalid controller run ID")
        run = RunManifest(
            id=run_id,
            kind=kind,
            parent_run_id=parent_run_id,
            binding=binding,
            external_origin=external_origin,
            events=[StateEvent(status=RunStatus.QUEUED, phase=None)],
        )
        if _run_id is not None:
            self._create_atomic_run(run)
        else:
            directory = self._run_dir(run_id)
            directory.mkdir(mode=0o700)
            (directory / "artifacts").mkdir(mode=0o700)
            self._save(run)
            sync_directory(self.root)
        return run

    def load(self, run_id: str) -> RunManifest:
        data = read_regular(self._run_dir(run_id) / "manifest.json", 8 * 1024 * 1024)
        manifest = RunManifest.model_validate_json(data)
        if manifest.id != run_id:
            raise ValueError("manifest ID mismatch")
        return manifest

    def recoverable_runs(self) -> list[RunManifest]:
        """Return interrupted controller runs without resuming native processes."""
        runs: list[RunManifest] = []
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and re.fullmatch(r"[a-f0-9]{32}", path.name):
                run = self.load(path.name)
                if run.status == RunStatus.RUNNING:
                    runs.append(run)
        return runs

    def children(self, parent_run_id: str) -> list[RunManifest]:
        """Return the exact direct-child inventory for a controller run."""
        self.load(parent_run_id)
        children: list[RunManifest] = []
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and re.fullmatch(r"[a-f0-9]{32}", path.name):
                run = self.load(path.name)
                if run.parent_run_id == parent_run_id:
                    children.append(run)
        return children

    def fail_interrupted(
        self, run_id: str, reason_code: str = "CONTROLLER_RESTARTED"
    ) -> RunManifest:
        if not re.fullmatch(r"[A-Z0-9_]{1,64}", reason_code):
            raise ValueError("invalid recovery reason")
        run = self.load(run_id)
        if run.status != RunStatus.RUNNING:
            raise ValueError("run is not recoverable")
        self.put(
            run_id,
            "recovery/interruption.json",
            (f'{{"reason_code":"{reason_code}"}}').encode(),
            self.visibility,
        )
        return self.transition(run_id, RunStatus.FAILED, None)

    def transition(
        self, run_id: str, status: RunStatus | str, phase: CurrentPhase | str | None
    ) -> RunManifest:
        target = RunStatus(status)
        active = CurrentPhase(phase) if phase is not None else None
        if (target == RunStatus.RUNNING) != (active is not None):
            raise ValueError("invalid status/phase pair")
        with self._lock(run_id):
            run = self.load(run_id)
            if run.status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
                raise ValueError("terminal run is immutable")
            if target == RunStatus.QUEUED or (
                run.status == RunStatus.QUEUED and target == RunStatus.COMPLETED
            ):
                raise ValueError("invalid transition")
            if (
                target == RunStatus.RUNNING
                and run.kind == "evaluation"
                and run.binding is not None
                and run.binding.purpose == "evaluation"
            ):
                raise ValueError("evaluation RUNNING requires signed schedule activation")
            if run.status == RunStatus.RUNNING and (
                target == RunStatus.COMPLETED
                or (target == RunStatus.RUNNING and active != run.current_phase)
            ):
                run.last_completed_phase = run.current_phase
            run.status, run.current_phase = target, active
            run.events.append(StateEvent(status=target, phase=active))
            self._save(run)
            return run

    def activate_evaluation(self, verifier: EvaluationScheduleVerifier, run_id: str) -> RunManifest:
        """Atomically verify and perform the only evaluation activation transition."""
        from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

        if type(verifier) is not EvaluationScheduleVerifier:
            raise ValueError("evaluation activation requires the native verifier")
        EvaluationScheduleVerifier.require_store(verifier, self)
        if self._evaluation_verifier is not verifier:
            raise ValueError("evaluation activation verifier is not store-bound")
        with self.evaluation_run_lease(run_id) as lease:
            run = lease.load()
            if (
                run.kind != "evaluation"
                or run.binding is None
                or run.binding.purpose != "evaluation"
            ):
                raise ValueError("run is not a bound evaluation")
            if run.status == RunStatus.RUNNING and run.current_phase == CurrentPhase.EXECUTING:
                EvaluationScheduleVerifier._verify_leased(verifier, lease)
                return run
            if run.status != RunStatus.QUEUED or run.current_phase is not None:
                raise ValueError("evaluation is not in the activatable QUEUED state")
            EvaluationScheduleVerifier._verify_leased(verifier, lease)
            run.status = RunStatus.RUNNING
            run.current_phase = CurrentPhase.EXECUTING
            run.events.append(StateEvent(status=RunStatus.RUNNING, phase=CurrentPhase.EXECUTING))
            lease.save(run)
            return run

    def validate_and_create_evaluation_child(
        self,
        verifier: EvaluationScheduleVerifier,
        unit: EvaluationUnitBinding,
    ) -> RunManifest:
        """Validate and reserve one diagnosis child while holding the parent lock."""
        from gpu_agent.benchmark.evaluation import EvaluationUnitBinding
        from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

        if (
            type(verifier) is not EvaluationScheduleVerifier
            or type(unit) is not EvaluationUnitBinding
        ):
            raise ValueError("evaluation child requires native authority models")
        EvaluationScheduleVerifier.require_store(verifier, self)
        if self._evaluation_verifier is not verifier:
            raise ValueError("evaluation child verifier is not store-bound")
        with self.evaluation_run_lease(unit.evaluation_run_id) as lease:
            EvaluationScheduleVerifier._validate_unit_leased(verifier, lease, unit)
            parent = lease.load()
            child_id = hashlib.sha256(
                f"evaluation-diagnosis-v1:{parent.id}:{unit.ordinal}".encode()
            ).hexdigest()[:32]
            child = RunManifest(
                id=child_id,
                kind="diagnosis",
                parent_run_id=parent.id,
                binding=parent.binding,
                external_origin=parent.external_origin,
                events=[StateEvent(status=RunStatus.QUEUED, phase=None)],
            )
            self._create_atomic_run(child)
            self.put(
                child.id,
                "evaluation/unit.json",
                unit.model_dump_json().encode(),
                self.visibility,
            )
            return self.load(child.id)

    def _validate_put(self, name: str, content: bytes, visibility: Visibility) -> None:
        label = PurePosixPath(name)
        if not name or label.is_absolute() or ".." in label.parts or "\\" in name:
            raise ValueError("invalid artifact name")
        if visibility != self.visibility:
            raise ValueError("artifact visibility requires a separate store")
        if len(content) > 64 * 1024 * 1024:
            raise ValueError("artifact too large")

    def _put_locked(
        self,
        run: RunManifest,
        name: str,
        content: bytes,
        visibility: Visibility,
    ) -> ArtifactRef:
        if run.status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
            raise ValueError("terminal run is immutable")
        artifact_id = new_id()
        relative = f"{run.id}/artifacts/{artifact_id}"
        target = self.root / relative
        reject_symlinks(target)
        ref = ArtifactRef(
            id=artifact_id,
            run_id=run.id,
            name=name,
            sha256=hashlib.sha256(content).hexdigest(),
            visibility=visibility,
            relative_path=relative,
            byte_count=len(content),
        )
        fd, temporary = tempfile.mkstemp(prefix=".blob-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o400)
            os.link(temporary, target)  # exclusive atomic publication, never overwrite
            sync_directory(target.parent)
            run.artifact_refs.append(ref)
            try:
                self._save(run)
            except OSError:
                # If replace succeeded but directory fsync failed, keep the registered blob.
                if ref not in self.load(run.id).artifact_refs:
                    target.unlink()
                raise
        finally:
            Path(temporary).unlink(missing_ok=True)
        return ref

    def put(self, run_id: str, name: str, content: bytes, visibility: Visibility) -> ArtifactRef:
        self._validate_put(name, content, visibility)
        with self._lock(run_id):
            run = self.load(run_id)
            return self._put_locked(run, name, content, visibility)

    def put_if_absent_exact(
        self, run_id: str, name: str, content: bytes, visibility: Visibility
    ) -> ArtifactRef:
        """Atomically install one named artifact or verify the existing exact value."""
        self._validate_put(name, content, visibility)
        with self._lock(run_id):
            run = self.load(run_id)
            refs = [ref for ref in run.artifact_refs if ref.name == name]
            if len(refs) > 1:
                raise ValueError("artifact name is ambiguous")
            if refs:
                if self.read(refs[0]) != content:
                    raise ValueError("existing artifact differs")
                return refs[0]
            return self._put_locked(run, name, content, visibility)

    def read(self, ref: ArtifactRef) -> bytes:
        if ref.visibility != self.visibility or ref not in self.load(ref.run_id).artifact_refs:
            raise ValueError("unregistered artifact or wrong visibility")
        if ref.relative_path != f"{ref.run_id}/artifacts/{ref.id}":
            raise ValueError("artifact path mismatch")
        data = read_regular(self.root / ref.relative_path, min(ref.byte_count, 64 * 1024 * 1024))
        if len(data) != ref.byte_count or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ValueError("artifact hash mismatch")
        return data
