"""Single-store atomic manifests and immutable blobs in a controller-owned root.

State audit events are committed inside the manifest to avoid a two-file transaction.
No candidate may write this root. This is not a sandbox against the owning OS user.
"""

import fcntl
import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from gpu_agent.contracts import (
    ArtifactRef,
    CurrentPhase,
    RunManifest,
    RunStatus,
    StateEvent,
    Visibility,
    new_id,
)


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

    def create_run(self, kind: str, parent_run_id: str | None = None) -> RunManifest:
        if parent_run_id is not None:
            self.load(parent_run_id)
        run_id = new_id()
        directory = self._run_dir(run_id)
        directory.mkdir(mode=0o700)
        (directory / "artifacts").mkdir(mode=0o700)
        run = RunManifest(
            id=run_id,
            kind=kind,
            parent_run_id=parent_run_id,
            events=[StateEvent(status=RunStatus.QUEUED, phase=None)],
        )
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
            if run.status == RunStatus.RUNNING and (
                target == RunStatus.COMPLETED
                or (target == RunStatus.RUNNING and active != run.current_phase)
            ):
                run.last_completed_phase = run.current_phase
            run.status, run.current_phase = target, active
            run.events.append(StateEvent(status=target, phase=active))
            self._save(run)
            return run

    def put(self, run_id: str, name: str, content: bytes, visibility: Visibility) -> ArtifactRef:
        label = PurePosixPath(name)
        if not name or label.is_absolute() or ".." in label.parts or "\\" in name:
            raise ValueError("invalid artifact name")
        if visibility != self.visibility:
            raise ValueError("artifact visibility requires a separate store")
        if len(content) > 64 * 1024 * 1024:
            raise ValueError("artifact too large")
        with self._lock(run_id):
            run = self.load(run_id)
            if run.status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
                raise ValueError("terminal run is immutable")
            artifact_id = new_id()
            relative = f"{run_id}/artifacts/{artifact_id}"
            target = self.root / relative
            reject_symlinks(target)
            ref = ArtifactRef(
                id=artifact_id,
                run_id=run_id,
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
                    if ref not in self.load(run_id).artifact_refs:
                        target.unlink()
                    raise
            finally:
                Path(temporary).unlink(missing_ok=True)
            return ref

    def read(self, ref: ArtifactRef) -> bytes:
        if ref.visibility != self.visibility or ref not in self.load(ref.run_id).artifact_refs:
            raise ValueError("unregistered artifact or wrong visibility")
        if ref.relative_path != f"{ref.run_id}/artifacts/{ref.id}":
            raise ValueError("artifact path mismatch")
        data = read_regular(self.root / ref.relative_path, min(ref.byte_count, 64 * 1024 * 1024))
        if len(data) != ref.byte_count or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ValueError("artifact hash mismatch")
        return data
