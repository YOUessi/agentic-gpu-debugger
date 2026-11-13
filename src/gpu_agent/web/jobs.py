"""Persistent web job metadata for asynchronous public repair workflows."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from gpu_agent.agent.provider import DevelopmentCallPolicy
from gpu_agent.contracts import RunManifest, new_id
from gpu_agent.public_task import PublicRepairInputError
from gpu_agent.repair import RepairPolicy
from gpu_agent.service import ApplicationService
from gpu_agent.store import read_regular, reject_symlinks, sync_directory
from gpu_agent.web.models import RepairJob, RepairJobStatus, RepairRequest

_JOB_ID = re.compile(r"^[a-f0-9]{32}$")
ServiceFactory = Callable[[], ApplicationService]
SourceResolver = Callable[[str], Path]


def _now() -> datetime:
    return datetime.now(UTC)


class RepairJobStore:
    """Controller-owned orchestration metadata; never an execution evidence source."""

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        reject_symlinks(self.root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
    def _path(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise ValueError("invalid web job ID")
        return self.root / f"{job_id}.json"

    def save(self, job: RepairJob) -> RepairJob:
        path = self._path(job.id)
        reject_symlinks(path)
        fd, temporary = tempfile.mkstemp(prefix=".job-", dir=self.root)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(job.model_dump_json(indent=2).encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            sync_directory(self.root)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return job.model_copy(deep=True)

    def load(self, job_id: str) -> RepairJob:
        path = self._path(job_id)
        return RepairJob.model_validate_json(read_regular(path, 256 * 1024))

    def recover_interrupted(self) -> None:
        for path in sorted(self.root.glob("*.json")):
            if not _JOB_ID.fullmatch(path.stem):
                continue
            try:
                job = self.load(path.stem)
            except (OSError, ValueError):
                continue
            if job.status not in {RepairJobStatus.QUEUED, RepairJobStatus.RUNNING}:
                continue
            self.save(
                job.model_copy(
                    update={
                        "status": RepairJobStatus.FAILED,
                        "updated_at": _now(),
                        "error_code": "WEB_CONTROLLER_RESTARTED",
                    }
                )
            )


class RepairJobManager:
    def __init__(
        self,
        *,
        job_root: Path,
        service_factory: ServiceFactory,
        source_resolver: SourceResolver,
        max_workers: int = 2,
    ) -> None:
        if not 1 <= max_workers <= 8:
            raise ValueError("invalid web repair worker count")
        self.store = RepairJobStore(job_root)
        self.store.recover_interrupted()
        self._service_factory = service_factory
        self._source_resolver = source_resolver
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="gpu-agent-web-repair",
        )
        self._lock = RLock()

    def get(self, job_id: str) -> RepairJob:
        with self._lock:
            return self.store.load(job_id)

    def _update(self, job_id: str, **changes: object) -> RepairJob:
        with self._lock:
            current = self.store.load(job_id)
            return self.store.save(
                current.model_copy(update={**changes, "updated_at": _now()})
            )

    def submit(self, request: RepairRequest) -> RepairJob:
        source = self._source_resolver(request.case_id)
        if not source.is_dir():
            raise ValueError("public repair case is unavailable")
        now = _now()
        job = RepairJob(
            id=new_id(),
            status=RepairJobStatus.QUEUED,
            case_id=request.case_id,
            mode=request.mode,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self.store.save(job)
        self._executor.submit(self._execute, job.id, request.model_copy(deep=True), source)
        return job

    def _execute(self, job_id: str, request: RepairRequest, source: Path) -> None:
        self._update(job_id, status=RepairJobStatus.RUNNING, error_code=None)
        try:
            service = self._service_factory()
            if request.allow_paid_calls:
                service.allow_development_paid_calls(
                    DevelopmentCallPolicy(max_llm_calls=request.max_llm_calls)
                )

            def bind_run(run: RunManifest) -> None:
                self._update(job_id, run_id=run.id)

            run, verified = service.repair(
                source,
                policy=RepairPolicy(max_candidates=request.max_candidates),
                mode=request.mode,
                on_run_created=bind_run,
            )
            self._update(
                job_id,
                status=RepairJobStatus.COMPLETED,
                run_id=run.id,
                verification_verdict=verified.verdict.value if verified else None,
                error_code=None,
            )
        except PublicRepairInputError as exc:
            self._update(
                job_id,
                status=RepairJobStatus.FAILED,
                error_code=exc.code,
            )
        except (OSError, ValueError):
            self._update(
                job_id,
                status=RepairJobStatus.FAILED,
                error_code="REPAIR_INPUT_INVALID",
            )
        except Exception:
            # Web orchestration must not serialize arbitrary upstream exception text.
            self._update(
                job_id,
                status=RepairJobStatus.FAILED,
                error_code="WEB_JOB_FAILED",
            )
