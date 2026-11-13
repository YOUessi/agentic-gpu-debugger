import stat
import time
from datetime import UTC, datetime

from gpu_agent.contracts import RunManifest
from gpu_agent.web.jobs import RepairJobManager, RepairJobStore
from gpu_agent.web.models import RepairJob, RepairJobStatus, RepairRequest


def test_async_repair_job_persists_run_binding_and_completion(tmp_path):
    source = tmp_path / "case"
    source.mkdir()
    seen = []

    class Service:
        def repair(self, selected, *, policy, mode, on_run_created):
            assert selected == source
            assert policy.max_candidates == 3
            assert mode == "E"
            run = RunManifest(id="a" * 32, kind="diagnosis")
            on_run_created(run)
            seen.append(run.id)
            time.sleep(0.02)
            return run.model_copy(update={"status": "COMPLETED"}), None

    manager = RepairJobManager(
        job_root=tmp_path / "jobs",
        service_factory=Service,
        source_resolver=lambda case_id: source,
        max_workers=1,
    )
    submitted = manager.submit(
        RepairRequest(
            case_id="case_0021",
            mode="E",
            max_candidates=3,
            max_llm_calls=8,
            allow_paid_calls=False,
        )
    )
    assert submitted.status == RepairJobStatus.QUEUED

    deadline = time.monotonic() + 2
    current = submitted
    while current.status not in {RepairJobStatus.COMPLETED, RepairJobStatus.FAILED}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
        current = manager.get(submitted.id)

    assert current.status == RepairJobStatus.COMPLETED
    assert current.run_id == "a" * 32
    assert current.error_code is None
    assert seen == ["a" * 32]
    job_path = tmp_path / "jobs" / f"{submitted.id}.json"
    assert job_path.is_file()
    assert stat.S_IMODE((tmp_path / "jobs").stat().st_mode) == 0o700
    assert stat.S_IMODE(job_path.stat().st_mode) == 0o600


def test_repair_job_store_marks_interrupted_jobs_failed(tmp_path):
    store = RepairJobStore(tmp_path / "jobs")
    now = datetime.now(UTC)
    job = RepairJob(
        id="b" * 32,
        status=RepairJobStatus.RUNNING,
        case_id="case_0021",
        mode="E",
        created_at=now,
        updated_at=now,
        run_id="c" * 32,
    )
    store.save(job)
    store.recover_interrupted()

    recovered = store.load(job.id)
    assert recovered.status == RepairJobStatus.FAILED
    assert recovered.run_id == "c" * 32
    assert recovered.error_code == "WEB_CONTROLLER_RESTARTED"
