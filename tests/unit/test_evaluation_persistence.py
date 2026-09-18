import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest
from schedule_authority_support import schedule_client_for_test

from gpu_agent.benchmark.evaluation import (
    EvaluationAttempt,
    EvaluationRunner,
    EvaluationScheduleItem,
    PublicEvaluationRecord,
)
from gpu_agent.contracts import RunStatus


def _runner(executor, execute_owner=None, **overrides) -> EvaluationRunner:
    binding = executor.service.binding
    assert binding is not None
    del execute_owner
    options = {
        "store": executor.service.store,
        "executor": executor,
        "schedule_client": schedule_client_for_test(executor),
        "commit": binding.repository.commit,
        "prompt_version": binding.prompt_version,
        "toolchain_hash": binding.toolchain_lock_hash,
        "model_config_hash": binding.model_config_hash,
        "binding": binding,
        "max_cost_usd": 0.0,
        "max_unit_cost_usd": 0.0,
        "random_seed": 7,
    }
    options.update(overrides)
    return EvaluationRunner(**options)


def _artifact(store, run_id: str, name: str) -> bytes:
    ref = next(ref for ref in store.load(run_id).artifact_refs if ref.name == name)
    return store.read(ref)


class _Proxy:
    def __init__(self, executor):
        self.executor = executor
        self.native_execute = executor.execute_scheduled
        self.calls = []

    def execute_scheduled(self, run_id, ordinal):
        record = self.native_execute(run_id, ordinal)
        self.calls.append(record.repeat)
        return record

    def validate_scheduled_record(self, record, item, attempt):
        return self.executor.validate_scheduled_record(record, item, attempt)


def test_record_is_durable_before_next_unit(native_evaluation_executor, monkeypatch):
    from gpu_agent.benchmark.executor import EvaluationExecutor

    executor = native_evaluation_executor
    native = EvaluationExecutor.execute_scheduled
    calls = []

    def observed(self, run_id, ordinal):
        active = executor.service.store.recoverable_runs()
        if calls:
            persisted = PublicEvaluationRecord.model_validate_json(
                _artifact(
                    executor.service.store,
                    active[0].id,
                    f"evaluation/records/{len(calls) - 1}.json",
                )
            )
            assert persisted.repeat == calls[-1]
        record = native(self, run_id, ordinal)
        calls.append(record.repeat)
        return record

    monkeypatch.setattr(EvaluationExecutor, "execute_scheduled", observed)
    result = _runner(executor).run("D", "development", 3)
    run = executor.service.store.load(result.run_id)
    names = {ref.name for ref in run.artifact_refs}
    assert {"evaluation/schedule.json", "evaluation/manifest.json"} <= names
    assert {f"evaluation/records/{index}.json" for index in range(3)} <= names
    assert (
        json.loads(_artifact(executor.service.store, result.run_id, "evaluation/manifest.json"))[
            "executed_units"
        ]
        == 3
    )
    assert run.status == RunStatus.COMPLETED


def test_unit_reservation_stops_before_cost_cap_can_be_exceeded(native_evaluation_executor):
    proxy = _Proxy(native_evaluation_executor)
    result = _runner(
        native_evaluation_executor,
        proxy,
        max_cost_usd=0.5,
        max_unit_cost_usd=0.6,
    ).run("D", "development", 3)
    assert result.stopped_reason == "COST_CAP_RESERVATION_REQUIRED"
    assert result.executed_units == 0 and proxy.calls == []


def test_unexpected_executor_failure_preserves_completed_records(
    native_evaluation_executor, monkeypatch
):
    from gpu_agent.benchmark.executor import EvaluationExecutor

    executor = native_evaluation_executor
    native = EvaluationExecutor.execute_scheduled
    calls = []

    def fails_second(self, run_id, ordinal):
        if calls:
            raise RuntimeError("executor died")
        calls.append(ordinal)
        return native(self, run_id, ordinal)

    monkeypatch.setattr(EvaluationExecutor, "execute_scheduled", fails_second)
    result = _runner(executor).run("D", "development", 3)
    assert result.stopped_reason == "EXECUTION_ERROR" and result.executed_units == 1
    assert "evaluation/records/1.json" not in {
        ref.name for ref in executor.service.store.load(result.run_id).artifact_refs
    }
    assert executor.service.store.load(result.run_id).status == RunStatus.FAILED


def test_resume_rejects_commit_or_schedule_mismatch(native_evaluation_executor, monkeypatch):
    from gpu_agent.benchmark.executor import EvaluationExecutor

    executor = native_evaluation_executor
    monkeypatch.setattr(
        EvaluationExecutor,
        "execute_scheduled",
        lambda self, run_id, ordinal: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    original = _runner(executor)
    with pytest.raises(KeyboardInterrupt):
        original.run("D", "development", 3)
    run_id = executor.service.store.recoverable_runs()[0].id
    with pytest.raises(ValueError):
        _runner(executor, commit="d" * 40).resume(run_id, "D", "development", 3)
    with pytest.raises(ValueError):
        original.resume(run_id, "D", "holdout", 3)


def test_started_attempt_without_record_fails_closed_without_resume_replay(
    native_evaluation_executor, monkeypatch
):
    from gpu_agent.benchmark.executor import EvaluationExecutor

    executor = native_evaluation_executor
    native = EvaluationExecutor.execute_scheduled
    monkeypatch.setattr(
        EvaluationExecutor,
        "execute_scheduled",
        lambda self, run_id, ordinal: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        _runner(executor).run("D", "development", 3)
    run_id = executor.service.store.recoverable_runs()[0].id
    monkeypatch.setattr(EvaluationExecutor, "execute_scheduled", native)
    result = _runner(executor).resume(run_id, "D", "development", 3)
    assert result.stopped_reason == "AMBIGUOUS_STARTED_ATTEMPT"
    assert result.executed_units == 0
    assert executor.service.store.load(run_id).status == RunStatus.FAILED


def test_successful_resume_continues_after_completed_ordinal(
    native_evaluation_executor, monkeypatch
):
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.store import RunStore

    executor = native_evaluation_executor
    runner = _runner(executor)
    store = executor.service.store
    put = RunStore.put

    def interrupt_after_first_record(self, run_id, name, content, visibility):
        result = put(self, run_id, name, content, visibility)
        if name == "evaluation/records/0.json":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(RunStore, "put", interrupt_after_first_record)
    with pytest.raises(KeyboardInterrupt):
        runner.run("D", "development", 3)
    run_id = store.recoverable_runs()[0].id
    monkeypatch.setattr(RunStore, "put", put)
    restarted = EvaluationExecutor(
        executor.service,
        executor.corpus,
        executor.sources,
        _corpus_family=executor._corpus_family,
        _schedule_verifier=executor._schedule_verifier,
    )
    result = _runner(restarted).resume(run_id, "D", "development", 3)
    assert [record.repeat for record in result.records] == [2, 0, 1]
    assert result.executed_units == 3 and result.stopped_reason is None


def test_record_serialization_failure_persists_failed_terminal_manifest(
    native_evaluation_executor, monkeypatch
):
    executor = native_evaluation_executor
    runner = _runner(executor)
    put = EvaluationRunner._put

    def fail_record(self, run_id, name, content):
        if name == "evaluation/records/0.json":
            raise TypeError("serialization failed")
        return put(self, run_id, name, content)

    monkeypatch.setattr(EvaluationRunner, "_put", fail_record)
    result = runner.run("D", "development", 3)
    assert result.stopped_reason == "RECORD_PERSISTENCE_ERROR"
    assert result.records == [] and result.executed_units == 0
    assert executor.service.store.load(result.run_id).status == RunStatus.FAILED


def test_native_record_cannot_be_replayed_for_another_schedule_unit(
    native_evaluation_executor, monkeypatch
):
    from gpu_agent.benchmark.executor import EvaluationExecutor

    executor = native_evaluation_executor
    native = EvaluationExecutor.execute_scheduled
    records = []

    def replay(self, run_id, ordinal):
        if not records:
            records.append(native(self, run_id, ordinal))
        return records[0]

    monkeypatch.setattr(EvaluationExecutor, "execute_scheduled", replay)
    result = _runner(executor).run("D", "development", 3)
    assert result.stopped_reason == "EXECUTION_ERROR"
    assert result.executed_units == 1


@pytest.mark.parametrize("fault", ["reservation", "extra_attempt", "record_without_attempt"])
def test_resume_recomputes_exact_attempt_set(native_evaluation_executor, fault):
    executor = native_evaluation_executor
    runner = _runner(executor)
    schedule = runner._schedule("D", "development", 3)
    store = executor.service.store
    run = store.create_run("evaluation", binding=executor.service.binding)
    runner._put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode())
    from gpu_agent.benchmark.schedule_authority import activate_schedule, seal_schedule

    seal_schedule(
        executor._corpus_family,
        store,
        run.id,
        schedule,
        executor.service.binding,
        schedule_client_for_test(executor),
        executor._schedule_verifier,
    )
    activate_schedule(store, executor._schedule_verifier, run.id)
    attempt = runner._attempt(run.id, schedule, schedule.items[0])
    if fault == "reservation":
        attempt = attempt.model_copy(update={"reserved_cost_usd": 1})
        runner._put(run.id, "evaluation/attempts/0.json", attempt.model_dump_json().encode())
    elif fault == "extra_attempt":
        extra = attempt.model_copy(update={"ordinal": len(schedule.items)})
        runner._put(
            run.id,
            f"evaluation/attempts/{len(schedule.items)}.json",
            extra.model_dump_json().encode(),
        )
    else:
        source_run = store.create_run("evaluation", binding=executor.service.binding)
        runner._put(
            source_run.id,
            "evaluation/schedule.json",
            schedule.model_dump_json().encode(),
        )
        seal_schedule(
            executor._corpus_family,
            store,
            source_run.id,
            schedule,
            executor.service.binding,
            schedule_client_for_test(executor),
            executor._schedule_verifier,
        )
        activate_schedule(store, executor._schedule_verifier, source_run.id)
        source_attempt = runner._attempt(source_run.id, schedule, schedule.items[0])
        runner._put(
            source_run.id,
            "evaluation/attempts/0.json",
            source_attempt.model_dump_json().encode(),
        )
        record = executor.execute_scheduled(source_run.id, schedule.items[0].ordinal)
        runner._put(run.id, "evaluation/records/0.json", record.public().model_dump_json().encode())
    with pytest.raises(ValueError):
        runner.resume(run.id, "D", "development", 3)


def test_runner_rejects_non_public_store(tmp_path, native_evaluation_executor):
    from gpu_agent.store import RunStore

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None
    with pytest.raises(ValueError):
        EvaluationRunner(
            RunStore(tmp_path / "evaluator", visibility="evaluator"),
            executor,
            commit=binding.repository.commit,
            prompt_version=binding.prompt_version or "",
            toolchain_hash=binding.toolchain_lock_hash or "",
            model_config_hash=binding.model_config_hash or "",
            binding=binding,
            max_cost_usd=0,
            max_unit_cost_usd=0,
        )


def test_runner_rejects_duck_typed_executor(native_evaluation_executor):
    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None

    class ForgedExecutor:
        execute_scheduled = executor.execute_scheduled
        validate_scheduled_record = executor.validate_scheduled_record

    with pytest.raises(ValueError, match="native executor implementation"):
        EvaluationRunner(
            executor.service.store,
            ForgedExecutor(),  # type: ignore[arg-type]
            commit=binding.repository.commit,
            prompt_version=binding.prompt_version or "",
            toolchain_hash=binding.toolchain_lock_hash or "",
            model_config_hash=binding.model_config_hash or "",
            binding=binding,
            max_cost_usd=0,
            max_unit_cost_usd=0,
        )


def test_concurrent_resume_claims_one_physical_evaluation_unit(
    native_evaluation_executor, monkeypatch
):
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.benchmark.schedule_authority import activate_schedule, seal_schedule

    executor = native_evaluation_executor

    class Blocking(_Proxy):
        def __init__(self, executor):
            super().__init__(executor)
            self.guard = Lock()
            self.started = Event()
            self.second_seen = Event()
            self.release = Event()

        def execute_scheduled(self, run_id, ordinal):
            item = schedule.items[ordinal]
            with self.guard:
                self.calls.append(item.repeat)
                if len(self.calls) == 1:
                    self.started.set()
                else:
                    self.second_seen.set()
            self.release.wait(2)
            return self.native_execute(run_id, ordinal)

    blocking = Blocking(executor)
    monkeypatch.setattr(
        EvaluationExecutor,
        "execute_scheduled",
        lambda self, run_id, ordinal: blocking.execute_scheduled(run_id, ordinal),
    )
    first_runner = _runner(executor)
    schedule = first_runner._schedule("D", "development", 3)
    store = executor.service.store
    run = store.create_run("evaluation", binding=executor.service.binding)
    first_runner._put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode())
    seal_schedule(
        executor._corpus_family,
        store,
        run.id,
        schedule,
        executor.service.binding,
        schedule_client_for_test(executor),
        executor._schedule_verifier,
    )
    activate_schedule(store, executor._schedule_verifier, run.id)
    second_runner = _runner(executor)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_runner.resume, run.id, "D", "development", 3)
        assert blocking.started.wait(1)
        second = pool.submit(second_runner.resume, run.id, "D", "development", 3)
        raced = blocking.second_seen.wait(0.2)
        blocking.release.set()
        outcomes = []
        for future in (first, second):
            try:
                outcomes.append(future.result())
            except ValueError:
                pass
    assert not raced
    assert len(blocking.calls) == 3
    assert len(outcomes) == 1 and outcomes[0].executed_units == 3


def test_low_level_executor_rejects_caller_constructed_schedule_unit(
    native_evaluation_executor,
):
    executor = native_evaluation_executor
    assert not hasattr(executor, "execute")
    assert not hasattr(executor, "_claim_scheduled")
    item = EvaluationScheduleItem(
        ordinal=0,
        case_id="case_0100",
        template_id="vector-add",
        mode="D",
        repeat=0,
        split="development",
    )
    attempt = EvaluationAttempt(
        run_id="a" * 32,
        ordinal=0,
        schedule_hash="b" * 64,
        idempotency_key="c" * 64,
        reserved_cost_usd=0,
    )
    with pytest.raises((TypeError, ValueError)):
        executor.execute_scheduled(item, attempt)
    with pytest.raises(ValueError):
        executor.execute_scheduled(attempt.run_id, attempt.ordinal)
    with pytest.raises(ValueError):
        executor._execute(attempt.run_id, attempt.ordinal)
    assert (
        executor.service.store.root
        / f".evaluation-execution-{attempt.run_id}-{attempt.ordinal}.lock"
    ).is_file()


def test_execution_lease_rejects_wrong_inode(native_evaluation_executor, monkeypatch):
    """The locked descriptor must still name the canonical per-unit lock path."""
    import gpu_agent.benchmark.executor as executor_module

    executor = native_evaluation_executor
    runner = _runner(executor)
    schedule = runner._schedule("D", "development", 3)
    store = executor.service.store
    run = store.create_run("evaluation", binding=executor.service.binding)
    runner._put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode())
    from gpu_agent.benchmark.schedule_authority import activate_schedule, seal_schedule

    seal_schedule(
        executor._corpus_family,
        store,
        run.id,
        schedule,
        executor.service.binding,
        schedule_client_for_test(executor),
        executor._schedule_verifier,
    )
    activate_schedule(store, executor._schedule_verifier, run.id)
    attempt = runner._attempt(run.id, schedule, schedule.items[0])
    runner._put(run.id, "evaluation/attempts/0.json", attempt.model_dump_json().encode())

    native_stat = executor_module.os.stat
    lock_path = store.root / f".evaluation-execution-{run.id}-0.lock"

    def wrong_inode(path, *args, **kwargs):
        result = native_stat(path, *args, **kwargs)
        if os.fspath(path) == os.fspath(lock_path):
            values = list(result)
            values[1] += 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(executor_module.os, "stat", wrong_inode)
    with pytest.raises(ValueError, match="lease path"):
        executor.execute_scheduled(run.id, 0)
    assert not store.children(run.id)


@pytest.mark.parametrize("replacement", ["regular", "symlink"])
def test_execution_lease_rejects_canonical_path_substitution(
    native_evaluation_executor, monkeypatch, tmp_path, replacement
):
    import gpu_agent.benchmark.executor as executor_module

    executor = native_evaluation_executor
    runner = _runner(executor)
    schedule = runner._schedule("D", "development", 3)
    store = executor.service.store
    run = store.create_run("evaluation", binding=executor.service.binding)
    runner._put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode())
    from gpu_agent.benchmark.schedule_authority import activate_schedule, seal_schedule

    seal_schedule(
        executor._corpus_family,
        store,
        run.id,
        schedule,
        executor.service.binding,
        schedule_client_for_test(executor),
        executor._schedule_verifier,
    )
    activate_schedule(store, executor._schedule_verifier, run.id)
    attempt = runner._attempt(run.id, schedule, schedule.items[0])
    runner._put(run.id, "evaluation/attempts/0.json", attempt.model_dump_json().encode())

    lock_path = store.root / f".evaluation-execution-{run.id}-0.lock"
    replacement_path = tmp_path / "replacement.lock"
    replacement_path.write_bytes(b"")
    replacement_path.chmod(0o600)
    native_flock = executor_module.fcntl.flock
    replaced = False

    def replace_after_flock(fd, operation):
        nonlocal replaced
        result = native_flock(fd, operation)
        if not replaced and operation == executor_module.fcntl.LOCK_EX:
            replaced = True
            lock_path.unlink()
            if replacement == "symlink":
                lock_path.symlink_to(replacement_path)
            else:
                os.replace(replacement_path, lock_path)
        return result

    monkeypatch.setattr(executor_module.fcntl, "flock", replace_after_flock)
    with pytest.raises(ValueError, match="symlink|lease path"):
        executor.execute_scheduled(run.id, 0)
    assert not store.children(run.id)


def test_runner_ignores_instance_replaced_trust_boundary_methods(
    native_evaluation_executor, monkeypatch
):
    """Instance callbacks cannot replace execution, validation, or persistence authority."""
    executor = native_evaluation_executor
    runner = _runner(executor)

    def forbidden(*args, **kwargs):
        raise AssertionError("instance replacement reached a trust boundary")

    for name in (
        "execute_scheduled",
        "validate_scheduled_record",
        "_ref",
    ):
        monkeypatch.setattr(executor, name, forbidden)
    for name in (
        "_schedule",
        "_attempt",
        "_validate_record",
        "_records",
        "_attempts",
        "_terminal",
        "_put",
        "_one_artifact",
    ):
        monkeypatch.setattr(runner, name, forbidden)

    result = runner.run("D", "development", 3)
    assert result.executed_units == 3 and result.stopped_reason is None


def test_executor_exposes_no_unleased_helper_or_wrapped_path(native_evaluation_executor):
    executor = native_evaluation_executor
    assert not hasattr(executor, "_unit_transaction")
    assert not hasattr(executor, "_EvaluationExecutor__execute_locked")
    assert not hasattr(executor, "_execute_leased")
    assert not hasattr(type(executor).execute_scheduled, "__wrapped__")
    assert not hasattr(type(executor)._execute, "__wrapped__")


def test_concurrent_direct_execution_dispatches_one_physical_diagnosis(
    native_evaluation_executor, monkeypatch
):
    from gpu_agent.benchmark.schedule_authority import activate_schedule, seal_schedule
    from gpu_agent.service import ApplicationService

    executor = native_evaluation_executor
    runner = _runner(executor)
    schedule = runner._schedule("D", "development", 3)
    store = executor.service.store
    run = store.create_run("evaluation", binding=executor.service.binding)
    runner._put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode())
    seal_schedule(
        executor._corpus_family,
        store,
        run.id,
        schedule,
        executor.service.binding,
        schedule_client_for_test(executor),
        executor._schedule_verifier,
    )
    activate_schedule(store, executor._schedule_verifier, run.id)
    attempt = runner._attempt(run.id, schedule, schedule.items[0])
    runner._put(run.id, "evaluation/attempts/0.json", attempt.model_dump_json().encode())

    native_diagnose = ApplicationService.diagnose
    guard = Lock()
    started = Event()
    release = Event()
    calls = 0

    def blocking_diagnose(self, *args, **kwargs):
        nonlocal calls
        with guard:
            calls += 1
        started.set()
        release.wait(2)
        return native_diagnose(self, *args, **kwargs)

    monkeypatch.setattr(ApplicationService, "diagnose", blocking_diagnose)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(type(executor).execute_scheduled, executor, run.id, 0)
        assert started.wait(1)
        second = pool.submit(type(executor)._execute, executor, run.id, 0)
        release.set()
        record = first.result()
        with pytest.raises(ValueError, match="execution state"):
            second.result()
    assert record.lineage.diagnosis_run_id
    assert calls == 1


def test_crash_after_physical_diagnosis_never_redispatches_ordinal(
    native_evaluation_executor, monkeypatch
):
    from gpu_agent.service import ApplicationService

    executor = native_evaluation_executor
    native_diagnose = ApplicationService.diagnose
    native_diagnosis = ApplicationService.diagnosis
    calls = 0

    def count_diagnose(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return native_diagnose(self, *args, **kwargs)

    def crash_after_diagnosis(self, run_id):
        native_diagnosis(self, run_id)
        raise KeyboardInterrupt

    monkeypatch.setattr(ApplicationService, "diagnose", count_diagnose)
    monkeypatch.setattr(ApplicationService, "diagnosis", crash_after_diagnosis)
    runner = _runner(executor)
    with pytest.raises(KeyboardInterrupt):
        runner.run("D", "development", 3)
    run_id = executor.service.store.recoverable_runs()[0].id
    monkeypatch.setattr(ApplicationService, "diagnosis", native_diagnosis)
    result = _runner(executor).resume(run_id, "D", "development", 3)
    assert result.stopped_reason == "AMBIGUOUS_STARTED_ATTEMPT"
    assert result.executed_units == 0
    assert calls == 1
