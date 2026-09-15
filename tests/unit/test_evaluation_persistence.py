import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

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
        "case_ids": {"case_0100": "vector-add"},
        "executor": executor,
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

    executor = native_evaluation_executor
    runner = _runner(executor)
    store = executor.service.store
    put = store.put

    def interrupt_after_first_record(run_id, name, content, visibility):
        result = put(run_id, name, content, visibility)
        if name == "evaluation/records/0.json":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(store, "put", interrupt_after_first_record)
    with pytest.raises(KeyboardInterrupt):
        runner.run("D", "development", 3)
    run_id = store.recoverable_runs()[0].id
    monkeypatch.setattr(store, "put", put)
    restarted = EvaluationExecutor(
        executor.service,
        executor.corpus,
        executor.sources,
        _corpus_family=executor._corpus_family,
    )
    result = _runner(restarted).resume(run_id, "D", "development", 3)
    assert [record.repeat for record in result.records] == [2, 0, 1]
    assert result.executed_units == 3 and result.stopped_reason is None


def test_record_serialization_failure_persists_failed_terminal_manifest(
    native_evaluation_executor, monkeypatch
):
    executor = native_evaluation_executor
    runner = _runner(executor)
    put = runner._put

    def fail_record(run_id, name, content):
        if name == "evaluation/records/0.json":
            raise TypeError("serialization failed")
        return put(run_id, name, content)

    monkeypatch.setattr(runner, "_put", fail_record)
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
    store.transition(run.id, RunStatus.RUNNING, "EXECUTING")
    runner._put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode())
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleAuthority

    authority = EvaluationScheduleAuthority.for_family(executor._corpus_family, store)
    authority.seal(store, run.id, schedule, executor.service.binding)
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
        store.transition(source_run.id, RunStatus.RUNNING, "EXECUTING")
        runner._put(
            source_run.id,
            "evaluation/schedule.json",
            schedule.model_dump_json().encode(),
        )
        authority.seal(store, source_run.id, schedule, executor.service.binding)
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
            {"case_0100": "vector-add"},
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
            {"case_0100": "vector-add"},
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
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleAuthority

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
    store.transition(run.id, RunStatus.RUNNING, "EXECUTING")
    first_runner._put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode())
    EvaluationScheduleAuthority.for_family(executor._corpus_family, store).seal(
        store, run.id, schedule, executor.service.binding
    )
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
    assert (executor.service.store.root / f".evaluation-execution-{attempt.run_id}.lock").is_file()
