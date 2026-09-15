import json

import pytest

from gpu_agent.benchmark.evaluation import EvaluationRecord, EvaluationRunner
from gpu_agent.contracts import RunStatus

COMMIT = "c" * 40
TOOLCHAIN_HASH = "d" * 64
MODEL_CONFIG_HASH = "e" * 64


def _record(
    case: str, template: str, mode: str, repeat: int, cost: float | None
) -> EvaluationRecord:
    return EvaluationRecord(
        record_id=f"{case}-{mode}-{repeat}",
        case_id=case,
        template_id=template,
        mode=mode,
        repeat=repeat,
        input_hash="a" * 64,
        evidence_hash="b" * 64,
        executed_checks={"memcheck": "CLEAN"},
        status="COMPLETED",
        diagnosis={},
        latency_ms=1,
        cost_usd=cost,
    )


def _runner(store, execute, **overrides) -> EvaluationRunner:
    options = {
        "store": store,
        "case_ids": {"case_0001": "index"},
        "execute": execute,
        "commit": COMMIT,
        "prompt_version": "v2",
        "toolchain_hash": TOOLCHAIN_HASH,
        "model_config_hash": MODEL_CONFIG_HASH,
        "max_cost_usd": 3.0,
        "max_unit_cost_usd": 1.0,
        "random_seed": 7,
    }
    options.update(overrides)
    return EvaluationRunner(**options)


def _artifact(store, run_id: str, name: str) -> bytes:
    ref = next(ref for ref in store.load(run_id).artifact_refs if ref.name == name)
    return store.read(ref)


def test_record_is_durable_before_next_unit(store):
    completed = []

    def execute(case, template, mode, repeat):
        active = store.recoverable_runs()
        if completed:
            persisted = EvaluationRecord.model_validate_json(
                _artifact(store, active[0].id, "evaluation/records/0.json")
            )
            assert persisted == completed[0]
        record = _record(case, template, mode, repeat, 1.0)
        completed.append(record)
        return record

    result = _runner(store, execute).run("E", "development", 3)

    run = store.load(result.run_id)
    names = {ref.name for ref in run.artifact_refs}
    assert {"evaluation/schedule.json", "evaluation/manifest.json"} <= names
    assert {
        "evaluation/records/0.json",
        "evaluation/records/1.json",
        "evaluation/records/2.json",
    } <= names
    manifest = json.loads(_artifact(store, result.run_id, "evaluation/manifest.json"))
    assert manifest["executed_units"] == 3
    assert run.status == RunStatus.COMPLETED


def test_unit_reservation_stops_before_cost_cap_can_be_exceeded(store):
    result = _runner(
        store,
        lambda case, template, mode, repeat: _record(case, template, mode, repeat, 0.6),
        max_cost_usd=1.0,
        max_unit_cost_usd=0.6,
    ).run("E", "development", 3)

    assert result.stopped_reason == "COST_CAP_RESERVATION_REQUIRED"
    assert result.executed_units == 1
    assert EvaluationRecord.model_validate_json(
        _artifact(store, result.run_id, "evaluation/records/0.json")
    ).cost_usd == 0.6
    assert store.load(result.run_id).status == RunStatus.COMPLETED


def test_unexpected_executor_failure_preserves_completed_records(store):
    completed = []

    def execute(case, template, mode, repeat):
        if completed:
            raise RuntimeError("executor died")
        record = _record(case, template, mode, repeat, 1.0)
        completed.append(record)
        return record

    result = _runner(store, execute).run("E", "development", 3)

    assert result.stopped_reason == "EXECUTION_ERROR"
    assert result.executed_units == 1
    assert EvaluationRecord.model_validate_json(
        _artifact(store, result.run_id, "evaluation/records/0.json")
    ) == completed[0]
    assert "evaluation/records/1.json" not in {
        ref.name for ref in store.load(result.run_id).artifact_refs
    }
    assert store.load(result.run_id).status == RunStatus.FAILED


def test_resume_rejects_commit_or_schedule_mismatch(store):
    def interrupt(*_args):
        raise KeyboardInterrupt

    original = _runner(store, interrupt)
    with pytest.raises(KeyboardInterrupt):
        original.run("E", "development", 3)
    commit_run_id = store.recoverable_runs()[0].id

    with pytest.raises(ValueError):
        _runner(store, interrupt, commit="d" * 40).resume(commit_run_id, "E", "development", 3)

    with pytest.raises(KeyboardInterrupt):
        original.run("E", "development", 3)
    schedule_run_id = next(
        run.id for run in store.recoverable_runs() if run.id != commit_run_id
    )

    with pytest.raises(ValueError):
        _runner(store, interrupt, case_ids={"case_0002": "race"}).resume(
            schedule_run_id, "E", "development", 3
        )
