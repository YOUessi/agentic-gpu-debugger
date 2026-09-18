"""Adversarial recovery checks for the immutable corpus cutoff."""

import shutil

import pytest
from schedule_authority_support import schedule_client_for_test

from gpu_agent.benchmark.evaluation import (
    EvaluationAttempt,
    EvaluationRunner,
    EvaluationSchedule,
    PublicEvaluationRecord,
)
from gpu_agent.benchmark.holdout import HoldoutController


def _runner(executor, **overrides) -> EvaluationRunner:
    binding = executor.service.binding
    assert binding is not None
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


def test_development_resume_uses_signed_cutoff_after_later_commit(
    native_evaluation_executor, monkeypatch
):
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.store import RunStore

    executor = native_evaluation_executor
    runner = _runner(executor)
    native_put = RunStore.put

    def interrupt_after_first_record(self, run_id, name, content, visibility):
        result = native_put(self, run_id, name, content, visibility)
        if name == "evaluation/records/0.json":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(RunStore, "put", interrupt_after_first_record)
    with pytest.raises(KeyboardInterrupt):
        runner.run("D", "development", 3)
    run_id = executor.service.store.recoverable_runs()[0].id
    schedule = EvaluationSchedule.model_validate_json(
        _artifact(executor.service.store, run_id, "evaluation/schedule.json")
    )
    assert schedule.corpus_cutoff == 1

    future = executor._register_future_case_for_test()
    assert future.id == "case_0101"
    assert len(executor._corpus_family.ledger.committed_through()) == 2
    monkeypatch.setattr(RunStore, "put", native_put)
    restarted = EvaluationExecutor(
        executor.service,
        executor.corpus,
        executor.sources,
        _corpus_family=executor._corpus_family,
        _schedule_verifier=executor._schedule_verifier,
    )
    manifest = _runner(restarted).resume(run_id, "D", "development", 3)
    assert manifest.corpus_cutoff == schedule.corpus_cutoff
    assert manifest.executed_units == 3


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_holdout_score_uses_alias_cutoff_after_later_commit(native_evaluation_executor):
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.benchmark.metrics import EvaluationLabels, Score, aggregate

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None
    controller = HoldoutController(
        executor.service.store,
        executor.corpus,
        binding=binding,
        _schedule_verifier=executor._schedule_verifier,
    )
    batch = controller.prepare()
    assert batch.corpus_cutoff == 1
    holdout_executor = EvaluationExecutor(
        executor.service,
        executor.corpus,
        executor.sources,
        holdout_controller=controller,
        holdout_batch=batch,
        _corpus_family=executor._corpus_family,
        _schedule_verifier=executor._schedule_verifier,
    )
    manifest = _runner(
        holdout_executor,
        max_cost_usd=3,
        max_unit_cost_usd=1,
        holdout_controller=controller,
        holdout_batch=batch,
    ).run("D", "holdout", 3)
    run = executor.service.store.load(manifest.run_id)
    record_ref = next(ref for ref in run.artifact_refs if ref.name == "evaluation/records/0.json")

    future = executor._register_future_case_for_test()
    assert future.id == "case_0101"
    assert len(executor._corpus_family.ledger.committed_through()) == 2
    persisted = controller.bind_score(
        batch,
        batch.aliases[0],
        record_ref,
        labels=EvaluationLabels(),
        score=Score(
            family_correct=True,
            root_cause_correct=True,
            location_correct=True,
            inconclusive_correct=True,
        ),
        should_be_inconclusive=False,
        private_holdout_passed=True,
    )
    assert persisted.corpus_cutoff == batch.corpus_cutoff
    summary = aggregate(
        [persisted],
        public_store=executor.service.store,
        evaluator_store=executor.corpus,
        run_binding=binding,
        schedule_verifier=executor._schedule_verifier,
    )
    assert summary.record_count == 1
    second_ref = next(ref for ref in run.artifact_refs if ref.name == "evaluation/records/1.json")
    second = controller.bind_score(
        batch,
        batch.aliases[0],
        second_ref,
        labels=EvaluationLabels(),
        score=Score(
            family_correct=True,
            root_cause_correct=True,
            location_correct=True,
            inconclusive_correct=True,
        ),
        should_be_inconclusive=False,
        private_holdout_passed=True,
    )
    combined = aggregate(
        [persisted, second],
        public_store=executor.service.store,
        evaluator_store=executor.corpus,
        run_binding=binding,
        schedule_verifier=executor._schedule_verifier,
    )
    assert combined.record_count == 2
    assert combined.corpus_cutoff == batch.corpus_cutoff
    assert combined.evaluation_authority_id == manifest.run_id
    with pytest.raises(ValueError, match="one corpus cutoff"):
        aggregate(
            [persisted, second.model_copy(update={"corpus_cutoff": 2})],
            public_store=executor.service.store,
            evaluator_store=executor.corpus,
            run_binding=binding,
            schedule_verifier=executor._schedule_verifier,
        )
    with pytest.raises(ValueError, match="signed evaluation authority"):
        aggregate(
            [persisted, second.model_copy(update={"public_evaluation_run_id": "f" * 32})],
            public_store=executor.service.store,
            evaluator_store=executor.corpus,
            run_binding=binding,
            schedule_verifier=executor._schedule_verifier,
        )
    with pytest.raises(ValueError, match="score transaction"):
        controller._load_metric_record(
            persisted.model_copy(update={"corpus_cutoff": persisted.corpus_cutoff + 1})
        )


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_mixed_or_future_cutoffs_fail_closed(native_evaluation_executor):
    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None
    controller = HoldoutController(
        executor.service.store,
        executor.corpus,
        binding=binding,
        _schedule_verifier=executor._schedule_verifier,
    )
    batch = controller.prepare()
    with pytest.raises(ValueError, match="cutoff|holdout binding"):
        controller.validate_batch(
            batch.model_copy(update={"corpus_cutoff": batch.corpus_cutoff + 1})
        )


def test_mixed_attempt_record_and_receipt_cutoffs_fail_closed(native_evaluation_executor):
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleReceipt

    executor = native_evaluation_executor
    manifest = _runner(executor).run("D", "development", 3)
    schedule = EvaluationSchedule.model_validate_json(
        _artifact(executor.service.store, manifest.run_id, "evaluation/schedule.json")
    )
    receipt = EvaluationScheduleReceipt.model_validate_json(
        _artifact(
            executor.service.store,
            manifest.run_id,
            "evaluation/schedule-receipt.json",
        )
    )
    attempt = EvaluationAttempt.model_validate_json(
        _artifact(executor.service.store, manifest.run_id, "evaluation/attempts/0.json")
    )
    record = PublicEvaluationRecord.model_validate_json(
        _artifact(executor.service.store, manifest.run_id, "evaluation/records/0.json")
    )
    assert {
        schedule.corpus_cutoff,
        receipt.request.corpus_cutoff,
        attempt.corpus_cutoff,
        record.corpus_cutoff,
        record.lineage.corpus_cutoff,
        manifest.corpus_cutoff,
    } == {1}
    with pytest.raises(ValueError, match="cutoff"):
        executor.validate_scheduled_record(
            record,
            schedule.items[0],
            attempt.model_copy(update={"corpus_cutoff": 2}),
        )
    with pytest.raises(ValueError, match="cutoff"):
        executor.validate_scheduled_record(
            record.model_copy(update={"corpus_cutoff": 2}),
            schedule.items[0],
            attempt,
        )
    with pytest.raises(ValueError, match="cutoff"):
        executor.validate_scheduled_record(
            record.model_copy(
                update={"lineage": record.lineage.model_copy(update={"corpus_cutoff": 2})}
            ),
            schedule.items[0],
            attempt,
        )


def test_unreserved_new_run_cannot_sign_a_stale_cutoff(native_evaluation_executor):
    from gpu_agent.benchmark.schedule_authority import seal_schedule

    executor = native_evaluation_executor
    runner = _runner(executor)
    binding = executor.service.binding
    assert binding is not None
    stale = runner._schedule("D", "development", 3)
    future = executor._register_future_case_for_test()
    assert future.id == "case_0101"
    run = runner.store.create_run("evaluation", binding=binding)
    runner._put(run.id, "evaluation/schedule.json", stale.model_dump_json().encode())
    with pytest.raises(ValueError, match="cutoff reservation"):
        seal_schedule(
            executor._corpus_family,
            runner.store,
            run.id,
            stale,
            binding,
            schedule_client_for_test(executor),
            executor._schedule_verifier,
        )


def test_reserved_pre_schedule_crash_recovers_at_old_cutoff(
    native_evaluation_executor, monkeypatch
):
    executor = native_evaluation_executor
    runner = _runner(executor)
    native_put = EvaluationRunner._put

    def crash_before_schedule(self, run_id, name, content):
        if name == "evaluation/schedule.json":
            raise KeyboardInterrupt
        return native_put(self, run_id, name, content)

    monkeypatch.setattr(EvaluationRunner, "_put", crash_before_schedule)
    with pytest.raises(KeyboardInterrupt):
        runner.run("D", "development", 3)
    run_id = next(
        path.name
        for path in runner.store.root.iterdir()
        if path.is_dir() and runner.store.load(path.name).kind == "evaluation"
    )
    reservation = executor._corpus_family.ledger.evaluation_cutoff_reservation(run_id)
    assert reservation.corpus_cutoff == 1 and reservation.schedule_hash is not None

    future = executor._register_future_case_for_test()
    assert future.id == "case_0101"
    monkeypatch.setattr(EvaluationRunner, "_put", native_put)
    manifest = runner.resume(run_id, "D", "development", 3)
    assert manifest.corpus_cutoff == 1
    assert manifest.executed_units == 3


def test_reservation_rejects_a_future_evaluation_run_id(native_evaluation_executor):
    from gpu_agent.benchmark.schedule_authority import reserve_evaluation_cutoff

    executor = native_evaluation_executor
    runner = _runner(executor)
    binding = executor.service.binding
    assert binding is not None
    with pytest.raises(ValueError):
        reserve_evaluation_cutoff(
            executor._corpus_family,
            runner.store,
            "f" * 32,
            binding,
            selection="D",
            modes=["D"],
            split="development",
            repeats=3,
            random_seed=runner.random_seed,
            max_cost_usd=runner.bindings.max_cost_usd,
            max_unit_cost_usd=runner.bindings.max_unit_cost_usd,
        )
    with pytest.raises(ValueError, match="unavailable"):
        executor._corpus_family.ledger.evaluation_cutoff_reservation("f" * 32)


def test_reservation_rejects_rebuilt_run_in_another_store(native_evaluation_executor, tmp_path):
    from gpu_agent.benchmark.schedule_authority import (
        bind_reserved_schedule,
        rebuild_reserved_schedule,
        reserve_evaluation_cutoff,
    )
    from gpu_agent.store import RunStore

    executor = native_evaluation_executor
    runner = _runner(executor)
    binding = executor.service.binding
    assert binding is not None
    run = runner.store.create_run("evaluation", binding=binding)
    reserve_evaluation_cutoff(
        executor._corpus_family,
        runner.store,
        run.id,
        binding,
        selection="D",
        modes=["D"],
        split="development",
        repeats=3,
        random_seed=runner.random_seed,
        max_cost_usd=runner.bindings.max_cost_usd,
        max_unit_cost_usd=runner.bindings.max_unit_cost_usd,
    )
    schedule = rebuild_reserved_schedule(executor._corpus_family, runner.store, run.id, binding)
    copied = RunStore(tmp_path / "copied-evaluation-store")
    copied.create_run("evaluation", binding=binding, _run_id=run.id)
    with pytest.raises(ValueError, match="identity|prestate"):
        bind_reserved_schedule(executor._corpus_family, copied, run.id, schedule, binding)


def test_wrong_schedule_cannot_win_or_poison_first_hash_cas(native_evaluation_executor):
    from gpu_agent.benchmark.schedule_authority import (
        bind_reserved_schedule,
        rebuild_reserved_schedule,
        reserve_evaluation_cutoff,
    )

    executor = native_evaluation_executor
    runner = _runner(executor)
    binding = executor.service.binding
    assert binding is not None
    run = runner.store.create_run("evaluation", binding=binding)
    reserve_evaluation_cutoff(
        executor._corpus_family,
        runner.store,
        run.id,
        binding,
        selection="D",
        modes=["D"],
        split="development",
        repeats=3,
        random_seed=runner.random_seed,
        max_cost_usd=runner.bindings.max_cost_usd,
        max_unit_cost_usd=runner.bindings.max_unit_cost_usd,
    )
    correct = rebuild_reserved_schedule(executor._corpus_family, runner.store, run.id, binding)
    reordered = list(reversed(correct.items))
    wrong = correct.model_copy(
        update={
            "items": [
                item.model_copy(update={"ordinal": ordinal})
                for ordinal, item in enumerate(reordered)
            ]
        }
    )
    with pytest.raises(ValueError, match="schedule differs"):
        bind_reserved_schedule(executor._corpus_family, runner.store, run.id, wrong, binding)
    unbound = executor._corpus_family.ledger.evaluation_cutoff_reservation(run.id)
    assert unbound.schedule_hash is None

    future = executor._register_future_case_for_test()
    assert future.id == "case_0101"
    bound = bind_reserved_schedule(executor._corpus_family, runner.store, run.id, correct, binding)
    assert bound.schedule_hash == runner._schedule_hash(correct)


def test_copytree_replacement_cannot_reuse_a_reserved_run_inode(
    native_evaluation_executor, tmp_path
):
    from gpu_agent.benchmark.schedule_authority import (
        rebuild_reserved_schedule,
        reserve_evaluation_cutoff,
    )

    executor = native_evaluation_executor
    runner = _runner(executor)
    binding = executor.service.binding
    assert binding is not None
    run = runner.store.create_run("evaluation", binding=binding)
    reservation = reserve_evaluation_cutoff(
        executor._corpus_family,
        runner.store,
        run.id,
        binding,
        selection="D",
        modes=["D"],
        split="development",
        repeats=3,
        random_seed=runner.random_seed,
        max_cost_usd=runner.bindings.max_cost_usd,
        max_unit_cost_usd=runner.bindings.max_unit_cost_usd,
    )
    original = runner.store.root / run.id
    displaced = tmp_path / "displaced-original-run"
    original.rename(displaced)
    shutil.copytree(displaced, original)

    with pytest.raises(ValueError, match="prestate"):
        rebuild_reserved_schedule(executor._corpus_family, runner.store, run.id, binding)
    observed = executor._corpus_family.ledger.evaluation_cutoff_reservation(run.id)
    assert observed == reservation and observed.schedule_hash is None


def test_swap_after_lock_acquisition_cannot_create_a_reservation(
    native_evaluation_executor, tmp_path, monkeypatch
):
    from gpu_agent.benchmark.schedule_authority import reserve_evaluation_cutoff

    executor = native_evaluation_executor
    runner = _runner(executor)
    binding = executor.service.binding
    assert binding is not None
    run = runner.store.create_run("evaluation", binding=binding)
    ledger = executor._corpus_family.ledger
    original_locked_state = ledger._locked_state
    swapped = False

    def swap_run_then_lock_ledger():
        nonlocal swapped
        if not swapped:
            swapped = True
            canonical = runner.store.root / run.id
            displaced = tmp_path / "locked-original-run"
            canonical.rename(displaced)
            shutil.copytree(displaced, canonical)
        return original_locked_state()

    monkeypatch.setattr(ledger, "_locked_state", swap_run_then_lock_ledger)
    with pytest.raises(ValueError, match="identity|canonical"):
        reserve_evaluation_cutoff(
            executor._corpus_family,
            runner.store,
            run.id,
            binding,
            selection="D",
            modes=["D"],
            split="development",
            repeats=3,
            random_seed=runner.random_seed,
            max_cost_usd=runner.bindings.max_cost_usd,
            max_unit_cost_usd=runner.bindings.max_unit_cost_usd,
        )
    with pytest.raises(ValueError, match="unavailable"):
        ledger.evaluation_cutoff_reservation(run.id)
