"""Adversarial recovery checks for the immutable corpus cutoff."""

import hashlib

import pytest
from schedule_authority_support import schedule_client_for_test

from gpu_agent.benchmark.evaluation import (
    EvaluationAttempt,
    EvaluationRunner,
    EvaluationSchedule,
    PublicEvaluationRecord,
)
from gpu_agent.benchmark.holdout import HoldoutController
from gpu_agent.benchmark.ledger import CorpusFamily


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


def _append_opposite_visibility_commit(family: CorpusFamily, visibility: str) -> int:
    """Advance the global ledger without changing the selected split's native store."""
    store = family.corpus_store(visibility)  # type: ignore[arg-type]
    marker = str(len(family.ledger.committed_through()) + 1).encode()
    transaction = family.ledger.prepare(
        b"future-case-" + marker,
        b"future-template-" + marker,
        b"future-source-pair-" + marker,
        store=store,
        manifest_hash=hashlib.sha256(b"future-manifest-" + marker).hexdigest(),
    )
    committed = family.ledger.commit(transaction)
    assert committed.commit_sequence is not None
    return committed.commit_sequence


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

    _append_opposite_visibility_commit(executor._corpus_family, "evaluator")
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

    _append_opposite_visibility_commit(executor._corpus_family, "public")
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
