import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from schedule_authority_support import TestScheduleCommitClient, schedule_client_for_test

from gpu_agent.benchmark.evaluation import EvaluationRunner, EvaluationUnitBinding
from gpu_agent.benchmark.schedule_authority import (
    EvaluationScheduleReceipt,
    EvaluationScheduleVerifier,
    activate_schedule,
    build_signing_request,
    seal_schedule,
)
from gpu_agent.contracts import CurrentPhase, RunStatus


def _runner(executor, *, client=True):
    binding = executor.service.binding
    assert binding is not None
    return EvaluationRunner(
        executor.service.store,
        executor,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
        random_seed=7,
        schedule_client=schedule_client_for_test(executor) if client else None,
    )


def _queued_schedule(executor):
    runner = _runner(executor)
    binding = executor.service.binding
    assert binding is not None
    schedule = runner._schedule("A", "development", 3)
    run = runner.store.create_run("evaluation", binding=binding)
    runner.store.put(
        run.id, "evaluation/schedule.json", schedule.model_dump_json().encode(), "public"
    )
    return runner, binding, schedule, run


def test_signing_request_requires_queued_parent(native_evaluation_executor):
    executor = native_evaluation_executor
    runner, binding, schedule, run = _queued_schedule(executor)
    seal_schedule(
        executor._corpus_family,
        runner.store,
        run.id,
        schedule,
        binding,
        schedule_client_for_test(executor),
        executor._schedule_verifier,
    )
    activate_schedule(runner.store, executor._schedule_verifier, run.id)

    with pytest.raises(ValueError, match="QUEUED"):
        build_signing_request(executor._corpus_family, runner.store, run.id, schedule, binding)


def test_queued_evaluation_rejects_children_and_running_without_receipt(
    native_evaluation_executor,
):
    executor = native_evaluation_executor
    runner, _binding, _schedule, run = _queued_schedule(executor)

    with pytest.raises(ValueError, match="signed schedule activation"):
        runner.store.transition(run.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)
    with pytest.raises(ValueError, match="QUEUED evaluation"):
        runner.store.create_run("diagnosis", parent_run_id=run.id)
    ordinary = runner.store.create_run("development")
    child = runner.store.create_run("diagnosis", parent_run_id=ordinary.id)
    assert child.parent_run_id == ordinary.id


def test_arbitrary_receipt_cannot_activate_evaluation(native_evaluation_executor):
    runner, _binding, _schedule, run = _queued_schedule(native_evaluation_executor)
    runner.store.put(run.id, "evaluation/schedule-receipt.json", b'{"state":"COMMITTED"}', "public")

    with pytest.raises(ValueError, match="signed schedule activation"):
        runner.store.transition(run.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)


def test_resume_recovers_queued_parent_before_receipt(native_evaluation_executor):
    executor = native_evaluation_executor
    runner, _binding, schedule, run = _queued_schedule(executor)

    result = runner.resume(run.id, "A", "development", 3)

    assert result.executed_units == len(schedule.items)
    assert runner.store.load(run.id).status == RunStatus.COMPLETED


def test_resume_recovers_external_commit_before_receipt_persistence(
    native_evaluation_executor,
):
    executor = native_evaluation_executor
    runner, _binding, schedule, run = _queued_schedule(executor)
    delegate = schedule_client_for_test(executor)

    class CrashAfterCommit:
        def commit(self, request):
            delegate.commit(request)
            raise RuntimeError("simulated controller crash after external commit")

    runner.schedule_client = CrashAfterCommit()
    with pytest.raises(RuntimeError, match="after external commit"):
        runner.resume(run.id, "A", "development", 3)
    assert runner.store.load(run.id).status == RunStatus.QUEUED
    assert not [
        ref
        for ref in runner.store.load(run.id).artifact_refs
        if ref.name == "evaluation/schedule-receipt.json"
    ]
    assert not runner.store.children(run.id)

    runner.schedule_client = delegate
    result = runner.resume(run.id, "A", "development", 3)
    assert result.executed_units == len(schedule.items)


def test_application_service_rejects_unclaimed_evaluation_unit(native_evaluation_executor):
    executor = native_evaluation_executor
    runner, binding, schedule, run = _queued_schedule(executor)
    seal_schedule(
        executor._corpus_family,
        runner.store,
        run.id,
        schedule,
        binding,
        schedule_client_for_test(executor),
        executor._schedule_verifier,
    )
    activate_schedule(runner.store, executor._schedule_verifier, run.id)
    attempt = runner._attempt(run.id, schedule, schedule.items[0])
    item = schedule.items[0]
    unit = EvaluationUnitBinding(
        evaluation_run_id=run.id,
        ordinal=item.ordinal,
        schedule_hash=attempt.schedule_hash,
        idempotency_key=attempt.idempotency_key,
        reserved_cost_usd=attempt.reserved_cost_usd,
        case_id=item.case_id,
        template_id=item.template_id,
        mode=item.mode,
        repeat=item.repeat,
        split=item.split,
        holdout_proof=item.holdout_proof,
    )

    with pytest.raises(ValueError, match="attempts/0|claimed"):
        executor.service.diagnose(
            executor.sources[item.case_id],
            mode=item.mode,
            evaluation_unit=unit,
        )


def test_application_service_rejects_queued_parent_without_receipt(
    native_evaluation_executor,
):
    executor = native_evaluation_executor
    runner, _binding, schedule, run = _queued_schedule(executor)
    item = schedule.items[0]
    attempt = runner._attempt(run.id, schedule, item)
    unit = EvaluationUnitBinding(
        evaluation_run_id=run.id,
        ordinal=item.ordinal,
        schedule_hash=attempt.schedule_hash,
        idempotency_key=attempt.idempotency_key,
        reserved_cost_usd=attempt.reserved_cost_usd,
        case_id=item.case_id,
        template_id=item.template_id,
        mode=item.mode,
        repeat=item.repeat,
        split=item.split,
        holdout_proof=item.holdout_proof,
    )

    with pytest.raises(ValueError, match="authority artifacts"):
        executor.service.diagnose(
            executor.sources[item.case_id], mode=item.mode, evaluation_unit=unit
        )
    assert not runner.store.children(run.id)


def test_signed_request_binds_exact_prefix_and_zero_children(native_evaluation_executor):
    executor = native_evaluation_executor
    runner, binding, schedule, run = _queued_schedule(executor)
    request = build_signing_request(
        executor._corpus_family, runner.store, run.id, schedule, binding
    )

    assert request.event_prefix_count == 1
    assert request.artifact_prefix_count == 1
    assert request.child_count == 0
    assert request.child_inventory_hash != "0" * 64
    assert request.queued_manifest_hash != request.schedule_hash


def test_receipt_rejects_artifact_added_after_signing(native_evaluation_executor):
    executor = native_evaluation_executor
    runner, binding, schedule, run = _queued_schedule(executor)
    request = build_signing_request(
        executor._corpus_family, runner.store, run.id, schedule, binding
    )
    receipt = schedule_client_for_test(executor).commit(request)
    runner.store.put(run.id, "evaluation/uncommitted-extra.json", b"{}", "public")
    runner.store.put(
        run.id,
        "evaluation/schedule-receipt.json",
        receipt.model_dump_json().encode(),
        "public",
    )

    with pytest.raises(ValueError, match="changed after schedule signing|prefix"):
        executor._schedule_verifier.verify(run.id)


def test_resume_recovers_receipt_before_running_and_running_before_work(
    native_evaluation_executor,
):
    executor = native_evaluation_executor
    for activate_first in (False, True):
        runner, binding, schedule, run = _queued_schedule(executor)
        seal_schedule(
            executor._corpus_family,
            runner.store,
            run.id,
            schedule,
            binding,
            schedule_client_for_test(executor),
            executor._schedule_verifier,
        )
        if activate_first:
            activate_schedule(runner.store, executor._schedule_verifier, run.id)

        result = runner.resume(run.id, "A", "development", 3)

        assert result.executed_units == len(schedule.items)
        assert runner.store.load(run.id).status == RunStatus.COMPLETED


def test_service_rejects_extra_child_before_claimed_unit(native_evaluation_executor):
    executor = native_evaluation_executor
    runner, binding, schedule, run = _queued_schedule(executor)
    seal_schedule(
        executor._corpus_family,
        runner.store,
        run.id,
        schedule,
        binding,
        schedule_client_for_test(executor),
        executor._schedule_verifier,
    )
    activate_schedule(runner.store, executor._schedule_verifier, run.id)
    attempt = runner._attempt(run.id, schedule, schedule.items[0])
    runner._put(run.id, "evaluation/attempts/0.json", attempt.model_dump_json().encode())
    runner.store.create_run("diagnosis", parent_run_id=run.id)

    with pytest.raises(ValueError, match="unexpected child"):
        executor.execute_scheduled(run.id, 0)


def test_post_execution_schedule_sealing_is_rejected(native_evaluation_executor):
    executor = native_evaluation_executor
    runner = _runner(executor)
    manifest = runner.run("A", "development", 3)
    run = runner.store.load(manifest.run_id)
    schedule_ref = next(ref for ref in run.artifact_refs if ref.name == "evaluation/schedule.json")
    from gpu_agent.benchmark.evaluation import EvaluationSchedule

    schedule = EvaluationSchedule.model_validate_json(runner.store.read(schedule_ref))

    with pytest.raises(ValueError, match="QUEUED"):
        seal_schedule(
            executor._corpus_family,
            runner.store,
            run.id,
            schedule,
            runner.binding,
            schedule_client_for_test(executor),
            executor._schedule_verifier,
        )


def test_production_runner_hard_closes_without_external_signer(native_evaluation_executor):
    runner = _runner(native_evaluation_executor, client=False)
    before = list(runner.store.root.iterdir())
    with pytest.raises(ValueError, match="external schedule authority"):
        runner.run("A", "development", 3)
    assert list(runner.store.root.iterdir()) == before


def test_test_receipt_is_explicit_and_production_verifier_rejects_it(
    native_evaluation_executor,
):
    executor = native_evaluation_executor
    manifest = _runner(executor).run("A", "development", 3)
    receipt = executor._schedule_verifier.verify(manifest.run_id)
    assert receipt.state == "COMMITTED"
    assert receipt.request.authority_profile == "TEST_ONLY"
    assert receipt.request.case_templates == {"case_0100": "vector-add"}
    with pytest.raises(ValueError, match="production schedule authority"):
        EvaluationScheduleVerifier.for_family(executor._corpus_family, executor.service.store)


def test_signed_receipt_is_run_bound_and_cannot_be_copied(native_evaluation_executor):
    executor = native_evaluation_executor
    manifest = _runner(executor).run("A", "development", 3)
    source = executor.service.store.load(manifest.run_id)
    copied = executor.service.store.create_run("evaluation", binding=source.binding)
    for name in ("evaluation/schedule.json", "evaluation/schedule-receipt.json"):
        ref = next(item for item in source.artifact_refs if item.name == name)
        executor.service.store.put(copied.id, name, executor.service.store.read(ref), "public")
    with pytest.raises(ValueError):
        executor._schedule_verifier.verify(copied.id)


def test_signature_and_subset_schedule_are_rejected(native_evaluation_executor):
    executor = native_evaluation_executor
    runner = _runner(executor)
    binding = executor.service.binding
    assert binding is not None
    schedule = runner._schedule("A", "development", 3)
    run = runner.store.create_run("evaluation", binding=binding)
    runner.store.put(
        run.id, "evaluation/schedule.json", schedule.model_dump_json().encode(), "public"
    )
    seal_schedule(
        executor._corpus_family,
        runner.store,
        run.id,
        schedule,
        binding,
        schedule_client_for_test(executor),
        executor._schedule_verifier,
    )
    receipt_ref = next(
        item
        for item in runner.store.load(run.id).artifact_refs
        if item.name == "evaluation/schedule-receipt.json"
    )
    receipt = EvaluationScheduleReceipt.model_validate_json(runner.store.read(receipt_ref))
    forged = runner.store.create_run("evaluation", binding=binding)
    forged_schedule = schedule.model_copy(
        update={
            "items": [
                item.model_copy(update={"ordinal": ordinal})
                for ordinal, item in enumerate(schedule.items[:-1])
            ]
        }
    )
    runner.store.put(
        forged.id,
        "evaluation/schedule.json",
        forged_schedule.model_dump_json().encode(),
        "public",
    )
    tampered = json.loads(receipt.model_dump_json())
    tampered["signature_hex"] = "0" * 128
    runner.store.put(
        forged.id,
        "evaluation/schedule-receipt.json",
        json.dumps(tampered).encode(),
        "public",
    )
    with pytest.raises(ValueError):
        executor._schedule_verifier.verify(forged.id)


def test_schedule_contains_entire_committed_universe(native_evaluation_executor):
    executor = native_evaluation_executor
    schedule = _runner(executor)._schedule("all", "development", 3)
    observed = {(item.case_id, item.template_id, item.mode, item.repeat) for item in schedule.items}
    expected = {
        ("case_0100", "vector-add", mode, repeat)
        for repeat in range(3)
        for mode in ("A", "B", "C", "D", "E")
    }
    assert observed == expected


def test_concurrent_exact_seal_and_crash_retry_are_idempotent(
    native_evaluation_executor, monkeypatch
):
    executor = native_evaluation_executor
    runner = _runner(executor)
    binding = executor.service.binding
    assert binding is not None
    schedule = runner._schedule("A", "development", 3)
    run = runner.store.create_run("evaluation", binding=binding)
    runner.store.put(
        run.id, "evaluation/schedule.json", schedule.model_dump_json().encode(), "public"
    )
    arguments = (
        executor._corpus_family,
        runner.store,
        run.id,
        schedule,
        binding,
        schedule_client_for_test(executor),
        executor._schedule_verifier,
    )
    original = runner.store.put_if_absent_exact
    crashed = False

    def crash_once(*args, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("simulated receipt persistence crash")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner.store, "put_if_absent_exact", crash_once)
    with pytest.raises(RuntimeError, match="persistence crash"):
        seal_schedule(*arguments)
    monkeypatch.setattr(runner.store, "put_if_absent_exact", original)
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda _: seal_schedule(*arguments), range(2)))
    assert receipts[0] == receipts[1]
    assert executor._schedule_verifier.verify(run.id) == receipts[0]


def test_family_pinned_key_rejects_self_signed_receipt(native_evaluation_executor, tmp_path):
    executor = native_evaluation_executor
    runner = _runner(executor)
    binding = executor.service.binding
    assert binding is not None
    schedule = runner._schedule("A", "development", 3)
    run = runner.store.create_run("evaluation", binding=binding)
    runner.store.put(
        run.id, "evaluation/schedule.json", schedule.model_dump_json().encode(), "public"
    )
    attacker = TestScheduleCommitClient.create(tmp_path / "attacker")
    with pytest.raises(ValueError, match="signature"):
        seal_schedule(
            executor._corpus_family,
            runner.store,
            run.id,
            schedule,
            binding,
            attacker,
            executor._schedule_verifier,
        )
