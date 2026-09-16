import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from schedule_authority_support import TestScheduleCommitClient, schedule_client_for_test

from gpu_agent.benchmark.evaluation import EvaluationRunner
from gpu_agent.benchmark.schedule_authority import (
    EvaluationScheduleReceipt,
    EvaluationScheduleVerifier,
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
    executor.service.store.transition(copied.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)
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
