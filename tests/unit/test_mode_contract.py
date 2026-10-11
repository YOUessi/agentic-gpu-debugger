"""Offline end-to-end checks of the A-E mode contract (docs/mode-contract.md).

Every mode runs through the real scheduler, executor, native replay validator, patch and
verification plumbing with a zero-cost mock provider and an offline container boundary.
"""

import pytest
from responses_support import _configure_responses_provider
from schedule_authority_support import schedule_client_for_test


def _run(executor, mode, repeats=3):  # the schedule minimum is 3 repeats
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    binding = executor.service.binding
    return EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version,
        toolchain_hash=binding.toolchain_lock_hash,
        model_config_hash=binding.model_config_hash,
        binding=binding,
        max_cost_usd=1000,
        max_unit_cost_usd=1,
        random_seed=7,
    ).run(mode, "development", repeats)


def _prompts_sent(store, run_id):
    from gpu_agent.agent.provider import Invocation

    run = store.load(run_id)
    return [
        Invocation.model_validate_json(store.read(ref)).kind
        for ref in run.artifact_refs
        if ref.name.startswith("provider/") and ref.name.endswith("/COMPLETED.json")
    ]


def test_record_only_policy_keeps_nonzero_costs_without_stopping(native_evaluation_executor):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding = executor.service.binding.model_copy(update={"cost_policy": "record_only"})
    executor.service._binding = binding
    runner = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version,
        toolchain_hash=binding.toolchain_lock_hash,
        model_config_hash=binding.model_config_hash,
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
    )
    result = runner.run("A", "development", 3)
    assert result.executed_units == 3 and result.stopped_reason is None
    assert all(record.cost_usd > 0 for record in result.records)
    assert executor.service.store.load(result.run_id).binding.cost_policy == "record_only"


def test_release_evaluation_accepts_native_claims_and_completions(
    native_evaluation_executor, monkeypatch, tmp_path
):
    from gpu_agent.benchmark.release import ReleaseEvidenceRoots, _ReleaseEvidenceResolver
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier
    from gpu_agent.service import ApplicationService

    executor = native_evaluation_executor
    binding = executor.service.binding.model_copy(update={"runtime_code_hash": "9" * 64})
    executor.service._binding = binding
    # This fixture has simulated GPU/model evidence, not a release checkout. Only
    # the code-attestation boundary is stubbed; artifact production/validation is native.
    monkeypatch.setattr(ApplicationService, "_attest_runtime_code", lambda self: "9" * 64)
    monkeypatch.setattr(
        EvaluationScheduleVerifier,
        "for_family",
        classmethod(lambda cls, family, store: executor._schedule_verifier),
    )
    result = _run(executor, "all")
    assert result.executed_units == 15 and result.stopped_reason is None
    resolver = _ReleaseEvidenceResolver(
        ReleaseEvidenceRoots(
            development_evaluation_run_id=result.run_id,
            holdout_evaluation_run_id="1" * 32,
            private_binding_run_id="2" * 32,
            release_test_run_id="3" * 32,
        ),
        executor.service.store,
        executor.service.evaluator_store,
        executor._corpus_family,
        tmp_path,
        binding.repository,
    )
    evidence = resolver._evaluation(result.run_id, "development")
    assert len(evidence.records) == 15


@pytest.mark.parametrize("mode", ["A", "B", "C", "D", "E"])
def test_every_mode_diagnoses_patches_and_verifies_with_the_shared_model(
    native_evaluation_executor, mode
):
    executor = native_evaluation_executor
    result = _run(executor, mode)
    assert result.stopped_reason is None and result.executed_units == 3
    for record in result.records:
        assert record.status == "COMPLETED" and record.failure_reason is None
        assert record.diagnosis["diagnostic_outcome"] == "DIAGNOSED"
        assert record.patch_hash is not None and record.verdict is not None
        assert record.lineage.verification_run_id is not None
        assert record.cost_usd is not None and record.cost_usd > 0
        kinds = _prompts_sent(executor.service.store, record.record_id)
        assert kinds.count("diagnose") == 1 and kinds.count("patch") == 1
        assert ("plan" in kinds) == (mode == "E")


def test_modes_d_and_e_share_diagnosis_and_patch_evidence_contract(native_evaluation_executor):
    executor = native_evaluation_executor
    d = _run(executor, "D").records[0]
    e = _run(executor, "E").records[0]
    # Same acquisition route here, so the whole diagnosis layer is identical apart from IDs.
    assert d.diagnosis["failure_family"] == e.diagnosis["failure_family"]
    assert d.diagnosis["source_locations"] == e.diagnosis["source_locations"]
    assert d.executed_checks.keys() == e.executed_checks.keys()


@pytest.mark.parametrize("mode", ["A", "D"])
def test_diagnosis_provider_failure_is_a_failed_record_and_the_batch_continues(
    native_evaluation_executor, monkeypatch, mode
):
    executor = native_evaluation_executor
    _configure_responses_provider(
        executor, monkeypatch, full_script=True, invalid_kinds=frozenset({"diagnose"})
    )
    result = _run(executor, mode)
    assert result.stopped_reason is None and result.executed_units == 3
    assert {record.status for record in result.records} == {"FAILED"}
    assert {record.failure_reason for record in result.records} == {"LLM_INVALID_OUTPUT"}
    assert all(record.patch_hash is None for record in result.records)


def test_patch_provider_failure_keeps_the_diagnosis_completed(
    native_evaluation_executor, monkeypatch
):
    executor = native_evaluation_executor
    _configure_responses_provider(
        executor, monkeypatch, full_script=True, invalid_kinds=frozenset({"patch"})
    )
    result = _run(executor, "C")
    assert result.stopped_reason is None and result.executed_units == 3
    for record in result.records:
        assert record.status == "COMPLETED" and record.failure_reason is None
        assert record.patch_hash is None and record.verdict is None
        assert record.diagnosis["limitations"][-1] == "LLM_INVALID_OUTPUT"


def test_model_declared_inconclusive_is_an_inconclusive_record(
    native_evaluation_executor, monkeypatch
):
    executor = native_evaluation_executor
    executor.scripted_provider.force_limitation = True
    executor.scripted_provider.limitation_canary = "ANY_MODEL_TEXT"
    result = _run(executor, "B")
    assert {record.status for record in result.records} == {"INCONCLUSIVE"}
    assert {record.failure_reason for record in result.records} == {"MODEL_DECLARED_INCONCLUSIVE"}


def test_preparation_failure_is_a_failed_record(native_evaluation_executor, monkeypatch):
    from gpu_agent.execution.models import BackendInfrastructureError

    executor = native_evaluation_executor
    backend = executor.service._backend_factory

    def unavailable(self, *args, **kwargs):
        raise BackendInfrastructureError("runtime attestation container unavailable")

    monkeypatch.setattr(backend, "_attest_runtime", unavailable)
    result = _run(executor, "A")
    assert result.stopped_reason is None and result.executed_units == 3
    assert {record.failure_reason for record in result.records} == {
        "EXECUTION_INFRASTRUCTURE_UNAVAILABLE"
    }
    assert {record.status for record in result.records} == {"FAILED"}


def test_records_carry_and_replay_the_bound_runtime_code(native_evaluation_executor):
    from pathlib import Path

    import gpu_agent
    from gpu_agent.provenance import runtime_code_fingerprint

    executor = native_evaluation_executor
    root = Path(gpu_agent.__file__).resolve().parents[2]
    executor.service._binding = executor.service.binding.model_copy(
        update={"runtime_code_hash": runtime_code_fingerprint(root)}
    )
    executor.service._repository_root = root
    result = _run(executor, "D")
    assert result.stopped_reason is None and result.executed_units == 3
    store = executor.service.store
    for record in result.records:
        run = store.load(record.record_id)
        assert any(ref.name == "agent/runtime-code.json" for ref in run.artifact_refs)


def test_development_report_summarizes_modes_with_frozen_family_labels(
    native_evaluation_executor,
):
    from gpu_agent.benchmark.dev_report import load_labels, summarize

    executor = native_evaluation_executor
    records = _run(executor, "C").records
    labels = load_labels()
    assert set(labels) == {f"case_{index:04d}" for index in range(1, 17)}
    summary = summarize(records, {"case_0100": {"failure_family": "out_of_bounds"}})
    row = summary["modes"]["C"]
    assert row["units"] == 3 and row["status"] == {"COMPLETED": 3}
    assert row["family_correct"] == row["family_scored"] == 3


def test_diagnosis_family_is_a_frozen_vocabulary():
    import pydantic

    from gpu_agent.agent.models import FAILURE_FAMILIES, DiagnosisResult
    from gpu_agent.benchmark.dev_report import load_labels

    assert {item["failure_family"] for item in load_labels().values()} <= set(FAILURE_FAMILIES)
    with pytest.raises(pydantic.ValidationError):
        DiagnosisResult(diagnostic_outcome="DIAGNOSED", failure_family="memory bug")


def test_planner_gets_one_replan_with_reason_codes_after_a_denial(
    native_evaluation_executor,
):
    from gpu_agent.agent.models import FinishAction, MemcheckAction, RetrieveDocsAction

    executor = native_evaluation_executor
    # Duplicate memcheck is denied; the replan sees the reason and proceeds.
    executor.scripted_provider.actions = [
        MemcheckAction(),
        MemcheckAction(),
        RetrieveDocsAction(typed_arguments={"query": "out of bounds", "k": 3}),
        FinishAction(),
    ]
    result = _run(executor, "E")
    assert result.stopped_reason is None and result.executed_units == 3
    record = result.records[0]
    assert record.status == "COMPLETED" and record.patch_hash is not None
    store = executor.service.store
    run = store.load(record.record_id)
    decisions = sorted(
        (ref for ref in run.artifact_refs if ref.name.endswith("/decision.json")),
        key=lambda ref: int(ref.name.split("/")[1]),
    )
    allowed = [__import__("json").loads(store.read(ref))["allowed"] for ref in decisions]
    assert allowed == [True, False, True, True]
