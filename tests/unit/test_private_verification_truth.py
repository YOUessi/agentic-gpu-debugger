"""Private registered truth is resolved evaluator-side and never publicly projected."""

import json

import pytest
from schedule_authority_support import schedule_client_for_test


@pytest.mark.parametrize("native_evaluation_executor", ["private_split"], indirect=True)
def test_private_registered_case_runs_verification_without_public_truth(
    native_evaluation_executor,
    private_split_executor,
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.verification.truth import VerificationTruth, resolve_run_truth, resolve_truth

    executor = private_split_executor
    binding = executor.service.binding
    result = EvaluationRunner(
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
        holdout_controller=executor.holdout_controller,
        holdout_batch=executor.holdout_batch,
    ).run("D", "holdout", 3)
    assert result.stopped_reason is None and result.executed_units == 3
    evaluator = executor.holdout_service.store
    audits = []
    # Audit nodes are deliberately accessible only in the evaluator store.
    for path in evaluator.root.iterdir():
        if not path.is_dir() or len(path.name) != 32:
            continue
        run = evaluator.load(path.name)
        if run.kind == "verification_audit":
            audits.append(run)
    assert len(audits) == 3
    for audit in audits:
        ref = next(ref for ref in audit.artifact_refs if ref.name == "case.json")
        truth = VerificationTruth.model_validate_json(evaluator.read(ref))
        assert truth.case_id == "case_0100"
        assert resolve_truth(truth.source_hashes) is None
        assert resolve_run_truth(evaluator, audit.parent_run_id, truth.source_hashes) == truth
        changed = {**truth.source_hashes, "kernel.cu": "0" * 64}
        assert resolve_run_truth(evaluator, audit.parent_run_id, changed) is None
        spec = next(
            ref for ref in audit.artifact_refs if ref.name == "verification/suite-spec.json"
        )
        assert json.loads(evaluator.read(spec))["expected_child_count"] > 1
    public = result.model_dump_json()
    assert "case_0100" not in public
    assert "private_seed" not in public
    assert "PRIVATE-SOURCE-CANARY" not in public
