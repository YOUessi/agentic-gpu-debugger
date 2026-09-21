"""Evaluator canaries never cross public store, report, source, or provider projections."""

import json

import pytest
from schedule_authority_support import schedule_client_for_test


@pytest.mark.release_evidence
def test_private_canary_is_absent_from_five_public_channels(tmp_path):
    from gpu_agent.agent.models import PublicEvidence, PublicSource
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.models import WorkspaceRequest
    from gpu_agent.reporting import ReportExporter
    from gpu_agent.store import RunStore

    canary = b"PRIVATE-HOLDOUT-CANARY-7f3a"
    public = RunStore(tmp_path / "public")
    evaluator = RunStore(tmp_path / "evaluator", visibility="evaluator")
    public_run = public.create_run("diagnosis")
    private_run = evaluator.create_run("holdout")
    private_ref = evaluator.put(private_run.id, "suite/canary.json", canary, "evaluator")

    with pytest.raises(ValueError):
        public.read(private_ref)
    assert private_ref.id.encode() not in ReportExporter(public).public(public_run.id)

    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = b"__global__ void kernel() {}\n"
    (source_root / "kernel.cu").write_bytes(source)
    backend = IsolatedGPUBackend(public, source_root, tmp_path / "tasks")
    with pytest.raises(ValueError):
        backend.prepare(
            WorkspaceRequest(
                run_id=public_run.id,
                source_manifest={"../evaluator/canary.json": "0" * 64},
            )
        )

    projection = (
        PublicEvidence(sources=[PublicSource(source_id="0" * 32, content=source.decode())])
        .model_dump_json()
        .encode()
    )
    assert canary not in projection
    assert private_ref.id.encode() not in projection

    next_run = public.create_run("next-diagnosis")
    assert all(canary not in public.read(ref) for ref in public.load(next_run.id).artifact_refs)
    assert canary in evaluator.read(private_ref)
    assert canary not in json.dumps(public.load(public_run.id).model_dump(mode="json")).encode()


@pytest.mark.parametrize("native_evaluation_executor", ["private_split"], indirect=True)
def test_holdout_execution_has_zero_private_bytes_in_public_store(
    private_split_executor, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner, HoldoutEvaluationLineage

    executor = private_split_executor
    binding = executor.service.binding
    assert binding is not None
    result = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash,
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
        random_seed=7,
        holdout_controller=executor.holdout_controller,
        holdout_batch=executor.holdout_batch,
    ).run("D", "holdout", 3)
    assert result.executed_units == 3
    assert all(isinstance(record.lineage, HoldoutEvaluationLineage) for record in result.records)
    public_bytes = b"\n".join(
        path.read_bytes()
        for path in sorted(executor.service.store.root.rglob("*"))
        if path.is_file()
    )
    private_canaries = (
        b"PRIVATE-SOURCE-CANARY-task3-a91e",
        b"case_0100",
        b"vector-add",
        str(executor.holdout_service.store.root).encode(),
    )
    for canary in private_canaries:
        assert canary not in public_bytes
    evaluator_bytes = b"\n".join(
        path.read_bytes()
        for path in sorted(executor.holdout_service.store.root.rglob("*"))
        if path.is_file()
    )
    assert b"PRIVATE-SOURCE-CANARY-task3-a91e" in evaluator_bytes
    assert b"case_0100" in evaluator_bytes
    assert b"vector-add" in evaluator_bytes
    for record in result.records:
        wire = record.model_dump_json().encode()
        assert b"case_0100" not in wire
        assert b"vector-add" not in wire
