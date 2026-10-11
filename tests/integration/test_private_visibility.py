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
@pytest.mark.parametrize("mode", ["A", "B", "C", "D", "E"])
def test_holdout_execution_has_zero_private_bytes_in_public_store(
    private_split_executor, native_evaluation_executor, mode, monkeypatch
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner, HoldoutEvaluationLineage

    executor = private_split_executor
    if mode == "E":
        from unit.test_evaluation_modes import _configure_responses_provider

        from gpu_agent.agent.models import InconclusiveAction
        from gpu_agent.benchmark.executor import EvaluationExecutor
        from gpu_agent.benchmark.holdout import HoldoutController

        executor.scripted_provider.actions = [InconclusiveAction() for _ in range(3)]
        binding, _ = _configure_responses_provider(executor, monkeypatch, full_script=True)
        assert executor.holdout_service is not None
        executor.holdout_service._binding = binding
        executor.holdout_service._provider = None
        executor.holdout_service._pricing_attestation = executor.service._pricing_attestation
        controller = HoldoutController(
            executor.service.store,
            executor.corpus,
            binding=binding,
            _schedule_verifier=executor._schedule_verifier,
            _schedule_family=executor._corpus_family,
        )
        batch = controller.prepare()
        executor = EvaluationExecutor(
            executor.service,
            executor.corpus,
            executor.sources,
            holdout_service=executor.holdout_service,
            holdout_controller=controller,
            holdout_batch=batch,
            _corpus_family=executor._corpus_family,
            _schedule_verifier=executor._schedule_verifier,
        )
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
        max_cost_usd=3,
        max_unit_cost_usd=1,
        random_seed=7,
        holdout_controller=executor.holdout_controller,
        holdout_batch=executor.holdout_batch,
    ).run(mode, "holdout", 3)
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
    evaluator_runs = [
        executor.holdout_service.store.load(path.name)
        for path in executor.holdout_service.store.root.iterdir()
        if path.is_dir() and len(path.name) == 32
    ]
    evaluator_private_ids = [
        run.id.encode() for run in evaluator_runs if run.kind in {"holdout_execution", "diagnosis"}
    ]
    mapping = next(run for run in evaluator_runs if run.kind == "holdout_alias_mapping")
    mapping_ref = next(
        ref for ref in mapping.artifact_refs if ref.name == "holdout/private-alias-map.json"
    )
    nonce = json.loads(executor.holdout_service.store.read(mapping_ref))["nonce_hex"].encode()
    for canary in [nonce, *evaluator_private_ids]:
        assert canary not in public_bytes
    for record in result.records:
        wire = record.model_dump_json().encode()
        assert b"case_0100" not in wire
        assert b"vector-add" not in wire
