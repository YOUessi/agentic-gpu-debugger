"""Evaluator canaries never cross public store, report, source, or provider projections."""

import json

import pytest


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
