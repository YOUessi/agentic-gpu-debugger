"""Real isolated memcheck evidence only. No synthetic logs or host fallback."""

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.container]


@pytest.mark.parametrize("case,want", [("case_0000", "CLEAN"), ("case_0001", "FINDING")])
def test_real_memcheck_clean_and_oob(tmp_path, request, case, want):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.models import (
        BuildRequest,
        ExecutionRequest,
        SanitizerRequest,
        WorkspaceRequest,
    )
    from gpu_agent.store import RunStore

    repo = Path(__file__).resolve().parents[2]
    root = request.config.getoption("--gpu-run-root")
    store = RunStore(Path(root) if root else tmp_path / "runs")
    # Source root is the explicitly public benchmark directory, never repository/private roots.
    public = repo / "benchmarks"
    backend = IsolatedGPUBackend(store, public, tmp_path / "workspaces")
    available = backend.availability()
    if not available.ready:
        pytest.skip(available.reason)
    names = [
        f"public/{case}/public_input/kernel.cu",
        "harness/vector_io.cpp",
        "harness/vector_api.h",
        "harness/vendor/json.hpp",
    ]
    manifest = {n: hashlib.sha256((public / n).read_bytes()).hexdigest() for n in names}
    run = store.create_run("isolated_memcheck_acceptance")
    handle = backend.prepare(
        WorkspaceRequest(run_id=run.id, source_manifest=manifest, trust_level="UNTRUSTED")
    )
    passed = False
    try:
        build = backend.build(BuildRequest(workspace_id=handle.id))
        assert build.success, build.model_dump_json(indent=2)
        n = 257
        ref = store.put(
            run.id,
            "input.json",
            json.dumps({"n": n, "a": [1] * n, "b": [2] * n}).encode(),
            "public",
        )
        ordinary = backend.run(ExecutionRequest(workspace_id=handle.id, stdin_ref=ref))
        if want == "CLEAN":
            assert ordinary.runtime_status == "SUCCESS"
            assert json.loads(store.read(ordinary.output_ref))["values"] == [3] * n
        # OOB need not make ordinary execution nonzero; sanitizer is the acceptance check.
        result = backend.run_sanitizer(
            SanitizerRequest(workspace_id=handle.id, stdin_ref=ref, timeout_seconds=120)
        )
        assert result.check_outcome == want, result.model_dump_json(indent=2)
        assert result.completed
        if want == "FINDING":
            assert any("Invalid __global__" in f.category for f in result.findings)
            assert any(f.source_location and f.source_location.line for f in result.findings)
            assert all(f.raw_ref and store.read(f.raw_ref) for f in result.findings)
        else:
            output = store.read(result.program_output_ref)
            assert json.loads(output)["values"] == [3] * n
        view = backend.evidence.public_view(run.id)
        assert view.sanitizer_results[-1] == result
        backend.cleanup(handle)
        handle = None
        for artifact in store.load(run.id).artifact_refs:
            store.read(artifact)
        passed = True
    finally:
        if handle is not None:
            backend.cleanup(handle)
        store.transition(run.id, "RUNNING", "FINALIZING")
        store.transition(run.id, "COMPLETED" if passed else "FAILED", None)
        print(f"isolated memcheck {case}: run={run.id}, root={store.root}")
