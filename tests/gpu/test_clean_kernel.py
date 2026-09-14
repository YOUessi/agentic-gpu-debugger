"""M0 live acceptance: exact trusted bytes -> build -> CPU-checked CUDA output."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_agent.config import Settings
from gpu_agent.environment import probe_environment

pytestmark = pytest.mark.gpu


def test_clean_kernel_real_gpu_and_persistent_evidence(tmp_path, request):
    report = probe_environment(Settings())
    if not report.ready:
        pytest.skip("CUDA metadata not ready: " + ", ".join(report.reason_codes))

    from gpu_agent.execution.local import LocalBackend
    from gpu_agent.execution.models import BuildRequest, ExecutionRequest, WorkspaceRequest
    from gpu_agent.store import RunStore

    repo = Path(__file__).resolve().parents[2]
    sources = {
        "benchmarks/public/case_0000/public_input/kernel.cu",
        "benchmarks/harness/vector_io.cpp",
        "benchmarks/harness/vector_api.h",
        "benchmarks/harness/vendor/json.hpp",
    }
    hashes = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in sources}
    configured_root = request.config.getoption("--gpu-run-root")
    store = RunStore(Path(configured_root) if configured_root else tmp_path / "runs")
    run = store.create_run("trusted_clean_acceptance")
    backend = LocalBackend(Settings(), store, repo, tmp_path / "workspaces")
    handle = None
    passed = False
    observations = []
    try:
        store.transition(run.id, "RUNNING", "PREPARING")
        store.put(run.id, "toolchain.json", report.model_dump_json().encode(), "public")
        handle = backend.prepare(WorkspaceRequest(run_id=run.id, source_manifest=hashes))
        store.transition(run.id, "RUNNING", "COMPILING")
        build = backend.build(BuildRequest(workspace_id=handle.id))
        assert build.success, build.model_dump_json(indent=2)
        assert build.binary_ref is not None
        assert hashlib.sha256(store.read(build.binary_ref)).hexdigest() == build.binary_ref.sha256
        store.transition(run.id, "RUNNING", "EXECUTING")
        for n in [1, 257, 1025]:
            # Dyadic values are exactly representable, avoiding a mirrored GPU implementation.
            a = [((i % 31) - 15) * 0.25 for i in range(n)]
            b = [((i % 17) - 8) * 0.5 for i in range(n)]
            expected = [x + y for x, y in zip(a, b, strict=True)]
            payload = json.dumps({"n": n, "a": a, "b": b}).encode()
            input_ref = store.put(run.id, f"input/n{n}.json", payload, "public")
            result = backend.run(ExecutionRequest(workspace_id=handle.id, stdin_ref=input_ref))
            assert result.runtime_status == "SUCCESS", result.model_dump_json(indent=2)
            actual = json.loads(store.read(result.output_ref))
            assert set(actual) == {"dtype", "shape", "values"}
            assert actual["dtype"] == "float32"
            assert actual["shape"] == [n]
            assert actual["values"] == expected
            assert store.read(input_ref) == payload
            observations.append(
                {
                    "n": n,
                    "oracle_passed": True,
                    "input_sha256": input_ref.sha256,
                    "output_sha256": result.output_ref.sha256,
                }
            )
        cleanup_handle, handle = handle, None
        backend.cleanup(cleanup_handle)
        # Integrity/source checks belong to acceptance, not after its success commit.
        reopened = RunStore(store.root)
        for ref in reopened.load(run.id).artifact_refs:
            assert len(reopened.read(ref)) == ref.byte_count
        assert hashes == {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in sources
        }
        store.put(
            run.id,
            "acceptance.json",
            json.dumps(
                {
                    "scope": "trusted_clean_kernel_only",
                    "execution_verified": True,
                    "sanitizer_verified": False,
                    "isolated_execution": False,
                    "binary_sha256": build.binary_ref.sha256,
                    "observations": observations,
                },
                indent=2,
            ).encode(),
            "public",
        )
        passed = True
    finally:
        try:
            if handle is not None:
                backend.cleanup(handle)
        finally:
            terminal = store.transition(run.id, "COMPLETED" if passed else "FAILED", None)
            print(f"CUDA acceptance run_id={run.id} status={terminal.status} root={store.root}")


@pytest.mark.parametrize("failure", ["integrity", "cleanup"])
def test_acceptance_cannot_persist_success_after_finalization_failure(
    tmp_path, monkeypatch, failure
):
    from gpu_agent.execution.local import LocalBackend
    from gpu_agent.store import RunStore

    created = []
    cleaned = False
    create_run, cleanup, read = RunStore.create_run, LocalBackend.cleanup, RunStore.read

    def record_run(self, *args, **kwargs):
        run = create_run(self, *args, **kwargs)
        created.append((self, run.id))
        return run

    def fail_cleanup(self, handle):
        nonlocal cleaned
        result = cleanup(self, handle)
        cleaned = True
        if failure == "cleanup":
            raise OSError("injected cleanup failure")
        return result

    def fail_read(self, ref):
        if cleaned and failure == "integrity":
            raise ValueError("injected artifact integrity failure")
        return read(self, ref)

    monkeypatch.setattr(RunStore, "create_run", record_run)
    monkeypatch.setattr(LocalBackend, "cleanup", fail_cleanup)
    monkeypatch.setattr(RunStore, "read", fail_read)
    isolated_request = SimpleNamespace(config=SimpleNamespace(getoption=lambda name: None))
    with pytest.raises((ValueError, OSError), match="injected"):
        test_clean_kernel_real_gpu_and_persistent_evidence(tmp_path, isolated_request)
    assert len(created) == 1
    store, run_id = created[0]
    manifest = store.load(run_id)
    assert manifest.status == "FAILED"
    assert manifest.current_phase is None
    assert not any(ref.name == "acceptance.json" for ref in manifest.artifact_refs)
