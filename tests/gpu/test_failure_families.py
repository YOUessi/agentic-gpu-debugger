"""Live evidence for four CUDA failure families and their clean reference."""

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.container, pytest.mark.release]

FAMILIES = [
    ("case_0001", "memcheck", 257, "Invalid __global__", 1),
    ("case_0002", "racecheck", 256, "Race reported", 5),
    ("case_0003", "initcheck", 32, "Uninitialized __global__", 1),
    ("case_0004", "synccheck", 32, "Barrier error detected", 1),
]


def _manifest(root: Path, case: str) -> dict[str, str]:
    names = [
        f"public/{case}/public_input/kernel.cu",
        "harness/vector_io.cpp",
        "harness/vector_api.h",
        "harness/vendor/json.hpp",
    ]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _exercise(backend, store, case: str, tool: str, n: int, repetitions: int):
    from gpu_agent.execution.models import BuildRequest, SanitizerRequest, WorkspaceRequest

    run = store.create_run(f"failure_family_{case}_{tool}")
    handle = backend.prepare(
        WorkspaceRequest(
            run_id=run.id,
            source_manifest=_manifest(backend.repo_root, case),
            trust_level="UNTRUSTED",
        )
    )
    try:
        build = backend.build(BuildRequest(workspace_id=handle.id))
        assert build.success, build.model_dump_json(indent=2)
        stdin = store.put(
            run.id,
            "input.json",
            json.dumps({"n": n, "a": [1.0] * n, "b": [2.0] * n}).encode(),
            "public",
        )
        precheck = None
        if tool != "memcheck":
            precheck = backend.run_sanitizer(
                SanitizerRequest(
                    workspace_id=handle.id,
                    stdin_ref=stdin,
                    tool="memcheck",
                    timeout_seconds=120,
                )
            )
        results = [
            backend.run_sanitizer(
                SanitizerRequest(
                    workspace_id=handle.id,
                    stdin_ref=stdin,
                    tool=tool,
                    timeout_seconds=120,
                )
            )
            for _ in range(repetitions)
        ]
        return precheck, results
    finally:
        backend.cleanup(handle)
        store.transition(run.id, "RUNNING", "FINALIZING")
        store.transition(run.id, "COMPLETED", None)


@pytest.mark.parametrize("case,tool,n,category,repetitions", FAMILIES)
def test_buggy_finding_and_clean_reference(tmp_path, request, case, tool, n, category, repetitions):
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.store import RunStore

    repo = Path(__file__).resolve().parents[2]
    root = request.config.getoption("--gpu-run-root")
    store = RunStore(Path(root) if root else tmp_path / "runs")
    backend = IsolatedGPUBackend(store, repo / "benchmarks", tmp_path / "workspaces")
    available = backend.availability()
    if not available.ready:
        pytest.skip(available.reason)

    buggy_precheck, buggy_results = _exercise(backend, store, case, tool, n, repetitions)
    if buggy_precheck is not None:
        assert buggy_precheck.completed and buggy_precheck.check_outcome == "CLEAN"
    outcomes = [result.check_outcome for result in buggy_results]
    detection_rate = outcomes.count("FINDING") / repetitions
    assert detection_rate == 1.0, outcomes
    assert all(result.completed for result in buggy_results)
    assert all(
        any(category in finding.category for finding in result.findings) for result in buggy_results
    )
    assert all(
        finding.raw_ref and store.read(finding.raw_ref)
        for result in buggy_results
        for finding in result.findings
    )

    clean_precheck, clean_results = _exercise(backend, store, "case_0000", tool, n, 1)
    if clean_precheck is not None:
        assert clean_precheck.completed and clean_precheck.check_outcome == "CLEAN"
    assert clean_results[0].completed and clean_results[0].check_outcome == "CLEAN"
    assert backend.active_containers() == []
    print(
        f"family={case} tool={tool} repetitions={repetitions} detection_rate={detection_rate:.3f}"
    )
