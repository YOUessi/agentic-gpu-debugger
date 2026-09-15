"""Live clean/mutant evidence is required before corpus registration."""

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.container]


@pytest.mark.parametrize(
    "number,tool,n,repetitions",
    [
        (1, "memcheck", 257, 1),
        (2, "racecheck", 256, 5),
        (3, "initcheck", 32, 1),
        (4, "synccheck", 32, 1),
    ],
)
def test_live_mutation_registration(tmp_path, number, tool, n, repetitions):
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.benchmark.models import CaseExecution
    from gpu_agent.contracts import RunBinding
    from gpu_agent.environment import load_toolchain_lock
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.models import (
        BuildRequest,
        ExecutionRequest,
        SanitizerRequest,
        WorkspaceRequest,
    )
    from gpu_agent.provenance import capture_repository_snapshot
    from gpu_agent.store import RunStore
    from gpu_agent.verification.oracle import parse_output

    root = Path(__file__).resolve().parents[2]
    public = root / "benchmarks"
    store = RunStore(tmp_path / "runs")
    backend = IsolatedGPUBackend(store, public, tmp_path / "tasks")
    if not backend.availability().ready:
        pytest.skip(backend.availability().reason)
    lock = load_toolchain_lock(root / "containers/toolchain.lock.json")
    binding = RunBinding(
        repository=capture_repository_snapshot(root),
        purpose="corpus_validation",
        toolchain_lock_hash=lock.lock_hash,
        prompt_version=None,
        model_config_hash=None,
    )
    input_bytes = json.dumps({"n": n, "a": [1.0] * n, "b": [2.0] * n}).encode()
    harness_names = ["harness/vector_io.cpp", "harness/vector_api.h", "harness/vendor/json.hpp"]
    harness_hash = hashlib.sha256(
        b"".join((public / name).read_bytes() for name in harness_names)
    ).hexdigest()

    def execute(case: str, repeats: int):
        names = [f"public/{case}/public_input/kernel.cu", *harness_names]
        manifest = {
            name: hashlib.sha256((public / name).read_bytes()).hexdigest() for name in names
        }
        run = store.create_run("mutation_validation", binding=binding)
        handle = backend.prepare(WorkspaceRequest(run_id=run.id, source_manifest=manifest))
        try:
            assert backend.build(BuildRequest(workspace_id=handle.id)).success
            ref = store.put(run.id, "input.json", input_bytes, "public")
            ordinary = backend.run(ExecutionRequest(workspace_id=handle.id, stdin_ref=ref))
            oracle_passed = (
                ordinary.runtime_status == "SUCCESS"
                and parse_output(store.read(ordinary.output_ref)) == [3.0] * n
            )
            precheck = None
            if tool != "memcheck":
                precheck = backend.run_sanitizer(
                    SanitizerRequest(workspace_id=handle.id, stdin_ref=ref)
                )
            results = [
                backend.run_sanitizer(
                    SanitizerRequest(
                        workspace_id=handle.id, stdin_ref=ref, tool=tool, timeout_seconds=120
                    )
                )
                for _ in range(repeats)
            ]
            clean = (precheck is None or precheck.check_outcome == "CLEAN") and all(
                result.check_outcome == "CLEAN" for result in results
            )
            return (
                run.id,
                manifest[Path(names[0]).as_posix()],
                oracle_passed,
                clean,
                [result.check_outcome for result in results],
            )
        finally:
            backend.cleanup(handle)
            store.transition(run.id, "RUNNING", "FINALIZING")
            store.transition(run.id, "COMPLETED", None)

    clean_run, clean_hash, clean_oracle, clean_checks, clean_outcomes = execute(
        "case_0000", repetitions
    )
    mutant_run, mutant_hash, mutant_oracle, _, mutant_outcomes = execute(
        f"case_{number:04d}", repetitions
    )
    common = dict(
        case_id=f"case_{number:04d}",
        template_id=f"vector-add-{number}",
        split="public",
        harness_hash=harness_hash,
        toolchain_hash=lock.lock_hash,
        input_set_hash=hashlib.sha256(input_bytes).hexdigest(),
        oracle_id="vector-add-cpu-v1",
        target_tool=tool,
        expected_finding=tool,
    )
    clean = CaseExecution(
        **common,
        mutation_id="clean",
        source_hash=clean_hash,
        run_ids=[clean_run],
        oracle_passed=clean_oracle,
        required_checks_clean=clean_checks,
        detection_outcomes=clean_outcomes,
    )
    mutant = CaseExecution(
        **common,
        mutation_id=f"mutation-{number}",
        source_hash=mutant_hash,
        run_ids=[mutant_run],
        oracle_passed=mutant_oracle,
        required_checks_clean=False,
        target_confirmed=all(x == "FINDING" for x in mutant_outcomes),
        detection_outcomes=mutant_outcomes,
    )
    manifest = BenchmarkBuilder(store).register(BenchmarkBuilder.validate(clean, mutant))
    assert manifest.validation_run_ids == [clean_run, mutant_run]
