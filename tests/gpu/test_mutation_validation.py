"""Live clean/mutant cases traverse the production native validation controller."""

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.container]


@pytest.mark.parametrize(
    "number,tool,n,repetitions,expected_finding",
    [
        (1, "memcheck", 257, 1, "Invalid __global__ write"),
        (2, "racecheck", 256, 5, "Race reported between Write access and Write access"),
        (3, "initcheck", 32, 1, "Uninitialized __global__ memory read"),
        (4, "synccheck", 32, 1, "Barrier error detected. Invalid arguments."),
    ],
)
def test_live_mutation_registration(tmp_path, number, tool, n, repetitions, expected_finding):
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.benchmark.models import AuthoritativeCaseRegistry, CaseExecutionPlan
    from gpu_agent.benchmark.validation import CaseValidationController
    from gpu_agent.contracts import RunBinding
    from gpu_agent.environment import load_toolchain_lock
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.provenance import capture_repository_snapshot
    from gpu_agent.store import RunStore

    root = Path(__file__).resolve().parents[2]
    public = root / "benchmarks"
    store = RunStore(tmp_path / "runs")
    backend = IsolatedGPUBackend(store, public, tmp_path / "tasks")
    if not backend.availability().ready:
        pytest.skip(backend.availability().reason)
    lock = load_toolchain_lock(root / "containers/toolchain.lock.json")
    registry_bytes = (root / "benchmarks/corpus-registry.json").read_bytes()
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()
    registry = AuthoritativeCaseRegistry.model_validate_json(registry_bytes)
    spec = next(case for case in registry.cases if case.case_id == f"case_{number:04d}")
    assert spec.expected_finding == expected_finding
    assert spec.sanitizer_repetitions == repetitions
    binding = RunBinding(
        repository=capture_repository_snapshot(root),
        purpose="corpus_validation",
        toolchain_lock_hash=lock.lock_hash,
        prompt_version=None,
        model_config_hash=None,
        case_registry_hash=registry_hash,
    )
    controller = CaseValidationController(store, backend, binding, root)
    input_bytes = json.dumps({"n": n, "a": [1.0] * n, "b": [2.0] * n}).encode()
    harness_names = ["harness/vector_io.cpp", "harness/vector_api.h", "harness/vendor/json.hpp"]

    def execute(role: str) -> str:
        case = "case_0000" if role == "clean" else f"case_{number:04d}"
        names = [f"public/{case}/public_input/kernel.cu", *harness_names]
        source_manifest = {
            name: hashlib.sha256((public / name).read_bytes()).hexdigest() for name in names
        }
        return controller.execute(
            CaseExecutionPlan(
                case_id=f"case_{number:04d}",
                template_id=spec.template_id,
                mutation_id="clean" if role == "clean" else spec.mutation_id,
                role=role,
                split="public",
                source_manifest=source_manifest,
                target_tool=tool,
                expected_finding=spec.expected_finding,
                sanitizer_repetitions=spec.sanitizer_repetitions,
                case_registry_hash=registry_hash,
                case_spec_hash=controller.spec_hash(spec),
                mutation_provenance_hash=spec.mutation_provenance_hash,
            ),
            input_bytes,
        )

    clean_run, mutant_run = execute("clean"), execute("mutant")
    builder = BenchmarkBuilder(store, ledger_root=tmp_path / "ledger")
    manifest = builder.register(builder.validate(clean_run, mutant_run))
    assert manifest.validation_run_ids == [clean_run, mutant_run]
