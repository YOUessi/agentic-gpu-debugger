"""Release corpus registration derives claims from native controller artifacts only."""

import hashlib
import json

import pytest


@pytest.fixture
def native_case(tmp_path):
    from gpu_agent.benchmark.models import CaseExecutionPlan
    from gpu_agent.benchmark.validation import CaseValidationController
    from gpu_agent.contracts import RepositorySnapshot, RunBinding
    from gpu_agent.environment import RuntimeToolchainAttestation, load_toolchain_lock
    from gpu_agent.execution.isolated import LOCK_PATH, IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture
    from gpu_agent.store import RunStore

    root = tmp_path / "sources"
    for role, kernel in (("clean", b"clean kernel\n"), ("mutant", b"mutant kernel\n")):
        directory = root / role
        directory.mkdir(parents=True)
        (directory / "kernel.cu").write_bytes(kernel)
    harness = root / "harness"
    harness.mkdir()
    for name in ("vector_io.cpp", "vector_api.h", "json.hpp"):
        (harness / name).write_bytes(("harness " + name).encode())

    store = RunStore(tmp_path / "corpus")
    lock = load_toolchain_lock(LOCK_PATH)
    binding = RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="corpus_validation",
        toolchain_lock_hash=lock.lock_hash,
        prompt_version=None,
        model_config_hash=None,
    )

    class FakeBackend(IsolatedGPUBackend):
        def _attest_runtime(self):
            assert self._expected_toolchain is not None
            return RuntimeToolchainAttestation(
                runtime_session_id=self._runtime_session_id,
                lock_hash=self._expected_toolchain.lock_hash,
                image_id=self._expected_toolchain.image_id,
                cuda_nvcc=self._expected_toolchain.cuda_nvcc,
                compute_sanitizer=self._expected_toolchain.compute_sanitizer,
                compute_capability="8.9",
                target_arch=self._expected_toolchain.target_arch,
                policy_hash=hashlib.sha256(self.policy.model_dump_json().encode()).hexdigest(),
            )

        def _container(self, path, operation, timeout, *, stdin=b"", cancel=None):
            mutant = (path / "kernel.cu").read_bytes().startswith(b"mutant")
            if operation == "build":
                return ProcessCapture(0, b"", b"", False), b"binary", b""
            if operation == "run":
                output = b'{"dtype":"float32","shape":[2],"values":[3.0,3.0]}'
                return ProcessCapture(0, output, b"", False), b"", b""
            log = (
                b"========= Invalid __global__ write of size 4 bytes\n"
                b"=========     at kernel in kernel.cu:1\n"
                b"========= ERROR SUMMARY: 1 error\n"
                if mutant
                else b"========= ERROR SUMMARY: 0 errors\n"
            )
            return ProcessCapture(86 if mutant else 0, b"", b"", False), b"", log

    backend = FakeBackend(store, root, tmp_path / "tasks")
    controller = CaseValidationController(store, backend, binding)
    input_bytes = json.dumps({"n": 2, "a": [1.0, 1.0], "b": [2.0, 2.0]}).encode()

    def execute(role, **changes):
        selected_controller = changes.pop("_controller", controller)
        selected_input = changes.pop("_input_bytes", input_bytes)
        names = [
            f"{role}/kernel.cu",
            *[f"harness/{name}" for name in ("vector_io.cpp", "vector_api.h", "json.hpp")],
        ]
        values = {
            "case_id": "case_0100",
            "template_id": "vector-add-index",
            "mutation_id": "clean" if role == "clean" else "delete-index-guard",
            "role": role,
            "split": "public",
            "source_manifest": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
            },
            "target_tool": "memcheck",
            "expected_finding": "Invalid __global__ write",
        }
        values.update(changes)
        return selected_controller.execute(CaseExecutionPlan(**values), selected_input)

    return store, controller, execute, input_bytes


def test_native_pair_registers_from_exact_terminal_artifacts(native_case):
    from gpu_agent.benchmark.builder import BenchmarkBuilder

    store, _, execute, _ = native_case
    clean_id, mutant_id = execute("clean"), execute("mutant")
    builder = BenchmarkBuilder(store)
    validation = builder.validate(clean_id, mutant_id)
    manifest = builder.register(validation)
    assert manifest.validation_run_ids == [clean_id, mutant_id]
    assert manifest.source_hash != store.load(clean_id).binding.repository.tracked_tree_hash
    observation_name = "validation/case-execution-observation.json"
    assert any(ref.name == observation_name for ref in store.load(clean_id).artifact_refs)


def test_claimant_summary_model_cannot_reach_registration(native_case):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError
    from gpu_agent.benchmark.models import CaseExecution, CaseValidation

    store, _, _, _ = native_case
    claimed = CaseExecution(
        case_id="case_0100",
        template_id="vector-add-index",
        mutation_id="clean",
        split="public",
        source_hash="1" * 64,
        harness_hash="2" * 64,
        toolchain_hash="3" * 64,
        input_set_hash="4" * 64,
        oracle_id="vector-add-cpu-v1",
        target_tool="memcheck",
        expected_finding="Invalid __global__ write",
        run_ids=["forged"],
        oracle_passed=True,
        required_checks_clean=True,
        detection_outcomes=["CLEAN"],
    )
    summary = CaseValidation(
        clean=claimed,
        mutant=claimed.model_copy(update={"mutation_id": "mutant"}),
        same_configuration=True,
        target_confirmed=True,
        clean_source_hash=claimed.source_hash,
        mutant_source_hash=claimed.source_hash,
    )
    with pytest.raises((UnvalidatedCaseError, TypeError, AttributeError)):
        BenchmarkBuilder(store).register(summary)  # type: ignore[arg-type]


def test_old_unbound_or_missing_observation_fails_closed(native_case):
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.benchmark.validation import CaseExecutionAttestationUnavailable

    store, _, execute, _ = native_case
    old = store.create_run("case_execution")
    store.transition(old.id, "RUNNING", "FINALIZING")
    store.transition(old.id, "COMPLETED", None)
    with pytest.raises(CaseExecutionAttestationUnavailable):
        BenchmarkBuilder(store).validate(old.id, execute("mutant"))


def test_role_swap_and_unrelated_identity_are_rejected(native_case):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError

    store, _, execute, _ = native_case
    clean_id, mutant_id = execute("clean"), execute("mutant")
    builder = BenchmarkBuilder(store)
    with pytest.raises(UnvalidatedCaseError):
        builder.validate(mutant_id, clean_id)
    unrelated = execute("mutant", case_id="case_0101")
    with pytest.raises(UnvalidatedCaseError):
        builder.validate(clean_id, unrelated)


@pytest.mark.parametrize(
    "change",
    [
        {"expected_finding": "claimant-selected-category"},
        {"target_tool": "racecheck"},
        {"_input_bytes": b'{"n":2,"a":[2.0,2.0],"b":[2.0,2.0]}'},
    ],
)
def test_finding_tool_and_input_substitution_are_rejected(native_case, change):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError
    from gpu_agent.benchmark.validation import CaseExecutionAttestationUnavailable

    store, _, execute, _ = native_case
    clean_id = execute("clean")
    mutant_id = execute("mutant", **change)
    with pytest.raises((UnvalidatedCaseError, CaseExecutionAttestationUnavailable)):
        BenchmarkBuilder(store).validate(clean_id, mutant_id)


def test_mixed_repository_binding_is_rejected(native_case):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError
    from gpu_agent.benchmark.validation import CaseValidationController

    store, controller, execute, _ = native_case
    different = controller.binding.model_copy(
        update={"repository": controller.binding.repository.model_copy(update={"commit": "c" * 40})}
    )
    other = CaseValidationController(store, controller.backend, different)
    with pytest.raises(UnvalidatedCaseError):
        BenchmarkBuilder(store).validate(execute("clean"), execute("mutant", _controller=other))


def test_timeout_artifact_is_rejected(native_case, monkeypatch):
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.benchmark.validation import CaseExecutionAttestationUnavailable
    from gpu_agent.execution.process import ProcessCapture

    store, controller, execute, _ = native_case
    clean_id = execute("clean")
    original = controller.backend._container

    def timed(path, operation, timeout, *, stdin=b"", cancel=None):
        if operation == "run":
            return ProcessCapture(0, b"", b"", True), b"", b""
        return original(path, operation, timeout, stdin=stdin, cancel=cancel)

    monkeypatch.setattr(controller.backend, "_container", timed)
    mutant_id = execute("mutant")
    with pytest.raises(CaseExecutionAttestationUnavailable):
        BenchmarkBuilder(store).validate(clean_id, mutant_id)


def test_validation_hash_and_run_pair_cannot_be_replayed(native_case):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError

    store, _, execute, _ = native_case
    builder = BenchmarkBuilder(store)
    validation = builder.validate(execute("clean"), execute("mutant"))
    with pytest.raises(UnvalidatedCaseError, match="hash"):
        builder.register(validation.model_copy(update={"clean_observation_hash": "f" * 64}))
    builder.register(validation)
    with pytest.raises(UnvalidatedCaseError, match="already"):
        builder.register(validation)


def test_private_plan_cannot_write_to_public_store(native_case):
    store, controller, _, input_bytes = native_case
    from gpu_agent.benchmark.models import CaseExecutionPlan

    with pytest.raises(ValueError, match="visibility"):
        controller.execute(
            CaseExecutionPlan(
                case_id="case_0100",
                template_id="vector-add-index",
                mutation_id="clean",
                role="clean",
                split="private",
                source_manifest={str(i): "0" * 64 for i in range(4)},
                target_tool="memcheck",
                expected_finding="Invalid __global__ write",
            ),
            input_bytes,
        )
    runs = [store.load(path.name) for path in store.root.iterdir() if len(path.name) == 32]
    assert all(run.kind != "benchmark_case" for run in runs)
