"""Release corpus registration derives claims from native controller artifacts only."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture
def native_case(tmp_path, monkeypatch):
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.models import (
        AuthoritativeCaseRegistry,
        AuthoritativeCaseSpec,
        CaseExecutionPlan,
    )
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

    input_bytes = json.dumps({"n": 2, "a": [1.0, 1.0], "b": [2.0, 2.0]}).encode()
    harness_hash = hashlib.sha256(
        json.dumps(
            sorted(
                (name, hashlib.sha256((harness / name).read_bytes()).hexdigest())
                for name in ("vector_io.cpp", "vector_api.h", "json.hpp")
            ),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    spec = AuthoritativeCaseSpec(
        case_id="case_0100",
        template_id="vector-add-index",
        mutation_id="delete-index-guard",
        split="public",
        clean_source_hash=hashlib.sha256((root / "clean/kernel.cu").read_bytes()).hexdigest(),
        mutant_source_hash=hashlib.sha256((root / "mutant/kernel.cu").read_bytes()).hexdigest(),
        harness_hash=harness_hash,
        input_set_hash=hashlib.sha256(input_bytes).hexdigest(),
        oracle_id="vector-add-cpu-v1",
        target_tool="memcheck",
        expected_finding="Invalid __global__ write",
        sanitizer_repetitions=1,
        mutation_provenance_hash="9" * 64,
    )
    registry_bytes = AuthoritativeCaseRegistry(cases=[spec]).model_dump_json().encode()
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()
    store = RunStore(tmp_path / "corpus")
    family = CorpusFamily.provision(
        tmp_path / "controller",
        public_store=store.root,
        evaluator_store=tmp_path / "evaluator",
        repository=tmp_path / "repository",
    )
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    lock = load_toolchain_lock(LOCK_PATH)
    binding = RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="corpus_validation",
        toolchain_lock_hash=lock.lock_hash,
        prompt_version=None,
        model_config_hash=None,
        case_registry_hash=registry_hash,
        corpus_ledger_namespace_hash=family.namespace_hash,
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
            output = b'{"dtype":"float32","shape":[2],"values":[3.0,3.0]}'
            return ProcessCapture(86 if mutant else 0, output, b"", False), b"", log

    backend = FakeBackend(store, root, tmp_path / "tasks")
    controller = CaseValidationController._for_test(store, backend, binding, registry_bytes)

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
            "sanitizer_repetitions": 1,
            "case_registry_hash": registry_hash,
            "case_spec_hash": controller.spec_hash(spec),
            "mutation_provenance_hash": spec.mutation_provenance_hash,
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


def test_oracle_policy_covers_ordinary_and_each_instrumented_output(native_case):
    from gpu_agent.benchmark.models import CaseExecutionObservation, CaseOracleObservation

    store, _, execute, _ = native_case
    run_id = execute("clean")
    run = store.load(run_id)
    observation_ref = next(
        ref for ref in run.artifact_refs if ref.name == "validation/case-execution-observation.json"
    )
    observation = CaseExecutionObservation.model_validate_json(store.read(observation_ref))
    refs = [observation.oracle_ref, *observation.sanitizer_oracle_refs]
    assertions = [CaseOracleObservation.model_validate_json(store.read(ref)) for ref in refs]
    assert [item.channel for item in assertions] == ["ordinary", "instrumented"]
    assert all(item.result.passed for item in assertions)
    assert {(item.result.atol, item.result.rtol) for item in assertions} == {(1e-5, 1e-5)}


def test_authoritative_repetition_schedule_cannot_be_shortened(native_case):
    _, _, execute, _ = native_case
    with pytest.raises(ValueError, match="authoritative"):
        execute("mutant", sanitizer_repetitions=5)


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
        BenchmarkBuilder(store).register(
            summary  # type: ignore[arg-type]
        )


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
    with pytest.raises(ValueError, match="authoritative"):
        execute("mutant", case_id="case_0101")


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
    with pytest.raises((ValueError, UnvalidatedCaseError, CaseExecutionAttestationUnavailable)):
        mutant_id = execute("mutant", **change)
        BenchmarkBuilder(store).validate(clean_id, mutant_id)


def test_mixed_repository_binding_is_rejected(native_case):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError
    from gpu_agent.benchmark.models import AuthoritativeCaseRegistry
    from gpu_agent.benchmark.validation import CaseValidationController

    store, controller, execute, _ = native_case
    different = controller.binding.model_copy(
        update={"repository": controller.binding.repository.model_copy(update={"commit": "c" * 40})}
    )
    registry_bytes = (
        AuthoritativeCaseRegistry(cases=list(controller.specs.values())).model_dump_json().encode()
    )
    other = CaseValidationController._for_test(store, controller.backend, different, registry_bytes)
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


def test_sanitizer_result_link_substitution_is_rejected(native_case, monkeypatch):
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.benchmark.validation import CaseExecutionAttestationUnavailable

    store, controller, execute, _ = native_case
    clean_id = execute("clean")
    original = controller.backend.run_sanitizer

    def mismatched(request, *, cancel=None):
        result = original(request, cancel=cancel)
        stderr = result.tool_result.typed_payload.program_stderr_ref
        assert stderr is not None
        return result.model_copy(update={"program_output_ref": stderr})

    monkeypatch.setattr(controller.backend, "run_sanitizer", mismatched)
    mutant_id = execute("mutant")
    with pytest.raises(CaseExecutionAttestationUnavailable):
        BenchmarkBuilder(store).validate(clean_id, mutant_id)


def test_validation_hash_is_checked_and_exact_retry_is_idempotent(native_case):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError

    store, _, execute, _ = native_case
    builder = BenchmarkBuilder(store)
    validation = builder.validate(execute("clean"), execute("mutant"))
    with pytest.raises(UnvalidatedCaseError, match="hash"):
        builder.register(validation.model_copy(update={"clean_observation_hash": "f" * 64}))
    first = builder.register(validation)
    assert builder.register(validation) == first


def test_registration_fails_closed_without_trusted_family_config(native_case, monkeypatch):
    from gpu_agent.benchmark.builder import BenchmarkBuilder

    store, _, _, _ = native_case
    monkeypatch.delenv("GPU_AGENT_CORPUS_FAMILY_ROOT")
    with pytest.raises(ValueError, match="trusted corpus family"):
        BenchmarkBuilder(store)


def test_registration_cannot_switch_to_a_new_ledger_namespace(native_case, monkeypatch, tmp_path):
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.validation import CaseExecutionAttestationUnavailable

    store, _, execute, _ = native_case
    validation = BenchmarkBuilder(store).validate(execute("clean"), execute("mutant"))
    BenchmarkBuilder(store).register(validation)
    with pytest.raises(ValueError, match="already pinned"):
        CorpusFamily.provision(
            tmp_path / "other-controller",
            public_store=store.root,
            evaluator_store=tmp_path / "evaluator",
            repository=tmp_path / "repository",
        )
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(tmp_path / "other-controller"))
    with pytest.raises((ValueError, CaseExecutionAttestationUnavailable)):
        BenchmarkBuilder(store)


def test_sanitizer_request_id_must_match_every_artifact_path(native_case, monkeypatch):
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.benchmark.validation import CaseExecutionAttestationUnavailable

    store, controller, execute, _ = native_case
    clean_id = execute("clean")
    original = controller.backend.run_sanitizer

    def spliced(request, *, cancel=None):
        result = original(request, cancel=cancel)
        tool_result = result.tool_result.model_copy(update={"request_id": "f" * 32})
        return result.model_copy(update={"tool_result": tool_result})

    monkeypatch.setattr(controller.backend, "run_sanitizer", spliced)
    mutant_id = execute("mutant")
    with pytest.raises(CaseExecutionAttestationUnavailable):
        BenchmarkBuilder(store).validate(clean_id, mutant_id)


def test_hidden_second_sanitizer_invocation_cannot_be_spliced_in(native_case, monkeypatch):
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.benchmark.validation import CaseExecutionAttestationUnavailable

    store, controller, execute, _ = native_case
    clean_id = execute("clean")
    original = controller.backend.run_sanitizer

    def hidden_first(request, *, cancel=None):
        original(request, cancel=cancel)
        return original(request, cancel=cancel)

    monkeypatch.setattr(controller.backend, "run_sanitizer", hidden_first)
    mutant_id = execute("mutant")
    with pytest.raises(CaseExecutionAttestationUnavailable):
        BenchmarkBuilder(store).validate(clean_id, mutant_id)


def test_runtime_request_id_must_match_output_and_log_paths(native_case, monkeypatch):
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.benchmark.validation import CaseExecutionAttestationUnavailable

    store, controller, execute, _ = native_case
    clean_id = execute("clean")
    original = controller.backend.run

    def spliced(request, *, cancel=None):
        result = original(request, cancel=cancel)
        tool_result = result.tool_result.model_copy(update={"request_id": "f" * 32})
        return result.model_copy(update={"tool_result": tool_result})

    monkeypatch.setattr(controller.backend, "run", spliced)
    mutant_id = execute("mutant")
    with pytest.raises(CaseExecutionAttestationUnavailable):
        BenchmarkBuilder(store).validate(clean_id, mutant_id)


def test_cleanup_exception_still_terminalizes_failed_run(native_case, monkeypatch):
    from gpu_agent.execution.models import CleanupResult

    store, controller, execute, _ = native_case
    calls = 0

    def broken_cleanup(handle):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated cleanup failure")
        return CleanupResult(workspace_id=handle.id, removed=True)

    before = {path.name for path in store.root.iterdir() if len(path.name) == 32}
    monkeypatch.setattr(controller.backend, "cleanup", broken_cleanup)
    with pytest.raises(RuntimeError, match="simulated cleanup failure"):
        execute("clean")
    created = [
        path.name
        for path in store.root.iterdir()
        if len(path.name) == 32 and path.name not in before
    ]
    assert len(created) == 1
    run = store.load(created[0])
    assert run.status.value == "FAILED"
    assert calls == 2
    assert any(ref.name == "validation/cleanup-error.json" for ref in run.artifact_refs)


def test_cleanup_exception_does_not_replace_primary_execution_failure(native_case, monkeypatch):
    store, controller, execute, _ = native_case

    def broken_build(_request):
        raise ValueError("primary build failure")

    def broken_cleanup(_handle):
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(controller.backend, "build", broken_build)
    monkeypatch.setattr(controller.backend, "cleanup", broken_cleanup)
    with pytest.raises(ValueError, match="primary build failure"):
        execute("clean")
    failed = [
        store.load(path.name)
        for path in store.root.iterdir()
        if len(path.name) == 32 and store.load(path.name).status.value == "FAILED"
    ]
    assert len(failed) == 1
    assert any(ref.name == "validation/cleanup-error.json" for ref in failed[0].artifact_refs)


def test_cleanup_removed_false_is_persisted_before_failed_terminal_state(native_case, monkeypatch):
    from gpu_agent.benchmark.validation import CaseExecutionAttestationUnavailable
    from gpu_agent.execution.models import CleanupResult

    store, controller, execute, _ = native_case
    monkeypatch.setattr(
        controller.backend,
        "cleanup",
        lambda handle: CleanupResult(workspace_id=handle.id, removed=False),
    )
    with pytest.raises(CaseExecutionAttestationUnavailable, match="cleanup"):
        execute("clean")
    failed = [
        store.load(path.name)
        for path in store.root.iterdir()
        if len(path.name) == 32 and store.load(path.name).status.value == "FAILED"
    ]
    assert len(failed) == 1
    error_ref = next(
        ref for ref in failed[0].artifact_refs if ref.name == "validation/cleanup-error.json"
    )
    assert json.loads(store.read(error_ref))["error_type"] == "CleanupResult"


def test_shared_ledger_is_atomic_and_contains_no_private_plaintext(tmp_path):
    from gpu_agent.benchmark.ledger import CorpusLedger
    from gpu_agent.store import RunStore

    root = tmp_path / "shared-ledger"
    store = RunStore(tmp_path / "public")
    private_identity = b"PRIVATE_case_9000_private-template"
    private_template = b"PRIVATE_template"
    private_pair = b"PRIVATE_source-harness-input-mutation"

    def reserve():
        return CorpusLedger(root).prepare(
            private_identity,
            private_template,
            private_pair,
            store=store,
            manifest_hash="a" * 64,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [future.result() for future in [pool.submit(reserve), pool.submit(reserve)]]
    assert outcomes[0] == outcomes[1]
    with pytest.raises(ValueError, match="already reserved"):
        CorpusLedger(root).prepare(
            b"different-case",
            b"different-template",
            private_pair,
            store=store,
            manifest_hash="b" * 64,
        )
    with pytest.raises(ValueError, match="already reserved"):
        CorpusLedger(root).prepare(
            private_identity,
            b"other-template",
            b"other-pair",
            store=store,
            manifest_hash="b" * 64,
        )
    with pytest.raises(ValueError, match="already reserved"):
        CorpusLedger(root).prepare(
            b"other-case",
            private_template,
            b"another-pair",
            store=store,
            manifest_hash="b" * 64,
        )
    raw = (root / "transactions.json").read_bytes()
    assert b"PRIVATE" not in raw and b"case_9000" not in raw


@pytest.mark.parametrize("crash_point", ["before_install", "after_install"])
def test_ledger_key_crash_never_exposes_partial_final_key(tmp_path, monkeypatch, crash_point):
    import gpu_agent.benchmark.ledger as ledger_module
    from gpu_agent.benchmark.ledger import CorpusLedger

    root = tmp_path / "ledger"
    crashed = False
    if crash_point == "before_install":
        original = ledger_module._atomic_create

        def crash_once(path, content, mode):
            nonlocal crashed
            if path.name == "identity.key" and not crashed:
                crashed = True
                raise RuntimeError("simulated key install crash")
            return original(path, content, mode)

        monkeypatch.setattr(ledger_module, "_atomic_create", crash_once)
    else:
        original_sync = ledger_module.sync_directory

        def crash_after_link(path):
            nonlocal crashed
            if (path / "identity.key").exists() and not crashed:
                crashed = True
                raise RuntimeError("simulated key install crash")
            return original_sync(path)

        monkeypatch.setattr(ledger_module, "sync_directory", crash_after_link)
    with pytest.raises(RuntimeError, match="key install"):
        CorpusLedger(root)
    if (root / "identity.key").exists():
        assert len((root / "identity.key").read_bytes()) == 32
    ledger = CorpusLedger(root)
    assert len((root / "identity.key").read_bytes()) == 32
    assert len(ledger.namespace_hash) == 64


def test_public_and_evaluator_stores_resolve_one_private_namespace(tmp_path, monkeypatch):
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.store import RunStore

    public = RunStore(tmp_path / "public")
    evaluator = RunStore(tmp_path / "evaluator", visibility="evaluator")
    family = CorpusFamily.provision(
        tmp_path / "controller",
        public_store=public.root,
        evaluator_store=evaluator.root,
        repository=tmp_path / "repository",
    )
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    assert CorpusFamily.configured(public).namespace_hash == family.namespace_hash
    assert CorpusFamily.configured(evaluator).namespace_hash == family.namespace_hash


def test_controller_key_root_cannot_overlap_publishable_store_or_repository(tmp_path):
    from gpu_agent.benchmark.ledger import CorpusFamily

    public = tmp_path / "public"
    with pytest.raises(ValueError, match="separate"):
        CorpusFamily.provision(
            public / "controller",
            public_store=public,
            evaluator_store=tmp_path / "evaluator",
            repository=tmp_path / "repository",
        )
    with pytest.raises(ValueError, match="repository"):
        CorpusFamily.provision(
            tmp_path / "repository" / "controller",
            public_store=public,
            evaluator_store=tmp_path / "evaluator",
            repository=tmp_path / "repository",
        )
    assert not (tmp_path / "repository" / "controller" / "ledger").exists()


@pytest.mark.parametrize(
    "crash_point",
    ["prepared", "create_publish", "created", "put", "terminalized", "before_commit"],
)
def test_registration_transaction_recovers_each_crash_point(native_case, monkeypatch, crash_point):
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.benchmark.ledger import CorpusLedger

    store, _, execute, _ = native_case
    builder = BenchmarkBuilder(store)
    validation = builder.validate(execute("clean"), execute("mutant"))
    crashed = False

    if crash_point == "prepared":
        original = BenchmarkBuilder._complete_registration

        def crash_after_prepared(self, *args, **kwargs):
            nonlocal crashed
            if not crashed:
                crashed = True
                raise RuntimeError("simulated prepared crash")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(BenchmarkBuilder, "_complete_registration", crash_after_prepared)
    elif crash_point == "create_publish":
        import gpu_agent.store as store_module

        original_rename = store_module.os.rename

        def crash_before_publish(source, target):
            nonlocal crashed
            if Path(target).name and not crashed:
                crashed = True
                raise RuntimeError("simulated create publish crash")
            return original_rename(source, target)

        monkeypatch.setattr(store_module.os, "rename", crash_before_publish)
    elif crash_point == "created":
        original_create = store.create_run

        def crash_after_create(*args, **kwargs):
            nonlocal crashed
            result = original_create(*args, **kwargs)
            if kwargs.get("_run_id") and not crashed:
                crashed = True
                raise RuntimeError("simulated create crash")
            return result

        monkeypatch.setattr(store, "create_run", crash_after_create)
    elif crash_point == "put":
        original_put = store.put

        def crash_after_put(run_id, name, content, visibility):
            nonlocal crashed
            result = original_put(run_id, name, content, visibility)
            if name == "validation/ledger-transaction.json" and not crashed:
                crashed = True
                raise RuntimeError("simulated put crash")
            return result

        monkeypatch.setattr(store, "put", crash_after_put)
    elif crash_point == "terminalized":
        original_transition = store.transition

        def crash_after_terminal(run_id, status, phase):
            nonlocal crashed
            result = original_transition(run_id, status, phase)
            if str(status) == "COMPLETED" and not crashed:
                crashed = True
                raise RuntimeError("simulated terminal crash")
            return result

        monkeypatch.setattr(store, "transition", crash_after_terminal)
    else:
        original_commit = CorpusLedger.commit

        def crash_before_commit(self, transaction):
            nonlocal crashed
            if not crashed:
                crashed = True
                raise RuntimeError("simulated pre-commit crash")
            return original_commit(self, transaction)

        monkeypatch.setattr(CorpusLedger, "commit", crash_before_commit)

    with pytest.raises(RuntimeError, match="simulated"):
        builder.register(validation)
    manifest = BenchmarkBuilder(store).register(validation)
    assert manifest.id == "case_0100"
    registrations = [
        store.load(path.name)
        for path in store.root.iterdir()
        if len(path.name) == 32 and store.load(path.name).kind == "benchmark_case"
    ]
    assert len(registrations) == 1
    assert registrations[0].status.value == "COMPLETED"
    state = json.loads((builder.family.ledger.root / "transactions.json").read_bytes())
    assert state["transactions"][0]["state"] == "COMMITTED"


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
                sanitizer_repetitions=1,
                case_registry_hash=controller.registry_hash,
                case_spec_hash="0" * 64,
                mutation_provenance_hash="0" * 64,
            ),
            input_bytes,
        )
    runs = [store.load(path.name) for path in store.root.iterdir() if len(path.name) == 32]
    assert all(run.kind != "benchmark_case" for run in runs)
