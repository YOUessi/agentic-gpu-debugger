"""A case is countable only after clean and mutant evidence pass the gate."""

import pytest


@pytest.fixture
def executions(tmp_path):
    from gpu_agent.benchmark.models import CaseExecution
    from gpu_agent.contracts import RepositorySnapshot, RunBinding
    from gpu_agent.store import RunStore

    store = RunStore(tmp_path / "corpus")
    binding = RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="corpus_validation",
        toolchain_lock_hash="3" * 64,
        prompt_version=None,
        model_config_hash=None,
    )
    clean_run = store.create_run("case_execution", binding=binding)
    mutant_run = store.create_run("case_execution", binding=binding)
    for run in (clean_run, mutant_run):
        store.transition(run.id, "RUNNING", "FINALIZING")
        store.transition(run.id, "COMPLETED", None)

    common = dict(
        case_id="case_0100",
        template_id="vector-add-index",
        split="public",
        harness_hash="2" * 64,
        toolchain_hash="3" * 64,
        input_set_hash="4" * 64,
        oracle_id="vector-add-cpu-v1",
        target_tool="memcheck",
        expected_finding="Invalid __global__ write",
        timed_out=False,
    )
    clean = CaseExecution(
        **common,
        mutation_id="clean",
        source_hash="0" * 64,
        run_ids=[clean_run.id],
        oracle_passed=True,
        required_checks_clean=True,
        detection_outcomes=["CLEAN"],
    )
    mutant = CaseExecution(
        **common,
        mutation_id="delete-index-guard",
        source_hash="1" * 64,
        run_ids=[mutant_run.id],
        oracle_passed=False,
        required_checks_clean=False,
        target_confirmed=True,
        detection_outcomes=["FINDING"],
    )
    return store, clean, mutant


def test_validated_case_is_registered_with_all_run_ids(tmp_path, executions):
    from gpu_agent.benchmark.builder import BenchmarkBuilder

    store, clean, mutant = executions
    builder = BenchmarkBuilder(store)
    validation = builder.validate(clean, mutant)
    manifest = builder.register(validation)
    assert manifest.validation_run_ids == [clean.run_ids[0], mutant.run_ids[0]]
    assert manifest.source_hash == "1" * 64
    registrations = [
        store.load(path.name)
        for path in store.root.iterdir()
        if store.load(path.name).kind == "benchmark_case"
    ]
    assert registrations[0].binding == store.load(clean.run_ids[0]).binding


@pytest.mark.parametrize("fault", ["no_finding", "clean_error", "timeout", "hash_mismatch"])
def test_invalid_evidence_never_registers(tmp_path, executions, fault):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError

    store, clean, mutant = executions
    validation = BenchmarkBuilder.validate(clean, mutant)
    if fault == "no_finding":
        validation = validation.model_copy(update={"target_confirmed": False})
    elif fault == "clean_error":
        validation = validation.model_copy(
            update={"clean": clean.model_copy(update={"required_checks_clean": False})}
        )
    elif fault == "timeout":
        validation = validation.model_copy(
            update={"mutant": mutant.model_copy(update={"timed_out": True})}
        )
    else:
        validation = validation.model_copy(
            update={"mutant": mutant.model_copy(update={"source_hash": "f" * 64})}
        )
    with pytest.raises((UnvalidatedCaseError, ValueError)):
        BenchmarkBuilder(store).register(validation)


def test_template_cannot_cross_splits(tmp_path, executions):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError

    store, clean, mutant = executions
    builder = BenchmarkBuilder(store)
    builder.register(builder.validate(clean, mutant))
    private_clean = clean.model_copy(update={"case_id": "case_0101", "split": "private"})
    private_mutant = mutant.model_copy(update={"case_id": "case_0101", "split": "private"})
    with pytest.raises(UnvalidatedCaseError, match="cross"):
        builder.register(builder.validate(private_clean, private_mutant))


def test_duplicate_case_id_cannot_overwrite_manifest(tmp_path, executions):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError

    store, clean, mutant = executions
    builder = BenchmarkBuilder(store)
    validation = builder.validate(clean, mutant)
    builder.register(validation)
    with pytest.raises(UnvalidatedCaseError, match="already"):
        builder.register(validation)


def test_registration_rejects_unbound_or_mixed_validation_runs(tmp_path, executions):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError
    from gpu_agent.store import RunStore

    _, clean, mutant = executions
    unrelated = RunStore(tmp_path / "unbound")
    clean_run = unrelated.create_run("case_execution")
    mutant_run = unrelated.create_run("case_execution")
    for run in (clean_run, mutant_run):
        unrelated.transition(run.id, "RUNNING", "FINALIZING")
        unrelated.transition(run.id, "COMPLETED", None)
    validation = BenchmarkBuilder.validate(
        clean.model_copy(update={"run_ids": [clean_run.id]}),
        mutant.model_copy(update={"run_ids": [mutant_run.id]}),
    )
    with pytest.raises(UnvalidatedCaseError, match="binding"):
        BenchmarkBuilder(unrelated).register(validation)


def test_registration_rejects_non_execution_summary_run(executions):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError

    store, clean, mutant = executions
    binding = store.load(clean.run_ids[0]).binding
    summary = store.create_run("release_evidence", binding=binding)
    store.transition(summary.id, "RUNNING", "FINALIZING")
    store.transition(summary.id, "COMPLETED", None)
    validation = BenchmarkBuilder.validate(
        clean,
        mutant.model_copy(update={"run_ids": [summary.id]}),
    )
    with pytest.raises(UnvalidatedCaseError, match="binding"):
        BenchmarkBuilder(store).register(validation)
