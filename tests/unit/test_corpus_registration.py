"""A case is countable only after clean and mutant evidence pass the gate."""

import pytest


@pytest.fixture
def executions():
    from gpu_agent.benchmark.models import CaseExecution

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
        run_ids=["clean-run"],
        oracle_passed=True,
        required_checks_clean=True,
        detection_outcomes=["CLEAN"],
    )
    mutant = CaseExecution(
        **common,
        mutation_id="delete-index-guard",
        source_hash="1" * 64,
        run_ids=["mutant-run"],
        oracle_passed=False,
        required_checks_clean=False,
        target_confirmed=True,
        detection_outcomes=["FINDING"],
    )
    return clean, mutant


def test_validated_case_is_registered_with_all_run_ids(tmp_path, executions):
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.store import RunStore

    builder = BenchmarkBuilder(RunStore(tmp_path / "corpus"))
    validation = builder.validate(*executions)
    manifest = builder.register(validation)
    assert manifest.validation_run_ids == ["clean-run", "mutant-run"]
    assert manifest.source_hash == "1" * 64


@pytest.mark.parametrize("fault", ["no_finding", "clean_error", "timeout", "hash_mismatch"])
def test_invalid_evidence_never_registers(tmp_path, executions, fault):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError
    from gpu_agent.store import RunStore

    clean, mutant = executions
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
        BenchmarkBuilder(RunStore(tmp_path / "corpus")).register(validation)


def test_template_cannot_cross_splits(tmp_path, executions):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError
    from gpu_agent.store import RunStore

    builder = BenchmarkBuilder(RunStore(tmp_path / "corpus"))
    builder.register(builder.validate(*executions))
    clean, mutant = executions
    private_clean = clean.model_copy(update={"case_id": "case_0101", "split": "private"})
    private_mutant = mutant.model_copy(update={"case_id": "case_0101", "split": "private"})
    with pytest.raises(UnvalidatedCaseError, match="cross"):
        builder.register(builder.validate(private_clean, private_mutant))


def test_duplicate_case_id_cannot_overwrite_manifest(tmp_path, executions):
    from gpu_agent.benchmark.builder import BenchmarkBuilder, UnvalidatedCaseError
    from gpu_agent.store import RunStore

    builder = BenchmarkBuilder(RunStore(tmp_path / "corpus"))
    validation = builder.validate(*executions)
    builder.register(validation)
    with pytest.raises(UnvalidatedCaseError, match="already"):
        builder.register(validation)
