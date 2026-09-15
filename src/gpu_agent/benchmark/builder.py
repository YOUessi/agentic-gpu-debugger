"""Fail-closed registration for reproducible, tool-confirmed mutations."""

from gpu_agent.benchmark.models import CaseExecution, CaseManifest, CaseValidation
from gpu_agent.contracts import RunBinding, RunStatus
from gpu_agent.store import RunStore


class UnvalidatedCaseError(ValueError):
    pass


class BenchmarkBuilder:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    @staticmethod
    def validate(clean_case: CaseExecution, mutant: CaseExecution) -> CaseValidation:
        same = all(
            left == right
            for left, right in (
                (clean_case.case_id, mutant.case_id),
                (clean_case.template_id, mutant.template_id),
                (clean_case.split, mutant.split),
                (clean_case.harness_hash, mutant.harness_hash),
                (clean_case.toolchain_hash, mutant.toolchain_hash),
                (clean_case.input_set_hash, mutant.input_set_hash),
                (clean_case.oracle_id, mutant.oracle_id),
                (clean_case.target_tool, mutant.target_tool),
            )
        )
        confirmed = (
            mutant.target_confirmed
            and not mutant.timed_out
            and bool(mutant.detection_outcomes)
            and all(outcome == "FINDING" for outcome in mutant.detection_outcomes)
        )
        return CaseValidation(
            clean=clean_case,
            mutant=mutant,
            same_configuration=same,
            target_confirmed=confirmed,
            clean_source_hash=clean_case.source_hash,
            mutant_source_hash=mutant.source_hash,
        )

    def register(self, validation: CaseValidation) -> CaseManifest:
        clean, mutant = validation.clean, validation.mutant
        run_ids = [*clean.run_ids, *mutant.run_ids]
        if len(run_ids) != len(set(run_ids)):
            raise UnvalidatedCaseError("clean and mutant validation runs must be unique")
        source_runs = []
        try:
            source_runs = [self.store.load(run_id) for run_id in run_ids]
        except ValueError as exc:
            raise UnvalidatedCaseError("validation run binding is unavailable") from exc
        bindings = {run.binding for run in source_runs}
        binding: RunBinding | None = source_runs[0].binding if source_runs else None
        if (
            binding is None
            or binding.purpose != "corpus_validation"
            or len(bindings) != 1
            or any(run.kind not in {"case_execution", "mutation_validation"} for run in source_runs)
            or any(run.status != RunStatus.COMPLETED for run in source_runs)
            or binding.toolchain_lock_hash != mutant.toolchain_hash
        ):
            raise UnvalidatedCaseError("validation run binding is missing or inconsistent")
        valid = (
            validation.same_configuration
            and validation.target_confirmed
            and mutant.target_confirmed
            and not mutant.timed_out
            and bool(mutant.detection_outcomes)
            and all(outcome == "FINDING" for outcome in mutant.detection_outcomes)
            and clean.oracle_passed
            and clean.required_checks_clean
            and not clean.timed_out
            and all(outcome == "CLEAN" for outcome in clean.detection_outcomes)
            and clean.source_hash != mutant.source_hash
            and validation.clean_source_hash == clean.source_hash
            and validation.mutant_source_hash == mutant.source_hash
        )
        if not valid:
            raise UnvalidatedCaseError("clean/mutant evidence does not satisfy registration gate")
        for path in self.store.root.iterdir():
            if not path.is_dir() or len(path.name) != 32:
                continue
            run = self.store.load(path.name)
            if run.kind != "benchmark_case":
                continue
            ref = next(item for item in run.artifact_refs if item.name == "case-manifest.json")
            existing = CaseManifest.model_validate_json(self.store.read(ref))
            if existing.id == mutant.case_id:
                raise UnvalidatedCaseError("case ID is already registered")
            if existing.template_id == mutant.template_id and existing.split != mutant.split:
                raise UnvalidatedCaseError("template cannot cross public/private splits")
        manifest = CaseManifest(
            id=mutant.case_id,
            source_hash=mutant.source_hash,
            harness_hash=mutant.harness_hash,
            mutation_id=mutant.mutation_id,
            template_id=mutant.template_id,
            split=mutant.split,
            oracle_id=mutant.oracle_id,
            target_tool=mutant.target_tool,
            expected_finding=mutant.expected_finding,
            validation_run_ids=[*clean.run_ids, *mutant.run_ids],
            toolchain_hash=mutant.toolchain_hash,
            input_set_hash=mutant.input_set_hash,
        )
        run = self.store.create_run("benchmark_case", binding=binding)
        self.store.put(
            run.id, "case-manifest.json", manifest.model_dump_json().encode(), self.store.visibility
        )
        self.store.transition(run.id, "RUNNING", "FINALIZING")
        self.store.transition(run.id, "COMPLETED", None)
        return manifest
