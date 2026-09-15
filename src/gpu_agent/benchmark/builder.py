"""Fail-closed registration derived only from native RunStore execution artifacts."""

import hashlib
from pathlib import PurePosixPath
from typing import Any

from gpu_agent.benchmark.models import (
    CaseExecutionObservation,
    CaseManifest,
    CaseOracleObservation,
    CaseValidationArtifact,
)
from gpu_agent.benchmark.validation import (
    CaseExecutionAttestationUnavailable,
    _validate_refs,
    derive_oracle,
    source_identities,
)
from gpu_agent.contracts import ArtifactRef, RunBinding, RunManifest, RunStatus, ToolResult
from gpu_agent.environment import RuntimeToolchainAttestation
from gpu_agent.evidence.sanitizer import parse_sanitizer
from gpu_agent.execution.models import BuildResult, ExecutionResult, SanitizerResult
from gpu_agent.execution.process import ProcessCapture
from gpu_agent.store import RunStore


class UnvalidatedCaseError(ValueError):
    pass


class _NativeExecution:
    def __init__(
        self,
        run: RunManifest,
        observation_ref: ArtifactRef,
        observation: CaseExecutionObservation,
        build: BuildResult,
        runtime: ExecutionResult,
        sanitizers: list[SanitizerResult],
        oracle: CaseOracleObservation,
    ) -> None:
        self.run = run
        self.observation_ref = observation_ref
        self.observation = observation
        self.build = build
        self.runtime = runtime
        self.sanitizers = sanitizers
        self.oracle = oracle


def _failed(result: ToolResult[Any] | None) -> bool:
    return result is None or bool(
        result.timed_out or result.cancelled or result.truncated or result.tool_error
    )


def _runtime_status(result: ToolResult[Any]) -> str:
    if result.cancelled:
        return "CANCELLED"
    if result.timed_out:
        return "TIMEOUT"
    if result.tool_error:
        return "TOOL_ERROR"
    if result.truncated:
        return "TRUNCATED"
    return "SUCCESS" if result.exit_code == 0 else "FAILED"


class BenchmarkBuilder:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    def _one(self, run: RunManifest, name: str) -> ArtifactRef:
        refs = [ref for ref in run.artifact_refs if ref.name == name]
        if len(refs) != 1:
            raise CaseExecutionAttestationUnavailable("CASE_EXECUTION_ATTESTATION_UNAVAILABLE")
        return refs[0]

    def _load(self, run_id: str) -> _NativeExecution:
        try:
            run = self.store.load(run_id)
            if (
                run.kind != "case_execution"
                or run.status != RunStatus.COMPLETED
                or run.binding is None
                or run.binding.purpose != "corpus_validation"
                or run.binding.toolchain_lock_hash is None
            ):
                raise ValueError
            observation_ref = self._one(run, "validation/case-execution-observation.json")
            observation = CaseExecutionObservation.model_validate_json(
                self.store.read(observation_ref)
            )
            _validate_refs(self.store, run_id, observation)
            build = BuildResult.model_validate_json(self.store.read(observation.build_ref))
            runtime = ExecutionResult.model_validate_json(self.store.read(observation.runtime_ref))
            sanitizers = [
                SanitizerResult.model_validate_json(self.store.read(ref))
                for ref in observation.sanitizer_refs
            ]
            oracle = CaseOracleObservation.model_validate_json(
                self.store.read(observation.oracle_ref)
            )
            for value in (build, runtime, sanitizers, oracle):
                _validate_refs(self.store, run_id, value)
        except (OSError, ValueError) as exc:
            if isinstance(exc, CaseExecutionAttestationUnavailable):
                raise
            raise CaseExecutionAttestationUnavailable(
                "CASE_EXECUTION_ATTESTATION_UNAVAILABLE"
            ) from exc
        self._validate_native(run, observation, build, runtime, sanitizers, oracle)
        return _NativeExecution(
            run, observation_ref, observation, build, runtime, sanitizers, oracle
        )

    def _validate_native(
        self,
        run: RunManifest,
        observation: CaseExecutionObservation,
        build: BuildResult,
        runtime: ExecutionResult,
        sanitizers: list[SanitizerResult],
        oracle: CaseOracleObservation,
    ) -> None:
        binding = run.binding
        assert binding is not None and binding.toolchain_lock_hash is not None
        expected_result_refs = (
            observation.build_ref.name == "validation/build-result.json"
            and observation.runtime_ref.name == "validation/runtime-result.json"
            and observation.oracle_ref.name == "validation/oracle-result.json"
            and len({ref.id for ref in observation.sanitizer_refs})
            == len(observation.sanitizer_refs)
            and [ref.name for ref in observation.sanitizer_refs]
            == [
                f"validation/sanitizer-{index:02d}.json"
                for index in range(len(observation.sanitizer_refs))
            ]
        )
        observed_sources = [ref for ref in run.artifact_refs if ref.name.startswith("sources/")]
        source_refs = {PurePosixPath(ref.name).name: ref.sha256 for ref in observed_sources}
        build_sources = {
            PurePosixPath(name).name: digest
            for name, digest in build.tool_result.typed_payload.source_manifest.items()
        }
        attestation_refs = [
            ref for ref in run.artifact_refs if ref.name == "environment/runtime-attestation.json"
        ]
        runtime_attestation = (
            RuntimeToolchainAttestation.model_validate_json(self.store.read(attestation_refs[0]))
            if len(attestation_refs) == 1
            else None
        )
        source_hash, harness_hash = source_identities(
            build.tool_result.typed_payload.source_manifest
        )
        build_ok = (
            build.success
            and build.binary_ref is not None
            and build.binary_ref == build.tool_result.typed_payload.binary_ref
            and build.tool_result.tool_name == "build"
            and _runtime_status(build.tool_result) == "SUCCESS"
            and not _failed(build.tool_result)
        )
        runtime_ok = (
            runtime.runtime_status in {"SUCCESS", "FAILED"}
            and runtime.runtime_status == _runtime_status(runtime.tool_result)
            and runtime.runtime_status == runtime.tool_result.typed_payload.runtime_status
            and runtime.output_ref == runtime.tool_result.typed_payload.output_ref
            and runtime.tool_result.tool_name == "run"
            and runtime.tool_result.typed_payload.binary_ref == build.binary_ref
            and not _failed(runtime.tool_result)
        )
        input_ref = runtime.tool_result.typed_payload.stdin_ref
        input_bytes = self.store.read(input_ref)
        input_hash = hashlib.sha256(input_bytes).hexdigest()
        recomputed_oracle = derive_oracle(
            input_bytes,
            self.store.read(runtime.output_ref),
            oracle.result.atol,
            oracle.result.rtol,
        )
        oracle_ok = (
            oracle.oracle_id == observation.oracle_id
            and oracle.result.oracle_id == observation.oracle_id
            and oracle.input_ref == input_ref
            and oracle.output_ref == runtime.output_ref
            and oracle.result == recomputed_oracle
        )
        sanitizer_ok = bool(sanitizers) and all(
            self._valid_sanitizer(result, observation, input_ref, build.binary_ref)
            for result in sanitizers
        )
        visibility = "public" if observation.split == "public" else "evaluator"
        if not (
            self.store.visibility == visibility
            and expected_result_refs
            and len(observed_sources) == 4
            and source_refs == build_sources
            and runtime_attestation is not None
            and runtime_attestation.lock_hash == binding.toolchain_lock_hash
            and binding.toolchain_lock_hash == observation.toolchain_hash
            and source_hash == observation.source_hash
            and harness_hash == observation.harness_hash
            and input_hash == observation.input_set_hash
            and build_ok
            and runtime_ok
            and oracle_ok
            and sanitizer_ok
        ):
            raise CaseExecutionAttestationUnavailable("CASE_EXECUTION_ATTESTATION_UNAVAILABLE")

    def _valid_sanitizer(
        self,
        result: SanitizerResult,
        observation: CaseExecutionObservation,
        input_ref: ArtifactRef,
        binary_ref: ArtifactRef | None,
    ) -> bool:
        tool_result = result.tool_result
        if tool_result is None:
            return False
        capture = ProcessCapture(
            exit_code=tool_result.exit_code,
            stdout=self.store.read(tool_result.stdout_artifact),
            stderr=self.store.read(tool_result.stderr_artifact),
            timed_out=tool_result.timed_out,
            elapsed_ms=tool_result.elapsed_ms,
            started_at=tool_result.started_at,
            finished_at=tool_result.finished_at,
            truncated=tool_result.truncated,
            cancelled=tool_result.cancelled,
            tool_error=tool_result.tool_error,
        )
        parsed = parse_sanitizer(observation.target_tool, capture)
        observed_findings = [
            finding.model_copy(update={"raw_ref": None}) for finding in result.findings
        ]
        return (
            result.completed
            and result.status == "COMPLETED"
            and tool_result.tool_name == "sanitizer"
            and tool_result.typed_payload.completed
            and tool_result.typed_payload.tool == observation.target_tool
            and tool_result.typed_payload.stdin_ref == input_ref
            and tool_result.typed_payload.binary_ref == binary_ref
            and result.check_outcome == tool_result.typed_payload.check_outcome
            and result.findings == tool_result.typed_payload.findings
            and result.check_outcome == parsed.check_outcome
            and result.completed == parsed.completed
            and observed_findings == parsed.findings
        )

    def validate(self, clean_run_id: str, mutant_run_id: str) -> CaseValidationArtifact:
        if clean_run_id == mutant_run_id:
            raise UnvalidatedCaseError("clean and mutant validation runs must be unique")
        clean, mutant = self._load(clean_run_id), self._load(mutant_run_id)
        left, right = clean.observation, mutant.observation
        same = all(
            first == second
            for first, second in (
                (left.case_id, right.case_id),
                (left.template_id, right.template_id),
                (left.split, right.split),
                (left.harness_hash, right.harness_hash),
                (left.input_set_hash, right.input_set_hash),
                (left.toolchain_hash, right.toolchain_hash),
                (left.oracle_id, right.oracle_id),
                (left.target_tool, right.target_tool),
                (left.expected_finding, right.expected_finding),
                (clean.run.binding, mutant.run.binding),
            )
        )
        clean_ok = (
            left.role == "clean"
            and left.mutation_id == "clean"
            and clean.runtime.runtime_status == "SUCCESS"
            and clean.oracle.result.passed
            and all(result.check_outcome == "CLEAN" for result in clean.sanitizers)
        )
        mutant_ok = (
            right.role == "mutant"
            and right.mutation_id != "clean"
            and right.source_hash != left.source_hash
            and all(result.check_outcome == "FINDING" for result in mutant.sanitizers)
            and all(
                any(finding.category == right.expected_finding for finding in result.findings)
                for result in mutant.sanitizers
            )
        )
        if not same or not clean_ok or not mutant_ok:
            raise UnvalidatedCaseError("native clean/mutant evidence does not satisfy gate")
        return CaseValidationArtifact(
            clean_run_id=clean_run_id,
            mutant_run_id=mutant_run_id,
            clean_observation_hash=clean.observation_ref.sha256,
            mutant_observation_hash=mutant.observation_ref.sha256,
        )

    def register(self, validation: CaseValidationArtifact) -> CaseManifest:
        if not isinstance(validation, CaseValidationArtifact):
            raise UnvalidatedCaseError("claimant summaries are not registration evidence")
        clean, mutant = self._load(validation.clean_run_id), self._load(validation.mutant_run_id)
        if (
            clean.observation_ref.sha256 != validation.clean_observation_hash
            or mutant.observation_ref.sha256 != validation.mutant_observation_hash
            or self.validate(validation.clean_run_id, validation.mutant_run_id) != validation
        ):
            raise UnvalidatedCaseError("validation artifact hash mismatch")
        selected = {validation.clean_run_id, validation.mutant_run_id}
        for path in self.store.root.iterdir():
            if not path.is_dir() or len(path.name) != 32:
                continue
            run = self.store.load(path.name)
            if run.kind != "benchmark_case":
                continue
            ref = self._one(run, "case-manifest.json")
            existing = CaseManifest.model_validate_json(self.store.read(ref))
            if selected.intersection(existing.validation_run_ids):
                raise UnvalidatedCaseError("validation run is already registered")
            if existing.id == mutant.observation.case_id:
                raise UnvalidatedCaseError("case ID is already registered")
            if (
                existing.template_id == mutant.observation.template_id
                and existing.split != mutant.observation.split
            ):
                raise UnvalidatedCaseError("template cannot cross public/private splits")
        observed = mutant.observation
        manifest = CaseManifest(
            id=observed.case_id,
            source_hash=observed.source_hash,
            harness_hash=observed.harness_hash,
            mutation_id=observed.mutation_id,
            template_id=observed.template_id,
            split=observed.split,
            oracle_id=observed.oracle_id,
            target_tool=observed.target_tool,
            expected_finding=observed.expected_finding,
            validation_run_ids=[validation.clean_run_id, validation.mutant_run_id],
            toolchain_hash=observed.toolchain_hash,
            input_set_hash=observed.input_set_hash,
        )
        binding: RunBinding | None = mutant.run.binding
        run = self.store.create_run("benchmark_case", binding=binding)
        self.store.put(
            run.id, "case-manifest.json", manifest.model_dump_json().encode(), self.store.visibility
        )
        self.store.transition(run.id, "RUNNING", "FINALIZING")
        self.store.transition(run.id, "COMPLETED", None)
        return manifest
