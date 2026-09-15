"""Fail-closed registration derived only from native RunStore execution artifacts."""

import hashlib
import json
from pathlib import PurePosixPath
from typing import Any

from gpu_agent.benchmark.ledger import CorpusFamily, CorpusTransaction
from gpu_agent.benchmark.models import (
    AuthoritativeCaseSpec,
    CaseExecutionObservation,
    CaseExecutionPlan,
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
        spec: AuthoritativeCaseSpec,
        plan: CaseExecutionPlan,
        build: BuildResult,
        runtime: ExecutionResult,
        sanitizers: list[SanitizerResult],
        oracle: CaseOracleObservation,
        sanitizer_oracles: list[CaseOracleObservation],
    ) -> None:
        self.run = run
        self.observation_ref = observation_ref
        self.observation = observation
        self.spec = spec
        self.plan = plan
        self.build = build
        self.runtime = runtime
        self.sanitizers = sanitizers
        self.oracle = oracle
        self.sanitizer_oracles = sanitizer_oracles


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
        self.family = CorpusFamily.configured(store)

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
                or run.binding.case_registry_hash is None
                or run.binding.corpus_ledger_namespace_hash != self.family.namespace_hash
            ):
                raise ValueError
            observation_ref = self._one(run, "validation/case-execution-observation.json")
            observation = CaseExecutionObservation.model_validate_json(
                self.store.read(observation_ref)
            )
            _validate_refs(self.store, run_id, observation)
            spec = AuthoritativeCaseSpec.model_validate_json(
                self.store.read(observation.case_spec_ref)
            )
            plan = CaseExecutionPlan.model_validate_json(self.store.read(observation.plan_ref))
            build = BuildResult.model_validate_json(self.store.read(observation.build_ref))
            runtime = ExecutionResult.model_validate_json(self.store.read(observation.runtime_ref))
            sanitizers = [
                SanitizerResult.model_validate_json(self.store.read(ref))
                for ref in observation.sanitizer_refs
            ]
            oracle = CaseOracleObservation.model_validate_json(
                self.store.read(observation.oracle_ref)
            )
            sanitizer_oracles = [
                CaseOracleObservation.model_validate_json(self.store.read(ref))
                for ref in observation.sanitizer_oracle_refs
            ]
            for value in (spec, plan, build, runtime, sanitizers, oracle, sanitizer_oracles):
                _validate_refs(self.store, run_id, value)
        except (OSError, ValueError) as exc:
            if isinstance(exc, CaseExecutionAttestationUnavailable):
                raise
            raise CaseExecutionAttestationUnavailable(
                "CASE_EXECUTION_ATTESTATION_UNAVAILABLE"
            ) from exc
        self._validate_native(
            run, observation, spec, plan, build, runtime, sanitizers, oracle, sanitizer_oracles
        )
        return _NativeExecution(
            run,
            observation_ref,
            observation,
            spec,
            plan,
            build,
            runtime,
            sanitizers,
            oracle,
            sanitizer_oracles,
        )

    def _validate_native(
        self,
        run: RunManifest,
        observation: CaseExecutionObservation,
        spec: AuthoritativeCaseSpec,
        plan: CaseExecutionPlan,
        build: BuildResult,
        runtime: ExecutionResult,
        sanitizers: list[SanitizerResult],
        oracle: CaseOracleObservation,
        sanitizer_oracles: list[CaseOracleObservation],
    ) -> None:
        binding = run.binding
        assert binding is not None and binding.toolchain_lock_hash is not None
        expected_result_refs = (
            observation.case_spec_ref.name == "validation/case-spec.json"
            and observation.plan_ref.name == "validation/execution-plan.json"
            and observation.build_ref.name == "validation/build-result.json"
            and observation.runtime_ref.name == "validation/runtime-result.json"
            and observation.oracle_ref.name == "validation/oracle-result.json"
            and len({ref.id for ref in observation.sanitizer_refs})
            == len(observation.sanitizer_refs)
            and [ref.name for ref in observation.sanitizer_refs]
            == [
                f"validation/sanitizer-{index:02d}.json"
                for index in range(len(observation.sanitizer_refs))
            ]
            and len(sanitizer_oracles) == len(sanitizers) == plan.sanitizer_repetitions
            and [ref.name for ref in observation.sanitizer_oracle_refs]
            == [
                f"validation/sanitizer-oracle-{index:02d}.json"
                for index in range(len(observation.sanitizer_oracle_refs))
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
            and build.tool_result.stdout_artifact.name
            == f"build/{build.tool_result.request_id}/stdout"
            and build.tool_result.stderr_artifact.name
            == f"build/{build.tool_result.request_id}/stderr"
            and build.binary_ref.name == f"build/{build.tool_result.request_id}/binary"
            and _runtime_status(build.tool_result) == "SUCCESS"
            and not _failed(build.tool_result)
        )
        runtime_ok = (
            runtime.runtime_status in {"SUCCESS", "FAILED"}
            and runtime.runtime_status == _runtime_status(runtime.tool_result)
            and runtime.runtime_status == runtime.tool_result.typed_payload.runtime_status
            and runtime.output_ref == runtime.tool_result.typed_payload.output_ref
            and runtime.output_ref == runtime.tool_result.stdout_artifact
            and runtime.output_ref.name == f"run/{runtime.tool_result.request_id}/stdout"
            and runtime.tool_result.stderr_artifact.name
            == f"run/{runtime.tool_result.request_id}/stderr"
            and runtime.tool_result.tool_name == "run"
            and runtime.tool_result.typed_payload.binary_ref == build.binary_ref
            and runtime.tool_result.typed_payload.stdin_ref.name == "validation/input.json"
            and not _failed(runtime.tool_result)
        )
        input_ref = runtime.tool_result.typed_payload.stdin_ref
        input_bytes = self.store.read(input_ref)
        input_hash = hashlib.sha256(input_bytes).hexdigest()
        recomputed_oracle = derive_oracle(
            input_bytes,
            self.store.read(runtime.output_ref),
            observation.oracle_id,
        )
        oracle_ok = (
            oracle.oracle_id == observation.oracle_id
            and oracle.channel == "ordinary"
            and oracle.sanitizer_result_ref is None
            and oracle.result.oracle_id == observation.oracle_id
            and oracle.input_ref == input_ref
            and oracle.output_ref == runtime.output_ref
            and oracle.result == recomputed_oracle
        )
        sanitizer_ok = bool(sanitizers) and all(
            self._valid_sanitizer(result, observation, input_ref, build.binary_ref)
            for result in sanitizers
        )
        sanitizer_request_ids = [
            result.tool_result.request_id for result in sanitizers if result.tool_result is not None
        ]
        sanitizer_invocations_unique = len(sanitizer_request_ids) == len(sanitizers) and len(
            set(sanitizer_request_ids)
        ) == len(sanitizer_request_ids)
        expected_native_paths = {
            f"build/{build.tool_result.request_id}/binary",
            f"build/{build.tool_result.request_id}/result.json",
            f"build/{build.tool_result.request_id}/stdout",
            f"build/{build.tool_result.request_id}/stderr",
            f"run/{runtime.tool_result.request_id}/stdout",
            f"run/{runtime.tool_result.request_id}/stderr",
            f"run/{runtime.tool_result.request_id}/result.json",
        }
        for result in sanitizers:
            if result.tool_result is not None:
                prefix = f"sanitizer/{result.tool_result.request_id}"
                expected_native_paths.update(
                    {
                        f"{prefix}/program.stdout",
                        f"{prefix}/program.stderr",
                        f"{prefix}/{observation.target_tool.value}.log",
                        f"{prefix}/result.json",
                    }
                )
        observed_native_paths = {
            ref.name
            for ref in run.artifact_refs
            if ref.name.startswith(("build/", "run/", "sanitizer/"))
        }
        native_invocation_paths_exact = observed_native_paths == expected_native_paths
        native_result_models_exact = (
            self.store.read(self._one(run, f"build/{build.tool_result.request_id}/result.json"))
            == build.tool_result.model_dump_json().encode()
            and self.store.read(self._one(run, f"run/{runtime.tool_result.request_id}/result.json"))
            == runtime.tool_result.model_dump_json().encode()
            and all(
                result.tool_result is not None
                and self.store.read(
                    self._one(
                        run,
                        f"sanitizer/{result.tool_result.request_id}/result.json",
                    )
                )
                == result.tool_result.model_dump_json().encode()
                for result in sanitizers
            )
        )
        instrumented_oracle_ok = all(
            item.oracle_id == observation.oracle_id
            and item.channel == "instrumented"
            and item.input_ref == input_ref
            and item.sanitizer_result_ref == sanitizer_ref
            and item.output_ref == sanitizer.program_output_ref
            and item.result
            == derive_oracle(
                input_bytes,
                self.store.read(item.output_ref),
                observation.oracle_id,
            )
            for item, sanitizer_ref, sanitizer in zip(
                sanitizer_oracles,
                observation.sanitizer_refs,
                sanitizers,
                strict=True,
            )
        )
        plan_ok = (
            plan.case_registry_hash == binding.case_registry_hash
            and plan.case_spec_hash == hashlib.sha256(spec.model_dump_json().encode()).hexdigest()
            and plan.mutation_provenance_hash == spec.mutation_provenance_hash
            and plan.sanitizer_repetitions == spec.sanitizer_repetitions
            and plan.case_id == spec.case_id
            and plan.template_id == spec.template_id
            and plan.split == spec.split
            and plan.oracle_id == spec.oracle_id
            and plan.target_tool == spec.target_tool
            and plan.expected_finding == spec.expected_finding
            and plan.mutation_id == ("clean" if plan.role == "clean" else spec.mutation_id)
            and observation.source_hash
            == (spec.clean_source_hash if plan.role == "clean" else spec.mutant_source_hash)
            and observation.harness_hash == spec.harness_hash
            and observation.input_set_hash == spec.input_set_hash
            and {PurePosixPath(name).name: digest for name, digest in plan.source_manifest.items()}
            == build_sources
        ) and all(
            getattr(plan, name) == getattr(observation, name)
            for name in (
                "case_id",
                "template_id",
                "mutation_id",
                "role",
                "split",
                "oracle_id",
                "target_tool",
                "expected_finding",
            )
        )
        visibility = "public" if observation.split == "public" else "evaluator"
        if not (
            self.store.visibility == visibility
            and expected_result_refs
            and plan_ok
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
            and sanitizer_invocations_unique
            and native_invocation_paths_exact
            and native_result_models_exact
            and instrumented_oracle_ok
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
            and tool_result.stdout_artifact.name
            == f"sanitizer/{tool_result.request_id}/program.stdout"
            and tool_result.stderr_artifact.name
            == f"sanitizer/{tool_result.request_id}/{observation.target_tool.value}.log"
            and tool_result.typed_payload.completed
            and tool_result.typed_payload.tool == observation.target_tool
            and tool_result.typed_payload.stdin_ref == input_ref
            and tool_result.typed_payload.binary_ref == binary_ref
            and result.program_output_ref == tool_result.typed_payload.program_output_ref
            and result.program_output_ref == tool_result.stdout_artifact
            and result.program_output_ref.name
            == f"sanitizer/{tool_result.request_id}/program.stdout"
            and tool_result.typed_payload.program_stderr_ref is not None
            and tool_result.typed_payload.program_stderr_ref.name
            == f"sanitizer/{tool_result.request_id}/program.stderr"
            and result.status == tool_result.typed_payload.status == parsed.status
            and result.parser_version
            == tool_result.typed_payload.parser_version
            == parsed.parser_version
            and result.check_outcome == tool_result.typed_payload.check_outcome
            and result.findings == tool_result.typed_payload.findings
            and all(finding.tool == observation.target_tool for finding in result.findings)
            and all(finding.raw_ref == tool_result.stderr_artifact for finding in result.findings)
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
            and all(item.result.passed for item in clean.sanitizer_oracles)
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
        observed = mutant.observation
        binding = mutant.run.binding
        assert (
            binding is not None
            and binding.case_registry_hash is not None
            and binding.corpus_ledger_namespace_hash == self.family.namespace_hash
        )
        identity = observed.case_id.encode()
        template_identity = observed.template_id.encode()
        source_pair = json.dumps(
            {
                "clean_source_hash": clean.observation.source_hash,
                "mutant_source_hash": mutant.observation.source_hash,
                "harness_hash": observed.harness_hash,
                "input_set_hash": observed.input_set_hash,
                "mutation_provenance_hash": mutant.spec.mutation_provenance_hash,
                "toolchain_hash": observed.toolchain_hash,
                "oracle_id": observed.oracle_id,
                "target_tool": observed.target_tool.value,
                "expected_finding": observed.expected_finding,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        ledger = self.family.ledger
        case_hash, template_hash, source_pair_hash = ledger.identities(
            identity, template_identity, source_pair
        )
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
            ledger_namespace_hash=ledger.namespace_hash,
            case_identity_hash=case_hash,
            template_identity_hash=template_hash,
            source_pair_hash=source_pair_hash,
        )
        manifest_bytes = manifest.model_dump_json().encode()
        try:
            transaction = ledger.prepare(
                identity,
                template_identity,
                source_pair,
                store=self.store,
                manifest_hash=hashlib.sha256(manifest_bytes).hexdigest(),
            )
            self._complete_registration(transaction, binding, manifest_bytes)
            if transaction.state == "PREPARED":
                ledger.commit(transaction)
        except ValueError as exc:
            raise UnvalidatedCaseError(str(exc)) from exc
        return manifest

    def _put_exact(self, run_id: str, name: str, content: bytes) -> None:
        run = self.store.load(run_id)
        refs = [ref for ref in run.artifact_refs if ref.name == name]
        if len(refs) > 1 or (refs and self.store.read(refs[0]) != content):
            raise ValueError("registration recovery artifact mismatch")
        if not refs:
            self.store.put(run_id, name, content, self.store.visibility)

    def _complete_registration(
        self, transaction: CorpusTransaction, binding: RunBinding, manifest_bytes: bytes
    ) -> None:
        if (
            transaction.target_store_hash != self.family.ledger.target_store_hash(self.store)
            or transaction.visibility != self.store.visibility
            or transaction.manifest_hash != hashlib.sha256(manifest_bytes).hexdigest()
            or binding.corpus_ledger_namespace_hash != self.family.namespace_hash
        ):
            raise ValueError("registration transaction binding mismatch")
        run_path = self.store.root / transaction.run_id
        if not run_path.exists():
            self.store.create_run("benchmark_case", binding=binding, _run_id=transaction.run_id)
        run = self.store.load(transaction.run_id)
        if run.kind != "benchmark_case" or run.binding != binding:
            raise ValueError("registration recovery run mismatch")
        reservation = json.dumps(
            {
                "schema_version": 2,
                "transaction_id": transaction.transaction_id,
                "owner_id": transaction.owner_id,
                "ledger_namespace_hash": self.family.namespace_hash,
                "case_identity_hash": transaction.case_hash,
                "template_identity_hash": transaction.template_hash,
                "source_pair_hash": transaction.source_pair_hash,
                "target_store_hash": transaction.target_store_hash,
                "visibility": transaction.visibility,
                "expected_manifest_hash": transaction.manifest_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if run.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
            self._put_exact(run.id, "validation/ledger-transaction.json", reservation)
            self._put_exact(run.id, "case-manifest.json", manifest_bytes)
            run = self.store.load(run.id)
            if run.status == RunStatus.QUEUED:
                self.store.transition(run.id, "RUNNING", "FINALIZING")
            elif run.current_phase != "FINALIZING":
                raise ValueError("registration recovery phase mismatch")
            self.store.transition(run.id, "COMPLETED", None)
            run = self.store.load(run.id)
        if run.status != RunStatus.COMPLETED:
            raise ValueError("registration recovery run is terminally invalid")
        expected = {
            "validation/ledger-transaction.json": reservation,
            "case-manifest.json": manifest_bytes,
        }
        if {ref.name for ref in run.artifact_refs} != set(expected):
            raise ValueError("registration recovery artifacts are ambiguous")
        for name, content in expected.items():
            ref = self._one(run, name)
            if self.store.read(ref) != content:
                raise ValueError("registration recovery artifact mismatch")
