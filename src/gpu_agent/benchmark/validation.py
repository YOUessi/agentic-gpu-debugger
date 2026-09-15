"""Native corpus execution producer; no claimant-supplied outcome summaries."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

from gpu_agent.benchmark.ledger import CorpusFamily
from gpu_agent.benchmark.models import (
    AuthoritativeCaseRegistry,
    AuthoritativeCaseSpec,
    CaseExecutionObservation,
    CaseExecutionPlan,
    CaseOracleObservation,
)
from gpu_agent.contracts import ArtifactRef, RunBinding, RunStatus
from gpu_agent.execution.backend import ExecutionBackend
from gpu_agent.execution.models import (
    BuildRequest,
    ExecutionRequest,
    SanitizerRequest,
    WorkspaceRequest,
)
from gpu_agent.provenance import capture_repository_snapshot
from gpu_agent.store import RunStore, read_regular
from gpu_agent.verification.models import OracleResult
from gpu_agent.verification.oracle import NumericOracle, parse_output, reference_add

ORACLE_POLICIES = {"vector-add-cpu-v1": (1e-5, 1e-5)}


class CaseExecutionAttestationUnavailable(ValueError):
    """A selected run lacks complete native, case-bound execution evidence."""


class _VectorInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    n: int = Field(ge=1, le=65536)
    a: list[StrictFloat]
    b: list[StrictFloat]

    @model_validator(mode="after")
    def shape(self) -> "_VectorInput":
        if len(self.a) != self.n or len(self.b) != self.n:
            raise ValueError("validation input shape mismatch")
        return self


def _validate_refs(store: RunStore, run_id: str, value: object) -> None:
    if isinstance(value, ArtifactRef):
        if value.run_id != run_id or value.visibility != store.visibility:
            raise CaseExecutionAttestationUnavailable("native artifact crosses run or store")
        store.read(value)
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _validate_refs(store, run_id, getattr(value, name))
    elif isinstance(value, list):
        for item in value:
            _validate_refs(store, run_id, item)
    elif isinstance(value, dict):
        for item in value.values():
            _validate_refs(store, run_id, item)


def _put_model(store: RunStore, run_id: str, name: str, value: BaseModel) -> ArtifactRef:
    _validate_refs(store, run_id, value)
    return store.put(run_id, name, value.model_dump_json().encode(), store.visibility)


def source_identities(source_manifest: dict[str, str]) -> tuple[str, str]:
    """Derive source and harness identities from the backend-observed build manifest."""
    normalized = {PurePosixPath(name).name: digest for name, digest in source_manifest.items()}
    if len(normalized) != 4 or set(normalized) != {
        "kernel.cu",
        "vector_io.cpp",
        "vector_api.h",
        "json.hpp",
    }:
        raise CaseExecutionAttestationUnavailable("build source manifest is incomplete")
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in normalized.values()
    ):
        raise CaseExecutionAttestationUnavailable("build source hash is invalid")
    harness = json.dumps(
        sorted((name, digest) for name, digest in normalized.items() if name != "kernel.cu"),
        separators=(",", ":"),
    ).encode()
    return normalized["kernel.cu"], hashlib.sha256(harness).hexdigest()


def derive_oracle(input_bytes: bytes, output_bytes: bytes, oracle_id: str) -> OracleResult:
    try:
        atol, rtol = ORACLE_POLICIES[oracle_id]
    except KeyError as exc:
        raise ValueError("oracle implementation is not registered") from exc
    try:
        vector_input = _VectorInput.model_validate_json(input_bytes)
        actual = parse_output(output_bytes)
        return NumericOracle(atol, rtol, False, False).check(
            actual, reference_add(vector_input.a, vector_input.b)
        )
    except ValueError:
        return OracleResult(
            passed=False,
            atol=atol,
            rtol=rtol,
            nan_policy="reject",
            inf_policy="reject",
            failure_reason="INVALID_OUTPUT",
        )


class CaseValidationController:
    """Execute one controller-owned case role and persist its observation before terminalization."""

    def __init__(
        self,
        store: RunStore,
        backend: ExecutionBackend,
        binding: RunBinding,
        repository: Path,
    ) -> None:
        before = capture_repository_snapshot(repository, expected_commit=binding.repository.commit)
        if before != binding.repository:
            raise ValueError("repository differs from the validation binding")
        registry_bytes = read_regular(repository / "benchmarks/corpus-registry.json", 1024 * 1024)
        after = capture_repository_snapshot(repository, expected_commit=binding.repository.commit)
        if after != before:
            raise ValueError("repository changed while loading case registry")
        family = CorpusFamily.configured(store)
        family.reject_repository_overlap(repository)
        self._configure(store, backend, binding, registry_bytes, family)

    def _configure(
        self,
        store: RunStore,
        backend: ExecutionBackend,
        binding: RunBinding,
        registry_bytes: bytes,
        family: CorpusFamily,
    ) -> None:
        registry_hash = hashlib.sha256(registry_bytes).hexdigest()
        if (
            binding.purpose != "corpus_validation"
            or binding.toolchain_lock_hash is None
            or binding.case_registry_hash != registry_hash
            or binding.corpus_ledger_namespace_hash != family.namespace_hash
        ):
            raise ValueError("corpus controller requires a complete validation binding")
        registry = AuthoritativeCaseRegistry.model_validate_json(registry_bytes)
        if len({case.case_id for case in registry.cases}) != len(registry.cases):
            raise ValueError("authoritative case registry has duplicate identities")
        family.require_store(store)
        self.store, self.backend, self.binding, self.family = store, backend, binding, family
        self.registry_hash = registry_hash
        self.specs = MappingProxyType({case.case_id: case for case in registry.cases})

    @classmethod
    def _for_test(
        cls,
        store: RunStore,
        backend: ExecutionBackend,
        binding: RunBinding,
        registry_bytes: bytes,
    ) -> "CaseValidationController":
        """Private dependency-injection seam; production callers supply a repository path."""
        controller = cls.__new__(cls)
        controller._configure(
            store, backend, binding, registry_bytes, CorpusFamily.configured(store)
        )
        return controller

    @staticmethod
    def spec_hash(spec: AuthoritativeCaseSpec) -> str:
        return hashlib.sha256(spec.model_dump_json().encode()).hexdigest()

    def execute(self, plan: CaseExecutionPlan, input_bytes: bytes) -> str:
        expected_visibility = "public" if plan.split == "public" else "evaluator"
        if self.store.visibility != expected_visibility:
            raise ValueError("case split requires its dedicated visibility store")
        try:
            _VectorInput.model_validate_json(input_bytes)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("invalid controller validation input") from exc
        spec = self.specs.get(plan.case_id)
        source_hash, harness_hash = source_identities(plan.source_manifest)
        expected_source = (
            spec.clean_source_hash
            if spec is not None and plan.role == "clean"
            else (spec.mutant_source_hash if spec is not None else "")
        )
        if (
            spec is None
            or plan.case_registry_hash != self.registry_hash
            or plan.case_spec_hash != self.spec_hash(spec)
            or plan.template_id != spec.template_id
            or plan.mutation_id != ("clean" if plan.role == "clean" else spec.mutation_id)
            or plan.split != spec.split
            or plan.oracle_id != spec.oracle_id
            or plan.target_tool != spec.target_tool
            or plan.expected_finding != spec.expected_finding
            or plan.sanitizer_repetitions != spec.sanitizer_repetitions
            or plan.mutation_provenance_hash != spec.mutation_provenance_hash
            or source_hash != expected_source
            or harness_hash != spec.harness_hash
            or hashlib.sha256(input_bytes).hexdigest() != spec.input_set_hash
        ):
            raise ValueError("execution plan differs from authoritative case registry")
        run = self.store.create_run("case_execution", binding=self.binding)
        self.store.transition(run.id, "RUNNING", "PREPARING")
        case_spec_ref = _put_model(self.store, run.id, "validation/case-spec.json", spec)
        plan_ref = _put_model(self.store, run.id, "validation/execution-plan.json", plan)
        handle = None
        try:
            handle = self.backend.prepare(
                WorkspaceRequest(
                    run_id=run.id,
                    source_manifest=plan.source_manifest,
                    trust_level="TRUSTED_LOCAL",
                )
            )
            self.store.transition(run.id, "RUNNING", "COMPILING")
            build = self.backend.build(BuildRequest(workspace_id=handle.id))
            build_ref = _put_model(self.store, run.id, "validation/build-result.json", build)
            if not build.success or build.binary_ref is None:
                raise CaseExecutionAttestationUnavailable("validation build did not succeed")
            self.store.transition(run.id, "RUNNING", "EXECUTING")
            input_ref = self.store.put(
                run.id, "validation/input.json", input_bytes, self.store.visibility
            )
            runtime = self.backend.run(
                ExecutionRequest(workspace_id=handle.id, stdin_ref=input_ref)
            )
            runtime_ref = _put_model(self.store, run.id, "validation/runtime-result.json", runtime)
            result = derive_oracle(
                input_bytes,
                self.store.read(runtime.output_ref),
                plan.oracle_id,
            )
            oracle = CaseOracleObservation(
                oracle_id=plan.oracle_id,
                channel="ordinary",
                input_ref=input_ref,
                output_ref=runtime.output_ref,
                result=result,
            )
            oracle_ref = _put_model(self.store, run.id, "validation/oracle-result.json", oracle)
            sanitizer_refs = []
            sanitizer_oracle_refs = []
            for index in range(plan.sanitizer_repetitions):
                sanitizer = self.backend.run_sanitizer(
                    SanitizerRequest(
                        workspace_id=handle.id,
                        tool=plan.target_tool.value,
                        stdin_ref=input_ref,
                    )
                )
                sanitizer_ref = _put_model(
                    self.store,
                    run.id,
                    f"validation/sanitizer-{index:02d}.json",
                    sanitizer,
                )
                sanitizer_refs.append(sanitizer_ref)
                if sanitizer.program_output_ref is None:
                    raise CaseExecutionAttestationUnavailable(
                        "instrumented runtime output is unavailable"
                    )
                sanitizer_oracle = CaseOracleObservation(
                    oracle_id=plan.oracle_id,
                    channel="instrumented",
                    input_ref=input_ref,
                    output_ref=sanitizer.program_output_ref,
                    sanitizer_result_ref=sanitizer_ref,
                    result=derive_oracle(
                        input_bytes,
                        self.store.read(sanitizer.program_output_ref),
                        plan.oracle_id,
                    ),
                )
                sanitizer_oracle_refs.append(
                    _put_model(
                        self.store,
                        run.id,
                        f"validation/sanitizer-oracle-{index:02d}.json",
                        sanitizer_oracle,
                    )
                )
            source_hash, harness_hash = source_identities(
                build.tool_result.typed_payload.source_manifest
            )
            toolchain_hash = self.binding.toolchain_lock_hash
            assert toolchain_hash is not None
            observation = CaseExecutionObservation(
                case_id=plan.case_id,
                template_id=plan.template_id,
                mutation_id=plan.mutation_id,
                role=plan.role,
                split=plan.split,
                source_hash=source_hash,
                harness_hash=harness_hash,
                input_set_hash=hashlib.sha256(input_bytes).hexdigest(),
                toolchain_hash=toolchain_hash,
                oracle_id=plan.oracle_id,
                target_tool=plan.target_tool,
                expected_finding=plan.expected_finding,
                case_spec_ref=case_spec_ref,
                plan_ref=plan_ref,
                build_ref=build_ref,
                runtime_ref=runtime_ref,
                sanitizer_refs=sanitizer_refs,
                oracle_ref=oracle_ref,
                sanitizer_oracle_refs=sanitizer_oracle_refs,
            )
            _put_model(
                self.store, run.id, "validation/case-execution-observation.json", observation
            )
            try:
                cleaned = self.backend.cleanup(handle)
            except BaseException as cleanup_error:
                self._record_cleanup_error(run.id, cleanup_error)
                raise
            handle = None
            if not cleaned.removed:
                self._record_cleanup_error(run.id, cleaned)
                raise CaseExecutionAttestationUnavailable("validation workspace cleanup failed")
            self.store.transition(run.id, "RUNNING", "FINALIZING")
            self.store.transition(run.id, "COMPLETED", None)
            return run.id
        except BaseException as primary_error:
            if handle is not None:
                try:
                    self.backend.cleanup(handle)
                except BaseException as cleanup_error:
                    try:
                        self._record_cleanup_error(run.id, cleanup_error)
                    except BaseException:
                        pass
            current = self.store.load(run.id)
            if current.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
                self.store.transition(run.id, "FAILED", None)
            raise primary_error

    def _record_cleanup_error(self, run_id: str, error: object) -> None:
        run = self.store.load(run_id)
        if any(ref.name == "validation/cleanup-error.json" for ref in run.artifact_refs):
            return
        self.store.put(
            run_id,
            "validation/cleanup-error.json",
            json.dumps(
                {
                    "reason_code": "WORKSPACE_CLEANUP_FAILED",
                    "error_type": type(error).__name__,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            self.store.visibility,
        )
