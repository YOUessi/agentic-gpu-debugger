"""ID-only verification against evaluator-owned references and private evidence.

Controller APIs register candidates; verification never accepts a command, source
path, input suite, oracle configuration, or backend from the agent. Every input
gets a fresh isolated workspace/build/run/memcheck and a private provenance record.
"""

import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import Field, StrictFloat, model_validator

from gpu_agent.contracts import ArtifactRef, ExternalRunOrigin, RunBinding, ToolResult
from gpu_agent.evidence.models import EvidenceBundle
from gpu_agent.evidence.repository import EvidenceRepository
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import (
    BuildRequest,
    ExecutionModel,
    ExecutionPayload,
    ExecutionRequest,
    Finding,
    SanitizerRequest,
    SanitizerResult,
    SanitizerTool,
    WorkspaceRequest,
)
from gpu_agent.patching import (
    PatchCandidate,
    SourceSnapshot,
    candidate_line_map,
    materialize_candidate,
)
from gpu_agent.store import RunStore, read_regular, reject_symlinks
from gpu_agent.verification.models import (
    OracleResult,
    VerificationAuditResult,
    VerificationObservation,
    VerificationResult,
)
from gpu_agent.verification.oracle import NumericOracle, parse_output, reference_add
from gpu_agent.verification.policy import (
    decide_verdict,
    finding_signature,
    original_presence,
    plan_checks,
)

TRUTH_ROOT = Path(__file__).resolve().parents[3] / "benchmarks/development_truth/case_0001"
Payload = TypeVar("Payload")


class _Case(ExecutionModel):
    oracle: Literal["vector-add-cpu-v1"]
    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)
    private_seed: int
    boundary_sizes: list[int]
    random_cases: int
    source_hashes: dict[str, str]


class _Input(ExecutionModel):
    n: int = Field(strict=True, ge=1, le=65536)
    a: list[StrictFloat]
    b: list[StrictFloat]

    @model_validator(mode="after")
    def shape(self) -> "_Input":
        if len(self.a) != self.n or len(self.b) != self.n:
            raise ValueError("invalid input shape")
        if any(not math.isfinite(v) or abs(v) > 3.4028234663852886e38 for v in self.a + self.b):
            raise ValueError("invalid float32 input")
        return self


def register_candidate(store: RunStore, candidate: PatchCandidate) -> str:
    """Controller registration returns the candidate's immutable RunStore ID.

    A separate registration lock makes the one-candidate rule atomic across controllers;
    the run/artifact itself is still persisted exclusively through RunStore.
    """
    if store.visibility != "public":
        raise ValueError("candidate registration requires public store")
    store.load(candidate.parent_run_id)
    lock_path = store.root / candidate.parent_run_id / ".candidate-lock"
    reject_symlinks(lock_path)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        for path in store.root.iterdir():
            if path.is_dir() and len(path.name) == 32:
                run = store.load(path.name)
                if run.kind == "candidate" and run.parent_run_id == candidate.parent_run_id:
                    raise ValueError("only one candidate is allowed per diagnosis")
        run = store.create_run("candidate", candidate.parent_run_id)
        store.put(run.id, "candidate.json", candidate.model_dump_json().encode(), "public")
        store.transition(run.id, "RUNNING", "FINALIZING")
        store.transition(run.id, "COMPLETED", None)
        return run.id
    finally:
        os.close(fd)


def _infrastructure_failure(result: ToolResult[Payload]) -> bool:
    return bool(result.tool_error or result.timed_out or result.cancelled or result.truncated)


class VerificationEngine:
    def __init__(self, store: RunStore, evaluator_root: Path) -> None:
        if store.visibility != "public":
            raise ValueError("public verification requires public RunStore")
        self._store = store
        self._root = evaluator_root.absolute()
        reject_symlinks(self._root)
        public = store.root.resolve()
        private = self._root.resolve()
        if private.is_relative_to(public) or public.is_relative_to(private):
            raise ValueError("evaluator root must be independent from public RunStore")
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._root.chmod(0o700)
        self._private = RunStore(self._root / "runs", visibility="evaluator")

    def _candidate(
        self, original_run_id: str, candidate_id: str, binding: RunBinding | None
    ) -> PatchCandidate:
        registration = self._store.load(candidate_id)
        if registration.kind != "candidate" or registration.parent_run_id != original_run_id:
            raise ValueError("candidate does not belong to original run")
        if registration.binding != binding:
            raise ValueError("candidate and original release bindings differ")
        refs = [r for r in registration.artifact_refs if r.name == "candidate.json"]
        if len(refs) != 1:
            raise ValueError("candidate registration must be unique")
        candidate = PatchCandidate.model_validate_json(self._store.read(refs[0]))
        if candidate.parent_run_id != original_run_id:
            raise ValueError("candidate parent mismatch")
        return candidate

    def _snapshot(self, original_run_id: str, bundle: EvidenceBundle, root: Path) -> SourceSnapshot:
        hashes: dict[str, str] = {}
        for ref in bundle.source_snapshot:
            name = Path(ref.name).name
            if (
                name not in {"kernel.cu", "vector_io.cpp", "vector_api.h", "json.hpp"}
                or name in hashes
            ):
                raise ValueError("ambiguous source snapshot")
            data = self._store.read(ref)
            IsolatedGPUBackend._write_snapshot(root / name, data)
            hashes[name] = ref.sha256
        if len(hashes) != 4:
            raise ValueError("incomplete source snapshot")
        return SourceSnapshot(parent_run_id=original_run_id, root=root, hashes=hashes)

    def _registered_tool(self, run_id: str, tool: ToolResult[Payload]) -> bool:
        """Bind bundle metadata to the immutable record written during execution."""
        refs = [
            ref
            for ref in self._store.load(run_id).artifact_refs
            if ref.name == f"{tool.tool_name}/{tool.request_id}/result.json"
        ]
        return len(refs) == 1 and self._store.read(refs[0]) == tool.model_dump_json().encode()

    def _baseline(
        self, run_id: str, bundle: EvidenceBundle, source_hashes: dict[str, str]
    ) -> tuple[list[Finding], ArtifactRef | None]:
        if len(bundle.source_snapshot) != 4 or set(source_hashes) != {
            "kernel.cu",
            "vector_io.cpp",
            "vector_api.h",
            "json.hpp",
        }:
            return [], None
        build = bundle.build_result
        if build is None or not build.success or build.binary_ref is None:
            return [], None
        built = build.tool_result
        if (
            built.tool_name != "build"
            or built.exit_code != 0
            or _infrastructure_failure(built)
            or built.typed_payload.binary_ref != build.binary_ref
            or built.typed_payload.source_manifest != source_hashes
            or not self._registered_tool(run_id, built)
        ):
            return [], None
        for result in reversed(bundle.sanitizer_results):
            tool = result.tool_result
            if tool is None:
                continue
            payload = tool.typed_payload
            if (
                tool.tool_name == "sanitizer"
                and payload.tool == "memcheck"
                and not _infrastructure_failure(tool)
                and tool.exit_code is not None
                and 0 <= tool.exit_code < 128
                and result.completed
                and payload.completed
                and result.check_outcome == payload.check_outcome == "FINDING"
                and result.findings
                and result.findings == payload.findings
                and all(
                    f.tool.value == "memcheck" and f.raw_ref == tool.stderr_artifact
                    for f in result.findings
                )
                and payload.binary_ref == build.binary_ref
                and payload.stdin_ref is not None
                and result.program_output_ref == payload.program_output_ref
                and self._registered_tool(run_id, tool)
            ):
                return result.findings, payload.stdin_ref
        return [], None

    @staticmethod
    def _suite(case: _Case) -> list[_Input]:
        rng = random.Random(case.private_seed)
        sizes = [*case.boundary_sizes, *(rng.randint(2, 2048) for _ in range(case.random_cases))]
        return [
            _Input(
                n=n,
                a=[rng.randint(-8192, 8192) / 8 for _ in range(n)],
                b=[rng.randint(-8192, 8192) / 8 for _ in range(n)],
            )
            for n in sizes
        ]

    def verify(
        self,
        original_run_id: str,
        candidate_id: str,
        mode: Literal["standard", "full"] = "standard",
    ) -> VerificationResult:
        if mode not in {"standard", "full"}:
            raise ValueError("M1 requires full public/private verification")
        original_manifest = self._store.load(original_run_id)
        candidate = self._candidate(original_run_id, candidate_id, original_manifest.binding)
        origin = ExternalRunOrigin(run_id=original_run_id, visibility="public")
        audit = self._private.create_run(
            "verification_audit",
            binding=original_manifest.binding,
            external_origin=origin,
        )
        self._private.transition(audit.id, "RUNNING", "PREPARING")
        bundle = EvidenceRepository(self._store).public_view(original_run_id)
        baseline_hashes = {Path(ref.name).name: ref.sha256 for ref in bundle.source_snapshot}
        original, input_ref = self._baseline(original_run_id, bundle, baseline_hashes)
        if input_ref is None:
            empty_observation = self._with_check_plan(VerificationObservation(), {}, mode)
            self._finish_audit(audit.id, empty_observation, 0, 0, 0, "", [])
            return self._publish(
                original_run_id,
                VerificationObservation(),
                candidate,
                {},
                [],
                0,
                "ORACLE_OR_BASELINE_UNAVAILABLE",
                mode,
                audit.id,
            )
        directory = Path(tempfile.mkdtemp(prefix="verification-", dir=self._root))
        base = directory / "base"
        base.mkdir()
        try:
            snapshot = self._snapshot(original_run_id, bundle, base)
            sources = materialize_candidate(snapshot, candidate)
            line_map = candidate_line_map(snapshot, candidate)
            case = _Case.model_validate_json(read_regular(TRUTH_ROOT / "case.json", 65536))
            if snapshot.hashes != case.source_hashes:
                empty_observation = self._with_check_plan(VerificationObservation(), {}, mode)
                self._finish_audit(audit.id, empty_observation, 0, 0, 0, "", [])
                return self._publish(
                    original_run_id,
                    VerificationObservation(),
                    candidate,
                    {},
                    [],
                    0,
                    "ORACLE_OR_BASELINE_UNAVAILABLE",
                    mode,
                    audit.id,
                )
            public_input = _Input.model_validate_json(self._store.read(input_ref))
            holdouts = self._suite(case)
            suite = [public_input, *holdouts]
            suite_bytes = json.dumps(
                [item.model_dump() for item in holdouts], sort_keys=True
            ).encode()
            suite_hash = hashlib.sha256(suite_bytes).hexdigest()
            self._private.put(audit.id, "private-suite.json", suite_bytes, "evaluator")
            self._private.put(audit.id, "case.json", case.model_dump_json().encode(), "evaluator")
            self._private.put(
                audit.id,
                "reference.cu",
                read_regular(TRUTH_ROOT / "reference.cu", 65536),
                "evaluator",
            )
            self._private.put(
                audit.id,
                "oracle-implementation.py",
                read_regular(Path(__file__).with_name("oracle.py"), 65536),
                "evaluator",
            )
            # Only the current a/b/n is sent on stdin; no reference/checker or suite is mounted.
            source_root = directory / "sources"
            source_root.mkdir()
            for name, content in sources.items():
                IsolatedGPUBackend._write_snapshot(source_root / name, content)
            backend = IsolatedGPUBackend(self._private, source_root, directory / "tasks")
            manifest = {
                name: hashlib.sha256(content).hexdigest() for name, content in sources.items()
            }
            oracle = NumericOracle(case.atol, case.rtol, False, False)
            observation = VerificationObservation()
            checks = dict(
                build="NOT_RUN",
                runtime="NOT_RUN",
                memcheck="NOT_RUN",
                public_oracle="NOT_RUN",
                private_oracle="NOT_RUN",
            )
            if mode == "full":
                checks.update(racecheck="NOT_RUN", initcheck="NOT_RUN", synccheck="NOT_RUN")
            binaries: list[str] = []
            public_passed = private_passed = 0
            child_run_ids: list[str] = []
            reason = "ALL_REQUIRED_CHECKS_PASSED"
            for index, input_data in enumerate(suite):
                run = self._private.create_run(
                    "verification_input",
                    parent_run_id=audit.id,
                    binding=original_manifest.binding,
                    external_origin=origin,
                )
                child_run_ids.append(run.id)
                self._private.put(
                    run.id,
                    "input-index.json",
                    json.dumps({"index": index}, separators=(",", ":")).encode(),
                    "evaluator",
                )
                input_content = (
                    self._store.read(input_ref)
                    if index == 0
                    else input_data.model_dump_json().encode()
                )
                stdin = self._private.put(run.id, "input.json", input_content, "evaluator")
                handle = backend.prepare(
                    WorkspaceRequest(
                        run_id=run.id, source_manifest=manifest, trust_level="UNTRUSTED"
                    )
                )
                try:
                    build = backend.build(BuildRequest(workspace_id=handle.id))
                    self._private.put(
                        run.id,
                        "provenance.json",
                        json.dumps(
                            {
                                "candidate_hash": candidate.patched_source_hash,
                                "source_manifest": manifest,
                                "binary_ref": (
                                    build.binary_ref.model_dump() if build.binary_ref else None
                                ),
                                "input_ref": stdin.model_dump(),
                            }
                        ).encode(),
                        "evaluator",
                    )
                    if not build.success:
                        missing = _infrastructure_failure(build.tool_result)
                        observation = observation.model_copy(
                            update={
                                "build_ok": None if missing else False,
                                "required_evidence_missing": missing,
                            }
                        )
                        checks["build"] = "TOOL_ERROR" if missing else "FAILED"
                        reason = "BUILD_TOOL_ERROR" if missing else "CANDIDATE_BUILD_FAILED"
                        break
                    observation = observation.model_copy(update={"build_ok": True})
                    checks["build"] = "CLEAN"
                    if build.binary_ref is None:
                        raise ValueError("successful build lacks binary provenance")
                    binaries.append(build.binary_ref.sha256)
                    expected = reference_add(input_data.a, input_data.b)
                    ordinary = backend.run(
                        ExecutionRequest(workspace_id=handle.id, stdin_ref=stdin)
                    )
                    sanitizer = backend.run_sanitizer(
                        SanitizerRequest(
                            workspace_id=handle.id, stdin_ref=stdin, timeout_seconds=120
                        )
                    )
                    sanitizers = [sanitizer]
                    if (
                        mode == "full"
                        and sanitizer.completed
                        and sanitizer.check_outcome == "CLEAN"
                    ):
                        sanitizers.extend(
                            backend.run_sanitizer(
                                SanitizerRequest(
                                    workspace_id=handle.id,
                                    stdin_ref=stdin,
                                    tool=tool.value,
                                    timeout_seconds=120,
                                )
                            )
                            for tool in (
                                SanitizerTool.RACECHECK,
                                SanitizerTool.INITCHECK,
                                SanitizerTool.SYNCCHECK,
                            )
                        )
                    for checked in sanitizers:
                        self._check_provenance(
                            build.binary_ref, stdin, ordinary.tool_result, checked
                        )
                    runtime_ok = ordinary.runtime_status == "SUCCESS"
                    runtime_infra = _infrastructure_failure(ordinary.tool_result)
                    sanitizer_infra = any(
                        not checked.completed
                        or checked.tool_result is None
                        or _infrastructure_failure(checked.tool_result)
                        for checked in sanitizers
                    )
                    infra = runtime_infra or sanitizer_infra
                    checks["runtime"] = (
                        "TOOL_ERROR" if runtime_infra else "CLEAN" if runtime_ok else "FAILED"
                    )
                    for checked in sanitizers:
                        assert checked.tool_result is not None
                        checks[checked.tool_result.typed_payload.tool] = (
                            "TOOL_ERROR"
                            if not checked.completed or _infrastructure_failure(checked.tool_result)
                            else checked.check_outcome
                        )
                    present = observation.original_finding_present
                    if index == 0:
                        present = original_presence(original, sanitizer, line_map, same_input=True)
                    original_signatures = {finding_signature(f) for f in original}
                    new_findings = [
                        f
                        for checked in sanitizers
                        for f in checked.findings
                        if index != 0
                        or finding_signature(f) not in original_signatures
                        or finding_signature(f) is None
                    ]
                    numeric_ok: bool | None = None
                    if runtime_ok and not infra:
                        try:
                            regular_check = oracle.check(
                                parse_output(self._private.read(ordinary.output_ref)), expected
                            )
                            if sanitizer.program_output_ref is None:
                                raise ValueError("missing instrumented output")
                            instrumented_checks: list[dict[str, object]] = []
                            instrumented_results: list[OracleResult] = []
                            for checked in sanitizers:
                                if (
                                    checked.program_output_ref is None
                                    or checked.tool_result is None
                                ):
                                    raise ValueError("missing instrumented output")
                                checked_oracle = oracle.check(
                                    parse_output(self._private.read(checked.program_output_ref)),
                                    expected,
                                )
                                instrumented_results.append(checked_oracle)
                                instrumented_checks.append(
                                    {
                                        "tool": checked.tool_result.typed_payload.tool,
                                        "result": checked_oracle.model_dump(),
                                    }
                                )
                            numeric_ok = regular_check.passed and all(
                                item.passed for item in instrumented_results
                            )
                            self._private.put(
                                run.id,
                                "oracle.json",
                                json.dumps(
                                    {
                                        "expected": expected,
                                        "ordinary": regular_check.model_dump(),
                                        "instrumented": instrumented_checks,
                                    }
                                ).encode(),
                                "evaluator",
                            )
                        except ValueError:
                            numeric_ok = False
                    key = "public_oracle" if index == 0 else "private_oracle"
                    checks[key] = (
                        "CLEAN"
                        if numeric_ok is True
                        else "FAILED"
                        if numeric_ok is False
                        else "NOT_RUN"
                    )
                    updates: dict[str, object] = {
                        "runtime_ok": runtime_ok,
                        "original_finding_present": present,
                        "required_evidence_missing": infra,
                        "new_blocking_findings": [
                            *observation.new_blocking_findings,
                            *new_findings,
                        ],
                        "public_oracle_passed"
                        if index == 0
                        else "private_holdout_passed": numeric_ok,
                    }
                    observation = observation.model_copy(update=updates)
                    passed = (
                        runtime_ok
                        and numeric_ok is True
                        and all(
                            checked.completed and checked.check_outcome == "CLEAN"
                            for checked in sanitizers
                        )
                    )
                    if index == 0:
                        public_passed += int(passed)
                    else:
                        private_passed += int(passed)
                    if not passed or infra or present is not False:
                        reason = (
                            "REQUIRED_EVIDENCE_MISSING"
                            if infra
                            else "ORIGINAL_FINDING_PRESENT"
                            if present
                            else "ORIGINAL_FINDING_UNKNOWN"
                            if present is None
                            else "NEW_BLOCKING_FINDING"
                            if new_findings
                            else "RUNTIME_FAILED"
                            if not runtime_ok
                            else "ORACLE_FAILED"
                            if numeric_ok is False
                            else "ORACLE_NOT_RUN"
                        )
                        break
                finally:
                    backend.cleanup(handle)
                    self._private.transition(run.id, "RUNNING", "FINALIZING")
                    self._private.transition(run.id, "COMPLETED", None)
            if len(child_run_ids) < len(suite) and observation.private_holdout_passed is True:
                observation = observation.model_copy(update={"private_holdout_passed": None})
                checks["private_oracle"] = "INCOMPLETE"
            observation = self._with_check_plan(observation, checks, mode)
            self._finish_audit(
                audit.id,
                observation,
                public_passed,
                private_passed,
                len(suite) - len(child_run_ids),
                suite_hash,
                child_run_ids,
            )
            return self._publish(
                original_run_id,
                observation,
                candidate,
                checks,
                binaries,
                public_passed,
                reason,
                mode,
                audit.id,
            )
        finally:
            shutil.rmtree(directory)

    @staticmethod
    def _check_provenance(
        binary: ArtifactRef,
        stdin: ArtifactRef,
        ordinary: ToolResult[ExecutionPayload],
        sanitizer: SanitizerResult,
    ) -> None:
        payload = ordinary.typed_payload
        if (
            getattr(payload, "binary_ref", None) != binary
            or getattr(payload, "stdin_ref", None) != stdin
        ):
            raise ValueError("runtime provenance mismatch")
        if sanitizer.tool_result is not None:
            checked = sanitizer.tool_result.typed_payload
            if checked.binary_ref != binary or checked.stdin_ref != stdin:
                raise ValueError("sanitizer provenance mismatch")

    def _publish(
        self,
        original_run_id: str,
        observation: VerificationObservation,
        candidate: PatchCandidate,
        checks: dict[str, str],
        binaries: list[str],
        public_passed: int,
        reason: str,
        mode: Literal["standard", "full"] = "standard",
        evaluator_audit_run_id: str | None = None,
    ) -> VerificationResult:
        observation = self._with_check_plan(observation, checks, mode)
        requirements = observation.check_requirements
        outcomes = observation.check_outcomes
        not_run_reasons = {
            item.tool: "UPSTREAM_CHECK_DID_NOT_PASS"
            for item in requirements
            if item.required and outcomes[item.tool] == "NOT_RUN"
        }
        # Explicit allowlist construction: never dump/copy private evidence into public storage.
        result = VerificationResult(
            verdict=decide_verdict(observation),
            failure_stage=None if reason == "ALL_REQUIRED_CHECKS_PASSED" else "verification",
            reason_code=reason,
            original_finding_present=observation.original_finding_present,
            public_oracle_passed=observation.public_oracle_passed,
            required_checks={
                key: value for key, value in checks.items() if key != "private_oracle"
            },
            check_requirements=requirements,
            check_outcomes=outcomes,
            not_run_reasons=not_run_reasons,
            new_findings=len(observation.new_blocking_findings),
            candidate_hash=candidate.patched_source_hash,
            binary_hashes=sorted(set(binaries)),
            public_passed_count=public_passed,
            evaluator_audit_run_id=evaluator_audit_run_id,
            limitations=["Containers share the host kernel and GPU driver."],
        )
        run = self._store.create_run("verification", original_run_id)
        self._store.put(
            run.id, "verification/result.json", result.model_dump_json().encode(), "public"
        )
        self._store.transition(run.id, "RUNNING", "FINALIZING")
        self._store.transition(run.id, "COMPLETED", None)
        return result

    @staticmethod
    def _with_check_plan(
        observation: VerificationObservation,
        checks: dict[str, str],
        mode: Literal["standard", "full"],
    ) -> VerificationObservation:
        requirements = plan_checks(
            SanitizerTool.MEMCHECK,
            "strict" if mode == "full" else "standard",
            {tool: "SUPPORTED" for tool in SanitizerTool},
        )
        outcomes = {item.tool: checks.get(item.tool.value, "NOT_RUN") for item in requirements}
        return observation.model_copy(
            update={"check_requirements": requirements, "check_outcomes": outcomes}
        )

    def _finish_audit(
        self,
        audit_run_id: str,
        observation: VerificationObservation,
        public_passed_count: int,
        private_passed_count: int,
        not_run_count: int,
        suite_hash: str,
        child_run_ids: list[str],
    ) -> ArtifactRef:
        self._private.put(
            audit_run_id,
            "observation.json",
            observation.model_dump_json().encode(),
            "evaluator",
        )
        result = VerificationAuditResult(
            observation=observation,
            public_passed_count=public_passed_count,
            private_passed_count=private_passed_count,
            not_run_count=not_run_count,
            suite_hash=suite_hash,
            child_run_ids=child_run_ids,
        )
        ref = self._private.put(
            audit_run_id,
            "verification/audit-result.json",
            result.model_dump_json().encode(),
            "evaluator",
        )
        self._private.transition(audit_run_id, "RUNNING", "FINALIZING")
        self._private.transition(audit_run_id, "COMPLETED", None)
        return ref
