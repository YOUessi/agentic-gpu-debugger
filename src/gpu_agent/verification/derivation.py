"""Single native-evidence derivation for evaluator and public verification views.

The evaluator view is derived from every ordered child.  The public projector is
deliberately handed only child zero, so a private outcome cannot influence any
public field (including verdict, reason, counts, checks, limitations, or hashes).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from gpu_agent.contracts import (
    ArtifactRef,
    ExternalRunOrigin,
    RunBinding,
    RunManifest,
    RunStatus,
    ToolResult,
)
from gpu_agent.evidence.repository import EvidenceRepository
from gpu_agent.evidence.sanitizer import parse_sanitizer
from gpu_agent.execution.models import (
    BuildPayload,
    ExecutionPayload,
    Finding,
    SanitizerPayload,
    SanitizerTool,
)
from gpu_agent.execution.process import ProcessCapture
from gpu_agent.store import RunStore, read_regular
from gpu_agent.verification.models import (
    CheckRequirement,
    OracleResult,
    VerificationAuditResult,
    VerificationObservation,
    VerificationResult,
    VerificationSuiteSpec,
)
from gpu_agent.verification.oracle import NumericOracle, parse_output, reference_add
from gpu_agent.verification.policy import decide_verdict, finding_signature, plan_checks

_TRUTH_CASE = (
    Path(__file__).resolve().parents[3] / "benchmarks/development_truth/case_0001/case.json"
)
_LIMITATIONS = ["Containers share the host kernel and GPU driver."]


@dataclass(frozen=True)
class _ChildFacts:
    index: int
    run_id: str
    build_state: str
    runtime_state: str
    sanitizer_states: dict[SanitizerTool, str]
    oracle_passed: bool | None
    findings: tuple[Finding, ...]
    memcheck_findings: tuple[Finding, ...]
    memcheck_completed: bool
    infrastructure_missing: bool
    binary_hash: str | None
    passed: bool


@dataclass(frozen=True)
class VerificationDerivation:
    audit: VerificationAuditResult
    public: VerificationResult


def _one_ref(run: RunManifest, name: str) -> ArtifactRef:
    refs = [ref for ref in run.artifact_refs if ref.name == name]
    if len(refs) != 1:
        raise ValueError("verification lineage artifact is missing or ambiguous")
    return refs[0]


def _infra(
    result: ToolResult[BuildPayload] | ToolResult[ExecutionPayload] | ToolResult[SanitizerPayload],
) -> bool:
    return bool(result.tool_error or result.timed_out or result.cancelled or result.truncated)


def _runtime_status(result: ToolResult[ExecutionPayload]) -> str:
    if result.cancelled:
        return "CANCELLED"
    if result.timed_out:
        return "TIMEOUT"
    if result.tool_error:
        return "TOOL_ERROR"
    if result.truncated:
        return "TRUNCATED"
    return "SUCCESS" if result.exit_code == 0 else "FAILED"


def _state(states: list[str], *, absent: str = "NOT_RUN") -> str:
    if not states:
        return absent
    if all(item == "NOT_RUN" for item in states):
        return absent
    if "TOOL_ERROR" in states:
        return "TOOL_ERROR"
    if all(item == "CLEAN" for item in states):
        return "CLEAN"
    if "FINDING" in states:
        return "FINDING"
    return "FAILED"


def _requirements(mode: str) -> list[CheckRequirement]:
    return plan_checks(
        SanitizerTool.MEMCHECK,
        "strict" if mode == "full" else "standard",
        {tool: "SUPPORTED" for tool in SanitizerTool},
    )


def _reason(
    facts: list[_ChildFacts],
    original_present: bool | None,
    new_findings: list[Finding],
    *,
    expected_count: int,
) -> str:
    if not facts:
        return "REQUIRED_EVIDENCE_MISSING"
    for fact in facts:
        if fact.build_state == "TOOL_ERROR":
            return "BUILD_TOOL_ERROR"
        if fact.build_state == "FAILED":
            return "CANDIDATE_BUILD_FAILED"
        if fact.runtime_state == "TOOL_ERROR":
            return "RUNTIME_TOOL_ERROR"
        if "TOOL_ERROR" in fact.sanitizer_states.values():
            return "SANITIZER_TOOL_ERROR"
        if fact.index == 0 and original_present is True:
            return "ORIGINAL_FINDING_PRESENT"
        if fact.index == 0 and original_present is None:
            return "ORIGINAL_FINDING_UNKNOWN"
        child_new = list(fact.findings) if fact.index > 0 else new_findings
        if child_new:
            return "NEW_BLOCKING_FINDING"
        if fact.runtime_state == "FAILED":
            return "RUNTIME_FAILED"
        if fact.oracle_passed is False:
            return "ORACLE_FAILED"
        if fact.oracle_passed is None:
            return "ORACLE_NOT_RUN"
        if not fact.passed:
            return "REQUIRED_CHECK_FAILED"
    if len(facts) != expected_count:
        return "PRIVATE_EVIDENCE_INCOMPLETE"
    return "ALL_REQUIRED_CHECKS_PASSED"


def _observation(
    facts: list[_ChildFacts],
    requirements: list[CheckRequirement],
    original_present: bool | None,
    new_findings: list[Finding],
    private_passed: bool | None,
) -> tuple[VerificationObservation, dict[str, str]]:
    checks: dict[str, str] = {
        "build": _state([item.build_state for item in facts]),
        "runtime": _state([item.runtime_state for item in facts]),
        "public_oracle": (
            "CLEAN"
            if facts and facts[0].oracle_passed is True
            else "FAILED"
            if facts and facts[0].oracle_passed is False
            else "NOT_RUN"
        ),
    }
    for tool in SanitizerTool:
        states = [item.sanitizer_states[tool] for item in facts if tool in item.sanitizer_states]
        if states or any(req.tool == tool and req.required for req in requirements):
            checks[tool.value] = _state(states)
    outcomes = {item.tool: checks.get(item.tool.value, "NOT_RUN") for item in requirements}
    build_states = [item.build_state for item in facts]
    runtime_states = [item.runtime_state for item in facts if item.runtime_state != "NOT_RUN"]
    observation = VerificationObservation(
        build_ok=(
            None
            if not build_states or "TOOL_ERROR" in build_states
            else all(item == "CLEAN" for item in build_states)
        ),
        runtime_ok=(
            None
            if not runtime_states or "TOOL_ERROR" in runtime_states
            else all(item == "CLEAN" for item in runtime_states)
        ),
        original_finding_present=original_present,
        public_oracle_passed=facts[0].oracle_passed if facts else None,
        private_holdout_passed=private_passed,
        required_evidence_missing=any(item.infrastructure_missing for item in facts),
        new_blocking_findings=new_findings,
        check_requirements=requirements,
        check_outcomes=outcomes,
    )
    return observation, checks


def _not_run(observation: VerificationObservation) -> dict[SanitizerTool, str]:
    return {
        item.tool: "UPSTREAM_CHECK_DID_NOT_PASS"
        for item in observation.check_requirements
        if item.required and observation.check_outcomes.get(item.tool) == "NOT_RUN"
    }


def _public_projection(
    fact: _ChildFacts,
    mode: str,
    candidate_hash: str,
    original_present: bool | None,
    public_new_findings: list[Finding],
    audit_run_id: str,
) -> VerificationResult:
    """Project only child-zero facts.  No full-suite object is accepted here."""
    requirements = _requirements(mode)
    observation, checks = _observation(
        [fact], requirements, original_present, public_new_findings, True
    )
    reason = _reason([fact], original_present, public_new_findings, expected_count=1)
    verdict = decide_verdict(observation)
    return VerificationResult(
        verdict=verdict,
        failure_stage=None if reason == "ALL_REQUIRED_CHECKS_PASSED" else "verification",
        reason_code=reason,
        original_finding_present=original_present,
        public_oracle_passed=fact.oracle_passed,
        required_checks=checks,
        check_requirements=requirements,
        check_outcomes=observation.check_outcomes,
        not_run_reasons=_not_run(observation),
        new_findings=len(public_new_findings),
        candidate_hash=candidate_hash,
        binary_hashes=[fact.binary_hash] if fact.binary_hash else [],
        public_passed_count=int(fact.passed),
        evaluator_audit_run_id=audit_run_id,
        limitations=_LIMITATIONS,
    )


def derive_verification(
    public: RunStore,
    evaluator: RunStore,
    audit_run_id: str,
    diagnosis_run_id: str,
    binding: RunBinding | None,
) -> VerificationDerivation:
    """Derive both views afresh from immutable native child artifacts."""
    if public.visibility != "public" or evaluator.visibility != "evaluator":
        raise ValueError("verification stores have invalid visibility")
    audit_run = evaluator.load(audit_run_id)
    if (
        audit_run.kind != "verification_audit"
        or audit_run.status not in {RunStatus.RUNNING, RunStatus.COMPLETED}
        or audit_run.binding != binding
        or audit_run.external_origin
        != ExternalRunOrigin(run_id=diagnosis_run_id, visibility="public")
    ):
        raise ValueError("evaluation verification audit is invalid")
    spec = VerificationSuiteSpec.model_validate_json(
        evaluator.read(_one_ref(audit_run, "verification/suite-spec.json"))
    )
    trusted_case = json.loads(read_regular(_TRUTH_CASE, 65536))
    if json.loads(evaluator.read(_one_ref(audit_run, "case.json"))) != trusted_case:
        raise ValueError("evaluation verification case policy is invalid")
    rng = random.Random(trusted_case["private_seed"])
    sizes = [
        *trusted_case["boundary_sizes"],
        *(rng.randint(2, 2048) for _ in range(trusted_case["random_cases"])),
    ]
    holdouts = [
        {
            "n": n,
            "a": [rng.randint(-8192, 8192) / 8 for _ in range(n)],
            "b": [rng.randint(-8192, 8192) / 8 for _ in range(n)],
        }
        for n in sizes
    ]
    suite_bytes = json.dumps(holdouts, sort_keys=True).encode()
    if (
        evaluator.read(_one_ref(audit_run, "private-suite.json")) != suite_bytes
        or spec.suite_hash != hashlib.sha256(suite_bytes).hexdigest()
        or spec.expected_child_count != 1 + len(holdouts)
    ):
        raise ValueError("evaluation verification suite is invalid")

    children: dict[int, RunManifest] = {}
    for path in evaluator.root.iterdir():
        if not path.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", path.name):
            continue
        child = evaluator.load(path.name)
        if child.kind != "verification_input" or child.parent_run_id != audit_run.id:
            continue
        index = json.loads(evaluator.read(_one_ref(child, "input-index.json"))).get("index")
        if (
            type(index) is not int
            or index < 0
            or index >= spec.expected_child_count
            or index in children
            or child.status != RunStatus.COMPLETED
            or child.binding != binding
            or child.external_origin
            != ExternalRunOrigin(run_id=diagnosis_run_id, visibility="public")
        ):
            raise ValueError("evaluation verification input lineage is invalid")
        children[index] = child
    if not children or sorted(children) != list(range(len(children))):
        raise ValueError("evaluation verification input sequence is invalid")

    baseline_bundle = EvidenceRepository(public).public_view(diagnosis_run_id)
    baseline_items = [
        item
        for item in baseline_bundle.sanitizer_results
        if item.tool_result is not None
        and item.tool_result.typed_payload.tool == SanitizerTool.MEMCHECK
    ]
    if len(baseline_items) != 1 or baseline_items[0].tool_result is None:
        raise ValueError("evaluation verification baseline sanitizer is invalid")
    baseline = baseline_items[0].tool_result
    baseline_capture = ProcessCapture(
        exit_code=baseline.exit_code,
        stdout=public.read(baseline.stdout_artifact),
        stderr=public.read(baseline.stderr_artifact),
        timed_out=baseline.timed_out,
        elapsed_ms=baseline.elapsed_ms,
        started_at=baseline.started_at,
        finished_at=baseline.finished_at,
        truncated=baseline.truncated,
        cancelled=baseline.cancelled,
        tool_error=baseline.tool_error,
    )
    baseline_reparsed = parse_sanitizer(SanitizerTool.MEMCHECK, baseline_capture)
    baseline_findings = [
        item.model_copy(update={"raw_ref": baseline.stderr_artifact})
        for item in baseline_reparsed.findings
    ]
    baseline_payload = baseline.typed_payload
    if (
        baseline_reparsed.status != baseline_payload.status
        or baseline_reparsed.completed != baseline_payload.completed
        or baseline_reparsed.parser_version != baseline_payload.parser_version
        or baseline_reparsed.check_outcome != baseline_payload.check_outcome
        or baseline_findings != baseline_payload.findings
        or baseline_items[0].findings != baseline_findings
        or baseline_payload.program_output_ref != baseline.stdout_artifact
    ):
        raise ValueError(
            "evaluation verification baseline sanitizer is not reproducible: "
            f"{baseline_reparsed.status != baseline_payload.status=}, "
            f"{baseline_reparsed.completed != baseline_payload.completed=}, "
            f"{baseline_reparsed.parser_version != baseline_payload.parser_version=}, "
            f"{baseline_reparsed.check_outcome != baseline_payload.check_outcome=}, "
            f"{baseline_findings != baseline_payload.findings=}, "
            f"{baseline_items[0].findings != baseline_findings=}, "
            f"{baseline_payload.program_output_ref != baseline.stdout_artifact=}"
        )
    baseline_signatures = {finding_signature(item) for item in baseline_findings}
    oracle = NumericOracle(trusted_case["atol"], trusted_case["rtol"], False, False)
    facts: list[_ChildFacts] = []

    for index, child in sorted(children.items()):
        input_ref = _one_ref(child, "input.json")
        input_raw = evaluator.read(input_ref)
        input_payload = json.loads(input_raw)
        if (
            set(input_payload) != {"n", "a", "b"}
            or type(input_payload["n"]) is not int
            or len(input_payload["a"]) != input_payload["n"]
            or len(input_payload["b"]) != input_payload["n"]
            or (index > 0 and input_payload != holdouts[index - 1])
        ):
            raise ValueError("evaluation verification input differs from frozen suite")
        expected = reference_add(input_payload["a"], input_payload["b"])
        provenance = json.loads(evaluator.read(_one_ref(child, "provenance.json")))
        result_refs = [
            ref
            for ref in child.artifact_refs
            if ref.name.startswith(("build/", "run/", "sanitizer/"))
            and ref.name.endswith("/result.json")
        ]
        native_refs = [
            ref
            for ref in result_refs
            if re.fullmatch(r"(build|run|sanitizer)/[a-f0-9]{32}/result\.json", ref.name)
        ]
        kinds = [ref.name.split("/", 1)[0] for ref in native_refs]
        if (
            native_refs != result_refs
            or kinds.count("build") != 1
            or kinds.count("run") > 1
            or (kinds.count("run") == 1 and kinds.count("sanitizer") < 1)
        ):
            raise ValueError("evaluation verification input lacks native tool output")
        build: ToolResult[BuildPayload] | None = None
        run: ToolResult[ExecutionPayload] | None = None
        sanitizers: list[ToolResult[SanitizerPayload]] = []
        sanitizer_states: dict[SanitizerTool, str] = {}
        findings: list[Finding] = []
        memcheck_findings: tuple[Finding, ...] = ()
        memcheck_completed = False
        for ref in native_refs:
            raw = evaluator.read(ref)
            payload = json.loads(raw)
            tool_name, request_id = ref.name.split("/")[:2]
            if payload.get("tool_name") != tool_name or payload.get("request_id") != request_id:
                raise ValueError("evaluation verification tool output is spliced")
            if tool_name == "build":
                build_model = ToolResult[BuildPayload].model_validate_json(raw)
                build = build_model
                native_artifacts = (build_model.stdout_artifact, build_model.stderr_artifact)
            elif tool_name == "run":
                run_model = ToolResult[ExecutionPayload].model_validate_json(raw)
                run = run_model
                native_artifacts = (run_model.stdout_artifact, run_model.stderr_artifact)
            else:
                sanitizer_model = ToolResult[SanitizerPayload].model_validate_json(raw)
                capture = ProcessCapture(
                    exit_code=sanitizer_model.exit_code,
                    stdout=evaluator.read(sanitizer_model.stdout_artifact),
                    stderr=evaluator.read(sanitizer_model.stderr_artifact),
                    timed_out=sanitizer_model.timed_out,
                    elapsed_ms=sanitizer_model.elapsed_ms,
                    started_at=sanitizer_model.started_at,
                    finished_at=sanitizer_model.finished_at,
                    truncated=sanitizer_model.truncated,
                    cancelled=sanitizer_model.cancelled,
                    tool_error=sanitizer_model.tool_error,
                )
                reparsed = parse_sanitizer(
                    SanitizerTool(sanitizer_model.typed_payload.tool), capture
                )
                raw_findings = [
                    item.model_copy(update={"raw_ref": sanitizer_model.stderr_artifact})
                    for item in reparsed.findings
                ]
                if (
                    reparsed.status != sanitizer_model.typed_payload.status
                    or reparsed.completed != sanitizer_model.typed_payload.completed
                    or reparsed.parser_version != sanitizer_model.typed_payload.parser_version
                    or reparsed.check_outcome != sanitizer_model.typed_payload.check_outcome
                    or raw_findings != sanitizer_model.typed_payload.findings
                ):
                    raise ValueError("evaluation verification sanitizer result is not reproducible")
                sanitizers.append(sanitizer_model)
                tool = SanitizerTool(sanitizer_model.typed_payload.tool)
                sanitizer_infra = _infra(sanitizer_model) or not reparsed.completed
                sanitizer_states[tool] = (
                    "TOOL_ERROR"
                    if sanitizer_infra
                    else "CLEAN"
                    if reparsed.check_outcome == "CLEAN"
                    else "FINDING"
                )
                findings.extend(raw_findings)
                if tool == SanitizerTool.MEMCHECK:
                    memcheck_findings = tuple(raw_findings)
                    memcheck_completed = reparsed.completed and not sanitizer_infra
                native_artifacts = (
                    sanitizer_model.stdout_artifact,
                    sanitizer_model.stderr_artifact,
                )
            for nested in native_artifacts:
                if nested.run_id != child.id or nested.visibility != "evaluator":
                    raise ValueError("evaluation verification tool artifact crosses lineage")
                evaluator.read(nested)
        if build is None:
            raise ValueError("evaluation verification build lineage is missing")
        binary = build.typed_payload.binary_ref
        build_infra = _infra(build)
        build_ok = build.exit_code == 0 and binary is not None and not build_infra
        build_state = "TOOL_ERROR" if build_infra else "CLEAN" if build_ok else "FAILED"
        expected_tools = [SanitizerTool.MEMCHECK]
        if (
            spec.mode == "full"
            and sanitizers
            and SanitizerTool(sanitizers[0].typed_payload.tool) == SanitizerTool.MEMCHECK
            and sanitizer_states.get(SanitizerTool.MEMCHECK) == "CLEAN"
        ):
            expected_tools.extend(
                [SanitizerTool.RACECHECK, SanitizerTool.INITCHECK, SanitizerTool.SYNCCHECK]
            )
        observed_tools = [SanitizerTool(item.typed_payload.tool) for item in sanitizers]
        if build_ok:
            if run is None or observed_tools != expected_tools:
                raise ValueError("evaluation verification required tool multiplicity is invalid")
        elif run is not None or sanitizers:
            raise ValueError("evaluation verification ran tools after a failed build")
        if binary is not None:
            evaluator.read(binary)
        if (
            provenance.get("candidate_hash") != spec.candidate_hash
            or provenance.get("source_manifest") != build.typed_payload.source_manifest
            or provenance.get("binary_ref")
            != (binary.model_dump(mode="json") if binary is not None else None)
            or provenance.get("input_ref") != input_ref.model_dump(mode="json")
        ):
            raise ValueError("evaluation verification provenance is invalid")
        if run is not None and (
            binary is None
            or run.typed_payload.binary_ref != binary
            or run.typed_payload.stdin_ref != input_ref
            or run.typed_payload.output_ref != run.stdout_artifact
            or run.typed_payload.runtime_status != _runtime_status(run)
        ):
            raise ValueError("evaluation verification runtime links are invalid")
        if any(
            binary is None
            or checked.typed_payload.binary_ref != binary
            or checked.typed_payload.stdin_ref != input_ref
            or checked.typed_payload.program_output_ref != checked.stdout_artifact
            for checked in sanitizers
        ):
            raise ValueError("evaluation verification sanitizer links are invalid")
        runtime_infra = bool(run and _infra(run))
        runtime_ok = bool(
            run
            and run.typed_payload.runtime_status == "SUCCESS"
            and run.typed_payload.output_ref == run.stdout_artifact
            and not runtime_infra
        )
        runtime_state = (
            "NOT_RUN"
            if run is None
            else "TOOL_ERROR"
            if runtime_infra
            else "CLEAN"
            if runtime_ok
            else "FAILED"
        )
        sanitizer_infra = any(value == "TOOL_ERROR" for value in sanitizer_states.values())
        oracle_passed: bool | None = None
        oracle_refs = [ref for ref in child.artifact_refs if ref.name == "oracle.json"]
        if len(oracle_refs) > 1:
            raise ValueError("evaluation verification oracle output is ambiguous")
        if runtime_ok and not runtime_infra and not sanitizer_infra:
            assert run is not None
            try:
                ordinary = oracle.check(parse_output(evaluator.read(run.stdout_artifact)), expected)
                instrumented: list[dict[str, object]] = []
                instrumented_results: list[OracleResult] = []
                for checked in sanitizers:
                    checked_oracle = oracle.check(
                        parse_output(evaluator.read(checked.stdout_artifact)), expected
                    )
                    instrumented_results.append(checked_oracle)
                    instrumented.append(
                        {
                            "tool": checked.typed_payload.tool,
                            "result": checked_oracle.model_dump(mode="json"),
                        }
                    )
                expected_oracle = {
                    "expected": expected,
                    "ordinary": ordinary.model_dump(mode="json"),
                    "instrumented": instrumented,
                }
                if (
                    len(oracle_refs) != 1
                    or json.loads(evaluator.read(oracle_refs[0])) != expected_oracle
                ):
                    raise ValueError("evaluation verification oracle result is not reproducible")
                oracle_passed = ordinary.passed and all(
                    item.passed for item in instrumented_results
                )
            except ValueError:
                if oracle_refs:
                    raise
                oracle_passed = False
        elif oracle_refs:
            raise ValueError("evaluation verification oracle ran without complete native evidence")
        infrastructure_missing = build_infra or runtime_infra or sanitizer_infra
        passed = bool(
            build_ok
            and runtime_ok
            and oracle_passed is True
            and sanitizers
            and all(value == "CLEAN" for value in sanitizer_states.values())
        )
        facts.append(
            _ChildFacts(
                index=index,
                run_id=child.id,
                build_state=build_state,
                runtime_state=runtime_state,
                sanitizer_states=sanitizer_states,
                oracle_passed=oracle_passed,
                findings=tuple(findings),
                memcheck_findings=memcheck_findings,
                memcheck_completed=memcheck_completed,
                infrastructure_missing=infrastructure_missing,
                binary_hash=binary.sha256 if binary else None,
                passed=passed,
            )
        )

    public_input = _one_ref(children[0], "input.json")
    same_public_input = bool(
        baseline_payload.stdin_ref is not None
        and baseline_payload.stdin_ref.sha256 == public_input.sha256
        and public.read(baseline_payload.stdin_ref) == evaluator.read(public_input)
    )
    public_fact = facts[0]
    original_present: bool | None = None
    if same_public_input and baseline_signatures and public_fact.memcheck_completed:
        candidate_signatures = {finding_signature(item) for item in public_fact.memcheck_findings}
        original_present = any(
            signature is not None and signature in candidate_signatures
            for signature in baseline_signatures
        )
    public_new = [
        item
        for item in public_fact.findings
        if finding_signature(item) not in baseline_signatures or finding_signature(item) is None
    ]
    private_findings = [item for fact in facts[1:] for item in fact.findings]
    full_new = [*public_new, *private_findings]
    private_facts = facts[1:]
    private_passed: bool | None = (
        False
        if any(not item.passed for item in private_facts)
        else True
        if len(private_facts) == spec.expected_child_count - 1
        else None
    )
    requirements = _requirements(spec.mode)
    observation, checks = _observation(
        facts, requirements, original_present, full_new, private_passed
    )
    checks["private_oracle"] = (
        "CLEAN" if private_passed is True else "FAILED" if private_passed is False else "INCOMPLETE"
    )
    reason = _reason(facts, original_present, public_new, expected_count=spec.expected_child_count)
    audit = VerificationAuditResult(
        observation=observation,
        public_passed_count=int(public_fact.passed),
        private_passed_count=sum(item.passed for item in private_facts),
        not_run_count=spec.expected_child_count - len(facts),
        suite_hash=spec.suite_hash,
        child_run_ids=[item.run_id for item in facts],
        verdict=decide_verdict(observation),
        failure_stage=None if reason == "ALL_REQUIRED_CHECKS_PASSED" else "verification",
        reason_code=reason,
        required_checks=checks,
        not_run_reasons=_not_run(observation),
        binary_hashes=sorted({item.binary_hash for item in facts if item.binary_hash}),
        limitations=_LIMITATIONS,
    )
    return VerificationDerivation(
        audit=audit,
        public=_public_projection(
            public_fact,
            spec.mode,
            spec.candidate_hash,
            original_present,
            public_new,
            audit_run.id,
        ),
    )


def validate_persisted_derivation(
    public: RunStore,
    evaluator: RunStore | None,
    diagnosis_run_id: str,
    result: VerificationResult,
    binding: RunBinding | None,
) -> VerificationAuditResult:
    """Resolve, freshly derive, and compare all stored summaries fail-closed."""
    if evaluator is None or result.evaluator_audit_run_id is None:
        raise ValueError("evaluation verification has no evaluator audit lineage")
    derived = derive_verification(
        public, evaluator, result.evaluator_audit_run_id, diagnosis_run_id, binding
    )
    audit_run = evaluator.load(result.evaluator_audit_run_id)
    stored_observation = VerificationObservation.model_validate_json(
        evaluator.read(_one_ref(audit_run, "observation.json"))
    )
    stored_audit = VerificationAuditResult.model_validate_json(
        evaluator.read(_one_ref(audit_run, "verification/audit-result.json"))
    )
    if stored_observation != derived.audit.observation or stored_audit != derived.audit:
        raise ValueError("evaluation verification audit differs from native derivation")
    if result != derived.public:
        raise ValueError("evaluation verification projection differs from native derivation")
    return derived.audit


def resolve_verified_public_projection(
    public: RunStore,
    evaluator: RunStore | None,
    diagnosis_run_id: str,
    result: VerificationResult,
    binding: RunBinding | None,
) -> VerificationResult:
    """Return a public result only after fresh native-evidence validation."""
    validate_persisted_derivation(public, evaluator, diagnosis_run_id, result, binding)
    return result
