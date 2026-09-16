"""Production adapter from registered controller cases to immutable public observations."""

import fcntl
import hashlib
import json
import os
import random
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from gpu_agent.agent.models import (
    ACTION_ADAPTER,
    AcquisitionUsage,
    AgentBudget,
    DiagnosisResult,
    PolicyDecision,
    PublicEvidence,
)
from gpu_agent.agent.orchestrator import public_evidence_from_bundle
from gpu_agent.agent.policy import decide_action
from gpu_agent.agent.provider import Invocation
from gpu_agent.agent.rule_router import RuleRouter
from gpu_agent.benchmark.evaluation import (
    EvaluationAttempt,
    EvaluationExecutionClaim,
    EvaluationLineage,
    EvaluationProviderPolicy,
    EvaluationRecord,
    EvaluationSchedule,
    EvaluationScheduleItem,
    EvaluationUnitBinding,
    PricingAttestation,
    PublicEvaluationRecord,
)
from gpu_agent.benchmark.models import CaseManifest
from gpu_agent.contracts import (
    ArtifactRef,
    CurrentPhase,
    ExternalRunOrigin,
    RunBinding,
    RunManifest,
    RunStatus,
    ToolResult,
)
from gpu_agent.evidence.models import EvidenceBundle
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
from gpu_agent.patching import PatchCandidate
from gpu_agent.service import ApplicationService
from gpu_agent.store import RunStore, read_regular, reject_symlinks
from gpu_agent.verification.models import (
    OracleResult,
    VerificationAuditResult,
    VerificationObservation,
    VerificationResult,
    VerificationVerdict,
)
from gpu_agent.verification.oracle import NumericOracle, parse_output, reference_add
from gpu_agent.verification.policy import decide_verdict, finding_signature, plan_checks

if TYPE_CHECKING:
    from gpu_agent.benchmark.holdout import HoldoutBatch, HoldoutController
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier


_TRUTH_CASE = (
    Path(__file__).resolve().parents[3] / "benchmarks/development_truth/case_0001/case.json"
)


class CostBoundUnavailable(ValueError):
    """A production monetary bound must be attested before any provider work is allowed."""


def _one_ref(run: RunManifest, name: str) -> ArtifactRef:
    refs = [ref for ref in run.artifact_refs if ref.name == name]
    if len(refs) != 1:
        raise ValueError("evaluation lineage artifact is missing or ambiguous")
    return refs[0]


def _validate_verification_audit(
    public: RunStore,
    evaluator: RunStore | None,
    diagnosis_run_id: str,
    result: VerificationResult,
    binding: RunBinding,
) -> VerificationAuditResult:
    """Resolve the public projection to evaluator-owned native execution output."""
    if (
        evaluator is None
        or evaluator.visibility != "evaluator"
        or result.evaluator_audit_run_id is None
    ):
        raise ValueError("evaluation verification has no evaluator audit lineage")
    audit = evaluator.load(result.evaluator_audit_run_id)
    observation_ref = _one_ref(audit, "observation.json")
    audit_result_ref = _one_ref(audit, "verification/audit-result.json")
    if (
        audit.kind != "verification_audit"
        or audit.status != RunStatus.COMPLETED
        or audit.binding != binding
        or audit.external_origin != ExternalRunOrigin(run_id=diagnosis_run_id, visibility="public")
    ):
        raise ValueError("evaluation verification audit is invalid")
    observation = VerificationObservation.model_validate_json(evaluator.read(observation_ref))
    audit_result = VerificationAuditResult.model_validate_json(evaluator.read(audit_result_ref))
    if (
        audit_result.observation != observation
        or decide_verdict(observation) != result.verdict
        or observation.original_finding_present != result.original_finding_present
        or observation.public_oracle_passed != result.public_oracle_passed
        or observation.check_requirements != result.check_requirements
        or observation.check_outcomes != result.check_outcomes
        or len(observation.new_blocking_findings) != result.new_findings
    ):
        raise ValueError("evaluation verification projection differs from evaluator audit")

    children: dict[int, RunManifest] = {}
    for path in evaluator.root.iterdir():
        if not path.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", path.name):
            continue
        child = evaluator.load(path.name)
        if child.kind != "verification_input" or child.parent_run_id != audit.id:
            continue
        index_ref = _one_ref(child, "input-index.json")
        index_payload = json.loads(evaluator.read(index_ref))
        index = index_payload.get("index")
        if (
            type(index) is not int
            or index < 0
            or index in children
            or child.status != RunStatus.COMPLETED
            or child.binding != binding
            or child.external_origin
            != ExternalRunOrigin(run_id=diagnosis_run_id, visibility="public")
        ):
            raise ValueError("evaluation verification input lineage is invalid")
        children[index] = child
    if children and sorted(children) != list(range(len(children))):
        raise ValueError("evaluation verification input sequence is invalid")
    if not children or audit_result.child_run_ids != [
        child.id for _, child in sorted(children.items())
    ]:
        raise ValueError("evaluation verification child selection is invalid")
    trusted_case = json.loads(read_regular(_TRUTH_CASE, 65536))
    persisted_case = json.loads(evaluator.read(_one_ref(audit, "case.json")))
    if persisted_case != trusted_case:
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
        evaluator.read(_one_ref(audit, "private-suite.json")) != suite_bytes
        or audit_result.suite_hash != hashlib.sha256(suite_bytes).hexdigest()
        or len(children) + audit_result.not_run_count != 1 + len(holdouts)
    ):
        raise ValueError("evaluation verification suite is invalid")

    public_oracle: bool | None = None
    private_oracles: list[bool] = []
    build_clean: list[bool] = []
    runtime_clean: list[bool] = []
    sanitizer_clean: dict[str, list[bool]] = {}
    build_states: list[str] = []
    runtime_states: list[str] = []
    sanitizer_states: dict[str, list[str]] = {}
    native_public_passed = 0
    native_private_passed = 0
    native_new_findings: list[Finding] = []
    candidate_public_findings: list[Finding] = []
    candidate_public_memcheck: SanitizerPayload | None = None
    standard_requirements = plan_checks(
        SanitizerTool.MEMCHECK,
        "standard",
        {tool: "SUPPORTED" for tool in SanitizerTool},
    )
    full_requirements = plan_checks(
        SanitizerTool.MEMCHECK,
        "strict",
        {tool: "SUPPORTED" for tool in SanitizerTool},
    )
    if result.check_requirements == standard_requirements:
        full_check_plan = False
    elif result.check_requirements == full_requirements:
        full_check_plan = True
    else:
        raise ValueError("evaluation verification check plan is not controller-derived")
    observed_binary_hashes: list[str] = []
    infrastructure_missing = False
    baseline_bundle = EvidenceRepository(public).public_view(diagnosis_run_id)
    baseline_memchecks = [
        item
        for item in baseline_bundle.sanitizer_results
        if item.tool_result is not None
        and item.tool_result.typed_payload.tool == SanitizerTool.MEMCHECK
    ]
    if len(baseline_memchecks) != 1 or baseline_memchecks[0].tool_result is None:
        raise ValueError("evaluation verification baseline sanitizer is invalid")
    baseline_memcheck = baseline_memchecks[0]
    baseline_tool_result = baseline_memcheck.tool_result
    if baseline_tool_result is None:
        raise ValueError("evaluation verification baseline sanitizer is invalid")
    baseline_capture = ProcessCapture(
        exit_code=baseline_tool_result.exit_code,
        stdout=public.read(baseline_tool_result.stdout_artifact),
        stderr=public.read(baseline_tool_result.stderr_artifact),
        timed_out=baseline_tool_result.timed_out,
        elapsed_ms=baseline_tool_result.elapsed_ms,
        started_at=baseline_tool_result.started_at,
        finished_at=baseline_tool_result.finished_at,
        truncated=baseline_tool_result.truncated,
        cancelled=baseline_tool_result.cancelled,
        tool_error=baseline_tool_result.tool_error,
    )
    baseline_reparsed = parse_sanitizer(SanitizerTool.MEMCHECK, baseline_capture)
    baseline_payload = baseline_tool_result.typed_payload
    if (
        baseline_reparsed.status != baseline_payload.status
        or baseline_reparsed.completed != baseline_payload.completed
        or baseline_reparsed.parser_version != baseline_payload.parser_version
        or baseline_reparsed.check_outcome != baseline_payload.check_outcome
        or [item.model_copy(update={"raw_ref": None}) for item in baseline_payload.findings]
        != baseline_reparsed.findings
        or baseline_memcheck.findings != baseline_payload.findings
        or any(
            item.raw_ref != baseline_tool_result.stderr_artifact
            for item in baseline_payload.findings
        )
        or baseline_payload.program_output_ref != baseline_tool_result.stdout_artifact
    ):
        raise ValueError("evaluation verification baseline sanitizer is not reproducible")
    baseline_signatures = {finding_signature(item) for item in baseline_reparsed.findings}
    oracle = NumericOracle(trusted_case["atol"], trusted_case["rtol"], False, False)
    for index, child in sorted(children.items()):
        input_ref = _one_ref(child, "input.json")
        input_payload = json.loads(evaluator.read(input_ref))
        if (
            set(input_payload) != {"n", "a", "b"}
            or type(input_payload["n"]) is not int
            or len(input_payload["a"]) != input_payload["n"]
            or len(input_payload["b"]) != input_payload["n"]
            or (index > 0 and input_payload != holdouts[index - 1])
        ):
            raise ValueError("evaluation verification input differs from frozen suite")
        expected_output = reference_add(input_payload["a"], input_payload["b"])
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
        build_model: ToolResult[BuildPayload] | None = None
        run_model: ToolResult[ExecutionPayload] | None = None
        sanitizer_models: list[ToolResult[SanitizerPayload]] = []
        for ref in native_refs:
            raw = evaluator.read(ref)
            payload = json.loads(raw)
            tool_name, request_id = ref.name.split("/")[:2]
            if payload.get("tool_name") != tool_name or payload.get("request_id") != request_id:
                raise ValueError("evaluation verification tool output is spliced")
            model: (
                ToolResult[BuildPayload]
                | ToolResult[ExecutionPayload]
                | ToolResult[SanitizerPayload]
            )
            if tool_name == "build":
                model = ToolResult[BuildPayload].model_validate_json(raw)
                build_model = model
                build_infra = bool(
                    model.tool_error or model.timed_out or model.cancelled or model.truncated
                )
                build_ok = bool(
                    model.exit_code == 0
                    and model.typed_payload.binary_ref is not None
                    and not build_infra
                )
                build_clean.append(build_ok)
                build_states.append(
                    "TOOL_ERROR" if build_infra else "CLEAN" if build_ok else "FAILED"
                )
            elif tool_name == "run":
                model = ToolResult[ExecutionPayload].model_validate_json(raw)
                run_model = model
                runtime_infra = bool(
                    model.tool_error or model.timed_out or model.cancelled or model.truncated
                )
                runtime_ok = bool(
                    model.typed_payload.runtime_status == "SUCCESS"
                    and model.typed_payload.output_ref == model.stdout_artifact
                    and not runtime_infra
                )
                runtime_clean.append(runtime_ok)
                runtime_states.append(
                    "TOOL_ERROR" if runtime_infra else "CLEAN" if runtime_ok else "FAILED"
                )
            else:
                model = ToolResult[SanitizerPayload].model_validate_json(raw)
                sanitizer_models.append(model)
                capture = ProcessCapture(
                    exit_code=model.exit_code,
                    stdout=evaluator.read(model.stdout_artifact),
                    stderr=evaluator.read(model.stderr_artifact),
                    timed_out=model.timed_out,
                    elapsed_ms=model.elapsed_ms,
                    started_at=model.started_at,
                    finished_at=model.finished_at,
                    truncated=model.truncated,
                    cancelled=model.cancelled,
                    tool_error=model.tool_error,
                )
                reparsed = parse_sanitizer(SanitizerTool(model.typed_payload.tool), capture)
                if (
                    reparsed.status != model.typed_payload.status
                    or reparsed.completed != model.typed_payload.completed
                    or reparsed.parser_version != model.typed_payload.parser_version
                    or reparsed.check_outcome != model.typed_payload.check_outcome
                    or [
                        item.model_copy(update={"raw_ref": None})
                        for item in model.typed_payload.findings
                    ]
                    != reparsed.findings
                    or any(
                        item.raw_ref != model.stderr_artifact
                        for item in model.typed_payload.findings
                    )
                ):
                    raise ValueError("evaluation verification sanitizer result is not reproducible")
                sanitizer_infra = bool(
                    model.tool_error
                    or model.timed_out
                    or model.cancelled
                    or model.truncated
                    or not model.typed_payload.completed
                )
                sanitizer_ok = bool(
                    model.typed_payload.completed
                    and model.typed_payload.check_outcome == "CLEAN"
                    and model.typed_payload.program_output_ref == model.stdout_artifact
                    and not sanitizer_infra
                )
                sanitizer_clean.setdefault(model.typed_payload.tool, []).append(sanitizer_ok)
                sanitizer_states.setdefault(model.typed_payload.tool, []).append(
                    "TOOL_ERROR" if sanitizer_infra else "CLEAN" if sanitizer_ok else "FINDING"
                )
            for nested in (model.stdout_artifact, model.stderr_artifact):
                if nested.run_id != child.id or nested.visibility != "evaluator":
                    raise ValueError("evaluation verification tool artifact crosses lineage")
                evaluator.read(nested)
        if build_model is None:
            raise ValueError("evaluation verification build lineage is missing")
        binary_ref = build_model.typed_payload.binary_ref
        if binary_ref is not None:
            evaluator.read(binary_ref)
            observed_binary_hashes.append(binary_ref.sha256)
        build_succeeded = bool(
            build_model.exit_code == 0
            and binary_ref is not None
            and not (
                build_model.tool_error
                or build_model.timed_out
                or build_model.cancelled
                or build_model.truncated
            )
        )
        child_infrastructure_missing = bool(
            build_model.tool_error
            or build_model.timed_out
            or build_model.cancelled
            or build_model.truncated
            or (
                run_model is not None
                and (
                    run_model.tool_error
                    or run_model.timed_out
                    or run_model.cancelled
                    or run_model.truncated
                )
            )
            or any(
                checked.tool_error
                or checked.timed_out
                or checked.cancelled
                or checked.truncated
                or not checked.typed_payload.completed
                for checked in sanitizer_models
            )
        )
        infrastructure_missing |= child_infrastructure_missing
        sanitizer_tools = [checked.typed_payload.tool for checked in sanitizer_models]
        expected_sanitizer_tools = [SanitizerTool.MEMCHECK.value]
        if (
            full_check_plan
            and sanitizer_models
            and sanitizer_models[0].typed_payload.tool == SanitizerTool.MEMCHECK
            and sanitizer_models[0].typed_payload.completed
            and sanitizer_models[0].typed_payload.check_outcome == "CLEAN"
        ):
            expected_sanitizer_tools.extend(
                [
                    SanitizerTool.RACECHECK.value,
                    SanitizerTool.INITCHECK.value,
                    SanitizerTool.SYNCCHECK.value,
                ]
            )
        if build_succeeded:
            if run_model is None or sanitizer_tools != expected_sanitizer_tools:
                raise ValueError("evaluation verification required tool multiplicity is invalid")
        elif run_model is not None or sanitizer_models:
            raise ValueError("evaluation verification ran tools after a failed build")
        if (
            provenance.get("candidate_hash") != result.candidate_hash
            or provenance.get("source_manifest") != build_model.typed_payload.source_manifest
            or provenance.get("binary_ref")
            != (binary_ref.model_dump(mode="json") if binary_ref is not None else None)
            or provenance.get("input_ref") != input_ref.model_dump(mode="json")
        ):
            raise ValueError("evaluation verification provenance is invalid")
        if run_model is not None and (
            binary_ref is None
            or run_model.typed_payload.binary_ref != binary_ref
            or run_model.typed_payload.stdin_ref != input_ref
            or run_model.typed_payload.output_ref != run_model.stdout_artifact
        ):
            raise ValueError("evaluation verification runtime links are invalid")
        if any(
            binary_ref is None
            or checked.typed_payload.binary_ref != binary_ref
            or checked.typed_payload.stdin_ref != input_ref
            or checked.typed_payload.program_output_ref != checked.stdout_artifact
            for checked in sanitizer_models
        ):
            raise ValueError("evaluation verification sanitizer links are invalid")
        for checked in sanitizer_models:
            findings = checked.typed_payload.findings
            if index == 0:
                candidate_public_findings.extend(findings)
                if checked.typed_payload.tool == SanitizerTool.MEMCHECK:
                    candidate_public_memcheck = checked.typed_payload
                native_new_findings.extend(
                    finding
                    for finding in findings
                    if finding_signature(finding) not in baseline_signatures
                    or finding_signature(finding) is None
                )
            else:
                native_new_findings.extend(findings)
        oracle_refs = [ref for ref in child.artifact_refs if ref.name == "oracle.json"]
        if len(oracle_refs) > 1:
            raise ValueError("evaluation verification oracle output is ambiguous")
        if oracle_refs:
            stored_oracle = json.loads(evaluator.read(oracle_refs[0]))
            if run_model is None:
                raise ValueError("evaluation verification oracle has no ordinary execution")
            ordinary = oracle.check(
                parse_output(evaluator.read(run_model.typed_payload.output_ref)), expected_output
            )
            instrumented: list[dict[str, object]] = []
            instrumented_results: list[OracleResult] = []
            for checked in sanitizer_models:
                output_ref = checked.typed_payload.program_output_ref
                if output_ref is None:
                    raise ValueError("evaluation verification instrumented output is missing")
                checked_oracle = oracle.check(
                    parse_output(evaluator.read(output_ref)), expected_output
                )
                instrumented_results.append(checked_oracle)
                instrumented.append(
                    {
                        "tool": checked.typed_payload.tool,
                        "result": checked_oracle.model_dump(mode="json"),
                    }
                )
            expected_oracle = {
                "expected": expected_output,
                "ordinary": ordinary.model_dump(mode="json"),
                "instrumented": instrumented,
            }
            if stored_oracle != expected_oracle:
                raise ValueError("evaluation verification oracle result is not reproducible")
            passed = ordinary.passed and all(item.passed for item in instrumented_results)
            if index == 0:
                public_oracle = passed
            else:
                private_oracles.append(passed)
        elif run_model is not None and run_model.typed_payload.runtime_status == "SUCCESS":
            raise ValueError("evaluation verification oracle output is missing")
        native_passed = (
            bool(
                run_model is not None
                and run_model.typed_payload.runtime_status == "SUCCESS"
                and passed
                and sanitizer_models
                and all(
                    checked.typed_payload.completed
                    and checked.typed_payload.check_outcome == "CLEAN"
                    for checked in sanitizer_models
                )
            )
            if oracle_refs
            else False
        )
        if index == 0:
            native_public_passed += int(native_passed)
        else:
            native_private_passed += int(native_passed)
    if public_oracle is not None and public_oracle != result.public_oracle_passed:
        raise ValueError("public oracle projection differs from native audit")
    derived_private_oracle = (
        None
        if not private_oracles or (len(private_oracles) < len(holdouts) and all(private_oracles))
        else all(private_oracles)
    )
    if derived_private_oracle != audit_result.observation.private_holdout_passed:
        raise ValueError("private oracle projection differs from native audit")
    baseline_input = baseline_tool_result.typed_payload.stdin_ref
    candidate_input_ref = _one_ref(children[0], "input.json")
    same_public_input = bool(
        baseline_input is not None
        and baseline_input.sha256 == candidate_input_ref.sha256
        and public.read(baseline_input) == evaluator.read(candidate_input_ref)
    )
    candidate_signatures = {finding_signature(item) for item in candidate_public_findings}
    derived_original: bool | None = None
    if same_public_input and baseline_signatures and candidate_public_memcheck is not None:
        derived_original = bool(
            any(
                signature is not None and signature in candidate_signatures
                for signature in baseline_signatures
            )
        )
        if not candidate_public_memcheck.completed:
            derived_original = None
    if (
        derived_original != audit_result.observation.original_finding_present
        or native_new_findings != audit_result.observation.new_blocking_findings
    ):
        raise ValueError("verification findings differ from native sanitizer results")

    def aggregate_state(states: list[str], *, absent: str) -> str:
        if not states:
            return absent
        if "TOOL_ERROR" in states:
            return "TOOL_ERROR"
        return "CLEAN" if all(state == "CLEAN" for state in states) else "FAILED"

    derived_checks: dict[str, str] = {
        "build": aggregate_state(build_states, absent="NOT_RUN"),
        "runtime": aggregate_state(runtime_states, absent="NOT_RUN"),
        "public_oracle": (
            "CLEAN" if public_oracle is True else "FAILED" if public_oracle is False else "NOT_RUN"
        ),
    }
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        states = sanitizer_states.get(tool, [])
        if states:
            derived_checks[tool] = (
                "TOOL_ERROR"
                if "TOOL_ERROR" in states
                else "CLEAN"
                if all(state == "CLEAN" for state in states)
                else "FINDING"
            )
        elif tool == "memcheck" or full_check_plan:
            derived_checks[tool] = "NOT_RUN"
    derived_outcomes = {
        requirement.tool: derived_checks.get(requirement.tool.value, "NOT_RUN")
        for requirement in result.check_requirements
    }
    derived_observation = audit_result.observation.model_copy(
        update={
            "build_ok": (
                None if "TOOL_ERROR" in build_states else all(build_clean) if build_clean else None
            ),
            "runtime_ok": (
                None if "TOOL_ERROR" in runtime_states or not runtime_clean else all(runtime_clean)
            ),
            "original_finding_present": derived_original,
            "public_oracle_passed": public_oracle,
            "private_holdout_passed": derived_private_oracle,
            "required_evidence_missing": infrastructure_missing,
            "new_blocking_findings": native_new_findings,
            "check_requirements": result.check_requirements,
            "check_outcomes": derived_outcomes,
        }
    )
    if "TOOL_ERROR" in build_states:
        derived_reason = "BUILD_TOOL_ERROR"
    elif build_states and build_states[0] == "FAILED":
        derived_reason = "CANDIDATE_BUILD_FAILED"
    elif infrastructure_missing:
        derived_reason = "REQUIRED_EVIDENCE_MISSING"
    elif derived_original is True:
        derived_reason = "ORIGINAL_FINDING_PRESENT"
    elif derived_original is None:
        derived_reason = "ORIGINAL_FINDING_UNKNOWN"
    elif native_new_findings:
        derived_reason = "NEW_BLOCKING_FINDING"
    elif runtime_states and runtime_states[0] == "FAILED":
        derived_reason = "RUNTIME_FAILED"
    elif public_oracle is False:
        derived_reason = "ORACLE_FAILED"
    elif public_oracle is None:
        derived_reason = "ORACLE_NOT_RUN"
    else:
        derived_reason = "ALL_REQUIRED_CHECKS_PASSED"
    derived_not_run = {
        requirement.tool: "UPSTREAM_CHECK_DID_NOT_PASS"
        for requirement in result.check_requirements
        if requirement.required and derived_outcomes[requirement.tool] == "NOT_RUN"
    }
    if (
        result.required_checks != derived_checks
        or result.check_outcomes != derived_outcomes
        or audit_result.observation != derived_observation
        or result.binary_hashes != sorted(set(observed_binary_hashes))
        or decide_verdict(derived_observation) != result.verdict
        or result.reason_code != derived_reason
        or result.failure_stage
        != (None if derived_reason == "ALL_REQUIRED_CHECKS_PASSED" else "verification")
        or result.not_run_reasons != derived_not_run
        or result.public_passed_count != native_public_passed
        or result.limitations != ["Containers share the host kernel and GPU driver."]
    ):
        raise ValueError("verification checks differ from native tool results")
    if (
        audit_result.public_passed_count != native_public_passed
        or audit_result.private_passed_count != native_private_passed
        or audit_result.not_run_count != 1 + len(holdouts) - len(children)
    ):
        raise ValueError("verification pass counts differ from native results")
    return audit_result


def validate_evaluation_record(
    store: RunStore,
    record: PublicEvaluationRecord | EvaluationRecord,
    item: EvaluationScheduleItem,
    attempt: EvaluationAttempt,
    binding: RunBinding,
    evaluator: RunStore | None = None,
    corpus: RunStore | None = None,
    corpus_family: "CorpusFamily | None" = None,
    registered_case_id: str | None = None,
) -> None:
    """Resolve one public record back to immutable native execution artifacts."""
    public = record.public() if isinstance(record, EvaluationRecord) else record
    lineage = public.lineage
    run = store.load(lineage.diagnosis_run_id)
    if (
        store.visibility != "public"
        or run.kind != "diagnosis"
        or run.status != RunStatus.COMPLETED
        or run.parent_run_id != attempt.run_id
        or run.binding != binding
        or public.record_id != run.id
    ):
        raise ValueError("evaluation lineage does not resolve to its scheduled diagnosis")
    unit = EvaluationUnitBinding.model_validate_json(
        store.read(_one_ref(run, "evaluation/unit.json"))
    )
    expected_unit = EvaluationUnitBinding(
        evaluation_run_id=attempt.run_id,
        ordinal=item.ordinal,
        schedule_hash=attempt.schedule_hash,
        idempotency_key=attempt.idempotency_key,
        reserved_cost_usd=attempt.reserved_cost_usd,
        case_id=item.case_id,
        template_id=item.template_id,
        mode=item.mode,
        repeat=item.repeat,
        split=item.split,
        holdout_proof=item.holdout_proof,
    )
    if unit != expected_unit:
        raise ValueError("evaluation unit differs from its frozen schedule")

    diagnosis_ref = _one_ref(run, "diagnosis.json")
    diagnosis = DiagnosisResult.model_validate_json(store.read(diagnosis_ref))
    if (
        diagnosis_ref.sha256 != lineage.diagnosis_hash
        or diagnosis.model_dump(mode="json") != public.diagnosis
        or (diagnosis.diagnostic_outcome == "DIAGNOSED" and not diagnosis.root_cause)
        or (diagnosis.diagnostic_outcome != "DIAGNOSED" and not diagnosis.limitations)
    ):
        raise ValueError("evaluation diagnosis lineage is invalid")

    evidence_refs = [ref for ref in run.artifact_refs if ref.name == "evidence/bundle.json"]
    if (
        not evidence_refs
        or evidence_refs[-1].sha256 != lineage.evidence_hash
        or public.evidence_hash != lineage.evidence_hash
    ):
        raise ValueError("evaluation evidence lineage is invalid")
    bundles = [EvidenceBundle.model_validate_json(store.read(ref)) for ref in evidence_refs]
    for previous, current in zip(bundles, bundles[1:], strict=False):
        acquired = bool(
            previous.build_result
            or previous.execution_result
            or previous.sanitizer_results
            or previous.retrieved_chunks
        )
        if acquired and (
            current.environment != previous.environment
            or current.source_snapshot != previous.source_snapshot
            or current.limitations != previous.limitations
        ):
            raise ValueError("evaluation evidence base changed after acquisition")
        if (
            previous.build_result is not None and current.build_result != previous.build_result
        ) or (
            previous.execution_result is not None
            and current.execution_result != previous.execution_result
        ):
            raise ValueError("evaluation evidence result was replaced")
        if (
            current.sanitizer_results[: len(previous.sanitizer_results)]
            != previous.sanitizer_results
            or current.retrieved_chunks[: len(previous.retrieved_chunks)]
            != previous.retrieved_chunks
            or current.source_locations[: len(previous.source_locations)]
            != previous.source_locations
        ):
            raise ValueError("evaluation evidence progression is not monotone")
        changed = sum(
            (
                previous.build_result != current.build_result,
                previous.execution_result != current.execution_result,
                len(previous.sanitizer_results) != len(current.sanitizer_results),
                len(previous.retrieved_chunks) != len(current.retrieved_chunks),
            )
        )
        if changed > 1:
            raise ValueError("evaluation evidence progression combines acquisitions")
    bundle = EvidenceRepository(store).public_view(run.id)
    if bundle != bundles[-1]:
        raise ValueError("evaluation final evidence projection is invalid")
    source_refs = [ref for ref in bundle.source_snapshot if ref.name.endswith("/kernel.cu")]
    if len(source_refs) != 1 or source_refs[0].sha256 != public.input_hash:
        raise ValueError("evaluation input lineage is invalid")
    budget = AgentBudget.model_validate_json(store.read(_one_ref(run, "agent/final-budget.json")))
    acquisition = AcquisitionUsage.model_validate_json(
        store.read(_one_ref(run, "agent/acquisition-usage.json"))
    )
    summary = json.loads(store.read(_one_ref(run, "agent/usage-summary.json")))
    if summary.get("physical_calls") != budget.llm_calls:
        raise ValueError("native evaluation usage artifacts disagree")

    trace = json.loads(store.read(_one_ref(run, "agent/controller-lineage.json")))
    expected_controller = "fixed" if item.mode in {"A", "B", "C"} else "rule_router"
    policy_ref = _one_ref(run, "agent/acquisition-policy.json")
    decision_refs = sorted(
        (
            ref
            for ref in run.artifact_refs
            if ref.name.startswith("actions/") and ref.name.endswith("/decision.json")
        ),
        key=lambda ref: int(ref.name.split("/")[1]),
    )
    decisions = [PolicyDecision.model_validate_json(store.read(ref)) for ref in decision_refs]
    if any(not decision.allowed for decision in decisions):
        raise ValueError("controller route contains a rejected decision")
    acquisition_policy = json.loads(store.read(policy_ref))
    if (
        set(acquisition_policy) != {"mode", "required_tools"}
        or acquisition_policy["mode"] != item.mode
        or not isinstance(acquisition_policy["required_tools"], list)
        or any(
            tool not in {value.value for value in SanitizerTool}
            for tool in acquisition_policy["required_tools"]
        )
    ):
        raise ValueError("acquisition policy differs from scheduled mode")
    expected_trace = {
        "schema_version": 1,
        "mode": item.mode,
        "controller": expected_controller if item.mode != "E" else "planner",
        "provider_calls_allowed": item.mode == "E",
        "acquisition_policy_ref": {"id": policy_ref.id, "sha256": policy_ref.sha256},
        "evidence_ref": {
            "id": evidence_refs[-1].id,
            "sha256": evidence_refs[-1].sha256,
        },
        "route_decision_refs": [
            {"id": ref.id, "name": ref.name, "sha256": ref.sha256} for ref in decision_refs
        ],
    }
    provider_refs = [ref for ref in run.artifact_refs if ref.name.startswith("provider/")]
    ordered_started: list[Invocation] = []
    terminal_hashes: list[str] = []
    invocations: dict[str, list[Invocation]] = {}
    logical_terminals: list[Invocation] = []
    for ref in provider_refs:
        invocation = Invocation.model_validate_json(store.read(ref))
        if (
            invocation.run_id != run.id
            or ref.name != f"provider/{invocation.invocation_id}/{invocation.state}.json"
        ):
            raise ValueError("provider invocation lineage is invalid")
        invocations.setdefault(invocation.invocation_id, []).append(invocation)
        if invocation.state == "STARTED":
            ordered_started.append(invocation)
        else:
            terminal_hashes.append(ref.sha256)
    policy: EvaluationProviderPolicy | None = None
    pricing: PricingAttestation | None = None
    sanitizer_actions = {
        "run_memcheck",
        "run_racecheck",
        "run_initcheck",
        "run_synccheck",
    }
    decision_types = [decision.action_type for decision in decisions]
    observed_sanitizers = len(bundle.sanitizer_results)
    observed_retrievals = len(bundle.retrieved_chunks)
    if item.mode == "A" and (
        observed_sanitizers
        or observed_retrievals
        or acquisition.sanitizer_calls
        or acquisition.retrieval_calls
        or budget.sanitizer_calls
        or budget.rag_calls
    ):
        raise ValueError("fixed controller evidence differs from scheduled mode")
    if item.mode == "B" and (
        observed_sanitizers
        or acquisition.sanitizer_calls
        or budget.sanitizer_calls
        or acquisition.retrieval_calls not in {0, 1}
        or bool(observed_retrievals) != bool(acquisition.retrieval_calls)
    ):
        raise ValueError("fixed controller evidence differs from scheduled mode")
    if corpus is None or corpus_family is None:
        raise ValueError("native corpus authority is required for evaluation validation")
    trusted_case = registered_cases(corpus, binding, corpus_family).get(
        registered_case_id or item.case_id
    )
    if trusted_case is None:
        raise ValueError("scheduled case is absent from the trusted corpus")
    if acquisition_policy["required_tools"] != [trusted_case.target_tool]:
        raise ValueError("acquisition policy differs from the trusted case manifest")
    required_tools = [SanitizerTool.MEMCHECK.value, trusted_case.target_tool]
    expected_c_tools = list(dict.fromkeys(required_tools))
    observed_tools = [
        result.tool_result.typed_payload.tool
        for result in bundle.sanitizer_results
        if result.tool_result is not None
    ]
    if observed_tools and observed_tools[0] == SanitizerTool.MEMCHECK.value:
        memcheck = bundle.sanitizer_results[0]
        if memcheck.check_outcome != "CLEAN":
            expected_c_tools = [SanitizerTool.MEMCHECK.value]
    if item.mode == "C" and (
        observed_retrievals
        or acquisition.retrieval_calls
        or budget.rag_calls
        or observed_tools != expected_c_tools
        or acquisition.sanitizer_calls != len(expected_c_tools)
        or observed_sanitizers != acquisition.sanitizer_calls
    ):
        raise ValueError("fixed controller evidence differs from scheduled mode")
    if item.mode == "D" and (
        sum(action in sanitizer_actions for action in decision_types) < acquisition.sanitizer_calls
        or decision_types.count("retrieve_official_docs") < acquisition.retrieval_calls
        or observed_sanitizers != acquisition.sanitizer_calls
        or bool(observed_retrievals) != bool(acquisition.retrieval_calls)
        or any(
            action
            not in sanitizer_actions
            | {"retrieve_official_docs", "finish_diagnosis", "declare_inconclusive"}
            for action in decision_types
        )
    ):
        raise ValueError("rule controller evidence differs from route decisions")
    if item.mode == "D":
        initial_budget = AgentBudget.model_validate_json(
            store.read(_one_ref(run, "agent/initial-budget.json"))
        )
        budget_audit = json.loads(store.read(_one_ref(run, "agent/budget-audit.json")))
        if not isinstance(budget_audit, list) or initial_budget != AgentBudget():
            raise ValueError("rule controller budget audit is invalid")
        step_refs = sorted(
            (
                ref
                for ref in run.artifact_refs
                if ref.name.startswith("actions/") and ref.name.endswith("/step.json")
            ),
            key=lambda ref: int(ref.name.split("/")[1]),
        )
        if len(step_refs) != len(decision_refs):
            raise ValueError("rule controller action steps are incomplete")
        seen: set[str] = set()
        expected_budget = initial_budget
        expected_audit: list[dict[str, object]] = []
        expected_sanitizer_physical = 0
        expected_retrieval_physical = 0
        expected_evidence_index: int | None = None
        evidence_by_id = {ref.id: index for index, ref in enumerate(evidence_refs)}
        acquisition_actions = sanitizer_actions | {
            "retrieve_official_docs",
            "inspect_source",
        }
        for index, (step_ref, decision_ref) in enumerate(
            zip(step_refs, decision_refs, strict=True)
        ):
            if (
                step_ref.name != f"actions/{index}/step.json"
                or decision_ref.name != f"actions/{index}/decision.json"
            ):
                raise ValueError("rule controller action sequence is invalid")
            step = json.loads(store.read(step_ref))
            if (
                set(step)
                != {
                    "schema_version",
                    "action",
                    "evidence_ref",
                    "evidence",
                    "budget",
                    "seen",
                }
                or step["schema_version"] != 1
            ):
                raise ValueError("rule controller action step is invalid")
            action = ACTION_ADAPTER.validate_python(step["action"])
            evidence = PublicEvidence.model_validate(step["evidence"])
            step_budget = AgentBudget.model_validate(step["budget"])
            evidence_ref = ArtifactRef.model_validate(step["evidence_ref"])
            evidence_index = evidence_by_id.get(evidence_ref.id)
            if evidence_ref not in evidence_refs or evidence_index is None:
                raise ValueError("rule controller evidence reference is invalid")
            if expected_evidence_index is None:
                expected_evidence_index = evidence_index
            if evidence_index != expected_evidence_index:
                raise ValueError("rule controller evidence snapshot is out of sequence")
            observed_bundle = EvidenceBundle.model_validate_json(store.read(evidence_ref))
            if evidence != public_evidence_from_bundle(store, observed_bundle):
                raise ValueError("rule controller evidence snapshot is invalid")
            if step["seen"] != sorted(seen):
                raise ValueError("rule controller seen state is invalid")
            expected_step_budget = expected_budget.model_copy(
                update={"remaining_seconds": step_budget.remaining_seconds}
            )
            if (
                step_budget != expected_step_budget
                or step_budget.remaining_seconds > expected_budget.remaining_seconds
                or step_budget.llm_calls != 0
            ):
                raise ValueError("rule controller budget state is invalid")
            proposed = RuleRouter().next_action(evidence, expected_step_budget)
            if (
                proposed.action_type != action.action_type
                or proposed.typed_arguments != action.typed_arguments
                or proposed.rationale != action.rationale
            ):
                raise ValueError("rule controller action differs from deterministic routing")
            expected_decision = decide_action(
                action,
                evidence,
                step_budget,
                CurrentPhase.DIAGNOSING,
                seen,
            )
            if PolicyDecision.model_validate_json(store.read(decision_ref)) != expected_decision:
                raise ValueError("rule controller policy decision is invalid")
            seen.add(action.action_type + action.typed_arguments.model_dump_json())
            updates: dict[str, object] = {
                "agent_steps": expected_step_budget.agent_steps + 1,
                "remaining_seconds": step_budget.remaining_seconds,
            }
            if action.action_type in acquisition_actions:
                reservation_id = len(expected_audit) // 3 + 1
                audit_slice: list[object] = budget_audit[
                    len(expected_audit) : len(expected_audit) + 3
                ]
                prefix: list[dict[str, object]] = [
                    {"id": reservation_id, "action": action.action_type, "state": "ATTEMPTED"},
                    {"id": reservation_id, "action": action.action_type, "state": "STARTED"},
                ]
                if len(audit_slice) != 3 or audit_slice[:2] != prefix:
                    raise ValueError("rule controller acquisition audit is invalid")
                terminal = audit_slice[2] if len(audit_slice) == 3 else None
                if terminal not in (
                    {"id": reservation_id, "action": action.action_type, "state": "COMPLETED"},
                    {"id": reservation_id, "action": action.action_type, "state": "FAILED"},
                ):
                    raise ValueError("rule controller acquisition audit is invalid")
                assert isinstance(terminal, dict)
                expected_audit.extend([*prefix, terminal])
                if action.action_type in sanitizer_actions:
                    updates["sanitizer_calls"] = expected_step_budget.sanitizer_calls + 1
                    expected_sanitizer_physical += 1
                elif action.action_type == "retrieve_official_docs":
                    updates["rag_calls"] = expected_step_budget.rag_calls + 1
                    if (
                        terminal["state"] == "COMPLETED"
                        or "KNOWLEDGE_UNAVAILABLE" not in diagnosis.limitations
                    ):
                        expected_retrieval_physical += 1
                elif action.action_type == "inspect_source":
                    updates["source_reads"] = expected_step_budget.source_reads + 1
                if (
                    terminal["state"] == "COMPLETED"
                    or expected_evidence_index < len(evidence_refs) - 1
                ):
                    expected_evidence_index += 1
                if terminal["state"] == "FAILED" and (
                    index != len(step_refs) - 1 or diagnosis.diagnostic_outcome != "INCONCLUSIVE"
                ):
                    raise ValueError("failed acquisition did not terminate inconclusively")
            expected_budget = expected_step_budget.model_copy(update=updates)
        final_budget = budget
        expected_final = expected_budget.model_copy(
            update={"remaining_seconds": final_budget.remaining_seconds}
        )
        if (
            final_budget != expected_final
            or final_budget.remaining_seconds > expected_budget.remaining_seconds
            or budget_audit != expected_audit
            or expected_evidence_index is None
            or expected_evidence_index != len(evidence_refs) - 1
            or acquisition.sanitizer_calls != expected_sanitizer_physical
            or acquisition.retrieval_calls != expected_retrieval_physical
        ):
            raise ValueError("rule controller final budget or audit is invalid")
    if item.mode in {"A", "B", "C", "D"}:
        if (
            trace != expected_trace
            or (item.mode in {"A", "B", "C"} and decision_refs)
            or (item.mode == "D" and not decision_refs)
            or provider_refs
            or lineage.provider_invocation_hashes
            or public.usage.get("physical_calls") != 0
            or lineage.candidate_run_id is not None
            or lineage.verification_run_id is not None
            or lineage.public_verification_hash is not None
        ):
            raise ValueError("deterministic evaluation mode has invalid lineage")
    else:
        policy_ref = _one_ref(run, "agent/provider-policy.json")
        policy = EvaluationProviderPolicy.model_validate_json(store.read(policy_ref))
        pricing = PricingAttestation.model_validate_json(
            store.read(_one_ref(run, "agent/pricing-attestation.json"))
        )
        if (
            policy_ref.sha256 != binding.model_config_hash
            or policy.sha256 != binding.model_config_hash
            or policy.prompt_version != binding.prompt_version
            or policy.allowed_response_models != [policy.configured_model]
            or policy.pricing_hash != pricing.rate_card_hash
            or pricing.provider != policy.provider
            or pricing.model != policy.configured_model
            or pricing.repository_commit != binding.repository.commit
            or pricing.model_config_hash != binding.model_config_hash
        ):
            raise ValueError("provider policy differs from immutable evaluation binding")
        if trace != expected_trace or not invocations:
            raise ValueError("agent evaluation mode has no provider lineage")
        if terminal_hashes != lineage.provider_invocation_hashes:
            raise ValueError("provider invocation hashes differ from native artifacts")
        terminal_by_id: dict[str, Invocation] = {}
        for history in invocations.values():
            started = [value for value in history if value.state == "STARTED"]
            terminal = [value for value in history if value.state != "STARTED"]
            if len(started) != 1 or len(terminal) != 1:
                raise ValueError("provider invocation is not terminal and unique")
            final = terminal[0]
            terminal_by_id[started[0].invocation_id] = final
            if (
                final.kind != started[0].kind
                or final.attempt != started[0].attempt
                or final.client_request_id != started[0].client_request_id
                or final.endpoint_host != started[0].endpoint_host
                or final.endpoint_host != policy.endpoint_host
                or final.configured_model != started[0].configured_model
                or final.started_at != started[0].started_at
                or final.finished_at is None
                or final.finished_at < final.started_at
                or final.elapsed_ms is None
                or final.elapsed_ms < 0
                or final.format_retry_of != started[0].format_retry_of
                or final.prompt_version != binding.prompt_version
                or final.configured_model != policy.configured_model
                or final.response_model not in policy.allowed_response_models
                or not final.store_false_sent
                or final.usage is None
                or (final.state == "COMPLETED") != (final.output_hash is not None)
            ):
                raise ValueError("provider invocation policy is invalid")
        sequence = 0
        index = 0
        logical_kinds: list[str] = []
        while index < len(ordered_started):
            first = ordered_started[index]
            first_final = terminal_by_id[first.invocation_id]
            expected_request_id = hashlib.sha256(
                f"{attempt.idempotency_key}:{sequence}:{first.kind}:0".encode()
            ).hexdigest()[:32]
            if (
                first.attempt != 0
                or first.format_retry_of is not None
                or first.client_request_id != expected_request_id
            ):
                raise ValueError("provider invocation sequence is invalid")
            if first_final.state == "COMPLETED":
                completed = first_final
                index += 1
            elif (
                first_final.state == "FAILED"
                and first_final.error_code == "LLM_INVALID_OUTPUT"
                and not first_final.retryable
                and index + 1 < len(ordered_started)
            ):
                retry = ordered_started[index + 1]
                retry_final = terminal_by_id[retry.invocation_id]
                expected_retry_id = hashlib.sha256(
                    f"{attempt.idempotency_key}:{sequence}:{first.kind}:1".encode()
                ).hexdigest()[:32]
                if (
                    retry.kind != first.kind
                    or retry.attempt != 1
                    or retry.format_retry_of != first.invocation_id
                    or retry.client_request_id != expected_retry_id
                    or retry_final.state != "COMPLETED"
                ):
                    raise ValueError("provider retry lineage is invalid")
                completed = retry_final
                index += 2
            else:
                raise ValueError("provider invocation did not complete")
            logical_kinds.append(first.kind)
            logical_terminals.append(completed)
            sequence += 1
        terminals = list(terminal_by_id.values())
        if logical_kinds.count("plan") != len(decision_refs) or len(
            {value.client_request_id for value in terminals}
        ) != len(terminals):
            raise ValueError("provider invocation sequence differs from controller route")
        plan_hashes = [value.output_hash for value in logical_terminals if value.kind == "plan"]
        if plan_hashes != [decision.action_hash for decision in decisions]:
            raise ValueError("provider plan outputs differ from controller decisions")
        diagnosis_hashes = [
            value.output_hash for value in logical_terminals if value.kind == "diagnose"
        ]
        expected_diagnosis_hash = hashlib.sha256(diagnosis.model_dump_json().encode()).hexdigest()
        if diagnosis_hashes not in ([], [expected_diagnosis_hash]):
            raise ValueError("provider diagnosis output differs from terminal diagnosis")
        if public.usage.get("physical_calls") != len(invocations):
            raise ValueError("provider usage differs from native invocations")

    child_runs = [
        store.load(path.name)
        for path in store.root.iterdir()
        if path.is_dir() and re.fullmatch(r"[a-f0-9]{32}", path.name)
    ]
    candidates = [
        child for child in child_runs if child.kind == "candidate" and child.parent_run_id == run.id
    ]
    verifications = [
        child
        for child in child_runs
        if child.kind == "verification" and child.parent_run_id == run.id
    ]
    if len(candidates) > 1 or len(verifications) > 1:
        raise ValueError("evaluation child lineage is ambiguous")
    candidate: PatchCandidate | None = None
    verification: VerificationResult | None = None
    verification_ref: ArtifactRef | None = None
    if candidates:
        candidate_run = candidates[0]
        candidate = PatchCandidate.model_validate_json(
            store.read(_one_ref(candidate_run, "candidate.json"))
        )
        if (
            candidate_run.status != RunStatus.COMPLETED
            or candidate_run.binding != binding
            or lineage.candidate_run_id != candidate_run.id
            or public.patch_hash != candidate.patched_source_hash
            or not verifications
            or policy is None
            or candidate.generated_by != "agent"
            or candidate.provider != policy.provider
            or candidate.model != policy.configured_model
            or candidate.prompt_version != policy.prompt_version
        ):
            raise ValueError("evaluation candidate lineage is invalid")
        patch_hashes = [value.output_hash for value in logical_terminals if value.kind == "patch"]
        expected_patch_hash = hashlib.sha256(
            json.dumps({"unified_diff": candidate.unified_diff}, separators=(",", ":")).encode()
        ).hexdigest()
        if patch_hashes != [expected_patch_hash]:
            raise ValueError("provider patch output differs from candidate")
        verification_run = verifications[0]
        verification_ref = _one_ref(verification_run, "verification/result.json")
        verification = VerificationResult.model_validate_json(store.read(verification_ref))
        if (
            verification_run.status != RunStatus.COMPLETED
            or verification_run.binding != binding
            or lineage.verification_run_id != verification_run.id
            or lineage.public_verification_hash != verification_ref.sha256
            or verification.candidate_hash != candidate.patched_source_hash
            or public.verdict != verification.verdict.value
        ):
            raise ValueError("evaluation verification lineage is invalid")
        _validate_verification_audit(store, evaluator, run.id, verification, binding)
    elif (
        verifications
        or lineage.candidate_run_id is not None
        or lineage.verification_run_id is not None
        or lineage.public_verification_hash is not None
        or public.patch_hash is not None
        or public.verdict is not None
    ):
        raise ValueError("evaluation record declares nonexistent child lineage")
    if item.mode == "E":
        expected_kinds = ["plan"] * len(decisions)
        if any(value.kind == "diagnose" for value in logical_terminals):
            expected_kinds.append("diagnose")
        if candidate is not None:
            expected_kinds.append("patch")
        if logical_kinds != expected_kinds:
            raise ValueError("provider call order differs from controller workflow")

    expected_checks: dict[str, str] = {
        result.tool_result.typed_payload.tool: result.check_outcome
        for result in bundle.sanitizer_results
        if result.tool_result is not None
    }
    if verification is not None:
        expected_checks.update(
            {
                f"verification/{key}": value
                for key, value in verification.required_checks.items()
                if key != "private_oracle"
            }
        )
    expected_usage: dict[str, int | None] = {
        "physical_calls": budget.llm_calls,
        "sanitizer_calls": acquisition.sanitizer_calls,
        "retrieval_calls": acquisition.retrieval_calls,
        "sanitizer_attempts": budget.sanitizer_calls,
        "retrieval_attempts": budget.rag_calls,
        "build_calls": int(bundle.build_result is not None),
        "runtime_calls": int(bundle.execution_result is not None),
    }
    diagnostic_calls = (
        acquisition.sanitizer_calls
        + acquisition.retrieval_calls
        + int(bundle.build_result is not None)
        + int(bundle.execution_result is not None)
    )
    expected_usage["diagnostic_tool_calls"] = diagnostic_calls
    expected_usage["tool_calls"] = None if verification is not None else diagnostic_calls
    expected_usage["total_sanitizer_calls"] = (
        None if verification is not None else acquisition.sanitizer_calls
    )
    finals = [history[-1] for history in invocations.values()]
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        values = [getattr(value.usage, field) if value.usage else None for value in finals]
        expected_usage[field] = (
            sum(value for value in values if value is not None)
            if values and len(values) == budget.llm_calls and None not in values
            else None
        )
    result = diagnosis
    reason = result.limitations[0] if result.limitations else None
    expected_status = "INCONCLUSIVE"
    finished = run.events[-1].at
    if verification is not None:
        reason = verification.reason_code
        finished = verifications[0].events[-1].at
        if verification.verdict != VerificationVerdict.INCONCLUSIVE:
            expected_status = "COMPLETED"
    elif reason and reason not in {
        "INVALID_DIAGNOSIS_EVIDENCE",
        "MODEL_DECLARED_INCONCLUSIVE",
        "SANITIZER_EVIDENCE_UNAVAILABLE",
        "KNOWLEDGE_UNAVAILABLE",
        "NO_INFORMATION_GAIN",
    }:
        expected_status = "TIMEOUT" if "TIMEOUT" in reason else "FAILED"
    if reason is not None and not re.fullmatch(r"[A-Z0-9_]{1,80}", reason):
        reason = "EVALUATION_FAILED"
    expected_latency = (finished - run.events[0].at).total_seconds() * 1000
    expected_cost: float | None = 0.0
    if item.mode == "E":
        input_tokens = expected_usage.get("input_tokens")
        output_tokens = expected_usage.get("output_tokens")
        expected_cost = (
            pricing.cost(input_tokens, output_tokens)
            if pricing is not None
            and isinstance(input_tokens, int)
            and isinstance(output_tokens, int)
            else None
        )
    if (
        public.executed_checks != expected_checks
        or public.status != expected_status
        or public.failure_reason != reason
        or public.usage != expected_usage
        or public.latency_ms != expected_latency
        or public.cost_usd != expected_cost
        or public.patch_hash != (candidate.patched_source_hash if candidate else None)
        or public.oracle_passed != (verification.public_oracle_passed if verification else None)
        or public.verdict != (verification.verdict.value if verification else None)
        or public.regression_detected
        != bool(verification and verification.verdict == VerificationVerdict.REGRESSION_DETECTED)
    ):
        raise ValueError("evaluation record summary differs from native artifacts")


def registered_cases(
    corpus: RunStore,
    evaluation_binding: RunBinding,
    family: "CorpusFamily | None" = None,
) -> dict[str, CaseManifest]:
    """Load only completed, hash-checked registrations from the controller corpus."""
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.benchmark.ledger import CorpusFamily

    if evaluation_binding.purpose != "evaluation":
        raise ValueError("evaluation binding is invalid")
    family = family or CorpusFamily.configured(corpus)
    builder = BenchmarkBuilder(corpus)
    cases: dict[str, CaseManifest] = {}
    target_hash = family.ledger.target_store_hash(corpus)
    transactions = [
        transaction
        for transaction in family.ledger.committed_through()
        if transaction.visibility == corpus.visibility
        and transaction.target_store_hash == target_hash
    ]
    for authoritative_transaction in transactions:
        run = corpus.load(authoritative_transaction.run_id)
        if (
            run.kind != "benchmark_case"
            or run.status != RunStatus.COMPLETED
            or run.binding is None
            or run.binding.purpose != "corpus_validation"
            or run.binding.toolchain_lock_hash is None
            or run.binding.case_registry_hash is None
            or run.binding.corpus_ledger_namespace_hash is None
        ):
            raise ValueError("case registration is not terminal")
        refs = [ref for ref in run.artifact_refs if ref.name == "case-manifest.json"]
        ledger_refs = [
            ref for ref in run.artifact_refs if ref.name == "validation/ledger-transaction.json"
        ]
        if len(refs) != 1 or len(ledger_refs) != 1:
            raise ValueError("case registration is ambiguous")
        case = CaseManifest.model_validate_json(corpus.read(refs[0]))
        transaction = json.loads(corpus.read(ledger_refs[0]))
        try:
            committed = family.ledger.committed(str(transaction.get("transaction_id", "")))
        except ValueError:
            raise ValueError("committed corpus transaction is unavailable") from None
        if (
            set(transaction)
            != {
                "schema_version",
                "transaction_id",
                "owner_id",
                "ledger_namespace_hash",
                "case_identity_hash",
                "template_identity_hash",
                "source_pair_hash",
                "target_store_hash",
                "visibility",
                "expected_manifest_hash",
            }
            or transaction["schema_version"] != 2
            or not re.fullmatch(r"[a-f0-9]{32}", transaction["transaction_id"])
            or not re.fullmatch(r"[a-f0-9]{32}", transaction["owner_id"])
            or transaction["ledger_namespace_hash"] != run.binding.corpus_ledger_namespace_hash
            or transaction["ledger_namespace_hash"] != case.ledger_namespace_hash
            or transaction["case_identity_hash"] != case.case_identity_hash
            or transaction["template_identity_hash"] != case.template_identity_hash
            or transaction["source_pair_hash"] != case.source_pair_hash
            or transaction["visibility"] != corpus.visibility
            or transaction["expected_manifest_hash"] != refs[0].sha256
            or committed.run_id != run.id
            or committed.owner_id != transaction["owner_id"]
            or committed.case_hash != transaction["case_identity_hash"]
            or committed.template_hash != transaction["template_identity_hash"]
            or committed.source_pair_hash != transaction["source_pair_hash"]
            or committed.target_store_hash != transaction["target_store_hash"]
            or committed.target_store_hash != target_hash
            or committed.visibility != corpus.visibility
            or committed.manifest_hash != refs[0].sha256
            or case.toolchain_hash != run.binding.toolchain_lock_hash
            or run.binding.repository != evaluation_binding.repository
            or run.binding.toolchain_lock_hash != evaluation_binding.toolchain_lock_hash
            or len(case.validation_run_ids) != 2
            or len(set(case.validation_run_ids)) != 2
            or not all(re.fullmatch(r"[a-f0-9]{32}", item) for item in case.validation_run_ids)
            or (case.split == "public") != (corpus.visibility == "public")
            or committed != authoritative_transaction
        ):
            raise ValueError("case registration provenance is invalid")
        try:
            validation = builder.validate(*case.validation_run_ids)
            mutant = builder._load(validation.mutant_run_id)
        except (ValueError, OSError):
            raise ValueError("native corpus validation is unavailable") from None
        if (
            validation.clean_observation_hash == validation.mutant_observation_hash
            or mutant.observation.case_id != case.id
            or mutant.observation.template_id != case.template_id
            or mutant.observation.mutation_id != case.mutation_id
            or mutant.observation.source_hash != case.source_hash
            or mutant.observation.harness_hash != case.harness_hash
            or mutant.observation.input_set_hash != case.input_set_hash
            or mutant.observation.oracle_id != case.oracle_id
            or mutant.observation.target_tool != case.target_tool
            or mutant.observation.expected_finding != case.expected_finding
        ):
            raise ValueError("native corpus validation differs from registration")
        if case.id in cases:
            raise ValueError("case registration is duplicated")
        cases[case.id] = case
    return cases


class EvaluationExecutor:
    """Resolve sources controller-side; derive results only from terminal RunStore artifacts.

    ``sources`` is a controller-owned mapping, never a model or serialized result input.
    Corpus descriptors (including private truth) are not copied into the public store.
    """

    def __init__(
        self,
        service: ApplicationService,
        corpus: RunStore,
        sources: Mapping[str, Path],
        *,
        holdout_controller: "HoldoutController | None" = None,
        holdout_batch: "HoldoutBatch | None" = None,
        _corpus_family: "CorpusFamily | None" = None,
        _schedule_verifier: "EvaluationScheduleVerifier | None" = None,
    ) -> None:
        if service.store.visibility != "public":
            raise ValueError("evaluation requires a public service store")
        self.service, self.corpus = service, corpus
        self.sources = {case_id: path.absolute() for case_id, path in sources.items()}
        if (holdout_controller is None) != (holdout_batch is None):
            raise ValueError("holdout controller and batch must be configured together")
        self.holdout_controller, self.holdout_batch = holdout_controller, holdout_batch
        if _corpus_family is None:
            from gpu_agent.benchmark.ledger import CorpusFamily

            _corpus_family = CorpusFamily.configured(corpus)
        _corpus_family.require_store(corpus)
        self._corpus_family = _corpus_family
        if _schedule_verifier is None:
            from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

            _schedule_verifier = EvaluationScheduleVerifier.for_family(
                _corpus_family, service.store
            )
        self._schedule_verifier = _schedule_verifier
        service.store.bind_evaluation_verifier(_schedule_verifier)
        service._bind_evaluation_schedule_verifier(_schedule_verifier)

    def _ref(self, run: RunManifest, name: str) -> ArtifactRef:
        refs = [ref for ref in run.artifact_refs if ref.name == name]
        if len(refs) != 1:
            raise ValueError("evaluation artifact is missing or ambiguous")
        return refs[0]

    def validate_scheduled_record(
        self,
        record: PublicEvaluationRecord | EvaluationRecord,
        item: EvaluationScheduleItem,
        attempt: EvaluationAttempt,
    ) -> None:
        if self.service.binding is None:
            raise ValueError("evaluation service is unbound")
        registered_case_id = item.case_id
        if item.split == "holdout":
            if self.holdout_controller is None or self.holdout_batch is None:
                raise ValueError("private evaluation requires validated holdout authority")
            registered_case_id, _ = self.holdout_controller.resolve_private(
                self.holdout_batch, item.case_id
            )
        validate_evaluation_record(
            self.service.store,
            record,
            item,
            attempt,
            self.service.binding,
            RunStore(self.service.evaluator_root / "runs", visibility="evaluator"),
            self.corpus,
            self._corpus_family,
            registered_case_id,
        )

    @contextmanager
    def _unit_transaction(self, evaluation_run_id: str, ordinal: int) -> Iterator[tuple[int, Path]]:
        del ordinal
        path = self.service.store.root / f".evaluation-execution-{evaluation_run_id}.lock"
        reject_symlinks(path)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_mode & 0o077:
                raise ValueError("evaluation transaction lock is unsafe")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield fd, path
        finally:
            os.close(fd)

    def execute_scheduled(self, evaluation_run_id: str, ordinal: int) -> EvaluationRecord:
        return EvaluationExecutor._execute(self, evaluation_run_id, ordinal)

    def _execute(self, evaluation_run_id: str, ordinal: int) -> EvaluationRecord:
        if not re.fullmatch(r"[a-f0-9]{32}", evaluation_run_id) or ordinal < 0:
            raise ValueError("evaluation run locator is invalid")
        with self._unit_transaction(evaluation_run_id, ordinal) as (fd, path):
            return EvaluationExecutor.__execute_locked(self, fd, path, evaluation_run_id, ordinal)

    def __execute_locked(
        self,
        fd: int,
        path: Path,
        evaluation_run_id: str,
        ordinal: int,
    ) -> EvaluationRecord:
        expected = self.service.store.root / f".evaluation-execution-{evaluation_run_id}.lock"
        if path != expected or not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("evaluation transaction lease is invalid")
        # Re-acquiring on the same open file description is non-blocking.  A direct
        # helper call therefore cannot execute without first owning the real flock.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = self.service.store.load(evaluation_run_id)
        if (
            parent.kind != "evaluation"
            or parent.status != RunStatus.RUNNING
            or self.service.binding is None
            or parent.binding != self.service.binding
        ):
            raise ValueError("evaluation run is not active and bound")
        from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

        EvaluationScheduleVerifier.verify(self._schedule_verifier, evaluation_run_id)
        schedule_ref = self._ref(parent, "evaluation/schedule.json")
        schedule = EvaluationSchedule.model_validate_json(self.service.store.read(schedule_ref))
        if ordinal >= len(schedule.items) or schedule.bindings.max_unit_cost_usd is None:
            raise ValueError("evaluation ordinal is outside the frozen schedule")
        schedule_hash = hashlib.sha256(
            json.dumps(
                schedule.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        attempt_ref = self._ref(parent, f"evaluation/attempts/{ordinal}.json")
        attempt = EvaluationAttempt.model_validate_json(self.service.store.read(attempt_ref))
        item = schedule.items[ordinal]
        expected_attempt = EvaluationAttempt(
            run_id=evaluation_run_id,
            ordinal=ordinal,
            schedule_hash=schedule_hash,
            idempotency_key=hashlib.sha256(
                f"{evaluation_run_id}:{schedule_hash}:{ordinal}".encode()
            ).hexdigest(),
            reserved_cost_usd=schedule.bindings.max_unit_cost_usd,
        )
        expected_ordinals = list(range(len(schedule.items)))
        if [scheduled.ordinal for scheduled in schedule.items] != expected_ordinals:
            raise ValueError("evaluation schedule ordinals are not canonical")
        attempt_ordinals: list[int] = []
        record_ordinals: list[int] = []
        claim_ordinals: list[int] = []
        for ref in parent.artifact_refs:
            for prefix, destination in (
                ("evaluation/attempts/", attempt_ordinals),
                ("evaluation/records/", record_ordinals),
                ("evaluation/claims/", claim_ordinals),
            ):
                if ref.name.startswith(prefix):
                    match = re.fullmatch(re.escape(prefix) + r"([0-9]+)\.json", ref.name)
                    if match is None:
                        raise ValueError("evaluation artifact namespace is invalid")
                    destination.append(int(match.group(1)))
        if (
            sorted(attempt_ordinals) != list(range(ordinal + 1))
            or sorted(record_ordinals) != list(range(ordinal))
            or sorted(claim_ordinals) != list(range(ordinal))
            or len(attempt_ordinals) != len(set(attempt_ordinals))
            or len(record_ordinals) != len(set(record_ordinals))
            or len(claim_ordinals) != len(set(claim_ordinals))
        ):
            raise ValueError("evaluation execution state is not canonical")
        if (
            item.ordinal != ordinal
            or attempt != expected_attempt
            or any(ref.name == f"evaluation/records/{ordinal}.json" for ref in parent.artifact_refs)
            or schedule.bindings.commit != self.service.binding.repository.commit
            or schedule.bindings.prompt_version != self.service.binding.prompt_version
            or schedule.bindings.toolchain_hash != self.service.binding.toolchain_lock_hash
            or schedule.bindings.model_config_hash != self.service.binding.model_config_hash
        ):
            raise ValueError("evaluation schedule locator is invalid")
        if item.split == "holdout":
            if self.holdout_controller is None or self.holdout_batch is None:
                raise ValueError("private evaluation requires validated holdout authority")
            if self.holdout_controller.validate_batch(self.holdout_batch) != item.holdout_proof:
                raise ValueError("holdout authority differs from scheduled proof")
        elif item.holdout_proof is not None:
            raise ValueError("development evaluation cannot carry holdout authority")
        unit = EvaluationUnitBinding(
            evaluation_run_id=attempt.run_id,
            ordinal=item.ordinal,
            schedule_hash=attempt.schedule_hash,
            idempotency_key=attempt.idempotency_key,
            reserved_cost_usd=attempt.reserved_cost_usd,
            case_id=item.case_id,
            template_id=item.template_id,
            mode=item.mode,
            repeat=item.repeat,
            split=item.split,
            holdout_proof=item.holdout_proof,
        )
        attempt_content = attempt.model_dump_json().encode()
        claim = EvaluationExecutionClaim(
            run_id=evaluation_run_id,
            ordinal=ordinal,
            schedule_hash=schedule_hash,
            attempt_hash=hashlib.sha256(attempt_content).hexdigest(),
        )
        self.service.store.put_if_absent_exact(
            evaluation_run_id,
            f"evaluation/claims/{ordinal}.json",
            claim.model_dump_json().encode(),
            "public",
        )
        case_id, template_id, mode, repeat = (
            item.case_id,
            item.template_id,
            item.mode,
            item.repeat,
        )
        registered_case_id, registered_template_id = case_id, template_id
        if self.holdout_controller is not None and self.holdout_batch is not None:
            if case_id != template_id:
                raise ValueError("holdout evaluation alias is invalid")
            registered_case_id, registered_template_id = self.holdout_controller.resolve_private(
                self.holdout_batch, case_id
            )
        if self.service.binding is None:
            raise ValueError("evaluation service is unbound")
        case = registered_cases(self.corpus, self.service.binding, self._corpus_family).get(
            registered_case_id
        )
        if (
            case is None
            or case.template_id != registered_template_id
            or registered_case_id not in self.sources
        ):
            raise ValueError("case or template is not registered")
        if (case.split == "private") != (
            unit.split == "holdout" and unit.holdout_proof is not None
        ):
            raise ValueError("case visibility does not match evaluation split")
        if repeat < 0 or mode not in {"A", "B", "C", "D", "E"}:
            raise ValueError("invalid evaluation unit")
        source = self.sources[registered_case_id]
        selected = source / "kernel.cu" if source.is_dir() else source
        if hashlib.sha256(read_regular(selected, 4 * 1024 * 1024)).hexdigest() != case.source_hash:
            raise ValueError("registered source hash mismatch")
        run = self.service.diagnose(
            source,
            mode=mode,
            required_tools=(case.target_tool,),
            expected_source_hash=case.source_hash,
            evaluation_unit=unit,
        )
        run = self.service.store.load(run.id)
        if run.status != RunStatus.COMPLETED:
            raise ValueError("diagnosis artifacts are not terminal")
        store = self.service.store
        result = self.service.diagnosis(run.id)
        # Reading the required ref prevents diagnosis()'s missing-result convenience fallback.
        diagnosis_ref = self._ref(run, "diagnosis.json")
        store.read(diagnosis_ref)
        bundle = EvidenceRepository(store).public_view(run.id)
        source_refs = [ref for ref in bundle.source_snapshot if ref.name.endswith("/kernel.cu")]
        if len(source_refs) != 1 or source_refs[0].sha256 != case.source_hash:
            raise ValueError("diagnosis input differs from registered case")
        budget = AgentBudget.model_validate_json(
            store.read(self._ref(run, "agent/final-budget.json"))
        )
        summary = json.loads(store.read(self._ref(run, "agent/usage-summary.json")))
        if summary["physical_calls"] != budget.llm_calls:
            raise ValueError("provider usage artifacts disagree")
        acquisition = AcquisitionUsage.model_validate_json(
            store.read(self._ref(run, "agent/acquisition-usage.json"))
        )
        if (
            acquisition.sanitizer_calls > budget.sanitizer_calls
            or acquisition.retrieval_calls > budget.rag_calls
        ):
            raise ValueError("physical acquisition exceeds reserved attempts")
        usage: dict[str, int | None] = {
            "physical_calls": budget.llm_calls,
            "sanitizer_calls": acquisition.sanitizer_calls,
            "retrieval_calls": acquisition.retrieval_calls,
            "sanitizer_attempts": budget.sanitizer_calls,
            "retrieval_attempts": budget.rag_calls,
            "build_calls": int(bundle.build_result is not None),
            "runtime_calls": int(bundle.execution_result is not None),
        }
        diagnostic_tool_calls = (
            acquisition.sanitizer_calls
            + acquisition.retrieval_calls
            + int(bundle.build_result is not None)
            + int(bundle.execution_result is not None)
        )
        usage["diagnostic_tool_calls"] = diagnostic_tool_calls
        usage["tool_calls"] = diagnostic_tool_calls
        usage["total_sanitizer_calls"] = acquisition.sanitizer_calls
        invocations: dict[str, Invocation] = {}
        terminal_invocation_hashes: list[str] = []
        for ref in run.artifact_refs:
            if ref.name.startswith("provider/") and ref.name.endswith(".json"):
                invocation = Invocation.model_validate_json(store.read(ref))
                invocations[invocation.invocation_id] = invocation
                if invocation.state != "STARTED":
                    terminal_invocation_hashes.append(ref.sha256)
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            values = [
                getattr(item.usage, field) if item.usage else None for item in invocations.values()
            ]
            usage[field] = (
                sum(value for value in values if value is not None)
                if values and len(values) == budget.llm_calls and None not in values
                else None
            )
        cost_usd: float | None = 0.0
        if mode == "E":
            pricing_refs = [
                ref for ref in run.artifact_refs if ref.name == "agent/pricing-attestation.json"
            ]
            cost_usd = None
            if len(pricing_refs) == 1:
                pricing = PricingAttestation.model_validate_json(store.read(pricing_refs[0]))
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                cost_usd = (
                    pricing.cost(input_tokens, output_tokens)
                    if isinstance(input_tokens, int) and isinstance(output_tokens, int)
                    else None
                )
        checks: dict[str, str] = {
            item.tool_result.typed_payload.tool: item.check_outcome
            for item in bundle.sanitizer_results
            if item.tool_result is not None
        }
        candidate_hash: str | None = None
        candidate_run_id: str | None = None
        verification_run_id: str | None = None
        public_verification_hash: str | None = None
        verification: VerificationResult | None = None
        verification_audit: VerificationAuditResult | None = None
        finished = run.events[-1].at
        candidates = self.service.candidates(run.id)
        if len(candidates) > 1:
            raise ValueError("evaluation candidate is ambiguous")
        if candidates:
            candidate_run_id = candidates[0]
            candidate_run = store.load(candidates[0])
            if candidate_run.status != RunStatus.COMPLETED:
                raise ValueError("candidate artifacts are not terminal")
            candidate = PatchCandidate.model_validate_json(
                store.read(self._ref(candidate_run, "candidate.json"))
            )
            if candidate.generated_by != "agent" or candidate.parent_run_id != run.id:
                raise ValueError("evaluation candidate is not agent generated")
            candidate_hash = candidate.patched_source_hash
            self.service.verify(run.id, candidates[0])
            verification_runs = [
                store.load(path.name)
                for path in store.root.iterdir()
                if path.is_dir() and re.fullmatch(r"[a-f0-9]{32}", path.name)
            ]
            matches = [
                item
                for item in verification_runs
                if item.kind == "verification" and item.parent_run_id == run.id
            ]
            if len(matches) != 1 or matches[0].status != RunStatus.COMPLETED:
                raise ValueError("verification artifacts are missing or ambiguous")
            verification = VerificationResult.model_validate_json(
                store.read(self._ref(matches[0], "verification/result.json"))
            )
            verification_ref = self._ref(matches[0], "verification/result.json")
            verification_run_id = matches[0].id
            public_verification_hash = verification_ref.sha256
            if verification.candidate_hash != candidate_hash:
                raise ValueError("verification candidate hash mismatch")
            verification_audit = _validate_verification_audit(
                self.service.store,
                RunStore(self.service.evaluator_root / "runs", visibility="evaluator"),
                run.id,
                verification,
                self.service.binding,
            )
            checks.update(
                {
                    f"verification/{key}": value
                    for key, value in verification.required_checks.items()
                }
            )
            finished = matches[0].events[-1].at
            # The verification result has outcomes, not physical invocation
            # counts. Preserve diagnostic components and report totals unknown.
            usage["tool_calls"] = None
            usage["total_sanitizer_calls"] = None
        reason = result.limitations[0] if result.limitations else None
        status: str = "INCONCLUSIVE"
        if verification is not None:
            reason = verification.reason_code
            if verification.verdict != VerificationVerdict.INCONCLUSIVE:
                status = "COMPLETED"
        elif reason and reason not in {
            "INVALID_DIAGNOSIS_EVIDENCE",
            "MODEL_DECLARED_INCONCLUSIVE",
            "SANITIZER_EVIDENCE_UNAVAILABLE",
            "KNOWLEDGE_UNAVAILABLE",
            "NO_INFORMATION_GAIN",
        }:
            status = "TIMEOUT" if "TIMEOUT" in reason else "FAILED"
        # Only bounded reason codes leave this adapter, never an exception message.
        if reason is not None and not re.fullmatch(r"[A-Z0-9_]{1,80}", reason):
            reason = "EVALUATION_FAILED"
        evidence_refs = [ref for ref in run.artifact_refs if ref.name == "evidence/bundle.json"]
        if not evidence_refs:
            raise ValueError("evaluation evidence is unavailable")
        store.read(evidence_refs[-1])
        lineage = EvaluationLineage(
            diagnosis_run_id=run.id,
            diagnosis_hash=diagnosis_ref.sha256,
            evidence_hash=evidence_refs[-1].sha256,
            provider_invocation_hashes=terminal_invocation_hashes,
            candidate_run_id=candidate_run_id,
            verification_run_id=verification_run_id,
            public_verification_hash=public_verification_hash,
        )
        return EvaluationRecord.model_validate(
            {
                "record_id": run.id,
                "lineage": lineage,
                "case_id": case_id,
                "template_id": template_id,
                "mode": mode,
                "repeat": repeat,
                "input_hash": case.source_hash,
                "evidence_hash": evidence_refs[-1].sha256,
                "executed_checks": checks,
                "status": status,
                "diagnosis": result.model_dump(mode="json"),
                "patch_hash": candidate_hash,
                "oracle_passed": verification.public_oracle_passed if verification else None,
                "private_holdout_passed": (
                    verification_audit.observation.private_holdout_passed
                    if verification_audit
                    else None
                ),
                "patch_compile_passed": (
                    {"CLEAN": True, "FAILED": False}.get(
                        verification.required_checks.get("build", "")
                    )
                    if verification
                    else None
                ),
                "verdict": verification.verdict.value if verification else None,
                "regression_detected": bool(
                    verification and verification.verdict == VerificationVerdict.REGRESSION_DETECTED
                ),
                "usage": usage,
                "latency_ms": (finished - run.events[0].at).total_seconds() * 1000,
                "cost_usd": cost_usd,
                "failure_reason": reason,
            }
        )
