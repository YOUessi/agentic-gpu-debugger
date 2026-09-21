"""Production adapter from registered controller cases to immutable public observations."""

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from gpu_agent._resources import runtime_resource
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
    EvaluationProviderPolicy,
    EvaluationRecord,
    EvaluationSchedule,
    EvaluationScheduleItem,
    EvaluationUnitBinding,
    NativeEvaluationLineage,
    PricingAttestation,
    PublicEvaluationRecord,
)
from gpu_agent.benchmark.models import CaseManifest
from gpu_agent.contracts import (
    ArtifactRef,
    CurrentPhase,
    RunBinding,
    RunManifest,
    RunStatus,
)
from gpu_agent.evidence.models import EvidenceBundle
from gpu_agent.evidence.repository import EvidenceRepository
from gpu_agent.execution.models import (
    SanitizerTool,
)
from gpu_agent.patching import PatchCandidate, SourceSnapshot, materialize_candidate
from gpu_agent.service import ApplicationService
from gpu_agent.store import RunStore, read_regular, reject_symlinks
from gpu_agent.verification.derivation import validate_persisted_derivation
from gpu_agent.verification.models import (
    VerificationAuditResult,
    VerificationResult,
    VerificationVerdict,
)

if TYPE_CHECKING:
    from gpu_agent.benchmark.holdout import HoldoutBatch, HoldoutController
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier


_TRUTH_CASE = runtime_resource("benchmarks/development_truth/case_0001/case.json")


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
    """Resolve the public projection from fresh evaluator-owned native output."""
    return validate_persisted_derivation(public, evaluator, diagnosis_run_id, result, binding)


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
    *,
    diagnosis_parent_run_id: str | None = None,
    expected_visibility: Literal["public", "evaluator"] = "public",
) -> PublicEvaluationRecord:
    """Resolve one record after loading its case from authoritative corpus state."""
    if corpus is None or corpus_family is None:
        raise ValueError("native corpus authority is required for evaluation validation")
    selected_case_id = registered_case_id or item.case_id
    trusted_case = registered_cases(
        corpus, binding, corpus_family, cutoff=attempt.corpus_cutoff
    ).get(selected_case_id)
    if trusted_case is None or trusted_case.id != selected_case_id:
        raise ValueError("scheduled case is absent from the trusted corpus")
    return _validate_evaluation_record_against_case(
        store,
        record,
        item,
        attempt,
        binding,
        trusted_case,
        selected_case_id,
        evaluator=evaluator,
        diagnosis_parent_run_id=diagnosis_parent_run_id,
        expected_visibility=expected_visibility,
    )


def _validate_evaluation_record_against_case(
    store: RunStore,
    record: PublicEvaluationRecord | EvaluationRecord,
    item: EvaluationScheduleItem,
    attempt: EvaluationAttempt,
    binding: RunBinding,
    trusted_case: CaseManifest,
    registered_case_id: str,
    *,
    evaluator: RunStore | None = None,
    diagnosis_parent_run_id: str | None = None,
    expected_visibility: Literal["public", "evaluator"] = "public",
) -> PublicEvaluationRecord:
    """Resolve one public record back to immutable native execution artifacts."""
    public = record.public() if isinstance(record, EvaluationRecord) else record
    lineage = public.lineage
    if not isinstance(lineage, NativeEvaluationLineage):
        raise ValueError("native evaluation validation requires native lineage")
    if (
        public.corpus_cutoff != attempt.corpus_cutoff
        or lineage.corpus_cutoff != attempt.corpus_cutoff
    ):
        raise ValueError("evaluation record corpus cutoff differs from its attempt")
    run = store.load(lineage.diagnosis_run_id)
    expected_parent = diagnosis_parent_run_id or attempt.run_id
    if (
        store.visibility != expected_visibility
        or run.kind != "diagnosis"
        or run.status != RunStatus.COMPLETED
        or run.parent_run_id != expected_parent
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
        corpus_cutoff=attempt.corpus_cutoff,
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
    bundle = EvidenceRepository(store, evaluator=store.visibility == "evaluator").view(run.id)
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
    rejected_decisions = [
        (index, decision) for index, decision in enumerate(decisions) if not decision.allowed
    ]
    rejected_decision: PolicyDecision | None = None
    if rejected_decisions:
        rejected_index, rejected = rejected_decisions[0]
        if (
            item.mode != "E"
            or len(rejected_decisions) != 1
            or rejected_index != len(decisions) - 1
            or diagnosis.diagnostic_outcome != "INCONCLUSIVE"
            or len(rejected.reason_codes) != 1
            or diagnosis.limitations != rejected.reason_codes
        ):
            raise ValueError("controller route contains an invalid rejected decision")
        rejected_decision = rejected
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
    if trusted_case.id != registered_case_id:
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
    fixed_audit = json.loads(store.read(_one_ref(run, "agent/budget-audit.json")))
    failed_first_sanitizer = (
        diagnosis.limitations == ["AGENT_BUDGET_EXHAUSTED"]
        and observed_tools == []
        and observed_sanitizers == 0
        and acquisition.sanitizer_calls == 0
        and budget.sanitizer_calls == 1
        and fixed_audit
        == [
            {"id": 1, "action": "run_memcheck", "state": "ATTEMPTED"},
            {"id": 1, "action": "run_memcheck", "state": "STARTED"},
            {"id": 1, "action": "run_memcheck", "state": "FAILED"},
        ]
    )
    if item.mode == "C" and (
        observed_retrievals
        or acquisition.retrieval_calls
        or budget.rag_calls
        or (
            not failed_first_sanitizer
            and (
                observed_tools != expected_c_tools
                or acquisition.sanitizer_calls != len(expected_c_tools)
            )
        )
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
        logical_physical_before: list[int] = []
        terminal_provider_failure: Invocation | None = None
        while index < len(ordered_started):
            physical_before = index
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
                ):
                    raise ValueError("provider retry lineage is invalid")
                if retry_final.state == "COMPLETED":
                    completed = retry_final
                    index += 2
                elif (
                    retry_final.state == "FAILED"
                    and retry_final.error_code == "LLM_INVALID_OUTPUT"
                    and not retry_final.retryable
                    and first.kind == "plan"
                    and index + 2 == len(ordered_started)
                    and diagnosis.diagnostic_outcome == "INCONCLUSIVE"
                    and diagnosis.limitations == ["LLM_INVALID_OUTPUT"]
                ):
                    terminal_provider_failure = retry_final
                    index += 2
                    sequence += 1
                    break
                else:
                    raise ValueError("provider retry lineage is invalid")
            else:
                raise ValueError("provider invocation did not complete")
            logical_kinds.append(first.kind)
            logical_terminals.append(completed)
            logical_physical_before.append(physical_before)
            sequence += 1
        terminals = list(terminal_by_id.values())
        if logical_kinds.count("plan") != len(decision_refs) or len(
            {value.client_request_id for value in terminals}
        ) != len(terminals):
            raise ValueError("provider invocation sequence differs from controller route")
        plan_hashes = [value.output_hash for value in logical_terminals if value.kind == "plan"]
        if plan_hashes != [decision.action_hash for decision in decisions]:
            raise ValueError("provider plan outputs differ from controller decisions")
        step_refs = sorted(
            (
                ref
                for ref in run.artifact_refs
                if ref.name.startswith("actions/") and ref.name.endswith("/step.json")
            ),
            key=lambda ref: int(ref.name.split("/")[1]),
        )
        if len(step_refs) != len(decision_refs):
            raise ValueError("agent controller action steps are incomplete")
        initial_budget = AgentBudget.model_validate_json(
            store.read(_one_ref(run, "agent/initial-budget.json"))
        )
        budget_audit = json.loads(store.read(_one_ref(run, "agent/budget-audit.json")))
        if initial_budget != AgentBudget() or not isinstance(budget_audit, list):
            raise ValueError("agent controller initial budget is invalid")
        plan_physical_before = [
            before
            for before, kind in zip(logical_physical_before, logical_kinds, strict=True)
            if kind == "plan"
        ]
        if len(plan_physical_before) != len(step_refs):
            raise ValueError("agent controller plan inventory is invalid")
        agent_seen: set[str] = set()
        agent_evidence_by_id = {ref.id: ref for ref in evidence_refs}
        initial_evidence_indices = [
            evidence_index
            for evidence_index, observed in enumerate(bundles)
            if observed.build_result is not None
            and observed.execution_result is not None
            and not observed.sanitizer_results
            and not observed.retrieved_chunks
        ]
        if not initial_evidence_indices:
            raise ValueError("agent controller initial evidence is unavailable")
        expected_evidence_index = max(initial_evidence_indices)
        expected_budget = initial_budget
        agent_expected_audit: list[dict[str, object]] = []
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
                raise ValueError("agent controller action sequence is invalid")
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
                raise ValueError("agent controller action step is invalid")
            action = ACTION_ADAPTER.validate_python(step["action"])
            evidence = PublicEvidence.model_validate(step["evidence"])
            step_budget = AgentBudget.model_validate(step["budget"])
            evidence_ref = ArtifactRef.model_validate(step["evidence_ref"])
            authoritative_ref = agent_evidence_by_id.get(evidence_ref.id)
            if (
                authoritative_ref != evidence_ref
                or evidence_ref != evidence_refs[expected_evidence_index]
            ):
                raise ValueError("agent controller evidence reference is invalid")
            observed_bundle = EvidenceBundle.model_validate_json(store.read(evidence_ref))
            if evidence != public_evidence_from_bundle(store, observed_bundle):
                raise ValueError("agent controller evidence snapshot is invalid")
            if step["seen"] != sorted(agent_seen):
                raise ValueError("agent controller seen state is invalid")
            expected_step_budget = expected_budget.model_copy(
                update={
                    "llm_calls": plan_physical_before[index],
                    "remaining_seconds": step_budget.remaining_seconds,
                }
            )
            if (
                step_budget != expected_step_budget
                or step_budget.remaining_seconds > expected_budget.remaining_seconds
            ):
                raise ValueError("agent controller step budget is invalid")
            planner_id = len(agent_expected_audit) // 3 + 1
            agent_expected_audit.extend(
                [
                    {"id": planner_id, "action": "planner_llm", "state": "ATTEMPTED"},
                    {"id": planner_id, "action": "planner_llm", "state": "STARTED"},
                    {"id": planner_id, "action": "planner_llm", "state": "COMPLETED"},
                ]
            )
            expected_decision = decide_action(
                action,
                evidence,
                step_budget,
                CurrentPhase.DIAGNOSING,
                agent_seen,
            )
            if decisions[index] != expected_decision:
                raise ValueError("agent controller decision differs from policy replay")
            if expected_decision.allowed:
                agent_seen.add(action.action_type + action.typed_arguments.model_dump_json())
                agent_updates: dict[str, object] = {
                    "agent_steps": step_budget.agent_steps + 1,
                }
                if action.action_type in acquisition_actions:
                    reservation_id = len(agent_expected_audit) // 3 + 1
                    agent_audit_slice = budget_audit[
                        len(agent_expected_audit) : len(agent_expected_audit) + 3
                    ]
                    agent_prefix: list[dict[str, object]] = [
                        {
                            "id": reservation_id,
                            "action": action.action_type,
                            "state": "ATTEMPTED",
                        },
                        {
                            "id": reservation_id,
                            "action": action.action_type,
                            "state": "STARTED",
                        },
                    ]
                    if len(agent_audit_slice) != 3 or agent_audit_slice[:2] != agent_prefix:
                        raise ValueError("agent controller acquisition audit is invalid")
                    terminal = agent_audit_slice[2]
                    if terminal not in (
                        {
                            "id": reservation_id,
                            "action": action.action_type,
                            "state": "COMPLETED",
                        },
                        {
                            "id": reservation_id,
                            "action": action.action_type,
                            "state": "FAILED",
                        },
                    ):
                        raise ValueError("agent controller acquisition audit is invalid")
                    assert isinstance(terminal, dict)
                    agent_expected_audit.extend([*agent_prefix, terminal])
                    if index + 1 < len(step_refs):
                        if terminal["state"] != "COMPLETED":
                            raise ValueError("agent controller continued after failed acquisition")
                    if terminal["state"] == "COMPLETED" and action.action_type != "inspect_source":
                        expected_evidence_index += 1
                    if action.action_type in sanitizer_actions:
                        agent_updates["sanitizer_calls"] = step_budget.sanitizer_calls + 1
                    elif action.action_type == "retrieve_official_docs":
                        agent_updates["rag_calls"] = step_budget.rag_calls + 1
                    else:
                        agent_updates["source_reads"] = step_budget.source_reads + 1
                elif action.action_type == "finish_diagnosis":
                    reservation_id = len(agent_expected_audit) // 3 + 1
                    agent_expected_audit.extend(
                        [
                            {
                                "id": reservation_id,
                                "action": "diagnosis_llm",
                                "state": "ATTEMPTED",
                            },
                            {
                                "id": reservation_id,
                                "action": "diagnosis_llm",
                                "state": "STARTED",
                            },
                            {
                                "id": reservation_id,
                                "action": "diagnosis_llm",
                                "state": "COMPLETED",
                            },
                        ]
                    )
                if index + 1 < len(step_refs) and action.action_type not in acquisition_actions:
                    raise ValueError("agent controller continued after terminal action")
            expected_budget = step_budget.model_copy(update=agent_updates)
        if terminal_provider_failure is not None:
            reservation_id = len(agent_expected_audit) // 3 + 1
            agent_expected_audit.extend(
                [
                    {"id": reservation_id, "action": "planner_llm", "state": "ATTEMPTED"},
                    {"id": reservation_id, "action": "planner_llm", "state": "STARTED"},
                    {"id": reservation_id, "action": "planner_llm", "state": "FAILED"},
                ]
            )
        expected_final_budget = expected_budget.model_copy(
            update={
                "llm_calls": len(invocations),
                "remaining_seconds": budget.remaining_seconds,
            }
        )
        if (
            budget != expected_final_budget
            or budget.remaining_seconds > expected_budget.remaining_seconds
            or budget_audit != agent_expected_audit
            or expected_evidence_index != len(evidence_refs) - 1
        ):
            raise ValueError("agent controller final budget or audit is invalid")
        if rejected_decision is not None and (
            acquisition.sanitizer_calls != expected_budget.sanitizer_calls
            or acquisition.retrieval_calls != expected_budget.rag_calls
            or len(bundles[-1].sanitizer_results) != acquisition.sanitizer_calls
        ):
            raise ValueError("agent controller denied after inconsistent acquisition usage")
        diagnosis_hashes = [
            value.output_hash for value in logical_terminals if value.kind == "diagnose"
        ]
        expected_diagnosis_hash = hashlib.sha256(diagnosis.model_dump_json().encode()).hexdigest()
        if diagnosis_hashes not in ([], [expected_diagnosis_hash]):
            raise ValueError("provider diagnosis output differs from terminal diagnosis")
        if public.usage.get("physical_calls") != len(invocations):
            raise ValueError("provider usage differs from native invocations")

    from gpu_agent.verification.engine import candidate_run_id, verification_run_id

    exact_candidate_id = candidate_run_id(run.id)
    candidates = (
        [store.load(exact_candidate_id)] if (store.root / exact_candidate_id).exists() else []
    )
    candidate: PatchCandidate | None = None
    verification: VerificationResult | None = None
    verification_ref: ArtifactRef | None = None
    if candidates:
        candidate_run = candidates[0]
        if candidate_run.external_origin != run.external_origin:
            raise ValueError("evaluation candidate topology is invalid")
        candidate = PatchCandidate.model_validate_json(
            store.read(_one_ref(candidate_run, "candidate.json"))
        )
        exact_verification_id = verification_run_id(run.id, candidate.patched_source_hash)
        verifications = (
            [store.load(exact_verification_id)]
            if (store.root / exact_verification_id).exists()
            else []
        )
        if (
            candidate_run.kind != "candidate"
            or candidate_run.parent_run_id != run.id
            or candidate_run.status != RunStatus.COMPLETED
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
        snapshot_hashes: dict[str, str] = {}
        snapshot_contents: dict[str, bytes] = {}
        for source_ref in bundle.source_snapshot:
            name = Path(source_ref.name).name
            if (
                name not in {"kernel.cu", "vector_io.cpp", "vector_api.h", "json.hpp"}
                or name in snapshot_hashes
            ):
                raise ValueError("evaluation candidate provenance is invalid")
            snapshot_hashes[name] = source_ref.sha256
            snapshot_contents[name] = store.read(source_ref)
        try:
            with tempfile.TemporaryDirectory(prefix="gpu-agent-evaluation-candidate-") as directory:
                root = Path(directory)
                for name, content in snapshot_contents.items():
                    (root / name).write_bytes(content)
                materialize_candidate(
                    SourceSnapshot(parent_run_id=run.id, root=root, hashes=snapshot_hashes),
                    candidate,
                )
        except (OSError, ValueError):
            raise ValueError("evaluation candidate provenance is invalid") from None
        patch_hashes = [value.output_hash for value in logical_terminals if value.kind == "patch"]
        expected_patch_hash = hashlib.sha256(
            json.dumps({"unified_diff": candidate.unified_diff}, separators=(",", ":")).encode()
        ).hexdigest()
        if patch_hashes != [expected_patch_hash]:
            raise ValueError("provider patch output differs from candidate")
        verification_run = verifications[0]
        if (
            verification_run.kind != "verification"
            or verification_run.parent_run_id != run.id
            or verification_run.external_origin != run.external_origin
        ):
            raise ValueError("evaluation verification topology is invalid")
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
        lineage.candidate_run_id is not None
        or lineage.verification_run_id is not None
        or lineage.public_verification_hash is not None
        or public.patch_hash is not None
        or public.verdict is not None
    ):
        raise ValueError("evaluation record declares nonexistent child lineage")
    if item.mode == "E":
        expected_kinds = ["plan"] * len(decisions)
        if rejected_decision is None:
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
    expected_status: Literal["COMPLETED", "FAILED", "TIMEOUT", "INCONCLUSIVE"] = "INCONCLUSIVE"
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
    validated = PublicEvaluationRecord(
        record_id=run.id,
        corpus_cutoff=attempt.corpus_cutoff,
        lineage=NativeEvaluationLineage(
            corpus_cutoff=attempt.corpus_cutoff,
            diagnosis_run_id=run.id,
            diagnosis_hash=diagnosis_ref.sha256,
            evidence_hash=evidence_refs[-1].sha256,
            provider_invocation_hashes=terminal_hashes,
            candidate_run_id=candidates[0].id if candidate is not None else None,
            verification_run_id=(verifications[0].id if verification is not None else None),
            public_verification_hash=(verification_ref.sha256 if verification_ref else None),
        ),
        case_id=item.case_id,
        template_id=item.template_id,
        mode=item.mode,
        repeat=item.repeat,
        input_hash=source_refs[0].sha256,
        evidence_hash=evidence_refs[-1].sha256,
        executed_checks=expected_checks,
        status=expected_status,
        diagnosis=diagnosis.model_dump(mode="json"),
        patch_hash=candidate.patched_source_hash if candidate else None,
        oracle_passed=verification.public_oracle_passed if verification else None,
        verdict=verification.verdict.value if verification else None,
        regression_detected=bool(
            verification and verification.verdict == VerificationVerdict.REGRESSION_DETECTED
        ),
        usage=expected_usage,
        latency_ms=expected_latency,
        cost_usd=expected_cost,
        failure_reason=reason,
    )
    if public != validated:
        raise ValueError("evaluation record summary differs from native artifacts")
    return validated


def registered_cases(
    corpus: RunStore,
    evaluation_binding: RunBinding,
    family: "CorpusFamily | None" = None,
    *,
    cutoff: int | None = None,
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
        for transaction in family.ledger.committed_through(cutoff)
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
        holdout_service: ApplicationService | None = None,
        holdout_controller: "HoldoutController | None" = None,
        holdout_batch: "HoldoutBatch | None" = None,
        _corpus_family: "CorpusFamily | None" = None,
        _schedule_verifier: "EvaluationScheduleVerifier | None" = None,
    ) -> None:
        if service.store.visibility != "public":
            raise ValueError("evaluation requires a public service store")
        self.service, self.corpus = service, corpus
        self.sources = {case_id: path.absolute() for case_id, path in sources.items()}
        holdout_parts = (holdout_service, holdout_controller, holdout_batch)
        if any(part is None for part in holdout_parts) and any(
            part is not None for part in holdout_parts
        ):
            raise ValueError("holdout service, controller, and batch must be configured together")
        self.holdout_service = holdout_service
        self.holdout_controller, self.holdout_batch = holdout_controller, holdout_batch
        if _corpus_family is None:
            from gpu_agent.benchmark.ledger import CorpusFamily

            _corpus_family = CorpusFamily.configured(corpus)
        _corpus_family.require_store(service.store)
        _corpus_family.require_store(corpus)
        _corpus_family.require_store(service.evaluator_store)
        self._corpus_family = _corpus_family
        if _schedule_verifier is None:
            from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

            _schedule_verifier = EvaluationScheduleVerifier.for_family(
                _corpus_family, service.store
            )
        from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

        EvaluationScheduleVerifier.require_store(_schedule_verifier, service.store)
        if holdout_service is None:
            if corpus.visibility != "public" or corpus.identity != service.store.identity:
                raise ValueError("development evaluation requires the exact public family store")
        else:
            _corpus_family.require_store(holdout_service.store)
            _corpus_family.require_store(holdout_service.evaluator_store)
            if (
                corpus.visibility != "evaluator"
                or holdout_service.store.visibility != "evaluator"
                or corpus.identity != holdout_service.store.identity
                or corpus.identity != holdout_service.evaluator_store.identity
                or corpus.identity != service.evaluator_store.identity
                or holdout_service.binding != service.binding
                or holdout_controller is None
                or holdout_batch is None
                or holdout_controller.binding != service.binding
                or holdout_controller.public.identity != service.store.identity
                or holdout_controller.evaluator.identity != corpus.identity
                or holdout_controller._schedule_verifier is not _schedule_verifier
            ):
                raise ValueError("holdout evaluation requires exact paired family services")
        self._schedule_verifier = _schedule_verifier
        service.store.bind_evaluation_verifier(_schedule_verifier)
        service._bind_evaluation_schedule_verifier(_schedule_verifier)

    def _ref(self, run: RunManifest, name: str) -> ArtifactRef:
        refs = [ref for ref in run.artifact_refs if ref.name == name]
        if len(refs) != 1:
            raise ValueError("evaluation artifact is missing or ambiguous")
        return refs[0]

    def _artifact_derived_native_record(
        self,
        service: ApplicationService,
        record: EvaluationRecord,
        item: EvaluationScheduleItem,
        attempt: EvaluationAttempt,
        registered_case_id: str,
        registered_template_id: str,
        *,
        diagnosis_parent_run_id: str | None = None,
    ) -> EvaluationRecord:
        """Apply the one native artifact parser to either selected execution store."""
        if service.binding is None:
            raise ValueError("evaluation service is unbound")
        native_item = item.model_copy(
            update={
                "case_id": registered_case_id,
                "template_id": registered_template_id,
            }
        )
        validated = validate_evaluation_record(
            service.store,
            record,
            native_item,
            attempt,
            service.binding,
            service.evaluator_store,
            self.corpus,
            self._corpus_family,
            registered_case_id,
            diagnosis_parent_run_id=diagnosis_parent_run_id,
            expected_visibility=service.store.visibility,
        )
        if validated != record.public():
            raise ValueError("native evaluation adapter is inconsistent")
        return record

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
            recovered = self.holdout_controller.recover_execution(self.holdout_batch, item, attempt)
            if recovered is None or recovered.model_dump_json() != record.model_dump_json():
                raise ValueError("holdout public record differs from evaluator transaction")
            return
        validate_evaluation_record(
            self.service.store,
            record,
            item,
            attempt,
            self.service.binding,
            self.service.evaluator_store,
            self.corpus,
            self._corpus_family,
            registered_case_id,
        )

    def execute_scheduled(
        self, evaluation_run_id: str, ordinal: int
    ) -> PublicEvaluationRecord | EvaluationRecord:
        if not re.fullmatch(r"[a-f0-9]{32}", evaluation_run_id) or ordinal < 0:
            raise ValueError("evaluation run locator is invalid")
        expected = (
            self.service.store.root / f".evaluation-execution-{evaluation_run_id}-{ordinal}.lock"
        )
        reject_symlinks(expected)
        fd = os.open(expected, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or opened.st_mode & 0o077:
                raise ValueError("evaluation transaction lock is unsafe")
            fcntl.flock(fd, fcntl.LOCK_EX)
            reject_symlinks(expected)
            located = os.stat(expected, follow_symlinks=False)
            if not stat.S_ISREG(located.st_mode) or (opened.st_dev, opened.st_ino) != (
                located.st_dev,
                located.st_ino,
            ):
                raise ValueError("evaluation transaction lease path changed")
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
            schedule_ref = EvaluationExecutor._ref(self, parent, "evaluation/schedule.json")
            schedule = EvaluationSchedule.model_validate_json(self.service.store.read(schedule_ref))
            if ordinal >= len(schedule.items) or schedule.bindings.max_unit_cost_usd is None:
                raise ValueError("evaluation ordinal is outside the frozen schedule")
            schedule_hash = hashlib.sha256(
                json.dumps(
                    schedule.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            attempt_ref = EvaluationExecutor._ref(
                self, parent, f"evaluation/attempts/{ordinal}.json"
            )
            attempt = EvaluationAttempt.model_validate_json(self.service.store.read(attempt_ref))
            item = schedule.items[ordinal]
            expected_attempt = EvaluationAttempt(
                run_id=evaluation_run_id,
                ordinal=ordinal,
                schedule_hash=schedule_hash,
                corpus_cutoff=schedule.corpus_cutoff,
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
                or any(
                    ref.name == f"evaluation/records/{ordinal}.json" for ref in parent.artifact_refs
                )
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
            attempt_content = attempt.model_dump_json().encode()
            claim = EvaluationExecutionClaim(
                run_id=evaluation_run_id,
                ordinal=ordinal,
                schedule_hash=schedule_hash,
                corpus_cutoff=schedule.corpus_cutoff,
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
            prepared = None
            execution_service = self.service
            if item.split == "holdout":
                if (
                    self.holdout_controller is None
                    or self.holdout_batch is None
                    or self.holdout_service is None
                ):
                    raise ValueError("private evaluation requires paired evaluator authority")
                if case_id != template_id:
                    raise ValueError("holdout evaluation alias is invalid")
                prepared = self.holdout_controller.reserve_execution(
                    self.holdout_batch,
                    evaluation_run_id=evaluation_run_id,
                    item=item,
                    attempt=attempt,
                )
                registered_case_id = prepared.binding.private_case_id
                registered_template_id = prepared.binding.private_template_id
                unit = prepared.evaluation_unit
                execution_service = self.holdout_service
            else:
                unit = EvaluationUnitBinding(
                    evaluation_run_id=attempt.run_id,
                    ordinal=item.ordinal,
                    schedule_hash=attempt.schedule_hash,
                    corpus_cutoff=attempt.corpus_cutoff,
                    idempotency_key=attempt.idempotency_key,
                    reserved_cost_usd=attempt.reserved_cost_usd,
                    case_id=item.case_id,
                    template_id=item.template_id,
                    mode=item.mode,
                    repeat=item.repeat,
                    split=item.split,
                    holdout_proof=item.holdout_proof,
                )
            if execution_service.binding is None:
                raise ValueError("evaluation service is unbound")
            case = registered_cases(
                self.corpus,
                execution_service.binding,
                self._corpus_family,
                cutoff=schedule.corpus_cutoff,
            ).get(registered_case_id)
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
            if (
                hashlib.sha256(read_regular(selected, 4 * 1024 * 1024)).hexdigest()
                != case.source_hash
            ):
                raise ValueError("registered source hash mismatch")
            if prepared is not None:
                assert self.holdout_controller is not None
                assert self.holdout_batch is not None
                run = self.holdout_controller.execute_reserved_diagnosis(
                    self.holdout_batch,
                    prepared,
                    execution_service,
                    source,
                    required_tools=(case.target_tool,),
                    expected_source_hash=case.source_hash,
                )
            else:
                run = execution_service.diagnose(
                    source,
                    mode=mode,
                    required_tools=(case.target_tool,),
                    expected_source_hash=case.source_hash,
                    evaluation_unit=unit,
                )
            run = execution_service.store.load(run.id)
            if run.status != RunStatus.COMPLETED:
                raise ValueError("diagnosis artifacts are not terminal")
            store = execution_service.store
            result = execution_service.diagnosis(run.id)
            # Reading the required ref prevents diagnosis()'s missing-result convenience fallback.
            diagnosis_ref = EvaluationExecutor._ref(self, run, "diagnosis.json")
            store.read(diagnosis_ref)
            bundle = EvidenceRepository(store, evaluator=store.visibility == "evaluator").view(
                run.id
            )
            source_refs = [ref for ref in bundle.source_snapshot if ref.name.endswith("/kernel.cu")]
            if len(source_refs) != 1 or source_refs[0].sha256 != case.source_hash:
                raise ValueError("diagnosis input differs from registered case")
            budget = AgentBudget.model_validate_json(
                store.read(EvaluationExecutor._ref(self, run, "agent/final-budget.json"))
            )
            summary = json.loads(
                store.read(EvaluationExecutor._ref(self, run, "agent/usage-summary.json"))
            )
            if summary["physical_calls"] != budget.llm_calls:
                raise ValueError("provider usage artifacts disagree")
            acquisition = AcquisitionUsage.model_validate_json(
                store.read(EvaluationExecutor._ref(self, run, "agent/acquisition-usage.json"))
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
                    getattr(item.usage, field) if item.usage else None
                    for item in invocations.values()
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
            candidates = execution_service.candidates(run.id)
            if len(candidates) > 1:
                raise ValueError("evaluation candidate is ambiguous")
            if candidates:
                candidate_run_id = candidates[0]
                candidate_run = store.load(candidates[0])
                if candidate_run.status != RunStatus.COMPLETED:
                    raise ValueError("candidate artifacts are not terminal")
                candidate = PatchCandidate.model_validate_json(
                    store.read(EvaluationExecutor._ref(self, candidate_run, "candidate.json"))
                )
                if candidate.generated_by != "agent" or candidate.parent_run_id != run.id:
                    raise ValueError("evaluation candidate is not agent generated")
                candidate_hash = candidate.patched_source_hash
                _, verification_run_id = execution_service.verify_exact(run.id, candidates[0])
                verification_run = store.load(verification_run_id)
                if (
                    verification_run.kind != "verification"
                    or verification_run.parent_run_id != run.id
                    or verification_run.status != RunStatus.COMPLETED
                ):
                    raise ValueError("verification artifacts are missing or ambiguous")
                verification = VerificationResult.model_validate_json(
                    store.read(
                        EvaluationExecutor._ref(self, verification_run, "verification/result.json")
                    )
                )
                verification_ref = EvaluationExecutor._ref(
                    self, verification_run, "verification/result.json"
                )
                public_verification_hash = verification_ref.sha256
                if verification.candidate_hash != candidate_hash:
                    raise ValueError("verification candidate hash mismatch")
                verification_audit = _validate_verification_audit(
                    execution_service.store,
                    execution_service.evaluator_store,
                    run.id,
                    verification,
                    execution_service.binding,
                )
                checks.update(
                    {
                        f"verification/{key}": value
                        for key, value in verification.required_checks.items()
                    }
                )
                finished = verification_run.events[-1].at
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
            lineage = NativeEvaluationLineage(
                corpus_cutoff=unit.corpus_cutoff,
                diagnosis_run_id=run.id,
                diagnosis_hash=diagnosis_ref.sha256,
                evidence_hash=evidence_refs[-1].sha256,
                provider_invocation_hashes=terminal_invocation_hashes,
                candidate_run_id=candidate_run_id,
                verification_run_id=verification_run_id,
                public_verification_hash=public_verification_hash,
            )
            native = EvaluationRecord.model_validate(
                {
                    "record_id": run.id,
                    "corpus_cutoff": unit.corpus_cutoff,
                    "lineage": lineage,
                    "case_id": registered_case_id,
                    "template_id": registered_template_id,
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
                        verification
                        and verification.verdict == VerificationVerdict.REGRESSION_DETECTED
                    ),
                    "usage": usage,
                    "latency_ms": (finished - run.events[0].at).total_seconds() * 1000,
                    "cost_usd": cost_usd,
                    "failure_reason": reason,
                }
            )
            native = EvaluationExecutor._artifact_derived_native_record(
                self,
                execution_service,
                native,
                item,
                attempt,
                registered_case_id,
                registered_template_id,
                diagnosis_parent_run_id=(prepared.execution_run_id if prepared else None),
            )
            if prepared is not None:
                assert self.holdout_controller is not None
                return self.holdout_controller.complete_execution(prepared, native)
            return native
        finally:
            os.close(fd)

    def recover_scheduled(
        self, evaluation_run_id: str, ordinal: int
    ) -> PublicEvaluationRecord | None:
        """Recover only an exact terminal evaluator projection for one public attempt."""
        if not re.fullmatch(r"[a-f0-9]{32}", evaluation_run_id) or ordinal < 0:
            raise ValueError("evaluation run locator is invalid")
        if self.holdout_controller is None or self.holdout_batch is None:
            return None
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
        schedule = EvaluationSchedule.model_validate_json(
            self.service.store.read(
                EvaluationExecutor._ref(self, parent, "evaluation/schedule.json")
            )
        )
        if ordinal >= len(schedule.items) or schedule.bindings.max_unit_cost_usd is None:
            raise ValueError("evaluation ordinal is outside the frozen schedule")
        schedule_hash = hashlib.sha256(
            json.dumps(
                schedule.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        attempt_ref = EvaluationExecutor._ref(self, parent, f"evaluation/attempts/{ordinal}.json")
        attempt_content = self.service.store.read(attempt_ref)
        attempt = EvaluationAttempt.model_validate_json(attempt_content)
        expected_attempt = EvaluationAttempt(
            run_id=evaluation_run_id,
            ordinal=ordinal,
            schedule_hash=schedule_hash,
            corpus_cutoff=schedule.corpus_cutoff,
            idempotency_key=hashlib.sha256(
                f"{evaluation_run_id}:{schedule_hash}:{ordinal}".encode()
            ).hexdigest(),
            reserved_cost_usd=schedule.bindings.max_unit_cost_usd,
        )
        item = schedule.items[ordinal]
        claim = EvaluationExecutionClaim.model_validate_json(
            self.service.store.read(
                EvaluationExecutor._ref(self, parent, f"evaluation/claims/{ordinal}.json")
            )
        )
        expected_claim = EvaluationExecutionClaim(
            run_id=evaluation_run_id,
            ordinal=ordinal,
            schedule_hash=schedule_hash,
            corpus_cutoff=schedule.corpus_cutoff,
            attempt_hash=hashlib.sha256(attempt_content).hexdigest(),
        )
        if (
            item.ordinal != ordinal
            or item.split != "holdout"
            or attempt != expected_attempt
            or claim != expected_claim
        ):
            return None
        return self.holdout_controller.recover_execution(self.holdout_batch, item, attempt)

    def _execute(
        self, evaluation_run_id: str, ordinal: int
    ) -> PublicEvaluationRecord | EvaluationRecord:
        return EvaluationExecutor.execute_scheduled(self, evaluation_run_id, ordinal)
