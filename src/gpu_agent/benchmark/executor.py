"""Production adapter from registered controller cases to immutable public observations."""

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from gpu_agent.agent.models import AcquisitionUsage, AgentBudget, DiagnosisResult, PolicyDecision
from gpu_agent.agent.provider import Invocation
from gpu_agent.benchmark.evaluation import (
    EvaluationAttempt,
    EvaluationLineage,
    EvaluationMode,
    EvaluationProviderPolicy,
    EvaluationRecord,
    EvaluationScheduleItem,
    EvaluationUnitBinding,
    PricingAttestation,
    PublicEvaluationRecord,
)
from gpu_agent.benchmark.models import CaseManifest
from gpu_agent.contracts import ArtifactRef, RunBinding, RunManifest, RunStatus
from gpu_agent.evidence.repository import EvidenceRepository
from gpu_agent.patching import PatchCandidate
from gpu_agent.service import ApplicationService
from gpu_agent.store import RunStore, read_regular
from gpu_agent.verification.models import VerificationResult, VerificationVerdict

if TYPE_CHECKING:
    from gpu_agent.benchmark.holdout import HoldoutBatch, HoldoutController
    from gpu_agent.benchmark.ledger import CorpusFamily


class CostBoundUnavailable(ValueError):
    """A production monetary bound must be attested before any provider work is allowed."""


def _one_ref(run: RunManifest, name: str) -> ArtifactRef:
    refs = [ref for ref in run.artifact_refs if ref.name == name]
    if len(refs) != 1:
        raise ValueError("evaluation lineage artifact is missing or ambiguous")
    return refs[0]


def validate_evaluation_record(
    store: RunStore,
    record: PublicEvaluationRecord | EvaluationRecord,
    item: EvaluationScheduleItem,
    attempt: EvaluationAttempt,
    binding: RunBinding,
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
    store.read(evidence_refs[-1])
    bundle = EvidenceRepository(store).public_view(run.id)
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
        (ref for ref in run.artifact_refs if ref.name.startswith("actions/")),
        key=lambda ref: int(ref.name.split("/")[1]),
    )
    decisions = [PolicyDecision.model_validate_json(store.read(ref)) for ref in decision_refs]
    if any(not decision.allowed for decision in decisions):
        raise ValueError("controller route contains a rejected decision")
    acquisition_policy = json.loads(store.read(policy_ref))
    if acquisition_policy != {"mode": item.mode}:
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
            {f"verification/{key}": value for key, value in verification.required_checks.items()}
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
    elif reason and reason not in {"INVALID_DIAGNOSIS_EVIDENCE", "MODEL_DECLARED_INCONCLUSIVE"}:
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
    for path in sorted(corpus.root.iterdir()):
        if not re.fullmatch(r"[a-f0-9]{32}", path.name) or not path.is_dir():
            continue
        run = corpus.load(path.name)
        if run.kind != "benchmark_case":
            continue
        if (
            run.status != RunStatus.COMPLETED
            or run.binding is None
            or run.binding.purpose != "corpus_validation"
            or run.binding.toolchain_lock_hash is None
            or run.binding.case_registry_hash is None
            or run.binding.corpus_ledger_namespace_hash is None
        ):
            raise ValueError("case registration is not terminal")
        refs = [ref for ref in run.artifact_refs if ref.name == "case-manifest.json"]
        transactions = [
            ref for ref in run.artifact_refs if ref.name == "validation/ledger-transaction.json"
        ]
        if len(refs) != 1 or len(transactions) != 1:
            raise ValueError("case registration is ambiguous")
        case = CaseManifest.model_validate_json(corpus.read(refs[0]))
        transaction = json.loads(corpus.read(transactions[0]))
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
            or committed.target_store_hash != family.ledger.target_store_hash(corpus)
            or committed.visibility != corpus.visibility
            or committed.manifest_hash != refs[0].sha256
            or case.toolchain_hash != run.binding.toolchain_lock_hash
            or run.binding.repository != evaluation_binding.repository
            or run.binding.toolchain_lock_hash != evaluation_binding.toolchain_lock_hash
            or len(case.validation_run_ids) != 2
            or len(set(case.validation_run_ids)) != 2
            or not all(re.fullmatch(r"[a-f0-9]{32}", item) for item in case.validation_run_ids)
            or (case.split == "public") != (corpus.visibility == "public")
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

    def _ref(self, run: RunManifest, name: str) -> ArtifactRef:
        refs = [ref for ref in run.artifact_refs if ref.name == name]
        if len(refs) != 1:
            raise ValueError("evaluation artifact is missing or ambiguous")
        return refs[0]

    def execute(
        self, case_id: str, template_id: str, mode: EvaluationMode, repeat: int
    ) -> EvaluationRecord:
        return self._execute(case_id, template_id, mode, repeat, None)

    def execute_scheduled(
        self, item: EvaluationScheduleItem, attempt: EvaluationAttempt
    ) -> EvaluationRecord:
        if attempt.ordinal != item.ordinal:
            raise ValueError("evaluation attempt ordinal does not match scheduled unit")
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
        return self._execute(item.case_id, item.template_id, item.mode, item.repeat, unit)

    def _execute(
        self,
        case_id: str,
        template_id: str,
        mode: EvaluationMode,
        repeat: int,
        unit: EvaluationUnitBinding | None,
    ) -> EvaluationRecord:
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
        if unit is not None and (
            (case.split == "private")
            != (unit.split == "holdout" and unit.holdout_proof is not None)
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
        elif reason and reason not in {"INVALID_DIAGNOSIS_EVIDENCE", "MODEL_DECLARED_INCONCLUSIVE"}:
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
                    verification.private_holdout_passed if verification else None
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
