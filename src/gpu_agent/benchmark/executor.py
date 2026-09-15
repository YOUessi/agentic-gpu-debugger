"""Production adapter from registered controller cases to immutable public observations."""

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from gpu_agent.agent.models import AgentBudget
from gpu_agent.agent.provider import Invocation
from gpu_agent.benchmark.evaluation import EvaluationMode, EvaluationRecord
from gpu_agent.benchmark.models import CaseManifest
from gpu_agent.contracts import ArtifactRef, RunManifest, RunStatus
from gpu_agent.evidence.repository import EvidenceRepository
from gpu_agent.patching import PatchCandidate
from gpu_agent.service import ApplicationService
from gpu_agent.store import RunStore, read_regular
from gpu_agent.verification.models import VerificationResult, VerificationVerdict


class CaseExecutionAttestationUnavailable(ValueError):
    """The current artifact contract cannot attest all CaseExecution claims."""


def registered_cases(corpus: RunStore) -> dict[str, CaseManifest]:
    """Load only completed, hash-checked registrations from the controller corpus."""
    cases: dict[str, CaseManifest] = {}
    for path in sorted(corpus.root.iterdir()):
        if not re.fullmatch(r"[a-f0-9]{32}", path.name) or not path.is_dir():
            continue
        run = corpus.load(path.name)
        if run.kind != "benchmark_case":
            continue
        if run.status != RunStatus.COMPLETED:
            raise ValueError("case registration is not terminal")
        refs = [ref for ref in run.artifact_refs if ref.name == "case-manifest.json"]
        if len(refs) != 1:
            raise ValueError("case registration is ambiguous")
        case = CaseManifest.model_validate_json(corpus.read(refs[0]))
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
        self, service: ApplicationService, corpus: RunStore, sources: Mapping[str, Path]
    ) -> None:
        if service.store.visibility != "public":
            raise ValueError("evaluation requires a public service store")
        self.service, self.corpus = service, corpus
        self.sources = {case_id: path.absolute() for case_id, path in sources.items()}

    def _ref(self, run: RunManifest, name: str) -> ArtifactRef:
        refs = [ref for ref in run.artifact_refs if ref.name == name]
        if len(refs) != 1:
            raise ValueError("evaluation artifact is missing or ambiguous")
        return refs[0]

    def execute(
        self, case_id: str, template_id: str, mode: EvaluationMode, repeat: int
    ) -> EvaluationRecord:
        case = registered_cases(self.corpus).get(case_id)
        if case is None or case.template_id != template_id or case_id not in self.sources:
            raise ValueError("case or template is not registered")
        if repeat < 0 or mode not in {"A", "B", "C", "D", "E"}:
            raise ValueError("invalid evaluation unit")
        source = self.sources[case_id]
        selected = source / "kernel.cu" if source.is_dir() else source
        if hashlib.sha256(read_regular(selected, 4 * 1024 * 1024)).hexdigest() != case.source_hash:
            raise ValueError("registered source hash mismatch")
        run = self.service.diagnose(
            source,
            mode=mode,
            required_tools=(case.target_tool,),
            expected_source_hash=case.source_hash,
        )
        run = self.service.store.load(run.id)
        if run.status != RunStatus.COMPLETED:
            raise ValueError("diagnosis artifacts are not terminal")
        store = self.service.store
        result = self.service.diagnosis(run.id)
        # Reading the required ref prevents diagnosis()'s missing-result convenience fallback.
        store.read(self._ref(run, "diagnosis.json"))
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
        usage: dict[str, int | None] = {
            "physical_calls": budget.llm_calls,
            "sanitizer_calls": budget.sanitizer_calls,
            "retrieval_calls": budget.rag_calls,
            "build_calls": int(bundle.build_result is not None),
            "runtime_calls": int(bundle.execution_result is not None),
        }
        invocations: dict[str, Invocation] = {}
        for ref in run.artifact_refs:
            if ref.name.startswith("provider/") and ref.name.endswith(".json"):
                invocation = Invocation.model_validate_json(store.read(ref))
                invocations[invocation.invocation_id] = invocation
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            values = [
                getattr(item.usage, field) if item.usage else None for item in invocations.values()
            ]
            usage[field] = (
                sum(value for value in values if value is not None)
                if values and len(values) == budget.llm_calls and None not in values
                else None
            )
        checks: dict[str, str] = {
            item.tool_result.typed_payload.tool: item.check_outcome
            for item in bundle.sanitizer_results
            if item.tool_result is not None
        }
        candidate_hash: str | None = None
        verification: VerificationResult | None = None
        finished = run.events[-1].at
        candidates = self.service.candidates(run.id)
        if len(candidates) > 1:
            raise ValueError("evaluation candidate is ambiguous")
        if candidates:
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
            if verification.candidate_hash != candidate_hash:
                raise ValueError("verification candidate hash mismatch")
            checks.update(
                {
                    f"verification/{key}": value
                    for key, value in verification.required_checks.items()
                }
            )
            finished = matches[0].events[-1].at
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
        return EvaluationRecord.model_validate(
            {
                "record_id": run.id,
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
                "verdict": verification.verdict.value if verification else None,
                "regression_detected": bool(
                    verification and verification.verdict == VerificationVerdict.REGRESSION_DETECTED
                ),
                "usage": usage,
                "latency_ms": (finished - run.events[0].at).total_seconds() * 1000,
                # Provider artifacts currently have tokens but no attested monetary cost.
                "cost_usd": None,
                "failure_reason": reason,
            }
        )
