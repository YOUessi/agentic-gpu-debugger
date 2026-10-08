"""Re-use *attested public* self-check observations for one unchanged candidate.

This module never reads private/evaluator state. It copies citation bytes into the
new child run (no cross-run ArtifactRefs), and retains original run/hash lineage.
Re-used observations are never counted as fresh Sanitizer acquisitions and cannot
replace the strict independent verifier.
"""

import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field

from gpu_agent.agent.models import PublicEvidence, PublicFinding, PublicRepairContext
from gpu_agent.contracts import ArtifactRef
from gpu_agent.evidence.repository import _evidence
from gpu_agent.execution.models import CheckOutcome, ExecutionModel, SanitizerTool, SourceLocation
from gpu_agent.store import RunStore


class ReusedObservation(ExecutionModel):
    tool: SanitizerTool
    outcome: Literal["CLEAN", "FINDING"]
    citation_ref: ArtifactRef
    origin_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    origin_log_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    findings: list[PublicFinding] = Field(default_factory=list)


class ReusedCandidateEvidence(ExecutionModel):
    version: Literal[1] = 1
    source_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    observations: list[ReusedObservation]


def _one(store: RunStore, run_id: str, name: str) -> ArtifactRef:
    found = [ref for ref in store.load(run_id).artifact_refs if ref.name == name]
    if len(found) != 1:
        raise ValueError("reused public check source/input reference is missing or ambiguous")
    return found[0]


def transfer_self_check_evidence(
    store: RunStore,
    target_run_id: str,
    self_check_run_id: str,
    candidate_source_sha256: str,
    stdin: bytes,
) -> None:
    """Attest + re-home Sanitizer logs from this candidate's completed public check."""
    if store.visibility != "public":
        raise ValueError("reused evidence must come from the public store")
    origin = store.load(self_check_run_id)
    target = store.load(target_run_id)
    if (
        origin.kind != "repair_self_check"
        or target.kind != "repair_reinvestigation"
        or origin.parent_run_id != target.parent_run_id
        or origin.status != "COMPLETED"
        or target.binding is not None
        or origin.binding is not None
    ):
        raise ValueError("reused evidence has invalid run lineage")
    source = store.read(_one(store, self_check_run_id, "sources/kernel.cu"))
    target_source = store.read(_one(store, target_run_id, "sources/kernel.cu"))
    if (
        hashlib.sha256(source).hexdigest() != candidate_source_sha256
        or source != target_source
        or store.read(_one(store, self_check_run_id, "public-input.json")) != stdin
        or store.read(_one(store, target_run_id, "public-input.json")) != stdin
    ):
        raise ValueError("reused evidence has inconsistent candidate source or input")

    summary = json.loads(store.read(_one(store, self_check_run_id, "self-check.json")))
    if summary.get("status") != "FAILED":
        return
    origin_bundle = _evidence(store).view(self_check_run_id)
    target_bundle = _evidence(store).view(target_run_id)
    origin_env, target_env = origin_bundle.environment, target_bundle.environment
    # Require the same attestable execution environment before trusting prior observations.
    for key in ("toolchain_lock_hash", "image_id", "target_arch"):
        if not origin_env.get(key) or origin_env.get(key) != target_env.get(key):
            return
    if not origin_bundle.sanitizer_results:
        return

    observations: list[ReusedObservation] = []
    seen_tools: set[SanitizerTool] = set()
    for result in origin_bundle.sanitizer_results:
        if not result.completed or result.check_outcome not in {"CLEAN", "FINDING"}:
            continue
        tool_result = result.tool_result
        if tool_result is None:
            continue
        tool = SanitizerTool(tool_result.typed_payload.tool)
        if tool in seen_tools or summary.get("checks", {}).get(tool.value) != result.check_outcome:
            continue
        seen_tools.add(tool)
        original_ref = next(
            (finding.raw_ref for finding in result.findings if finding.raw_ref is not None),
            tool_result.stderr_artifact,
        )
        copied = store.put(
            target_run_id,
            f"repair/reused-self-check/{tool.value}.log",
            store.read(original_ref),
            "public",
        )
        findings = []
        for finding in result.findings:
            location = finding.source_location
            if location is not None and PurePosixPath(location.path).name != "kernel.cu":
                continue
            findings.append(
                PublicFinding(
                    artifact_id=copied.id,
                    category=finding.category,
                    source_location=(
                        SourceLocation(path="kernel.cu", line=location.line)
                        if location is not None
                        else None
                    ),
                )
            )
        observations.append(
            ReusedObservation(
                tool=tool,
                outcome=result.check_outcome,
                citation_ref=copied,
                origin_run_id=self_check_run_id,
                origin_log_sha256=original_ref.sha256,
                findings=findings,
            )
        )
    if observations:
        record = ReusedCandidateEvidence(
            source_run_id=self_check_run_id,
            source_sha256=candidate_source_sha256,
            input_sha256=hashlib.sha256(stdin).hexdigest(),
            observations=observations,
        )
        store.put(
            target_run_id,
            "repair/reused-evidence.json",
            record.model_dump_json().encode(),
            "public",
        )


def candidate_reused_evidence(
    store: RunStore,
    run_id: str,
    evidence: PublicEvidence,
    context: PublicRepairContext,
) -> PublicEvidence:
    """Project only locally owned, verified re-used evidence into the planner view."""
    found = [
        ref for ref in store.load(run_id).artifact_refs
        if ref.name == "repair/reused-evidence.json"
    ]
    if not found:
        return evidence
    if len(found) != 1 or store.visibility != "public":
        raise ValueError("ambiguous reused public observations")
    record = ReusedCandidateEvidence.model_validate_json(store.read(found[0]))
    stdin = store.read(_one(store, run_id, "public-input.json"))
    if (
        record.source_sha256 != context.candidate_source_sha256
        or record.input_sha256 != hashlib.sha256(stdin).hexdigest()
        or record.source_run_id not in {
            r.id for r in store.children(store.load(run_id).parent_run_id or "0" * 32)
            if r.kind == "repair_self_check"
        }
    ):
        raise ValueError("reused observation provenance mismatch")
    outcomes = dict(evidence.sanitizer_outcomes)
    findings = list(evidence.tool_findings)
    for observation in record.observations:
        if observation.citation_ref.run_id != run_id:
            raise ValueError("reused citation crosses run")
        raw = store.read(observation.citation_ref)
        if hashlib.sha256(raw).hexdigest() != observation.origin_log_sha256:
            raise ValueError("reused citation content hash mismatch")
        original_run = store.load(observation.origin_run_id)
        if original_run.kind != "repair_self_check" or original_run.parent_run_id != store.load(run_id).parent_run_id:
            raise ValueError("reused observation not from sibling public check")
        if observation.tool in outcomes and outcomes[observation.tool] != observation.outcome:
            raise ValueError("reused observation conflicts with fresh current evidence")
        outcomes[observation.tool] = observation.outcome
        findings.extend(observation.findings)
    return evidence.model_copy(
        update={
            "sanitizer_outcomes": outcomes,
            "tool_findings": list({(x.artifact_id, x.category): x for x in findings}.values()),
            "limitations": [*evidence.limitations, "REUSED_ATTESTED_PUBLIC_SELF_CHECK"],
        }
    )
