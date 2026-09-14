"""Human report from allowlisted public records; no arbitrary artifact/log dumping."""

import hashlib
import json

from gpu_agent.agent.models import DiagnosisResult
from gpu_agent.agent.provider import Invocation
from gpu_agent.patching import PatchCandidate
from gpu_agent.store import RunStore
from gpu_agent.verification.models import VerificationResult


def render_report(store: RunStore, run_id: str) -> str:
    manifest = store.load(run_id)
    refs = [r for r in manifest.artifact_refs if r.name == "diagnosis.json"]
    diagnosis = (
        DiagnosisResult.model_validate_json(store.read(refs[-1]))
        if refs
        else (DiagnosisResult.inconclusive("DIAGNOSIS_NOT_COMPLETED"))
    )
    lines = [
        f"Run {run_id}",
        f"Status: {manifest.status.value}",
        f"Diagnosis: {diagnosis.diagnostic_outcome}",
        f"Root cause: {diagnosis.root_cause or 'Not established'}",
    ]
    for title, claims in [
        ("Observed facts", diagnosis.observed_facts),
        ("Tool findings", diagnosis.tool_findings),
        ("Documentation evidence", diagnosis.documentation_evidence),
    ]:
        lines.append(title + ":")
        lines.extend(f"- {c.text} [{', '.join(c.citation_ids)}]" for c in claims)
        if not claims:
            lines.append("- Not established")
    lines.append("Model inferences (not observed facts):")
    lines.extend(f"- {inference}" for inference in diagnosis.model_inferences)
    lines.append(f"Confidence label: {diagnosis.confidence_label} (not a calibrated probability)")
    candidates: list[PatchCandidate] = []
    verifications: list[VerificationResult] = []
    for path in sorted(store.root.iterdir()):
        if not path.is_dir() or len(path.name) != 32:
            continue
        child = store.load(path.name)
        if child.parent_run_id != run_id:
            continue
        for ref in child.artifact_refs:
            if child.kind == "candidate" and ref.name == "candidate.json":
                candidates.append(PatchCandidate.model_validate_json(store.read(ref)))
            if child.kind == "verification" and ref.name == "verification/result.json":
                verifications.append(VerificationResult.model_validate_json(store.read(ref)))
    lines.append(
        "Single candidate: " + (candidates[0].patched_source_hash if candidates else "none")
    )
    if candidates:
        lines.append(
            "Patch SHA256: " + hashlib.sha256(candidates[0].unified_diff.encode()).hexdigest()
        )
    lines.append(
        "VERIFIED_FIXED requires build/runtime success, original finding absent, "
        "no blocking new findings and public/private oracle checks passed."
    )
    if not verifications:
        lines.append("Verification: NOT_RUN")
    for result in verifications:
        lines.append(f"Verification: {result.verdict.value} ({result.reason_code})")
        lines.extend(f"- {name}: {status}" for name, status in result.required_checks.items())
        lines.append(f"Not run count: {result.not_run_count}")
    usage = [r for r in manifest.artifact_refs if r.name == "agent/usage-summary.json"]
    if usage:
        summary = json.loads(store.read(usage[-1]))
        lines.append(
            f"Physical LLM calls: {summary['physical_calls']}; synthetic: {summary['synthetic']}"
        )
    invocations: dict[str, Invocation] = {}
    for ref in manifest.artifact_refs:
        if ref.name.startswith("provider/"):
            record = Invocation.model_validate_json(store.read(ref))
            invocations[record.invocation_id] = record
    for record in invocations.values():
        state = "UNCERTAIN" if record.state == "STARTED" else record.state
        lines.append(
            f"LLM {record.kind}: {state}; model={record.configured_model}; "
            f"response_model={record.response_model}; response_id={record.response_id}; "
            f"request_id={record.provider_request_id}; "
            f"client_request_id={record.client_request_id}; "
            f"tokens={record.usage.total_tokens if record.usage else 'unknown'}; store=false"
        )
    limitations = [*diagnosis.limitations]
    limitations.extend(item for result in verifications for item in result.limitations)
    if not verifications:
        limitations.append("VERIFICATION_NOT_RUN")
    if any(
        i.response_model and i.response_model != i.configured_model for i in invocations.values()
    ):
        limitations.append("PROVIDER_MODEL_MISMATCH")
    lines.append(
        "Unfinished/limitations: " + (", ".join(dict.fromkeys(limitations)) or "none reported")
    )
    return "\n".join(lines)
