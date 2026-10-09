"""Evidence reuse must remain public, source/input-scoped and independently auditable."""

import hashlib
import json
from datetime import UTC, datetime

import pytest

from gpu_agent.agent.models import DiagnosisResult, PublicRepairContext
from gpu_agent.agent.orchestrator import public_evidence
from gpu_agent.contracts import ToolResult
from gpu_agent.evidence.models import EvidenceBundle
from gpu_agent.evidence.repository import _evidence
from gpu_agent.execution.models import (
    Finding,
    SanitizerPayload,
    SanitizerResult,
    SanitizerTool,
    SourceLocation,
)
from gpu_agent.repair_evidence import transfer_self_check_evidence


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tool_result(store, run_id, tool, findings):
    raw = store.put(run_id, f"raw/{tool.value}.log", (tool.value + "-native").encode(), "public")
    stdout = store.put(run_id, f"stdout/{tool.value}", b"", "public")
    payload = SanitizerPayload(
        tool=tool.value,
        completed=True,
        check_outcome="FINDING" if findings else "CLEAN",
        findings=findings,
    )
    clock = datetime.now(UTC)
    result = ToolResult[SanitizerPayload](
        tool_name="compute-sanitizer",
        request_id="a" * 32,
        started_at=clock,
        finished_at=clock,
        elapsed_ms=1,
        exit_code=0,
        timed_out=False,
        stdout_artifact=stdout,
        stderr_artifact=raw,
        typed_payload=payload,
    )
    return SanitizerResult(
        tool_result=result,
        status="SUCCESS",
        findings=findings,
        completed=True,
        check_outcome=payload.check_outcome,
    )


def _runs(store):
    kernel = b"__global__ void kernel() { /* checked */ }\n"
    stdin = b'{"n":1,"a":[1],"b":[2]}'
    parent = store.create_run("diagnosis")
    check = store.create_run("repair_self_check", parent_run_id=parent.id)
    child = store.create_run("repair_reinvestigation", parent_run_id=parent.id)
    store.transition(check.id, "RUNNING", "PREPARING")
    store.put(check.id, "sources/kernel.cu", kernel, "public")
    store.put(check.id, "public-input.json", stdin, "public")
    raw = store.put(check.id, "raw/race-finding.log", b"race-check finding", "public")
    finding = Finding(
        tool=SanitizerTool.RACECHECK,
        category="Race reported between Write access and Write access",
        source_location=SourceLocation(path="kernel.cu", line=1),
        raw_ref=raw,
    )
    results = [
        _tool_result(store, check.id, SanitizerTool.MEMCHECK, []),
        _tool_result(store, check.id, SanitizerTool.RACECHECK, [finding]),
    ]
    environment = {
        "toolchain_lock_hash": "b" * 64,
        "image_id": "sha256:" + "c" * 64,
        "target_arch": "sm_89",
    }
    _evidence(store).save(
        check.id,
        EvidenceBundle(
            environment=environment,
            sanitizer_results=results,
            source_snapshot=[store.put(check.id, "snapshot/kernel.cu", kernel, "public")],
        ),
    )
    store.put(
        check.id,
        "self-check.json",
        json.dumps(
            {"status": "FAILED", "checks": {"memcheck": "CLEAN", "racecheck": "FINDING"}}
        ).encode(),
        "public",
    )
    store.transition(check.id, "COMPLETED", None)
    store.transition(child.id, "RUNNING", "PREPARING")
    child_source = store.put(child.id, "snapshot/kernel.cu", kernel, "public")
    _evidence(store).save(
        child.id, EvidenceBundle(source_snapshot=[child_source], environment=environment)
    )
    store.put(child.id, "public-input.json", stdin, "public")
    context = PublicRepairContext(
        repair_round=1,
        original_source_sha256=_sha(kernel),
        candidate_source_sha256=_sha(kernel),
        previous_diagnosis_source_sha256=_sha(kernel),
        previous_diagnosis=DiagnosisResult.inconclusive("PREVIOUS"),
        public_checks={"racecheck": "FINDING"},
        public_feedback=[],
    )
    store.put(child.id, "repair/context.json", context.model_dump_json().encode(), "public")
    return child, check, kernel, stdin


def test_native_self_check_observation_is_locally_copied_and_citable(store):
    child, check, kernel, stdin = _runs(store)
    transfer_self_check_evidence(store, child.id, check.id, _sha(kernel), stdin)
    projected = public_evidence(store, child.id)
    assert projected.sanitizer_outcomes == {"memcheck": "CLEAN", "racecheck": "FINDING"}
    assert len(projected.tool_findings) == 1
    assert projected.tool_findings[0].source_location.line == 1
    assert "REUSED_ATTESTED_PUBLIC_SELF_CHECK" in projected.limitations
    assert any(r.name == "repair/reused-evidence.json" for r in store.load(child.id).artifact_refs)
    exported = projected.tool_findings[0].artifact_id
    copied = next(r for r in store.load(child.id).artifact_refs if r.id == exported)
    assert copied.run_id == child.id
    assert store.read(copied) == b"race-check finding"
    # Re-use never schedules a second sanitizer call or increments its acquisition ledger.
    assert not any(r.name.startswith("sanitizer/") for r in store.load(child.id).artifact_refs)


@pytest.mark.parametrize("changed", ["input", "source", "toolchain"])
def test_reuse_refuses_untrusted_source_input_and_environment(store, changed):
    child, check, kernel, stdin = _runs(store)
    if changed == "source":
        kernel = b"__global__ void kernel() { /* changed */ }\n"
    if changed == "input":
        stdin = b'{"n":2,"a":[1],"b":[2]}'
    if changed == "toolchain":
        bundle = _evidence(store).view(child.id)
        _evidence(store).save(
            child.id,
            bundle.model_copy(
                update={"environment": {**bundle.environment, "toolchain_lock_hash": "f" * 64}}
            ),
        )
    if changed in ("source", "input"):
        with pytest.raises(ValueError, match="inconsistent candidate source or input"):
            transfer_self_check_evidence(store, child.id, check.id, _sha(kernel), stdin)
    else:
        transfer_self_check_evidence(store, child.id, check.id, _sha(kernel), stdin)
        assert not any(
            ref.name == "repair/reused-evidence.json" for ref in store.load(child.id).artifact_refs
        )
