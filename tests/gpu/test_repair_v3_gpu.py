"""Real GPU public-repair smoke; scripted decisions do not measure LLM effectiveness.

No GPU subprocess, sanitizer output, or private verifier is mocked. The test drives
the public repair coordinator directly, so it never reads a private oracle or claims
VERIFIED_FIXED. Use --require-live to make missing container/GPU support a failure.
"""

import difflib
import hashlib
import json
from pathlib import Path

import pytest

from gpu_agent.agent.models import (
    DiagnosisResult,
    EvidenceClaim,
    FinishAction,
    MemcheckAction,
    RetrieveDocsAction,
)
from gpu_agent.agent.orchestrator import AgentOrchestrator, public_evidence
from gpu_agent.agent.policy import LLMCallGate, validate_diagnosis
from gpu_agent.agent.provider import FakeProvider
from gpu_agent.environment import load_toolchain_lock
from gpu_agent.evidence.repository import _evidence
from gpu_agent.execution.isolated import LOCK_PATH, IsolatedGPUBackend
from gpu_agent.execution.models import (
    BuildRequest,
    ExecutionRequest,
    SanitizerTool,
    SourceLocation,
    WorkspaceRequest,
)
from gpu_agent.knowledge.models import make_chunk
from gpu_agent.knowledge.retrieve import KnowledgeIndex
from gpu_agent.patching import SourceSnapshot, apply_generated_candidate, materialize_candidate
from gpu_agent.public_task import PublicTask, check_public_output
from gpu_agent.repair import RepairPolicy, repair_candidates
from gpu_agent.repair_coordinator import RepairCoordinator
from gpu_agent.store import RunStore

pytestmark = [pytest.mark.gpu, pytest.mark.container]


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _diff(original, changed):
    return "".join(
        difflib.unified_diff(
            original.splitlines(True),
            changed.splitlines(True),
            fromfile="a/kernel.cu",
            tofile="b/kernel.cu",
        )
    )


def _artifact(store, run_id, name):
    refs = [ref for ref in store.load(run_id).artifact_refs if ref.name == name]
    assert refs, (run_id, name)
    return json.loads(store.read(refs[-1]))


class _ScriptedRepairProvider(FakeProvider):
    """Human-authored control flow; every citation comes from current native evidence."""

    def __init__(self, original, wrong, correct):
        super().__init__([], DiagnosisResult.inconclusive("SCRIPT_NOT_RUN"), _diff(original, wrong))
        self.original, self.wrong, self.correct = original, wrong, correct

    def plan(self, evidence, budget, feedback=None, state=None):
        if SanitizerTool.MEMCHECK not in evidence.sanitizer_outcomes:
            action = MemcheckAction()
        elif evidence.tool_findings and not evidence.documentation:
            action = RetrieveDocsAction(typed_arguments={"query": "out of bounds", "k": 1})
        else:
            action = FinishAction()
        self.actions.append(action)
        return super().plan(evidence, budget, feedback, state)

    def diagnose(self, evidence):
        if evidence.repair_context is None:
            assert evidence.sources[0].content == self.original
            assert evidence.sanitizer_outcomes[SanitizerTool.MEMCHECK] == "FINDING"
            located = [f.source_location for f in evidence.tool_findings if f.source_location]
            assert located and evidence.documentation
            locations = [located[0]]
            family, cause = "out_of_bounds", "The original kernel does not guard the final block."
        else:
            assert evidence.sources[0].content == self.wrong
            assert evidence.repair_context.public_functional_failure
            assert evidence.sanitizer_outcomes[SanitizerTool.MEMCHECK] == "CLEAN"
            assert not evidence.tool_findings
            assert any("NUMERIC_MISMATCH" in fact.text for fact in evidence.observed_facts)
            line = next(
                i for i, text in enumerate(self.wrong.splitlines(), 1) if "a[i] - b[i]" in text
            )
            locations = [SourceLocation(path="kernel.cu", line=line)]
            family, cause = "other", "The guarded candidate subtracts b[i] instead of adding it."
        self.result = DiagnosisResult(
            diagnostic_outcome="DIAGNOSED",
            failure_family=family,
            root_cause=cause,
            source_locations=locations,
            observed_facts=evidence.observed_facts,
            tool_findings=[
                EvidenceClaim(text=f.category, citation_ids=[f.artifact_id])
                for f in evidence.tool_findings
            ],
            documentation_evidence=[
                EvidenceClaim(text=d.text, citation_ids=[d.chunk_id])
                for d in evidence.documentation
            ],
            recommended_change="Guard i < n and compute out[i] = a[i] + b[i].",
            confidence_label="high",
            limitations=["SCRIPTED_PROVIDER_CONTROL_FLOW_ONLY"],
        )
        return super().diagnose(evidence)

    def revise_patch(self, public_source, diagnosis, feedback):
        assert public_source.content == self.original
        assert feedback["previous_candidate_source"] == self.wrong
        assert feedback["diagnosis_source"] == self.wrong
        assert feedback["diagnosis_source_sha256"] == _sha(self.wrong.encode())
        assert feedback["original_source_sha256"] == _sha(self.original.encode())
        assert diagnosis.failure_family == "other" and not diagnosis.tool_findings
        self.diff = _diff(self.original, self.correct)
        return super().revise_patch(public_source, diagnosis, feedback)


def _native_bundle(store, run_id, sources, toolchain):
    """Check source/binary lineage using real backend artifacts, never a synthetic log."""
    bundle = _evidence(store).public_view(run_id)
    assert bundle.environment["backend"] == "IsolatedGPUBackend"
    assert bundle.environment["toolchain_lock_hash"] == toolchain.lock_hash
    assert bundle.environment["image_id"] == toolchain.image_id
    assert bundle.environment["target_arch"] == toolchain.target_arch
    assert {Path(ref.name).name: store.read(ref) for ref in bundle.source_snapshot} == sources
    build, execution = bundle.build_result, bundle.execution_result
    assert build is not None and build.success and build.binary_ref is not None
    assert store.read(build.binary_ref).startswith(b"\x7fELF")
    assert build.tool_result.typed_payload.source_manifest == {
        name: _sha(content) for name, content in sources.items()
    }
    assert execution is not None
    assert execution.tool_result.typed_payload.binary_ref == build.binary_ref
    for result in bundle.sanitizer_results:
        assert result.completed and result.tool_result is not None
        assert result.tool_result.typed_payload.binary_ref == build.binary_ref
        for finding in result.findings:
            assert finding.raw_ref is not None and store.read(finding.raw_ref)
    for ref in store.load(run_id).artifact_refs:
        assert ref.visibility == "public"
        store.read(ref)  # RunStore verifies the persisted bytes against their recorded hash.
    return bundle


def test_public_repair_v3_reinvestigates_numeric_failure_on_real_gpu(tmp_path, request):
    gate = LLMCallGate()
    repo = Path(__file__).resolve().parents[2]
    configured_root = request.config.getoption("--gpu-run-root")
    store = RunStore(Path(configured_root) if configured_root else tmp_path / "public-runs")
    source_root = tmp_path / "public-source"
    source_root.mkdir()
    backends = []

    def native_backend(public_store, sources, tasks):
        backend = IsolatedGPUBackend(public_store, sources, tasks)
        backends.append(backend)
        return backend

    backend = native_backend(store, source_root, tmp_path / "original-workspace")
    available = backend.availability()
    if not available.ready:
        pytest.skip(available.reason)
    toolchain = load_toolchain_lock(LOCK_PATH)
    # Only the public clean host wrapper and declared public harness files are read.
    correct = (repo / "benchmarks/public/case_0000/public_input/kernel.cu").read_text()
    guarded = "    if (i < n) {\n        out[i] = a[i] + b[i];\n    }"
    assert correct.count(guarded) == 1
    original = correct.replace(guarded, "    out[i] = a[i] + b[i];", 1)
    wrong = correct.replace("out[i] = a[i] + b[i];", "out[i] = a[i] - b[i];", 1)
    sources = {"kernel.cu": original.encode()}
    for name in ("vector_io.cpp", "vector_api.h", "json.hpp"):
        relative = "vendor/json.hpp" if name == "json.hpp" else name
        sources[name] = (repo / "benchmarks/harness" / relative).read_bytes()
    for name, content in sources.items():
        IsolatedGPUBackend._write_snapshot(source_root / name, content)
    # n is not divisible by the block size; both input arrays have nonzero, exact values.
    stdin = json.dumps({"n": 257, "a": [1.0] * 257, "b": [2.0] * 257}).encode()
    task = PublicTask(source_sha256=_sha(sources["kernel.cu"]), algorithm="vector-add-cpu-v1")
    sanitizer_version = toolchain.compute_sanitizer.split()[0]
    chunk = make_chunk(
        source_id="smoke-memcheck",
        document_title="Compute Sanitizer (fixed smoke-test paraphrase)",
        document_version=sanitizer_version,
        section_title="Memcheck",
        source_url="https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html",
        retrieved_at="2026-10-08T00:00:00Z",
        text="Memcheck detects out of bounds global memory reads and writes.",
        block_ordinal=0,
        compatibility={
            "cuda": f"=={toolchain.cuda_nvcc}",
            "compute_sanitizer": f"=={sanitizer_version}",
        },
    )
    provider = _ScriptedRepairProvider(original, wrong, correct)
    provider.gate = gate
    run = store.create_run("diagnosis")
    store.transition(run.id, "RUNNING", "PREPARING")
    store.put(run.id, "public-task.json", task.model_dump_json().encode(), "public")
    store.put(
        run.id,
        "agent/acquisition-policy.json",
        b'{"mode":"E","required_tools":["memcheck"]}',
        "public",
    )
    handle, passed = None, False
    smoke = {
        "scope": "real_gpu_public_repair_control_flow",
        "provider": "scripted_fake",
        "llm_effectiveness_evaluated": False,
        "private_verification": "NOT_RUN",
        "verified_fixed": False,
        "run_id": run.id,
        "artifact_root": str(store.root),
        "toolchain_lock_sha256": toolchain.lock_hash,
        "original_source_sha256": _sha(original.encode()),
        "failed_candidate_source_sha256": _sha(wrong.encode()),
        "corrected_source_sha256": _sha(correct.encode()),
    }
    try:
        hashes = {name: _sha(content) for name, content in sources.items()}
        snapshot = SourceSnapshot(parent_run_id=run.id, root=source_root, hashes=hashes)
        handle = backend.prepare(
            WorkspaceRequest(run_id=run.id, source_manifest=hashes, trust_level="UNTRUSTED")
        )
        bundle = _evidence(store).view(run.id)
        _evidence(store).save(run.id, bundle.model_copy(update={"public_task": task}))
        store.transition(run.id, "RUNNING", "COMPILING")
        built = backend.build(
            BuildRequest(workspace_id=handle.id, timeout_seconds=gate.timeout(120))
        )
        assert built.success, built.model_dump_json()
        input_ref = store.put(run.id, "public-input.json", stdin, "public")
        store.transition(run.id, "RUNNING", "EXECUTING")
        execution = backend.run(
            ExecutionRequest(
                workspace_id=handle.id, stdin_ref=input_ref, timeout_seconds=gate.timeout(30)
            )
        )
        assert execution.runtime_status in {"SUCCESS", "FAILED"}, execution.model_dump_json()
        orchestrator = AgentOrchestrator(
            store,
            provider,
            backend,
            handle,
            input_ref,
            KnowledgeIndex([chunk]),
            f"cuda={toolchain.cuda_nvcc};compute-sanitizer={sanitizer_version}",
        )
        diagnosis = orchestrator.investigate(run.id, mode="E")
        assert diagnosis.diagnostic_outcome == "DIAGNOSED", diagnosis.model_dump_json()
        store.put(run.id, "diagnosis.json", diagnosis.model_dump_json().encode(), "public")
        original_evidence = public_evidence(store, run.id)
        assert validate_diagnosis(diagnosis, original_evidence)
        store.transition(run.id, "RUNNING", "PATCH_GENERATING")
        first = apply_generated_candidate(
            snapshot, provider.propose_patch(original_evidence.sources[0], diagnosis)
        ).model_copy(update={"generated_by": "agent", "provider": provider.provider_name})
        ledger = orchestrator.ledger
        coordinator = RepairCoordinator(
            store,
            orchestrator,
            native_backend,
            task,
            original_evidence.sources[0],
            diagnosis,
            max_reinvestigations=1,
            mode="E",
        )
        selected = repair_candidates(
            store,
            snapshot,
            first,
            original_evidence.sources[0],
            diagnosis,
            stdin,
            provider,
            native_backend,
            RepairPolicy(version="public-repair-v3", max_candidates=2),
            task,
            coordinator=coordinator,
        )
        summary = _artifact(store, run.id, "repair/summary.json")
        assert summary["stop_reason"] == "PUBLIC_CHECKS_PASSED", summary
        assert summary["reinvestigations"] == 1 and len(summary["rounds"]) == 2
        first_check, final_check = (item["check"] for item in summary["rounds"])
        assert first_check["status"] == "FAILED"
        assert first_check["checks"]["functional"] == "NUMERIC_MISMATCH"
        assert summary["rounds"][0]["decision"]["action"] == "REINVESTIGATE"
        assert summary["rounds"][0]["decision"]["reason"] == "PUBLIC_FUNCTIONAL_FAILURE"
        assert final_check["status"] == "PASSED"
        assert final_check["checks"]["functional"] == "PASSED"
        assert all(final_check["checks"][tool.value] == "CLEAN" for tool in SanitizerTool)
        (child_id,) = summary["investigation_runs"]
        children = store.children(run.id)
        assert len(children) == 3 and all(child.status == "COMPLETED" for child in children)
        assert {child.id for child in children if child.kind == "repair_reinvestigation"} == {
            child_id
        }
        scoped = _artifact(store, run.id, "repair/1/reinvestigation.json")
        assert scoped["run_id"] == child_id and scoped["source_role"] == "failed_candidate"
        assert scoped["source_sha256"] == _sha(wrong.encode())
        child_diagnosis = DiagnosisResult.model_validate(scoped["diagnosis"])
        child_evidence = public_evidence(store, child_id)
        assert validate_diagnosis(child_diagnosis, child_evidence)
        assert not validate_diagnosis(child_diagnosis, original_evidence)
        assert _artifact(store, run.id, "diagnosis.json") == diagnosis.model_dump(mode="json")
        assert selected.base_source_hash == first.base_source_hash
        assert materialize_candidate(snapshot, selected) == {
            **sources,
            "kernel.cu": correct.encode(),
        }
        assert (source_root / "kernel.cu").read_text() == original
        parent_bundle = _native_bundle(store, run.id, sources, toolchain)
        assert parent_bundle.sanitizer_results[0].check_outcome == "FINDING"
        for native_id, kernel in (
            (first_check["run_id"], wrong),
            (child_id, wrong),
            (final_check["run_id"], correct),
        ):
            native = _native_bundle(
                store, native_id, {**sources, "kernel.cu": kernel.encode()}, toolchain
            )
            assert native.execution_result.runtime_status == "SUCCESS"
            expected = "PASSED" if kernel == correct else "NUMERIC_MISMATCH"
            assert (
                check_public_output(task, stdin, store.read(native.execution_result.output_ref))
                == expected
            )
            if native_id == child_id:
                assert len(native.sanitizer_results) == 1
                assert native.sanitizer_results[0].check_outcome == "CLEAN"
            elif native_id == first_check["run_id"]:
                assert native.sanitizer_results == []
            else:
                assert len(native.sanitizer_results) == 4
        assert provider.gate is gate and orchestrator.ledger is ledger
        assert provider.kinds == [
            "plan",
            "plan",
            "plan",
            "diagnose",
            "patch",
            "plan",
            "plan",
            "diagnose",
            "patch",
        ]
        assert gate.snapshot().llm_calls == 9
        assert summary["acquisition_usage"] == {
            "schema_version": 1,
            "sanitizer_calls": 2,
            "retrieval_calls": 1,
        }
        assert summary["self_check_sanitizer_calls"] == 4
        child_before = _artifact(store, child_id, "agent/initial-budget.json")
        child_after = _artifact(store, child_id, "agent/budget.json")
        assert child_before["sanitizer_calls"] == 1 and child_after["sanitizer_calls"] == 2
        assert child_after["remaining_seconds"] < child_before["remaining_seconds"]
        smoke.update(
            {
                "public_check_statuses": [first_check["status"], final_check["status"]],
                "reinvestigation_run_id": child_id,
                "selected_hash": selected.patched_source_hash,
                "scripted_provider_calls": gate.snapshot().llm_calls,
                "acquisition_usage": summary["acquisition_usage"],
                "self_check_sanitizer_calls": summary["self_check_sanitizer_calls"],
            }
        )
        passed = True
    finally:
        try:
            if handle is not None:
                backend.cleanup(handle)
            assert all(item.active_containers() == [] for item in backends)
            assert gate.remaining() > 0, "The shared 600-second task deadline was exceeded."
        except BaseException:
            passed = False
            raise
        finally:
            smoke["status"] = "PASSED" if passed else "FAILED"
            smoke["remaining_seconds"] = round(gate.remaining(), 3)
            encoded = json.dumps(smoke, sort_keys=True, separators=(",", ":"))
            store.put(run.id, "repair/gpu-smoke.json", encoded.encode(), "public")
            store.transition(run.id, "RUNNING", "FINALIZING")
            store.transition(run.id, "COMPLETED" if passed else "FAILED", None)
            print(encoded)
