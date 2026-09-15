"""Acquisition modes use the real service, persistence and isolated backend plumbing."""

import hashlib
import json

import pytest


def rule_retrieval_corpus(service):
    from gpu_agent.knowledge.models import make_chunk
    from gpu_agent.knowledge.retrieve import KnowledgeIndex

    chunk = service.knowledge.chunks[0]
    fields = chunk.model_dump(exclude={"chunk_id", "content_hash", "text"})
    service.knowledge = KnowledgeIndex(
        [make_chunk(**fields, text=chunk.text + " Invalid __global__ write is a memcheck finding.")]
    )


def observation(service, provider, source, mode):
    run = service.diagnose(source, mode=mode)
    payload = next(
        payload["evidence"]
        for kind, payload in zip(provider.kinds, provider.inputs, strict=True)
        if kind == "diagnose"
    )
    budget = json.loads(
        service.store.read(
            next(ref for ref in run.artifact_refs if ref.name == "agent/final-budget.json")
        )
    )
    assert budget["llm_calls"] == len(provider.kinds)
    assert (
        json.loads(
            service.store.read(
                next(
                    ref for ref in run.artifact_refs if ref.name == "agent/acquisition-policy.json"
                )
            )
        )["mode"]
        == mode
    )
    return run, payload, budget


def test_mode_a_exposes_only_source_and_runtime(oob_service):
    service, provider, source = oob_service
    run, evidence, budget = observation(service, provider, source, "A")
    assert evidence["sources"] and evidence["observed_facts"]
    assert not evidence["tool_findings"] and not evidence["documentation"]
    assert provider.kinds == ["diagnose"]
    assert budget["sanitizer_calls"] == budget["rag_calls"] == 0
    assert service.diagnosis(run.id).limitations == ["INVALID_DIAGNOSIS_EVIDENCE"]
    assert not service.candidates(run.id)


def test_mode_b_adds_frozen_retrieval_without_tools(oob_service):
    service, provider, source = oob_service
    _, evidence, budget = observation(service, provider, source, "B")
    assert evidence["documentation"] and not evidence["tool_findings"]
    assert provider.kinds == ["diagnose"]
    assert budget["sanitizer_calls"] == 0 and budget["rag_calls"] == 1


def test_mode_c_uses_precollected_tools_without_planner(oob_service):
    service, provider, source = oob_service
    _, evidence, budget = observation(service, provider, source, "C")
    assert evidence["tool_findings"] and not evidence["documentation"]
    assert evidence["sanitizer_outcomes"] == {"memcheck": "FINDING"}
    assert provider.kinds == ["diagnose"]
    assert budget["sanitizer_calls"] == 1 and budget["rag_calls"] == 0


def test_mode_d_uses_rule_router_and_never_planner(oob_service):
    service, provider, source = oob_service
    rule_retrieval_corpus(service)
    provider.actions = []
    run, evidence, budget = observation(service, provider, source, "D")
    assert evidence["tool_findings"] and evidence["documentation"]
    assert provider.kinds == ["diagnose", "patch"]
    assert budget["sanitizer_calls"] == budget["rag_calls"] == 1
    assert service.diagnosis(run.id).diagnostic_outcome == "DIAGNOSED"
    assert len(service.candidates(run.id)) == 1


def test_mode_e_uses_planner_and_never_rule_substitution(oob_service):
    service, provider, source = oob_service
    provider.actions = []
    run = service.diagnose(source, mode="E")
    assert provider.kinds == ["plan"]
    assert service.diagnosis(run.id).limitations == ["FAKE_SCRIPT_EXHAUSTED"]
    assert not service.candidates(run.id)


def test_mode_b_missing_frozen_corpus_preserves_failure(oob_service):
    service, provider, source = oob_service
    service.knowledge = None
    run = service.diagnose(source, mode="B")
    assert service.diagnosis(run.id).limitations == ["KNOWLEDGE_UNAVAILABLE"]
    assert provider.kinds == []


def test_invalid_mode_rejected_before_provider_or_run(oob_service):
    service, provider, source = oob_service
    with pytest.raises(ValueError):
        service.diagnose(source, mode="fallback")
    assert provider.kinds == []
    assert list(service.store.root.iterdir()) == []


def registered_executor(oob_service, tmp_path):
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.benchmark.models import CaseManifest
    from gpu_agent.store import RunStore

    service, _, source = oob_service
    rule_retrieval_corpus(service)
    corpus = RunStore(tmp_path / "corpus", visibility="evaluator")
    manifest = CaseManifest(
        id="case_0100",
        source_hash=hashlib.sha256((source / "kernel.cu").read_bytes()).hexdigest(),
        harness_hash="2" * 64,
        mutation_id="delete-guard",
        template_id="vector-add",
        split="private",
        oracle_id="vector-add-cpu-v1",
        target_tool="memcheck",
        expected_finding="PRIVATE_TRUTH_CANARY",
        validation_run_ids=["a" * 32, "b" * 32],
        toolchain_hash="3" * 64,
        input_set_hash="4" * 64,
    )
    registration = corpus.create_run("benchmark_case")
    corpus.put(
        registration.id, "case-manifest.json", manifest.model_dump_json().encode(), "evaluator"
    )
    corpus.transition(registration.id, "RUNNING", "FINALIZING")
    corpus.transition(registration.id, "COMPLETED", None)
    return EvaluationExecutor(service, corpus, {"case_0100": source}), source


def test_executor_uses_persisted_diagnosis_candidate_and_verification(oob_service, tmp_path):
    executor, _ = registered_executor(oob_service, tmp_path)
    record = executor.execute("case_0100", "vector-add", "D", 0)
    service, provider, _ = oob_service
    assert record.mode == "D" and record.diagnosis["diagnostic_outcome"] == "DIAGNOSED"
    assert record.patch_hash and record.verdict == "INCONCLUSIVE"
    assert record.status == "INCONCLUSIVE" and record.oracle_passed is None
    assert record.usage["physical_calls"] == 2 and record.usage["sanitizer_calls"] == 1
    assert record.cost_usd is None  # No invented pricing for unpriced physical calls.
    assert record.input_hash and record.evidence_hash
    assert record.diagnosis == service.diagnosis(record.record_id).model_dump(mode="json")
    assert provider.kinds == ["diagnose", "patch"]
    wire = record.model_dump_json() + json.dumps(provider.inputs)
    assert "PRIVATE_TRUTH_CANARY" not in wire and str(tmp_path) not in wire


def test_executor_preserves_mode_failure(oob_service, tmp_path):
    executor, _ = registered_executor(oob_service, tmp_path)
    oob_service[1].actions = []
    record = executor.execute("case_0100", "vector-add", "E", 0)
    assert record.mode == "E" and record.status == "FAILED"
    assert record.failure_reason == "FAKE_SCRIPT_EXHAUSTED"
    assert record.patch_hash is None and record.usage["physical_calls"] == 1


@pytest.mark.parametrize("fault", ["case", "template", "source"])
def test_executor_refuses_unregistered_or_changed_inputs(oob_service, tmp_path, fault):
    executor, source = registered_executor(oob_service, tmp_path)
    if fault == "source":
        (source / "kernel.cu").write_text("changed source")
    with pytest.raises(ValueError):
        executor.execute(
            "case_9999" if fault == "case" else "case_0100",
            "wrong" if fault == "template" else "vector-add",
            "A",
            0,
        )
    assert oob_service[1].kinds == []


@pytest.mark.parametrize(
    "verdict, status, success",
    [
        ("VERIFIED_FIXED", "COMPLETED", 1),
        ("NOT_FIXED", "COMPLETED", 0),
        ("REGRESSION_DETECTED", "COMPLETED", 0),
        ("INCONCLUSIVE", "INCONCLUSIVE", 0),
    ],
)
def test_executor_verdict_aggregation(oob_service, tmp_path, monkeypatch, verdict, status, success):
    from gpu_agent.benchmark.metrics import aggregate
    from gpu_agent.patching import PatchCandidate
    from gpu_agent.verification.models import VerificationResult

    executor, _ = registered_executor(oob_service, tmp_path)
    service = oob_service[0]

    def persist_verification(run_id, candidate_id):
        candidate_run = service.store.load(candidate_id)
        candidate_ref = next(
            ref for ref in candidate_run.artifact_refs if ref.name == "candidate.json"
        )
        candidate = PatchCandidate.model_validate_json(service.store.read(candidate_ref))
        result = VerificationResult(
            verdict=verdict,
            failure_stage=None,
            reason_code="TEST_VERDICT",
            original_finding_present=verdict == "NOT_FIXED",
            public_oracle_passed=verdict == "VERIFIED_FIXED",
            private_holdout_passed=verdict == "VERIFIED_FIXED",
            required_checks={"memcheck": "CLEAN"},
            candidate_hash=candidate.patched_source_hash,
            suite_hash="a" * 64,
        )
        run = service.store.create_run("verification", run_id)
        service.store.put(
            run.id, "verification/result.json", result.model_dump_json().encode(), "public"
        )
        service.store.transition(run.id, "RUNNING", "FINALIZING")
        service.store.transition(run.id, "COMPLETED", None)
        return result

    monkeypatch.setattr(service, "verify", persist_verification)
    record = executor.execute("case_0100", "vector-add", "D", 0)
    assert record.status == status and record.verdict == verdict
    summary = aggregate([record])
    assert summary.end_to_end_success.numerator == success
    assert summary.failure_ids == ([] if success else [record.record_id])


def test_missing_knowledge_has_zero_physical_retrievals(oob_service, tmp_path):
    executor, _ = registered_executor(oob_service, tmp_path)
    oob_service[0].knowledge = None
    record = executor.execute("case_0100", "vector-add", "B", 0)
    assert record.usage["retrieval_calls"] == 0
    assert record.usage["retrieval_attempts"] == 1


def test_timeout_before_backend_has_zero_physical_sanitizer_calls(
    oob_service, tmp_path, monkeypatch
):
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import ProviderError

    executor, _ = registered_executor(oob_service, tmp_path)
    original = LLMCallGate.timeout
    timeouts = []

    def reject_sanitizer(self, limit):
        timeouts.append(limit)
        if len(timeouts) == 3:  # Build and ordinary runtime finish before sanitizer request.
            raise ProviderError("AGENT_BUDGET_EXHAUSTED")
        return original(self, limit)

    monkeypatch.setattr(LLMCallGate, "timeout", reject_sanitizer)
    record = executor.execute("case_0100", "vector-add", "C", 0)
    assert record.failure_reason == "AGENT_BUDGET_EXHAUSTED"
    assert record.usage["sanitizer_calls"] == 0
    assert record.usage["sanitizer_attempts"] == 1
    assert record.executed_checks == {}


def test_successful_acquisition_persists_physical_calls(oob_service, tmp_path):
    executor, _ = registered_executor(oob_service, tmp_path)
    record = executor.execute("case_0100", "vector-add", "D", 0)
    assert record.usage["sanitizer_calls"] == record.usage["retrieval_calls"] == 1
    store = oob_service[0].store
    refs = [
        ref
        for ref in store.load(record.record_id).artifact_refs
        if ref.name == "agent/acquisition-usage.json"
    ]
    assert len(refs) == 1
    assert json.loads(store.read(refs[0])) == {
        "schema_version": 1,
        "sanitizer_calls": 1,
        "retrieval_calls": 1,
    }


@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"sanitizer_calls": 0},
        {"sanitizer_calls": 0, "retrieval_calls": True},
        {"sanitizer_calls": 5, "retrieval_calls": 0},
    ],
)
def test_executor_rejects_incomplete_or_unbounded_acquisition_usage(
    oob_service, tmp_path, monkeypatch, usage
):
    executor, _ = registered_executor(oob_service, tmp_path)
    store = oob_service[0].store
    original = store.put

    def malformed(run_id, name, content, visibility):
        if name == "agent/acquisition-usage.json":
            content = json.dumps(usage).encode()
        return original(run_id, name, content, visibility)

    monkeypatch.setattr(store, "put", malformed)
    with pytest.raises(ValueError):
        executor.execute("case_0100", "vector-add", "B", 0)
