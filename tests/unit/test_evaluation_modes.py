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
    from gpu_agent.agent.orchestrator import public_evidence

    run = service.diagnose(source, mode=mode)
    payload = public_evidence(service.store, run.id).model_dump(mode="json")
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
    assert provider.kinds == []
    assert budget["sanitizer_calls"] == budget["rag_calls"] == 0
    assert service.diagnosis(run.id).limitations == ["DETERMINISTIC_NO_FINDING"]
    assert not service.candidates(run.id)


def test_mode_b_adds_frozen_retrieval_without_tools(oob_service):
    service, provider, source = oob_service
    _, evidence, budget = observation(service, provider, source, "B")
    assert evidence["documentation"] and not evidence["tool_findings"]
    assert provider.kinds == []
    assert budget["sanitizer_calls"] == 0 and budget["rag_calls"] == 1


def test_mode_c_uses_precollected_tools_without_planner(oob_service):
    service, provider, source = oob_service
    _, evidence, budget = observation(service, provider, source, "C")
    assert evidence["tool_findings"] and not evidence["documentation"]
    assert evidence["sanitizer_outcomes"] == {"memcheck": "FINDING"}
    assert provider.kinds == []
    assert budget["sanitizer_calls"] == 1 and budget["rag_calls"] == 0


def test_mode_d_uses_rule_router_and_never_planner(oob_service):
    service, provider, source = oob_service
    rule_retrieval_corpus(service)
    provider.actions = []
    run, evidence, budget = observation(service, provider, source, "D")
    assert evidence["tool_findings"] and evidence["documentation"]
    assert provider.kinds == []
    assert budget["sanitizer_calls"] == budget["rag_calls"] == 1
    assert service.diagnosis(run.id).diagnostic_outcome == "DIAGNOSED"
    assert not service.candidates(run.id)
    trace = json.loads(
        service.store.read(
            next(ref for ref in run.artifact_refs if ref.name == "agent/controller-lineage.json")
        )
    )
    assert trace["controller"] == "rule_router"
    assert trace["mode"] == "D" and trace["provider_calls_allowed"] is False
    assert trace["evidence_ref"]["sha256"]
    assert trace["acquisition_policy_ref"]["sha256"]
    assert [item["name"] for item in trace["route_decision_refs"]] == [
        "actions/0/decision.json",
        "actions/1/decision.json",
        "actions/2/decision.json",
    ]


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
    from gpu_agent.agent.prompts import PROMPT_VERSION
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.benchmark.models import CaseManifest
    from gpu_agent.contracts import RepositorySnapshot, RunBinding
    from gpu_agent.environment import load_toolchain_lock
    from gpu_agent.execution.isolated import LOCK_PATH
    from gpu_agent.store import RunStore

    service, _, source = oob_service
    toolchain_hash = load_toolchain_lock(LOCK_PATH).lock_hash
    service._binding = RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="evaluation",
        toolchain_lock_hash=toolchain_hash,
        prompt_version=PROMPT_VERSION,
        model_config_hash="5" * 64,
    )
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
        toolchain_hash=toolchain_hash,
        input_set_hash="4" * 64,
        ledger_namespace_hash="7" * 64,
        case_identity_hash="8" * 64,
        template_identity_hash="9" * 64,
        source_pair_hash="0" * 64,
    )
    registration = corpus.create_run(
        "benchmark_case",
        binding=RunBinding(
            repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
            purpose="corpus_validation",
            toolchain_lock_hash=toolchain_hash,
            case_registry_hash="6" * 64,
            corpus_ledger_namespace_hash="7" * 64,
        ),
    )
    manifest_bytes = manifest.model_dump_json().encode()
    corpus.put(
        registration.id,
        "validation/ledger-transaction.json",
        json.dumps(
            {
                "schema_version": 2,
                "transaction_id": "1" * 32,
                "owner_id": "2" * 32,
                "ledger_namespace_hash": "7" * 64,
                "case_identity_hash": "8" * 64,
                "template_identity_hash": "9" * 64,
                "source_pair_hash": "0" * 64,
                "target_store_hash": "3" * 64,
                "visibility": "evaluator",
                "expected_manifest_hash": hashlib.sha256(manifest_bytes).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        "evaluator",
    )
    corpus.put(registration.id, "case-manifest.json", manifest_bytes, "evaluator")
    corpus.transition(registration.id, "RUNNING", "FINALIZING")
    corpus.transition(registration.id, "COMPLETED", None)
    return EvaluationExecutor(service, corpus, {"case_0100": source}), source


def test_executor_uses_persisted_diagnosis_candidate_and_verification(oob_service, tmp_path):
    executor, _ = registered_executor(oob_service, tmp_path)
    record = executor.execute("case_0100", "vector-add", "E", 0)
    service, provider, _ = oob_service
    assert record.mode == "E" and record.diagnosis["diagnostic_outcome"] == "DIAGNOSED"
    assert record.patch_hash and record.verdict == "INCONCLUSIVE"
    assert record.status == "INCONCLUSIVE" and record.oracle_passed is None
    assert record.usage["physical_calls"] == 5 and record.usage["sanitizer_calls"] == 1
    # One build, ordinary execution, sanitizer invocation, and documentation retrieval.
    assert record.usage["diagnostic_tool_calls"] == 4
    assert record.cost_usd is None  # No invented pricing for unpriced physical calls.
    assert record.input_hash and record.evidence_hash
    assert record.diagnosis == service.diagnosis(record.record_id).model_dump(mode="json")
    assert provider.kinds == ["plan", "plan", "plan", "diagnose", "patch"]
    wire = record.model_dump_json() + json.dumps(provider.inputs)
    assert "PRIVATE_TRUTH_CANARY" not in wire and str(tmp_path) not in wire


def test_executor_preserves_mode_failure(oob_service, tmp_path):
    executor, _ = registered_executor(oob_service, tmp_path)
    oob_service[1].actions = []
    record = executor.execute("case_0100", "vector-add", "E", 0)
    assert record.mode == "E" and record.status == "FAILED"
    assert record.failure_reason == "FAKE_SCRIPT_EXHAUSTED"
    assert record.patch_hash is None and record.usage["physical_calls"] == 1


def test_runner_persists_only_schedule_bound_native_lineage(oob_service, tmp_path):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor, _ = registered_executor(oob_service, tmp_path)
    binding = executor.service.binding
    assert binding is not None and binding.prompt_version is not None
    assert binding.toolchain_lock_hash is not None and binding.model_config_hash is not None
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor.execute,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version,
        toolchain_hash=binding.toolchain_lock_hash,
        model_config_hash=binding.model_config_hash,
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
        random_seed=7,
    ).run("D", "development", 3)
    assert result.stopped_reason is None and result.executed_units == 3
    for record in result.records:
        run = executor.service.store.load(record.lineage.diagnosis_run_id)
        assert run.parent_run_id == result.run_id
        assert record.record_id == run.id
        assert record.lineage.provider_invocation_hashes == []


@pytest.mark.parametrize("forgery", ["record_id", "diagnosis_hash", "evidence_hash", "diagnosis"])
def test_runner_rejects_forged_native_lineage(oob_service, tmp_path, monkeypatch, forgery):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor, _ = registered_executor(oob_service, tmp_path)
    binding = executor.service.binding
    assert binding is not None and binding.prompt_version is not None
    assert binding.toolchain_lock_hash is not None and binding.model_config_hash is not None

    original = executor.execute_scheduled

    def forged(item, attempt):
        record = original(item, attempt)
        if forgery == "record_id":
            return record.model_copy(update={"record_id": "f" * 32})
        if forgery == "diagnosis_hash":
            lineage = record.lineage.model_copy(update={"diagnosis_hash": "f" * 64})
            return record.model_copy(update={"lineage": lineage})
        if forgery == "evidence_hash":
            lineage = record.lineage.model_copy(update={"evidence_hash": "f" * 64})
            return record.model_copy(update={"lineage": lineage, "evidence_hash": "f" * 64})
        return record.model_copy(update={"diagnosis": {}})

    monkeypatch.setattr(executor, "execute_scheduled", forged)
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor.execute,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version,
        toolchain_hash=binding.toolchain_lock_hash,
        model_config_hash=binding.model_config_hash,
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
        random_seed=7,
    ).run("D", "development", 3)
    assert result.stopped_reason == "EXECUTION_ERROR"
    assert result.records == []


def _configure_responses_provider(
    executor, monkeypatch, *, response_model="eval-model", usage=True
):
    from pydantic import SecretStr

    import gpu_agent.service as service_module
    from gpu_agent.agent.models import InconclusiveAction
    from gpu_agent.agent.prompts import PROMPT_VERSION
    from gpu_agent.agent.provider import (
        OpenAIProviderSettings,
        OpenAIResponsesProvider,
        ResponseMetadata,
        SDKResult,
        Usage,
    )

    class Port:
        def call(self, request):
            return SDKResult(
                value={"action": InconclusiveAction().model_dump(mode="json")},
                metadata=ResponseMetadata(
                    response_id="response-1",
                    provider_request_id="request-1",
                    response_model=response_model,
                    usage=(
                        Usage(input_tokens=3, output_tokens=2, total_tokens=5) if usage else None
                    ),
                    http_status=200,
                ),
            )

    settings = OpenAIProviderSettings(
        endpoint="https://api.openai.com/v1",
        model="eval-model",
        api_key=SecretStr("fixture-only"),
        supports_store_false=True,
    )
    policy = {
        "schema_version": 1,
        "provider": "openai-responses",
        "configured_model": "eval-model",
        "allowed_response_models": ["eval-model"],
        "prompt_version": PROMPT_VERSION,
        "store_false_required": True,
    }
    policy_hash = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    binding = executor.service.binding.model_copy(update={"model_config_hash": policy_hash})
    executor.service._binding = binding
    executor.service._provider = None
    monkeypatch.setattr(service_module.OpenAIProviderSettings, "from_environment", lambda: settings)
    monkeypatch.setattr(
        service_module,
        "OpenAIResponsesProvider",
        lambda settings, gate, store, run_id, **kwargs: OpenAIResponsesProvider(
            settings, gate, store, run_id, port=Port(), **kwargs
        ),
    )
    return binding, policy


def test_mode_e_binds_native_provider_policy_invocation_and_usage(
    oob_service, tmp_path, monkeypatch
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor, _ = registered_executor(oob_service, tmp_path)
    binding, policy = _configure_responses_provider(executor, monkeypatch)
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor.execute,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version,
        toolchain_hash=binding.toolchain_lock_hash,
        model_config_hash=binding.model_config_hash,
        binding=binding,
        max_cost_usd=1,
        max_unit_cost_usd=1,
        random_seed=7,
    ).run("E", "development", 3)
    assert result.stopped_reason == "COST_UNKNOWN" and result.executed_units == 1
    record = result.records[0]
    assert record.usage["physical_calls"] == 1
    assert len(record.lineage.provider_invocation_hashes) == 1
    run = executor.service.store.load(record.lineage.diagnosis_run_id)
    policy_ref = next(ref for ref in run.artifact_refs if ref.name == "agent/provider-policy.json")
    assert json.loads(executor.service.store.read(policy_ref)) == policy


def test_mode_e_rejects_forged_model_config_binding(oob_service, tmp_path, monkeypatch):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor, _ = registered_executor(oob_service, tmp_path)
    binding, _ = _configure_responses_provider(executor, monkeypatch)
    forged = binding.model_copy(update={"model_config_hash": "f" * 64})
    executor.service._binding = forged
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor.execute,
        commit=forged.repository.commit,
        prompt_version=forged.prompt_version,
        toolchain_hash=forged.toolchain_lock_hash,
        model_config_hash=forged.model_config_hash,
        binding=forged,
        max_cost_usd=1,
        max_unit_cost_usd=1,
        random_seed=7,
    ).run("E", "development", 3)
    assert result.stopped_reason == "EXECUTION_ERROR"
    assert result.records == []


@pytest.mark.parametrize(
    "response_model,usage", [("substituted-model", True), ("eval-model", False)]
)
def test_mode_e_rejects_unattested_response_policy(
    oob_service, tmp_path, monkeypatch, response_model, usage
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor, _ = registered_executor(oob_service, tmp_path)
    binding, _ = _configure_responses_provider(
        executor, monkeypatch, response_model=response_model, usage=usage
    )
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor.execute,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=1,
        max_unit_cost_usd=1,
        random_seed=7,
    ).run("E", "development", 3)
    assert result.stopped_reason == "EXECUTION_ERROR"
    assert result.records == []


def test_holdout_alias_and_score_binding_never_publish_private_identity(oob_service, tmp_path):
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.benchmark.holdout import HoldoutController
    from gpu_agent.benchmark.metrics import EvaluationLabels, Score
    from gpu_agent.store import RunStore

    executor, _ = registered_executor(oob_service, tmp_path)
    binding = executor.service.binding
    assert binding is not None
    evaluator = RunStore(tmp_path / "private-evaluator", visibility="evaluator")
    controller = HoldoutController(executor.service.store, evaluator, binding=binding)
    batch = controller.prepare([("case_0100", "vector-add")])
    alias = batch.aliases[0]
    holdout_executor = EvaluationExecutor(
        executor.service,
        executor.corpus,
        executor.sources,
        holdout_controller=controller,
        holdout_batch=batch,
    )
    manifest = EvaluationRunner(
        executor.service.store,
        {alias: alias},
        holdout_executor.execute,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=3,
        max_unit_cost_usd=1,
        random_seed=7,
    ).run("D", "holdout", 3)
    evaluation_run = executor.service.store.load(manifest.run_id)
    record_ref = next(
        ref for ref in evaluation_run.artifact_refs if ref.name == "evaluation/records/0.json"
    )
    assert alias not in {"case_0100", "vector-add"}
    private_canary = "PRIVATE-LABEL-CANARY"
    result = controller.bind_score(
        batch,
        alias,
        record_ref,
        labels=EvaluationLabels(claim_support={private_canary: True}),
        score=Score(
            family_correct=True,
            root_cause_correct=True,
            location_correct=True,
            inconclusive_correct=False,
        ),
    )
    assert result.public_record_id == manifest.records[0].record_id
    assert result.private_case_id == "case_0100"
    public_bytes = b"".join(
        path.read_bytes() for path in executor.service.store.root.rglob("*") if path.is_file()
    )
    assert b"case_0100" not in public_bytes
    assert b'"vector-add"' not in public_bytes
    assert private_canary.encode() not in public_bytes
    private_bytes = b"".join(
        path.read_bytes() for path in evaluator.root.rglob("*") if path.is_file()
    )
    assert b"case_0100" in private_bytes
    assert b"vector-add" in private_bytes
    assert private_canary.encode() in private_bytes
    with pytest.raises(ValueError, match="already scored"):
        controller.bind_score(
            batch,
            alias,
            record_ref,
            labels=EvaluationLabels(claim_support={private_canary: True}),
            score=Score(
                family_correct=True,
                root_cause_correct=True,
                location_correct=True,
                inconclusive_correct=False,
            ),
        )


def test_deterministic_mode_rejects_unexpected_verification_child(
    oob_service, tmp_path, monkeypatch
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor, _ = registered_executor(oob_service, tmp_path)
    binding = executor.service.binding
    assert binding is not None
    original = executor.execute_scheduled

    def injected(item, attempt):
        record = original(item, attempt)
        child = executor.service.store.create_run("verification", record.record_id)
        executor.service.store.transition(child.id, "RUNNING", "FINALIZING")
        executor.service.store.transition(child.id, "COMPLETED", None)
        return record

    monkeypatch.setattr(executor, "execute_scheduled", injected)
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor.execute,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
        random_seed=7,
    ).run("D", "development", 3)
    assert result.stopped_reason == "EXECUTION_ERROR" and result.records == []


def test_holdout_rejects_missing_private_labels_and_forged_public_record(oob_service, tmp_path):
    from gpu_agent.benchmark.holdout import HoldoutController
    from gpu_agent.benchmark.metrics import EvaluationLabels, Score
    from gpu_agent.contracts import ArtifactRef
    from gpu_agent.store import RunStore

    executor, _ = registered_executor(oob_service, tmp_path)
    binding = executor.service.binding
    assert binding is not None
    evaluator = RunStore(tmp_path / "private-evaluator", visibility="evaluator")
    controller = HoldoutController(executor.service.store, evaluator, binding=binding)
    batch = controller.prepare([("private-case", "private-template")])
    forged = ArtifactRef(
        id="a" * 32,
        run_id="b" * 32,
        name="evaluation/records/0.json",
        sha256="c" * 64,
        visibility="public",
        relative_path="b" * 32 + "/artifacts/" + "a" * 32,
        byte_count=1,
    )
    private_score = Score(
        family_correct=None,
        root_cause_correct=None,
        location_correct=None,
        inconclusive_correct=True,
    )
    with pytest.raises(ValueError, match="private evaluation labels"):
        controller.bind_score(batch, batch.aliases[0], forged, labels=None, score=private_score)
    with pytest.raises(ValueError, match="public evaluation record"):
        controller.bind_score(
            batch,
            batch.aliases[0],
            forged,
            labels=EvaluationLabels(),
            score=private_score,
        )


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
    "verdict, status, success, build_outcome, compiled",
    [
        ("VERIFIED_FIXED", "COMPLETED", 1, "CLEAN", True),
        ("NOT_FIXED", "COMPLETED", 0, "FAILED", False),
        ("REGRESSION_DETECTED", "COMPLETED", 0, "CLEAN", True),
        ("INCONCLUSIVE", "INCONCLUSIVE", 0, "TOOL_ERROR", None),
        ("INCONCLUSIVE", "INCONCLUSIVE", 0, "NOT_RUN", None),
    ],
)
def test_executor_verdict_aggregation(
    oob_service, tmp_path, monkeypatch, verdict, status, success, build_outcome, compiled
):
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
            required_checks={"memcheck": "CLEAN", "build": build_outcome},
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
    record = executor.execute("case_0100", "vector-add", "E", 0)
    assert record.status == status and record.verdict == verdict
    assert record.patch_compile_passed is compiled
    assert record.private_holdout_passed == (verdict == "VERIFIED_FIXED")
    assert "private_holdout_passed" not in record.public().model_dump()
    assert record.usage["tool_calls"] is None
    assert record.usage["total_sanitizer_calls"] is None
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
