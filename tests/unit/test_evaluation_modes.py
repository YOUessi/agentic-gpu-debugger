"""Acquisition modes use the real service, persistence and isolated backend plumbing."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

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


@pytest.mark.parametrize(
    "case_id,target_tool",
    [
        ("case_0002", "racecheck"),
        ("case_0003", "initcheck"),
        ("case_0004", "synccheck"),
    ],
)
def test_mode_c_clean_memcheck_runs_registered_target(
    oob_service, monkeypatch, case_id, target_tool
):
    from gpu_agent.execution.models import SanitizerTool
    from gpu_agent.execution.process import ProcessCapture

    service, provider, source = oob_service
    backend_type = service._backend_factory
    original = backend_type._container

    def clean_sanitizers(self, path, operation, timeout, *, stdin=b"", cancel=None):
        if operation in {tool.value for tool in SanitizerTool}:
            return (
                ProcessCapture(0, b'{"values":[3]}', b"", False),
                b"",
                b"========= ERROR SUMMARY: 0 errors\n",
            )
        return original(self, path, operation, timeout, stdin=stdin, cancel=cancel)

    monkeypatch.setattr(backend_type, "_container", clean_sanitizers)
    run = service.diagnose(
        source,
        mode="C",
        required_tools=(SanitizerTool(target_tool),),
    )
    from gpu_agent.agent.orchestrator import public_evidence

    evidence = public_evidence(service.store, run.id)
    assert list(evidence.sanitizer_outcomes) == [
        SanitizerTool.MEMCHECK,
        SanitizerTool(target_tool),
    ], case_id
    assert provider.kinds == []


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


def _execute_claimed_test_unit(
    executor, mode, *, case_id="case_0100", template_id="vector-add", repeat=0
):
    """Exercise the production schedule/attempt claim path for one unit.

    Component tests intentionally stop after the native executor returns; batch
    persistence and terminalization are covered by EvaluationRunner tests.
    """
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.contracts import CurrentPhase, RunStatus

    binding = executor.service.binding
    assert binding is not None
    runner = EvaluationRunner(
        executor.service.store,
        {case_id: template_id},
        executor,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=1,
        max_unit_cost_usd=1,
        random_seed=7,
    )
    schedule = runner._schedule(mode, "development", 3)
    item = next(value for value in schedule.items if value.repeat == repeat)
    ordered = [item, *(value for value in schedule.items if value is not item)]
    schedule = schedule.model_copy(
        update={
            "items": [
                value.model_copy(update={"ordinal": index}) for index, value in enumerate(ordered)
            ]
        }
    )
    item = schedule.items[0]
    run = runner.store.create_run("evaluation", binding=binding)
    runner.store.transition(run.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)
    runner._put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode())
    attempt = runner._attempt(run.id, schedule, item)
    runner._put(
        run.id,
        f"evaluation/attempts/{item.ordinal}.json",
        attempt.model_dump_json().encode(),
    )
    return executor.execute_scheduled(run.id, item.ordinal)


def test_executor_rejects_verification_without_native_children(
    oob_service, native_evaluation_executor
):
    executor = native_evaluation_executor
    with pytest.raises(ValueError, match="child selection"):
        _execute_claimed_test_unit(executor, "E")
    assert oob_service[1].kinds == ["plan", "plan", "plan", "diagnose", "patch"]


def test_evaluation_rejects_self_authored_corpus_receipt(oob_service, tmp_path):
    with pytest.raises(ValueError, match="trusted corpus family|committed corpus transaction"):
        registered_executor(oob_service, tmp_path)


def test_evaluation_rejects_prepared_corpus_transaction(native_evaluation_executor):
    executor = native_evaluation_executor
    state_path = executor._corpus_family.ledger.root / "transactions.json"
    state = json.loads(state_path.read_text())
    state["transactions"][0]["state"] = "PREPARED"
    state_path.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="committed corpus transaction"):
        _execute_claimed_test_unit(executor, "D")


def test_evaluation_rejects_corpus_from_another_repository(native_evaluation_executor):
    from gpu_agent.contracts import RepositorySnapshot

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None
    executor.service._binding = binding.model_copy(
        update={
            "repository": RepositorySnapshot(
                commit="f" * 40, tracked_tree_hash="e" * 64, clean=True
            )
        }
    )
    with pytest.raises(ValueError, match="provenance"):
        _execute_claimed_test_unit(executor, "D")


def test_executor_preserves_mode_failure(oob_service, tmp_path, native_evaluation_executor):
    executor = native_evaluation_executor
    oob_service[1].actions = []
    record = _execute_claimed_test_unit(executor, "E")
    assert record.mode == "E" and record.status == "FAILED"
    assert record.failure_reason == "FAKE_SCRIPT_EXHAUSTED"
    assert record.patch_hash is None and record.usage["physical_calls"] == 1


def test_runner_persists_only_schedule_bound_native_lineage(
    oob_service, tmp_path, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None and binding.prompt_version is not None
    assert binding.toolchain_lock_hash is not None and binding.model_config_hash is not None
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor,
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


@pytest.mark.parametrize("mode", ["A", "B", "C", "D"])
def test_deterministic_modes_reject_extra_controller_actions(
    mode, monkeypatch, native_evaluation_executor
):
    from gpu_agent.agent.models import PolicyDecision
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None
    store = executor.service.store
    original = store.put
    injected = False

    def add_extra(run_id, name, content, visibility):
        nonlocal injected
        if name == "agent/controller-lineage.json" and not injected:
            injected = True
            original(
                run_id,
                "actions/99/decision.json",
                PolicyDecision(
                    action_type="retrieve_official_docs",
                    action_hash="f" * 64,
                    allowed=True,
                )
                .model_dump_json()
                .encode(),
                visibility,
            )
        return original(run_id, name, content, visibility)

    monkeypatch.setattr(store, "put", add_extra)
    result = EvaluationRunner(
        store,
        {"case_0100": "vector-add"},
        executor,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=1,
        max_unit_cost_usd=1,
        random_seed=7,
    ).run(mode, "development", 3)
    assert result.stopped_reason == "EXECUTION_ERROR" and result.records == []


@pytest.mark.parametrize(
    "forgery",
    [
        "record_id",
        "diagnosis_hash",
        "evidence_hash",
        "diagnosis",
        "status",
        "failure_reason",
        "latency",
        "cost",
        "usage",
        "checks",
        "oracle",
        "regression",
    ],
)
def test_runner_rejects_forged_native_lineage(
    oob_service, tmp_path, monkeypatch, forgery, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None and binding.prompt_version is not None
    assert binding.toolchain_lock_hash is not None and binding.model_config_hash is not None

    original = executor.execute_scheduled

    def forged(run_id, ordinal):
        record = original(run_id, ordinal)
        if forgery == "record_id":
            return record.model_copy(update={"record_id": "f" * 32})
        if forgery == "diagnosis_hash":
            lineage = record.lineage.model_copy(update={"diagnosis_hash": "f" * 64})
            return record.model_copy(update={"lineage": lineage})
        if forgery == "evidence_hash":
            lineage = record.lineage.model_copy(update={"evidence_hash": "f" * 64})
            return record.model_copy(update={"lineage": lineage, "evidence_hash": "f" * 64})
        if forgery == "diagnosis":
            return record.model_copy(update={"diagnosis": {}})
        updates = {
            "status": {"status": "COMPLETED"},
            "failure_reason": {"failure_reason": "FORGED_FAILURE"},
            "latency": {"latency_ms": record.latency_ms + 1},
            "cost": {"cost_usd": 0.5},
            "usage": {"usage": {**record.usage, "tool_calls": 999}},
            "checks": {"executed_checks": {"memcheck": "FORGED"}},
            "oracle": {"oracle_passed": True},
            "regression": {"regression_detected": True},
        }
        return record.model_copy(update=updates[forgery])

    monkeypatch.setattr(executor, "execute_scheduled", forged)
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor,
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
    executor,
    monkeypatch,
    *,
    response_model="eval-model",
    usage=True,
    mock_provider=True,
    full_script=False,
):
    from pydantic import SecretStr

    import gpu_agent.service as service_module
    from gpu_agent.agent.models import DiagnosisResult, EvidenceClaim, InconclusiveAction
    from gpu_agent.agent.prompts import PROMPT_VERSION
    from gpu_agent.agent.provider import (
        MockResponsesProvider,
        OpenAIProviderSettings,
        OpenAIResponsesProvider,
        ResponseMetadata,
        SDKResult,
        Usage,
    )
    from gpu_agent.benchmark.evaluation import PricingAttestation
    from gpu_agent.execution.models import SourceLocation

    scripted = executor.service._provider

    class Port:
        def __init__(self):
            self.plan_index = 0

        def call(self, request):
            if full_script and request.kind == "plan":
                value = {"action": scripted.actions[self.plan_index].model_dump(mode="json")}
                self.plan_index += 1
            elif full_script and request.kind == "diagnose":
                evidence = request.payload["evidence"]
                value = DiagnosisResult(
                    diagnostic_outcome="DIAGNOSED",
                    failure_family="out_of_bounds",
                    root_cause="The thread index can exceed the input length.",
                    source_locations=[SourceLocation(path="kernel.cu", line=9)],
                    observed_facts=[
                        EvidenceClaim.model_validate(item) for item in evidence["observed_facts"]
                    ],
                    tool_findings=[
                        EvidenceClaim(text=item["category"], citation_ids=[item["artifact_id"]])
                        for item in evidence["tool_findings"]
                    ],
                    documentation_evidence=[
                        EvidenceClaim(text=item["text"], citation_ids=[item["chunk_id"]])
                        for item in evidence["documentation"]
                    ],
                    model_inferences=["An index guard may prevent the reported write."],
                    recommended_change="Guard the write with i < n.",
                    confidence_label="high",
                ).model_dump(mode="json")
            elif full_script and request.kind == "patch":
                value = {"unified_diff": scripted.diff}
            else:
                value = {"action": InconclusiveAction().model_dump(mode="json")}
            return SDKResult(
                value=value,
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
    provider_name = "mock-responses" if mock_provider else "openai-responses"
    rate_card = PricingAttestation._for_test(
        provider_name,
        "eval-model",
        executor.service.binding.repository.commit,
        "0" * 64,
    )
    policy = {
        "schema_version": 1,
        "provider": provider_name,
        "endpoint_host": "api.openai.com",
        "configured_model": "eval-model",
        "allowed_response_models": ["eval-model"],
        "prompt_version": PROMPT_VERSION,
        "pricing_hash": rate_card.rate_card_hash,
        "store_false_required": True,
    }
    policy_hash = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    binding = executor.service.binding.model_copy(update={"model_config_hash": policy_hash})
    executor.service._binding = binding
    executor.service._pricing_attestation = PricingAttestation._for_test(
        provider_name, "eval-model", binding.repository.commit, policy_hash
    )
    executor.service._provider = None
    monkeypatch.setattr(service_module.OpenAIProviderSettings, "from_environment", lambda: settings)
    monkeypatch.setattr(
        service_module,
        "OpenAIResponsesProvider",
        lambda settings, gate, store, run_id, **kwargs: (
            MockResponsesProvider if mock_provider else OpenAIResponsesProvider
        )(settings, gate, store, run_id, port=Port(), **kwargs),
    )
    return binding, policy


def test_test_pricing_cannot_unlock_real_provider(
    monkeypatch, tmp_path, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding, _ = _configure_responses_provider(executor, monkeypatch, mock_provider=False)
    forged = tmp_path / "caller-pricing"
    forged.mkdir(mode=0o700)
    (forged / "registry.key").write_bytes(b"x" * 32)
    (forged / "attestations.json").write_text('{"schema_version":1,"attestations":[]}')
    monkeypatch.setenv("GPU_AGENT_PRICING_REGISTRY_ROOT", str(forged))
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=1,
        max_unit_cost_usd=1,
        random_seed=7,
    ).run("E", "development", 3)
    assert result.stopped_reason == "EXECUTION_ERROR" and result.records == []
    assert not any(
        ref.name.startswith("provider/")
        for run_dir in executor.service.store.root.iterdir()
        if run_dir.is_dir() and len(run_dir.name) == 32
        for ref in executor.service.store.load(run_dir.name).artifact_refs
    )


def test_mode_e_binds_native_provider_policy_invocation_and_usage(
    oob_service, tmp_path, monkeypatch, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding, policy = _configure_responses_provider(executor, monkeypatch)
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version,
        toolchain_hash=binding.toolchain_lock_hash,
        model_config_hash=binding.model_config_hash,
        binding=binding,
        max_cost_usd=1,
        max_unit_cost_usd=1,
        random_seed=7,
    ).run("E", "development", 3)
    assert result.stopped_reason == "COST_CAP_RESERVATION_REQUIRED"
    assert result.executed_units == 1
    record = result.records[0]
    assert record.usage["physical_calls"] == 1
    assert record.cost_usd == 0.000005
    assert len(record.lineage.provider_invocation_hashes) == 1
    run = executor.service.store.load(record.lineage.diagnosis_run_id)
    policy_ref = next(ref for ref in run.artifact_refs if ref.name == "agent/provider-policy.json")
    assert json.loads(executor.service.store.read(policy_ref)) == policy


@pytest.mark.parametrize("native_evaluation_executor", ["public_exact"], indirect=True)
def test_scheduled_repair_resolves_native_private_verification(
    monkeypatch, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding, _ = _configure_responses_provider(executor, monkeypatch, full_script=True)
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=1,
        max_unit_cost_usd=1,
        random_seed=7,
    ).run("E", "development", 3)
    assert result.executed_units == 1
    assert result.records[0].lineage.verification_run_id is not None
    assert "verification/runtime" in result.records[0].executed_checks
    assert "verification/private_oracle" not in result.records[0].executed_checks
    assert result.stopped_reason == "COST_CAP_RESERVATION_REQUIRED"
    verification_run = executor.service.store.load(result.records[0].lineage.verification_run_id)
    public_result = json.loads(
        executor.service.store.read(
            next(
                ref
                for ref in verification_run.artifact_refs
                if ref.name == "verification/result.json"
            )
        )
    )
    assert {
        "private_holdout_passed",
        "private_passed_count",
        "not_run_count",
        "suite_hash",
    }.isdisjoint(public_result)


def test_scheduled_repair_rejects_public_only_verification_summary(
    monkeypatch, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.patching import PatchCandidate
    from gpu_agent.verification.models import VerificationResult

    executor = native_evaluation_executor
    binding, _ = _configure_responses_provider(executor, monkeypatch, full_script=True)

    def fake_verify(run_id, candidate_id):
        candidate_run = executor.service.store.load(candidate_id)
        candidate = PatchCandidate.model_validate_json(
            executor.service.store.read(
                next(ref for ref in candidate_run.artifact_refs if ref.name == "candidate.json")
            )
        )
        value = VerificationResult(
            verdict="INCONCLUSIVE",
            failure_stage="verification",
            reason_code="FORGED_SUMMARY",
            original_finding_present=None,
            public_oracle_passed=None,
            required_checks={},
            candidate_hash=candidate.patched_source_hash,
        )
        child = executor.service.store.create_run("verification", run_id)
        executor.service.store.put(
            child.id,
            "verification/result.json",
            value.model_dump_json().encode(),
            "public",
        )
        executor.service.store.transition(child.id, "RUNNING", "FINALIZING")
        executor.service.store.transition(child.id, "COMPLETED", None)
        return value

    monkeypatch.setattr(executor.service, "verify", fake_verify)
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=1,
        max_unit_cost_usd=1,
        random_seed=7,
    ).run("E", "development", 3)
    assert result.stopped_reason == "EXECUTION_ERROR" and result.records == []


def test_mode_e_rejects_forged_model_config_binding(
    oob_service, tmp_path, monkeypatch, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding, _ = _configure_responses_provider(executor, monkeypatch)
    forged = binding.model_copy(update={"model_config_hash": "f" * 64})
    executor.service._binding = forged
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor,
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


def test_mode_e_requires_pricing_attestation_before_provider_call(
    oob_service, tmp_path, monkeypatch, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding, _ = _configure_responses_provider(executor, monkeypatch)
    executor.service._pricing_attestation = None
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=1,
        max_unit_cost_usd=1,
        random_seed=7,
    ).run("E", "development", 3)
    assert result.stopped_reason == "EXECUTION_ERROR" and result.records == []
    diagnosis_runs = [
        executor.service.store.load(path.name)
        for path in executor.service.store.root.iterdir()
        if path.is_dir() and len(path.name) == 32
    ]
    assert not any(
        ref.name.startswith("provider/") for run in diagnosis_runs for ref in run.artifact_refs
    )


def test_unscheduled_mode_e_cannot_reach_paid_provider(
    oob_service, monkeypatch, native_evaluation_executor
):
    executor = native_evaluation_executor
    _configure_responses_provider(executor, monkeypatch)
    run = executor.service.diagnose(oob_service[2], mode="E")
    assert executor.service.diagnosis(run.id).limitations == ["PRICING_ATTESTATION_REQUIRED"]
    assert not any(ref.name.startswith("provider/") for ref in run.artifact_refs)


@pytest.mark.parametrize(
    "response_model,usage", [("substituted-model", True), ("eval-model", False)]
)
def test_mode_e_rejects_unattested_response_policy(
    oob_service, tmp_path, monkeypatch, response_model, usage, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding, _ = _configure_responses_provider(
        executor, monkeypatch, response_model=response_model, usage=usage
    )
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor,
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


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_holdout_alias_and_score_binding_never_publish_private_identity(
    oob_service, tmp_path, monkeypatch, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.benchmark.holdout import HoldoutController
    from gpu_agent.benchmark.metrics import EvaluationLabels, Score
    from gpu_agent.store import RunStore

    executor = native_evaluation_executor
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
        _corpus_family=executor._corpus_family,
    )
    manifest = EvaluationRunner(
        executor.service.store,
        {alias: alias},
        holdout_executor,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=3,
        max_unit_cost_usd=1,
        random_seed=7,
        holdout_controller=controller,
        holdout_batch=batch,
    ).run("D", "holdout", 3)
    evaluation_run = executor.service.store.load(manifest.run_id)
    record_ref = next(
        ref for ref in evaluation_run.artifact_refs if ref.name == "evaluation/records/0.json"
    )
    assert alias not in {"case_0100", "vector-add"}
    private_canary = "PRIVATE-LABEL-CANARY"
    labels = EvaluationLabels(claim_support={private_canary: True})
    private_score = Score(
        family_correct=True,
        root_cause_correct=True,
        location_correct=True,
        inconclusive_correct=False,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: controller.bind_score(
                    batch,
                    alias,
                    record_ref,
                    labels=labels,
                    score=private_score,
                    should_be_inconclusive=False,
                    private_holdout_passed=True,
                ),
                range(2),
            )
        )
    result = results[0]
    assert results == [result, result]
    assert result.public_record_id == manifest.records[0].record_id
    assert result.private_case_id == "case_0100"
    from gpu_agent.benchmark.metrics import (
        HiddenTruth,
        Rubric,
        aggregate,
        aggregate_grouped,
    )
    from gpu_agent.benchmark.metrics import (
        score as metric_score,
    )

    metric_kwargs = {
        "public_store": executor.service.store,
        "evaluator_store": evaluator,
        "run_binding": binding,
    }
    summary = aggregate([result], **metric_kwargs)
    assert summary.record_count == summary.case_count == summary.template_count == 1
    assert summary.family_accuracy.value == summary.root_cause_accuracy.value == 1
    assert summary.private_holdout_pass_rate.value == 1
    grouped = aggregate_grouped([result], **metric_kwargs)
    assert grouped.overall == summary and grouped.by_mode["D"] == summary
    rescored = metric_score(
        result,
        HiddenTruth(
            failure_family="out_of_bounds",
            root_cause_labels=["out_of_bounds"],
            source_path="kernel.cu",
            line_start=1,
            line_end=20,
        ),
        Rubric(),
        **metric_kwargs,
    )
    assert rescored.inconclusive_correct is True
    assert rescored.location_correct is True
    copied_run = executor.service.store.create_run("evaluation", binding=binding)
    executor.service.store.transition(copied_run.id, "RUNNING", "EXECUTING")
    original_refs = {ref.name: ref for ref in evaluation_run.artifact_refs}
    for name in (
        "evaluation/schedule.json",
        "evaluation/attempts/0.json",
        "evaluation/records/0.json",
    ):
        executor.service.store.put(
            copied_run.id,
            name,
            executor.service.store.read(original_refs[name]),
            "public",
        )
    executor.service.store.transition(copied_run.id, "RUNNING", "FINALIZING")
    executor.service.store.transition(copied_run.id, "COMPLETED", None)
    copied_ref = next(
        ref
        for ref in executor.service.store.load(copied_run.id).artifact_refs
        if ref.name == "evaluation/records/0.json"
    )
    with pytest.raises(ValueError, match="public evaluation record"):
        controller.bind_score(
            batch,
            alias,
            copied_ref,
            labels=labels,
            score=private_score,
            should_be_inconclusive=False,
            private_holdout_passed=True,
        )
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
    assert (
        controller.bind_score(
            batch,
            alias,
            record_ref,
            labels=labels,
            score=private_score,
            should_be_inconclusive=False,
            private_holdout_passed=True,
        )
        == result
    )

    second_ref = next(
        ref for ref in evaluation_run.artifact_refs if ref.name == "evaluation/records/1.json"
    )
    original_put = evaluator.put_if_absent_exact
    crashed = False

    def crash_after_private_score(run_id, name, content, visibility):
        nonlocal crashed
        ref = original_put(run_id, name, content, visibility)
        if name == "holdout/private-score.json" and not crashed:
            crashed = True
            raise RuntimeError("simulated controller crash")
        return ref

    monkeypatch.setattr(evaluator, "put_if_absent_exact", crash_after_private_score)
    with pytest.raises(RuntimeError, match="simulated controller crash"):
        controller.bind_score(
            batch,
            alias,
            second_ref,
            labels=labels,
            score=private_score,
            should_be_inconclusive=False,
            private_holdout_passed=True,
        )
    monkeypatch.setattr(evaluator, "put_if_absent_exact", original_put)
    recovered = controller.bind_score(
        batch,
        alias,
        second_ref,
        labels=labels,
        score=private_score,
        should_be_inconclusive=False,
        private_holdout_passed=True,
    )
    assert controller._load_metric_record(recovered).record_id == recovered.public_record_id


def test_deterministic_mode_rejects_unexpected_verification_child(
    oob_service, tmp_path, monkeypatch, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None
    original = executor.execute_scheduled

    def injected(run_id, ordinal):
        record = original(run_id, ordinal)
        child = executor.service.store.create_run("verification", record.record_id)
        executor.service.store.transition(child.id, "RUNNING", "FINALIZING")
        executor.service.store.transition(child.id, "COMPLETED", None)
        return record

    monkeypatch.setattr(executor, "execute_scheduled", injected)
    result = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor,
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


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_private_case_cannot_enter_public_schedule_without_holdout_alias(
    native_evaluation_executor,
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None
    runner = EvaluationRunner(
        executor.service.store,
        {"case_0100": "vector-add"},
        executor,
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
    )
    with pytest.raises(ValueError, match="holdout alias proof"):
        runner.run("D", "holdout", 3)


def test_holdout_rejects_missing_private_labels_and_forged_public_record(
    oob_service, tmp_path, native_evaluation_executor
):
    from gpu_agent.benchmark.holdout import HoldoutController
    from gpu_agent.benchmark.metrics import EvaluationLabels, Score
    from gpu_agent.contracts import ArtifactRef
    from gpu_agent.store import RunStore

    executor = native_evaluation_executor
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
        controller.bind_score(
            batch,
            batch.aliases[0],
            forged,
            labels=None,
            score=private_score,
            should_be_inconclusive=False,
            private_holdout_passed=False,
        )
    with pytest.raises(ValueError, match="public evaluation record"):
        controller.bind_score(
            batch,
            batch.aliases[0],
            forged,
            labels=EvaluationLabels(),
            score=private_score,
            should_be_inconclusive=False,
            private_holdout_passed=False,
        )


@pytest.mark.parametrize("fault", ["case", "template", "source"])
def test_executor_refuses_unregistered_or_changed_inputs(
    oob_service, tmp_path, fault, native_evaluation_executor
):
    executor, source = native_evaluation_executor, oob_service[2]
    if fault == "source":
        (source / "kernel.cu").write_text("changed source")
    with pytest.raises(ValueError):
        _execute_claimed_test_unit(
            executor,
            "A",
            case_id="case_9999" if fault == "case" else "case_0100",
            template_id="wrong" if fault == "template" else "vector-add",
        )
    assert oob_service[1].kinds == []


def test_executor_rejects_self_authored_verification_summary(
    oob_service, monkeypatch, native_evaluation_executor
):
    from gpu_agent.patching import PatchCandidate
    from gpu_agent.verification.models import VerificationResult

    executor = native_evaluation_executor
    service = oob_service[0]

    def persist_verification(run_id, candidate_id):
        candidate_run = service.store.load(candidate_id)
        candidate_ref = next(
            ref for ref in candidate_run.artifact_refs if ref.name == "candidate.json"
        )
        candidate = PatchCandidate.model_validate_json(service.store.read(candidate_ref))
        result = VerificationResult(
            verdict="VERIFIED_FIXED",
            failure_stage=None,
            reason_code="TEST_VERDICT",
            original_finding_present=False,
            public_oracle_passed=True,
            required_checks={"memcheck": "CLEAN", "build": "CLEAN"},
            candidate_hash=candidate.patched_source_hash,
        )
        run = service.store.create_run("verification", run_id)
        service.store.put(
            run.id, "verification/result.json", result.model_dump_json().encode(), "public"
        )
        service.store.transition(run.id, "RUNNING", "FINALIZING")
        service.store.transition(run.id, "COMPLETED", None)
        return result

    monkeypatch.setattr(service, "verify", persist_verification)
    with pytest.raises(ValueError, match="audit lineage"):
        _execute_claimed_test_unit(executor, "E")


def test_missing_knowledge_has_zero_physical_retrievals(
    oob_service, tmp_path, native_evaluation_executor
):
    executor = native_evaluation_executor
    oob_service[0].knowledge = None
    record = _execute_claimed_test_unit(executor, "B")
    assert record.usage["retrieval_calls"] == 0
    assert record.usage["retrieval_attempts"] == 1


def test_timeout_before_backend_has_zero_physical_sanitizer_calls(
    oob_service, tmp_path, monkeypatch, native_evaluation_executor
):
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import ProviderError

    executor = native_evaluation_executor
    original = LLMCallGate.timeout
    timeouts = []

    def reject_sanitizer(self, limit):
        timeouts.append(limit)
        if len(timeouts) == 3:  # Build and ordinary runtime finish before sanitizer request.
            raise ProviderError("AGENT_BUDGET_EXHAUSTED")
        return original(self, limit)

    monkeypatch.setattr(LLMCallGate, "timeout", reject_sanitizer)
    record = _execute_claimed_test_unit(executor, "C")
    assert record.failure_reason == "AGENT_BUDGET_EXHAUSTED"
    assert record.usage["sanitizer_calls"] == 0
    assert record.usage["sanitizer_attempts"] == 1
    assert record.executed_checks == {}


def test_successful_acquisition_persists_physical_calls(
    oob_service, tmp_path, native_evaluation_executor
):
    executor = native_evaluation_executor
    record = _execute_claimed_test_unit(executor, "D")
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
    oob_service, tmp_path, monkeypatch, usage, native_evaluation_executor
):
    executor = native_evaluation_executor
    store = oob_service[0].store
    original = store.put

    def malformed(run_id, name, content, visibility):
        if name == "agent/acquisition-usage.json":
            content = json.dumps(usage).encode()
        return original(run_id, name, content, visibility)

    monkeypatch.setattr(store, "put", malformed)
    with pytest.raises(ValueError):
        _execute_claimed_test_unit(executor, "B")
