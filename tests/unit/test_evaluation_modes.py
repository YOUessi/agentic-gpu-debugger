"""Acquisition modes use the real service, persistence and isolated backend plumbing."""

import hashlib
import json
import stat
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from schedule_authority_support import reserve_schedule_for_test, schedule_client_for_test


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


def test_development_lineage_and_source_remain_public(native_evaluation_executor):
    from gpu_agent.benchmark.evaluation import EvaluationRunner, NativeEvaluationLineage

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None
    result = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash,
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
        random_seed=7,
    ).run("D", "development", 3)
    assert result.executed_units == 3
    assert all(isinstance(record.lineage, NativeEvaluationLineage) for record in result.records)
    for record in result.records:
        run = executor.service.store.load(record.lineage.diagnosis_run_id)
        source_ref = next(ref for ref in run.artifact_refs if ref.name == "sources/kernel.cu")
        assert source_ref.visibility == "public"
        assert (
            executor.service.store.read(source_ref)
            == (executor.sources[record.case_id] / "kernel.cu").read_bytes()
        )


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


def _execute_claimed_test_unit(executor, mode, *, case_id="case_0100", template_id="vector-add"):
    """Exercise the production schedule/attempt claim path for one unit.

    Component tests intentionally stop after the native executor returns; batch
    persistence and terminalization are covered by EvaluationRunner tests.
    """
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    binding = executor.service.binding
    assert binding is not None
    runner = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
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
    if case_id != "case_0100" or template_id != "vector-add":
        schedule = schedule.model_copy(
            update={
                "items": [
                    value.model_copy(update={"case_id": case_id, "template_id": template_id})
                    for value in schedule.items
                ]
            }
        )
    item = schedule.items[0]
    run = runner.store.create_run("evaluation", binding=binding)
    reserve_schedule_for_test(executor, runner, run.id, schedule)
    runner._put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode())
    from gpu_agent.benchmark.schedule_authority import activate_schedule, seal_schedule

    seal_schedule(
        executor._corpus_family,
        runner.store,
        run.id,
        schedule,
        binding,
        schedule_client_for_test(executor),
        executor._schedule_verifier,
    )
    activate_schedule(runner.store, executor._schedule_verifier, run.id)
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
    with pytest.raises(ValueError, match="lineage artifact|child selection"):
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


def test_executor_rejects_synthetic_mode_failure_without_native_provider_lineage(
    oob_service, tmp_path, native_evaluation_executor
):
    executor = native_evaluation_executor
    oob_service[1].actions = []
    with pytest.raises(ValueError, match="lineage artifact"):
        _execute_claimed_test_unit(executor, "E")
    assert oob_service[1].kinds == ["plan"]


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
        executor,
        schedule_client=schedule_client_for_test(executor),
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
        executor,
        schedule_client=schedule_client_for_test(executor),
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
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version,
        toolchain_hash=binding.toolchain_lock_hash,
        model_config_hash=binding.model_config_hash,
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
        random_seed=7,
    ).run("D", "development", 3)
    assert result.stopped_reason is None
    assert len(result.records) == 3


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
                if getattr(scripted, "force_limitation", False):
                    value = DiagnosisResult.inconclusive(scripted.limitation_canary).model_dump(
                        mode="json"
                    )
                else:
                    value = DiagnosisResult(
                        diagnostic_outcome="DIAGNOSED",
                        failure_family="out_of_bounds",
                        root_cause=(
                            "The thread index can exceed the input length."
                            + getattr(scripted, "response_canary", "")
                        ),
                        source_locations=[SourceLocation(path="kernel.cu", line=9)],
                        observed_facts=[
                            EvidenceClaim.model_validate(item)
                            for item in evidence["observed_facts"]
                        ],
                        tool_findings=[
                            EvidenceClaim(
                                text=item["category"],
                                citation_ids=[item["artifact_id"]],
                            )
                            for item in evidence["tool_findings"]
                        ],
                        documentation_evidence=[
                            EvidenceClaim(text=item["text"], citation_ids=[item["chunk_id"]])
                            for item in evidence["documentation"]
                        ],
                        model_inferences=[
                            "An index guard may prevent the reported write."
                            + getattr(scripted, "path_canary", "")
                        ],
                        recommended_change="Guard the write with i < n.",
                        confidence_label="high",
                        limitations=(
                            [scripted.limitation_canary]
                            if getattr(scripted, "limitation_canary", "")
                            else []
                        ),
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
        executor,
        schedule_client=schedule_client_for_test(executor),
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
        executor,
        schedule_client=schedule_client_for_test(executor),
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


def test_mode_e_persists_policy_denied_duplicate_as_bounded_failure(
    oob_service, monkeypatch, native_evaluation_executor
):
    from gpu_agent.agent.models import MemcheckAction
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    oob_service[1].actions = [MemcheckAction(), MemcheckAction()]
    binding, _ = _configure_responses_provider(executor, monkeypatch, full_script=True)

    result = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
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
    assert result.records[0].status == "FAILED"
    assert result.records[0].failure_reason == "DUPLICATE_NO_BENEFIT"
    assert result.stopped_reason == "COST_CAP_RESERVATION_REQUIRED"


def test_mode_e_rejects_self_consistent_forged_policy_denial(
    oob_service, monkeypatch, native_evaluation_executor
):
    from gpu_agent.agent.models import DiagnosisResult, InconclusiveAction, PolicyDecision
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    oob_service[1].actions = [InconclusiveAction()]
    binding, _ = _configure_responses_provider(executor, monkeypatch, full_script=True)
    store = executor.service.store
    original_put = store.put

    def forge_denial(run_id, name, content, visibility):
        if name == "actions/0/decision.json":
            decision = PolicyDecision.model_validate_json(content)
            content = (
                decision.model_copy(
                    update={"allowed": False, "reason_codes": ["DUPLICATE_NO_BENEFIT"]}
                )
                .model_dump_json()
                .encode()
            )
        elif name == "diagnosis.json":
            content = (
                DiagnosisResult.inconclusive("DUPLICATE_NO_BENEFIT").model_dump_json().encode()
            )
        return original_put(run_id, name, content, visibility)

    monkeypatch.setattr(store, "put", forge_denial)
    result = EvaluationRunner(
        store,
        executor,
        schedule_client=schedule_client_for_test(executor),
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


def test_mode_e_rejects_policy_denial_from_forged_step_budget(
    oob_service, monkeypatch, native_evaluation_executor
):
    from gpu_agent.agent.models import DiagnosisResult, InconclusiveAction, PolicyDecision
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    oob_service[1].actions = [InconclusiveAction()]
    binding, _ = _configure_responses_provider(executor, monkeypatch, full_script=True)
    store = executor.service.store
    original_put = store.put

    def forge_exhausted_budget(run_id, name, content, visibility):
        if name == "actions/0/step.json":
            step = json.loads(content)
            step["budget"]["agent_steps"] = step["budget"]["max_agent_steps"]
            content = json.dumps(step, sort_keys=True, separators=(",", ":")).encode()
        elif name == "actions/0/decision.json":
            decision = PolicyDecision.model_validate_json(content)
            content = (
                decision.model_copy(
                    update={"allowed": False, "reason_codes": ["AGENT_BUDGET_EXHAUSTED"]}
                )
                .model_dump_json()
                .encode()
            )
        elif name == "diagnosis.json":
            content = (
                DiagnosisResult.inconclusive("AGENT_BUDGET_EXHAUSTED").model_dump_json().encode()
            )
        return original_put(run_id, name, content, visibility)

    monkeypatch.setattr(store, "put", forge_exhausted_budget)
    result = EvaluationRunner(
        store,
        executor,
        schedule_client=schedule_client_for_test(executor),
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


def test_mode_e_rejects_provider_diagnosis_after_forged_terminal_denial(
    oob_service, monkeypatch, native_evaluation_executor
):
    from gpu_agent.agent.models import PolicyDecision
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    oob_service[1].force_limitation = True
    oob_service[1].limitation_canary = "DUPLICATE_NO_BENEFIT"
    binding, _ = _configure_responses_provider(executor, monkeypatch, full_script=True)
    store = executor.service.store
    original_put = store.put

    def forge_terminal_denial(run_id, name, content, visibility):
        if name == "actions/2/step.json":
            step = json.loads(content)
            action = step["action"]
            signature = action["action_type"] + json.dumps(
                action["typed_arguments"], sort_keys=True, separators=(",", ":")
            )
            step["seen"] = sorted([*step["seen"], signature])
            content = json.dumps(step, sort_keys=True, separators=(",", ":")).encode()
        elif name == "actions/2/decision.json":
            decision = PolicyDecision.model_validate_json(content)
            content = (
                decision.model_copy(
                    update={"allowed": False, "reason_codes": ["DUPLICATE_NO_BENEFIT"]}
                )
                .model_dump_json()
                .encode()
            )
        return original_put(run_id, name, content, visibility)

    monkeypatch.setattr(store, "put", forge_terminal_denial)
    result = EvaluationRunner(
        store,
        executor,
        schedule_client=schedule_client_for_test(executor),
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


def test_mode_e_rejects_evidence_appended_after_terminal_denial(
    oob_service, monkeypatch, native_evaluation_executor
):
    from gpu_agent.agent.models import MemcheckAction
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.evidence.models import EvidenceBundle

    executor = native_evaluation_executor
    oob_service[1].actions = [MemcheckAction(), MemcheckAction()]
    binding, _ = _configure_responses_provider(executor, monkeypatch, full_script=True)
    store = executor.service.store
    original_put = store.put
    appended = False

    def append_post_denial_evidence(run_id, name, content, visibility):
        nonlocal appended
        if name == "agent/controller-lineage.json" and not appended:
            appended = True
            run = store.load(run_id)
            evidence_refs = [ref for ref in run.artifact_refs if ref.name == "evidence/bundle.json"]
            bundle = EvidenceBundle.model_validate_json(store.read(evidence_refs[-1]))
            assert bundle.sanitizer_results
            forged = bundle.model_copy(
                update={
                    "sanitizer_results": [
                        *bundle.sanitizer_results,
                        bundle.sanitizer_results[-1],
                    ]
                }
            )
            forged_ref = original_put(
                run_id,
                "evidence/bundle.json",
                forged.model_dump_json().encode(),
                visibility,
            )
            lineage = json.loads(content)
            lineage["evidence_ref"] = {"id": forged_ref.id, "sha256": forged_ref.sha256}
            content = json.dumps(lineage, sort_keys=True, separators=(",", ":")).encode()
        return original_put(run_id, name, content, visibility)

    monkeypatch.setattr(store, "put", append_post_denial_evidence)
    result = EvaluationRunner(
        store,
        executor,
        schedule_client=schedule_client_for_test(executor),
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


@pytest.mark.parametrize("native_evaluation_executor", ["public_exact"], indirect=True)
def test_scheduled_repair_resolves_native_private_verification(
    monkeypatch, native_evaluation_executor
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = native_evaluation_executor
    binding, _ = _configure_responses_provider(executor, monkeypatch, full_script=True)
    result = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
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
    public_bytes = b"".join(
        path.read_bytes() for path in executor.service.store.root.rglob("*") if path.is_file()
    )
    assert b"private_holdout_passed" not in public_bytes
    assert b"private_passed_count" not in public_bytes

    from gpu_agent.benchmark.evaluation import EvaluationAttempt, EvaluationSchedule
    from gpu_agent.contracts import ToolResult
    from gpu_agent.execution.models import SanitizerPayload
    from gpu_agent.store import RunStore

    evaluation_run = executor.service.store.load(result.run_id)
    schedule_ref = next(
        ref for ref in evaluation_run.artifact_refs if ref.name == "evaluation/schedule.json"
    )
    attempt_ref = next(
        ref for ref in evaluation_run.artifact_refs if ref.name == "evaluation/attempts/0.json"
    )
    schedule = EvaluationSchedule.model_validate_json(executor.service.store.read(schedule_ref))
    attempt = EvaluationAttempt.model_validate_json(executor.service.store.read(attempt_ref))
    audit_id = public_result["evaluator_audit_run_id"]
    evaluator = RunStore(executor.service.evaluator_root / "runs", visibility="evaluator")
    audit = evaluator.load(audit_id)
    child_id = json.loads(
        evaluator.read(
            next(ref for ref in audit.artifact_refs if ref.name == "verification/audit-result.json")
        )
    )["child_run_ids"][0]
    child = evaluator.load(child_id)
    candidate_sanitizer_ref = next(
        ref
        for ref in child.artifact_refs
        if ref.name.startswith("sanitizer/") and ref.name.endswith("/result.json")
    )
    candidate_sanitizer = ToolResult[SanitizerPayload].model_validate_json(
        evaluator.read(candidate_sanitizer_ref)
    )
    original_read = RunStore.read

    def validate_with_replacement(target_id, replacement):
        def forged_read(self, ref):
            if ref.id == target_id:
                return replacement
            return original_read(self, ref)

        monkeypatch.setattr(RunStore, "read", forged_read)
        with pytest.raises(ValueError):
            executor.validate_scheduled_record(result.records[0], schedule.items[0], attempt)
        monkeypatch.setattr(RunStore, "read", original_read)

    validate_with_replacement(
        candidate_sanitizer.stderr_artifact.id,
        b"========= Invalid __global__ write of size 4 bytes\n========= ERROR SUMMARY: 1 error\n",
    )
    diagnosis_run = executor.service.store.load(result.records[0].record_id)
    baseline_sanitizer_ref = next(
        ref
        for ref in diagnosis_run.artifact_refs
        if ref.name.startswith("sanitizer/") and ref.name.endswith("/result.json")
    )
    baseline_sanitizer = ToolResult[SanitizerPayload].model_validate_json(
        executor.service.store.read(baseline_sanitizer_ref)
    )
    validate_with_replacement(
        baseline_sanitizer.stderr_artifact.id,
        b"========= ERROR SUMMARY: 0 errors\n",
    )
    candidate_input = next(ref for ref in child.artifact_refs if ref.name == "input.json")
    validate_with_replacement(
        candidate_input.id,
        b" " + evaluator.read(candidate_input),
    )


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
        executor,
        schedule_client=schedule_client_for_test(executor),
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
        executor,
        schedule_client=schedule_client_for_test(executor),
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
        executor,
        schedule_client=schedule_client_for_test(executor),
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
        executor,
        schedule_client=schedule_client_for_test(executor),
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

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None
    evaluator = executor.corpus
    controller = HoldoutController(
        executor.service.store,
        evaluator,
        binding=binding,
        _schedule_verifier=executor._schedule_verifier,
    )
    batch = controller.prepare()
    alias = batch.aliases[0]
    holdout_executor = EvaluationExecutor(
        executor.service,
        executor.corpus,
        executor.sources,
        holdout_service=executor.holdout_service,
        holdout_controller=controller,
        holdout_batch=batch,
        _corpus_family=executor._corpus_family,
        _schedule_verifier=executor._schedule_verifier,
    )
    manifest = EvaluationRunner(
        executor.service.store,
        holdout_executor,
        schedule_client=schedule_client_for_test(holdout_executor),
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
    before_preparation = {
        str(path): path.read_bytes()
        for root in (executor.service.store.root, evaluator.root)
        for path in root.rglob("*")
        if path.is_file()
    }
    prepared = controller.prepare_score(
        batch,
        alias,
        record_ref,
        labels=labels,
        score=private_score,
        should_be_inconclusive=False,
        private_holdout_passed=True,
    )
    assert prepared.binding.public_record_id == manifest.records[0].record_id
    assert prepared.binding.private_case_id == "case_0100"
    assert {
        str(path): path.read_bytes()
        for root in (executor.service.store.root, evaluator.root)
        for path in root.rglob("*")
        if path.is_file()
    } == before_preparation
    evaluation_lock = executor.service.store.root / manifest.run_id / ".lock"
    displaced_lock = tmp_path / "preserved-evaluation.lock"
    evaluation_lock.rename(displaced_lock)
    try:
        with pytest.raises(ValueError):
            controller.prepare_score(
                batch,
                alias,
                record_ref,
                labels=labels,
                score=private_score,
                should_be_inconclusive=False,
                private_holdout_passed=True,
            )
        assert not evaluation_lock.exists()
    finally:
        evaluation_lock.unlink(missing_ok=True)
        displaced_lock.rename(evaluation_lock)
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
    assert result == prepared.binding
    persisted_score_ref = next(
        ref
        for ref in evaluator.load(prepared.run_id).artifact_refs
        if ref.name == "holdout/private-score.json"
    )
    assert evaluator.read(persisted_score_ref) == prepared.private_score_content
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
        "schedule_verifier": executor._schedule_verifier,
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
    original_refs = {ref.name: ref for ref in evaluation_run.artifact_refs}
    for name in ("evaluation/schedule.json", "evaluation/schedule-receipt.json"):
        executor.service.store.put(
            copied_run.id,
            name,
            executor.service.store.read(original_refs[name]),
            "public",
        )
    for name in (
        "evaluation/attempts/0.json",
        "evaluation/records/0.json",
    ):
        executor.service.store.put(
            copied_run.id,
            name,
            executor.service.store.read(original_refs[name]),
            "public",
        )
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
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
        random_seed=7,
    ).run("D", "development", 3)
    assert result.stopped_reason is None and len(result.records) == 3


def test_mode_d_replay_rejects_future_evidence_and_forged_budget(
    native_evaluation_executor, monkeypatch
):
    from gpu_agent.benchmark.evaluation import (
        EvaluationAttempt,
        EvaluationRunner,
        EvaluationSchedule,
    )
    from gpu_agent.store import RunStore

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None
    result = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
        random_seed=7,
    ).run("D", "development", 3)
    evaluation_run = executor.service.store.load(result.run_id)
    schedule = EvaluationSchedule.model_validate_json(
        executor.service.store.read(
            next(
                ref
                for ref in evaluation_run.artifact_refs
                if ref.name == "evaluation/schedule.json"
            )
        )
    )
    attempt = EvaluationAttempt.model_validate_json(
        executor.service.store.read(
            next(
                ref
                for ref in evaluation_run.artifact_refs
                if ref.name == "evaluation/attempts/0.json"
            )
        )
    )
    diagnosis = executor.service.store.load(result.records[0].record_id)
    steps = sorted(
        (ref for ref in diagnosis.artifact_refs if ref.name.endswith("/step.json")),
        key=lambda ref: ref.name,
    )
    assert len(steps) >= 2
    first = json.loads(executor.service.store.read(steps[0]))
    second = json.loads(executor.service.store.read(steps[1]))
    original_read = RunStore.read

    for forged in (
        {**first, "evidence_ref": second["evidence_ref"]},
        {**first, "budget": {**first["budget"], "sanitizer_calls": 3}},
        {**first, "action": second["action"]},
    ):
        content = json.dumps(forged, sort_keys=True, separators=(",", ":")).encode()

        def forged_read(self, ref, *, _content=content):
            if ref.id == steps[0].id:
                return _content
            return original_read(self, ref)

        monkeypatch.setattr(RunStore, "read", forged_read)
        with pytest.raises(ValueError):
            executor.validate_scheduled_record(result.records[0], schedule.items[0], attempt)
        monkeypatch.setattr(RunStore, "read", original_read)

    initial_ref = next(
        ref for ref in diagnosis.artifact_refs if ref.name == "agent/initial-budget.json"
    )
    initial = json.loads(executor.service.store.read(initial_ref))

    def forged_initial(self, ref):
        if ref.id == initial_ref.id:
            return json.dumps({**initial, "agent_steps": 1}).encode()
        return original_read(self, ref)

    monkeypatch.setattr(RunStore, "read", forged_initial)
    with pytest.raises(ValueError, match="budget audit"):
        executor.validate_scheduled_record(result.records[0], schedule.items[0], attempt)
    monkeypatch.setattr(RunStore, "read", original_read)


@pytest.mark.parametrize("failure", ["sanitizer", "docs"])
def test_mode_d_failed_acquisition_is_terminal_inconclusive(
    native_evaluation_executor, monkeypatch, failure
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.execution.process import ProcessCapture

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None
    if failure == "docs":
        executor.service.knowledge = None
        expected = "KNOWLEDGE_UNAVAILABLE"
    else:
        backend = executor.service._backend_factory
        original_container = backend._container

        def fail_sanitizer(self, path, operation, timeout, *, stdin=b"", cancel=None):
            if operation in {"memcheck", "racecheck", "initcheck", "synccheck"}:
                return (
                    ProcessCapture(None, b"", b"", False, tool_error="CONTAINER_ERROR"),
                    b"",
                    b"",
                )
            return original_container(self, path, operation, timeout, stdin=stdin, cancel=cancel)

        monkeypatch.setattr(backend, "_container", fail_sanitizer)
        expected = "SANITIZER_EVIDENCE_UNAVAILABLE"
    result = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=0,
        max_unit_cost_usd=0,
        random_seed=7,
    ).run("D", "development", 3)
    assert result.stopped_reason is None and len(result.records) == 3
    assert all(record.status == "INCONCLUSIVE" for record in result.records)
    assert all(expected in record.diagnosis["limitations"] for record in result.records)
    for record in result.records:
        run = executor.service.store.load(record.record_id)
        audit_ref = next(ref for ref in run.artifact_refs if ref.name == "agent/budget-audit.json")
        assert json.loads(executor.service.store.read(audit_ref))[-1]["state"] == "FAILED"


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
        executor,
        schedule_client=schedule_client_for_test(executor),
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


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_holdout_rejects_missing_private_labels_and_forged_public_record(
    oob_service, tmp_path, native_evaluation_executor
):
    from gpu_agent.benchmark.holdout import HoldoutController
    from gpu_agent.benchmark.metrics import EvaluationLabels, Score
    from gpu_agent.contracts import ArtifactRef

    executor = native_evaluation_executor
    binding = executor.service.binding
    assert binding is not None
    evaluator = executor.corpus
    controller = HoldoutController(
        executor.service.store,
        evaluator,
        binding=binding,
        _schedule_verifier=executor._schedule_verifier,
    )
    batch = controller.prepare()
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
    with pytest.raises(ValueError, match="verification lineage"):
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


@pytest.fixture
def prepared_holdout_execution(native_evaluation_executor, monkeypatch):
    from gpu_agent.agent.models import AcquisitionUsage, AgentBudget
    from gpu_agent.benchmark import evaluation as evaluation_module
    from gpu_agent.benchmark.evaluation import EvaluationAttempt, EvaluationScheduleItem
    from gpu_agent.benchmark.executor import registered_cases
    from gpu_agent.benchmark.holdout import HoldoutController
    from gpu_agent.evidence.repository import EvidenceRepository
    from gpu_agent.service import ApplicationService

    executor = native_evaluation_executor
    controller = HoldoutController(
        executor.service.store,
        executor.corpus,
        binding=executor.service.binding,
        _schedule_verifier=executor._schedule_verifier,
    )
    batch = controller.prepare()
    item = EvaluationScheduleItem(
        ordinal=0,
        case_id=batch.aliases[0],
        template_id=batch.aliases[0],
        mode="D",
        repeat=0,
        split="holdout",
        holdout_proof=controller.validate_batch(batch),
    )
    attempt = EvaluationAttempt(
        run_id="a" * 32,
        ordinal=0,
        schedule_hash="b" * 64,
        corpus_cutoff=batch.corpus_cutoff,
        idempotency_key="c" * 64,
        reserved_cost_usd=1,
    )
    prepared = controller.reserve_execution(
        batch, evaluation_run_id=attempt.run_id, item=item, attempt=attempt
    )
    evaluator_service = ApplicationService(
        executor.corpus,
        executor.corpus,
        provider=executor.service._provider,
        backend_factory=executor.service._backend_factory,
        knowledge=executor.service.knowledge,
        knowledge_version=executor.service.knowledge_version,
        _binding=executor.service.binding,
        _evaluation_schedule_verifier=executor._schedule_verifier,
    )
    monkeypatch.setattr(
        executor.corpus,
        "validate_and_create_evaluation_child",
        lambda verifier, unit: (
            executor.corpus.load(prepared.diagnosis_run_id)
            if unit == prepared.evaluation_unit
            else (_ for _ in ()).throw(ValueError("unexpected evaluation unit"))
        ),
    )
    case = registered_cases(
        executor.corpus,
        executor.service.binding,
        executor._corpus_family,
        cutoff=batch.corpus_cutoff,
    )["case_0100"]
    run = evaluator_service.diagnose(
        executor.sources["case_0100"],
        mode=item.mode,
        required_tools=(case.target_tool,),
        expected_source_hash=case.source_hash,
        evaluation_unit=prepared.evaluation_unit,
    )
    diagnosis = evaluator_service.diagnosis(run.id)
    diagnosis_ref = next(ref for ref in run.artifact_refs if ref.name == "diagnosis.json")
    evidence_refs = [ref for ref in run.artifact_refs if ref.name == "evidence/bundle.json"]
    bundle = EvidenceRepository(executor.corpus, evaluator=True).view(run.id)
    budget = AgentBudget.model_validate_json(
        executor.corpus.read(
            next(ref for ref in run.artifact_refs if ref.name == "agent/final-budget.json")
        )
    )
    acquisition = AcquisitionUsage.model_validate_json(
        executor.corpus.read(
            next(ref for ref in run.artifact_refs if ref.name == "agent/acquisition-usage.json")
        )
    )
    diagnostic_calls = (
        acquisition.sanitizer_calls
        + acquisition.retrieval_calls
        + int(bundle.build_result is not None)
        + int(bundle.execution_result is not None)
    )
    usage = {
        "physical_calls": budget.llm_calls,
        "sanitizer_calls": acquisition.sanitizer_calls,
        "retrieval_calls": acquisition.retrieval_calls,
        "sanitizer_attempts": budget.sanitizer_calls,
        "retrieval_attempts": budget.rag_calls,
        "build_calls": int(bundle.build_result is not None),
        "runtime_calls": int(bundle.execution_result is not None),
        "diagnostic_tool_calls": diagnostic_calls,
        "tool_calls": diagnostic_calls,
        "total_sanitizer_calls": acquisition.sanitizer_calls,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    private_case_id, private_template_id = controller.resolve_private(batch, item.case_id)
    native = evaluation_module.EvaluationRecord(
        record_id=prepared.diagnosis_run_id,
        corpus_cutoff=batch.corpus_cutoff,
        lineage=evaluation_module.NativeEvaluationLineage(
            corpus_cutoff=batch.corpus_cutoff,
            diagnosis_run_id=prepared.diagnosis_run_id,
            diagnosis_hash=diagnosis_ref.sha256,
            evidence_hash=evidence_refs[-1].sha256,
            provider_invocation_hashes=[],
        ),
        case_id=private_case_id,
        template_id=private_template_id,
        mode=item.mode,
        repeat=item.repeat,
        input_hash=case.source_hash,
        evidence_hash=evidence_refs[-1].sha256,
        executed_checks={
            result.tool_result.typed_payload.tool: result.check_outcome
            for result in bundle.sanitizer_results
            if result.tool_result is not None
        },
        status="INCONCLUSIVE",
        diagnosis=diagnosis.model_dump(mode="json"),
        usage=usage,
        latency_ms=(run.events[-1].at - run.events[0].at).total_seconds() * 1000,
        cost_usd=0,
        failure_reason=None,
    )
    return SimpleNamespace(
        controller=controller,
        batch=batch,
        item=item,
        attempt=attempt,
        prepared=prepared,
        native=native,
        evaluator=executor.corpus,
    )


@pytest.fixture
def completed_holdout_execution(prepared_holdout_execution):
    case = prepared_holdout_execution
    case.public = case.controller.complete_execution(case.prepared, case.native)
    return case


@pytest.fixture
def prepared_holdout_repair_execution(native_evaluation_executor, monkeypatch):
    from test_holdout_scoring import evaluator_native_record

    from gpu_agent.benchmark.evaluation import EvaluationAttempt, EvaluationScheduleItem
    from gpu_agent.benchmark.executor import registered_cases
    from gpu_agent.benchmark.holdout import HoldoutController
    from gpu_agent.service import ApplicationService

    executor = native_evaluation_executor
    binding, _ = _configure_responses_provider(executor, monkeypatch, full_script=True)
    controller = HoldoutController(
        executor.service.store,
        executor.corpus,
        binding=binding,
        _schedule_verifier=executor._schedule_verifier,
    )
    batch = controller.prepare()
    item = EvaluationScheduleItem(
        ordinal=0,
        case_id=batch.aliases[0],
        template_id=batch.aliases[0],
        mode="E",
        repeat=0,
        split="holdout",
        holdout_proof=controller.validate_batch(batch),
    )
    attempt = EvaluationAttempt(
        run_id="a" * 32,
        ordinal=0,
        schedule_hash="b" * 64,
        corpus_cutoff=batch.corpus_cutoff,
        idempotency_key="c" * 64,
        reserved_cost_usd=1,
    )
    prepared = controller.reserve_execution(
        batch, evaluation_run_id=attempt.run_id, item=item, attempt=attempt
    )
    evaluator_service = ApplicationService(
        executor.corpus,
        executor.corpus,
        provider=executor.service._provider,
        backend_factory=executor.service._backend_factory,
        knowledge=executor.service.knowledge,
        knowledge_version=executor.service.knowledge_version,
        _binding=binding,
        _evaluation_schedule_verifier=executor._schedule_verifier,
    )
    evaluator_service._pricing_attestation = executor.service._pricing_attestation
    monkeypatch.setattr(
        executor.corpus,
        "validate_and_create_evaluation_child",
        lambda verifier, unit: (
            executor.corpus.load(prepared.diagnosis_run_id)
            if unit == prepared.evaluation_unit
            else (_ for _ in ()).throw(ValueError("unexpected evaluation unit"))
        ),
    )
    cases = registered_cases(
        executor.corpus,
        binding,
        executor._corpus_family,
        cutoff=batch.corpus_cutoff,
    )
    case = cases[prepared.binding.private_case_id]
    evaluator_service.diagnose(
        executor.sources[case.id],
        mode=item.mode,
        required_tools=(case.target_tool,),
        expected_source_hash=case.source_hash,
        evaluation_unit=prepared.evaluation_unit,
    )
    native = evaluator_native_record(
        evaluator_service,
        executor.corpus,
        prepared,
        item,
        case,
    )
    assert native.lineage.candidate_run_id is not None
    assert native.lineage.verification_run_id is not None
    return SimpleNamespace(
        controller=controller,
        batch=batch,
        item=item,
        attempt=attempt,
        prepared=prepared,
        native=native,
        evaluator=executor.corpus,
    )


@pytest.mark.parametrize("native_evaluation_executor", ["private_exact"], indirect=True)
def test_task2_holdout_repair_transaction_completes_and_recovers_exactly(
    prepared_holdout_repair_execution, native_evaluation_executor
):
    case = prepared_holdout_repair_execution
    public = case.controller.complete_execution(case.prepared, case.native)
    assert case.controller.recover_execution(case.batch, case.item, case.attempt) == public


@pytest.mark.parametrize("native_evaluation_executor", ["private_exact"], indirect=True)
def test_holdout_public_projection_drops_provider_diagnosis(
    prepared_holdout_repair_execution, native_evaluation_executor
):
    case = prepared_holdout_repair_execution
    assert case.native.diagnosis
    public = case.controller.complete_execution(case.prepared, case.native)
    assert public.diagnosis == {}
    assert public.failure_reason is None


@pytest.mark.parametrize("native_evaluation_executor", ["private_split"], indirect=True)
def test_forged_reserved_diagnosis_capability_has_zero_side_effects(
    private_split_executor, native_evaluation_executor, monkeypatch
):
    from gpu_agent.benchmark.evaluation import EvaluationUnitBinding
    from gpu_agent.benchmark.holdout import (
        HoldoutExecutionBinding,
        PreparedHoldoutExecution,
    )
    from gpu_agent.contracts import ExternalRunOrigin, RunStatus
    from gpu_agent.execution.models import SanitizerTool

    executor = private_split_executor
    service = executor.holdout_service
    controller = executor.holdout_controller
    batch = executor.holdout_batch
    assert service is not None and controller is not None and batch is not None
    private_case_id, private_template_id = controller.resolve_private(batch, batch.aliases[0])
    unit = EvaluationUnitBinding(
        evaluation_run_id="f" * 32,
        ordinal=0,
        schedule_hash="e" * 64,
        corpus_cutoff=batch.corpus_cutoff,
        idempotency_key="d" * 64,
        reserved_cost_usd=1,
        case_id=private_case_id,
        template_id=private_template_id,
        mode="E",
        repeat=0,
        split="holdout",
        holdout_proof=controller.validate_batch(batch),
    )
    store = service.store
    backend_calls = []
    backend_type = service._backend_factory
    native_container = backend_type._container

    def counted_container(*args, **kwargs):
        backend_calls.append(args[2])
        return native_container(*args, **kwargs)

    monkeypatch.setattr(backend_type, "_container", counted_container)
    parent = store.create_run(
        "holdout_execution",
        binding=service.binding,
        external_origin=ExternalRunOrigin(run_id=unit.evaluation_run_id, visibility="public"),
    )
    store.transition(parent.id, RunStatus.RUNNING, "PREPARING")
    diagnosis = store.create_run("diagnosis", parent_run_id=parent.id)
    store.put(
        diagnosis.id,
        "evaluation/unit.json",
        unit.model_dump_json().encode(),
        "evaluator",
    )
    before_provider = list(service._provider.kinds)
    selected = executor.sources[private_case_id] / "kernel.cu"
    with pytest.raises(TypeError):
        service.diagnose(
            executor.sources[private_case_id],
            mode="E",
            required_tools=(SanitizerTool.MEMCHECK,),
            expected_source_hash=hashlib.sha256(selected.read_bytes()).hexdigest(),
            evaluation_unit=unit,
            _reserved_run_id=diagnosis.id,
        )
    store.transition(diagnosis.id, RunStatus.RUNNING, "PREPARING")
    prepared = PreparedHoldoutExecution(
        execution_run_id=parent.id,
        diagnosis_run_id=diagnosis.id,
        binding=HoldoutExecutionBinding(
            public_evaluation_run_id=unit.evaluation_run_id,
            alias_mapping_run_id=batch.evaluator_run_id,
            ordinal=unit.ordinal,
            schedule_hash=unit.schedule_hash,
            attempt_hash="c" * 64,
            corpus_cutoff=unit.corpus_cutoff,
            alias=batch.aliases[0],
            private_case_id=unit.case_id,
            private_template_id=unit.template_id,
            diagnosis_run_id=diagnosis.id,
        ),
        evaluation_unit=unit,
    )
    assert not hasattr(controller, "_HoldoutController__register_reserved_capability")
    with pytest.raises(ValueError, match="holdout execution|authority"):
        service._diagnose_reserved(
            executor.sources[private_case_id],
            controller=controller,
            batch=batch,
            prepared=prepared,
            mode="E",
            required_tools=(SanitizerTool.MEMCHECK,),
            expected_source_hash=hashlib.sha256(selected.read_bytes()).hexdigest(),
        )
    assert store.load(diagnosis.id).status == RunStatus.RUNNING
    assert service._provider.kinds == before_provider
    assert backend_calls == []


@pytest.mark.parametrize("native_evaluation_executor", ["private_split"], indirect=True)
@pytest.mark.parametrize(
    "mismatch",
    ["controller_binding", "controller_verifier", "service_evaluator", "holdout_evaluator"],
)
def test_holdout_constructor_rejects_identity_mismatch_before_mutation(
    private_split_executor, native_evaluation_executor, tmp_path, monkeypatch, mismatch
):
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.service import ApplicationService
    from gpu_agent.store import RunStore

    executor = private_split_executor
    service = executor.service
    holdout_service = executor.holdout_service
    assert holdout_service is not None
    before_public = sorted(
        path.relative_to(service.store.root) for path in service.store.root.rglob("*")
    )
    before_evaluator = sorted(
        path.relative_to(executor.corpus.root) for path in executor.corpus.root.rglob("*")
    )
    wrong_store = RunStore(tmp_path / "wrong-evaluator", visibility="evaluator")
    mismatched = holdout_service
    if mismatch == "holdout_evaluator":
        mismatched = ApplicationService(
            holdout_service.store,
            wrong_store,
            provider=holdout_service._provider,
            backend_factory=holdout_service._backend_factory,
            _binding=holdout_service.binding,
        )
    elif mismatch == "service_evaluator":
        monkeypatch.setattr(service, "evaluator_store", wrong_store)
    elif mismatch == "controller_binding":
        monkeypatch.setattr(
            executor.holdout_controller,
            "binding",
            service.binding.model_copy(update={"prompt_version": "mismatched"}),
        )
    else:
        monkeypatch.setattr(executor.holdout_controller, "_schedule_verifier", object())
    with pytest.raises(ValueError, match="configured corpus family|paired family"):
        EvaluationExecutor(
            service,
            executor.corpus,
            executor.sources,
            holdout_service=mismatched,
            holdout_controller=executor.holdout_controller,
            holdout_batch=executor.holdout_batch,
            _corpus_family=executor._corpus_family,
            _schedule_verifier=executor._schedule_verifier,
        )
    assert (
        sorted(path.relative_to(service.store.root) for path in service.store.root.rglob("*"))
        == before_public
    )
    assert (
        sorted(path.relative_to(executor.corpus.root) for path in executor.corpus.root.rglob("*"))
        == before_evaluator
    )


@pytest.mark.parametrize("native_evaluation_executor", ["private_exact_canary"], indirect=True)
def test_holdout_mode_e_uses_exact_candidate_and_verification_ids_without_root_scan(
    native_evaluation_executor, monkeypatch, tmp_path
):
    import difflib
    import shutil

    import gpu_agent.verification.derivation as verification_derivation
    import gpu_agent.verification.engine as verification_engine
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.benchmark.holdout import HoldoutController
    from gpu_agent.benchmark.metrics import EvaluationLabels
    from gpu_agent.service import ApplicationService

    original = native_evaluation_executor
    source_canary = b"PRIVATE-SOURCE-CANARY-task3-a91e"
    provider_canary = b"PROVIDER-RESPONSE-CANARY-task3-b82f"
    candidate_canary = b"CANDIDATE-CANARY-task3-c73d"
    verification_canary = b"VERIFICATION-CANARY-task3-d64c"
    label_canary = b"LABEL-CANARY-task3-e55b"
    path_canary = b"PATH-CANARY-task3-f46a"
    scripted = original.service._provider
    scripted.response_canary = " " + provider_canary.decode()
    scripted.path_canary = " " + path_canary.decode()
    source_path = original.sources["case_0100"] / "kernel.cu"
    source_text = source_path.read_text()
    fixed_text = (
        source_text.replace("out[i] = a[i] + b[i];", "if (i < n) out[i] = a[i] + b[i];").replace(
            "n != 257", "n == 0"
        )
        + f"\n// {candidate_canary.decode()}\n"
    )
    scripted.diff = "".join(
        difflib.unified_diff(
            source_text.splitlines(True),
            fixed_text.splitlines(True),
            fromfile="a/kernel.cu",
            tofile="b/kernel.cu",
        )
    )
    truth_root = tmp_path / "canary-truth"
    shutil.copytree(verification_engine.TRUTH_ROOT, truth_root)
    truth_case = json.loads((truth_root / "case.json").read_bytes())
    truth_case["source_hashes"]["kernel.cu"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    (truth_root / "case.json").write_text(json.dumps(truth_case))
    (truth_root / "reference.cu").write_bytes(
        (truth_root / "reference.cu").read_bytes() + b"\n// " + verification_canary + b"\n"
    )
    monkeypatch.setattr(verification_engine, "TRUTH_ROOT", truth_root)
    monkeypatch.setattr(verification_derivation, "_TRUTH_CASE", truth_root / "case.json")
    binding, _ = _configure_responses_provider(original, monkeypatch, full_script=True)
    original.service.evaluator_store = original.corpus
    marker = original.corpus.create_run("holdout_private_canaries", binding=binding)
    original.corpus.put(
        marker.id,
        "labels/private-labels.json",
        EvaluationLabels(evidence_relevance={label_canary.decode(): True})
        .model_dump_json()
        .encode(),
        "evaluator",
    )
    original.corpus.put(
        marker.id,
        "private/evaluator-path.txt",
        path_canary + b"\n" + str(original.corpus.root).encode(),
        "evaluator",
    )
    original.corpus.transition(marker.id, "RUNNING", "FINALIZING")
    original.corpus.transition(marker.id, "COMPLETED", None)
    controller = HoldoutController(
        original.service.store,
        original.corpus,
        binding=binding,
        _schedule_verifier=original._schedule_verifier,
    )
    batch = controller.prepare()
    mapping = original.corpus.load(batch.evaluator_run_id)
    mapping_ref = next(
        ref for ref in mapping.artifact_refs if ref.name == "holdout/private-alias-map.json"
    )
    nonce_canary = json.loads(original.corpus.read(mapping_ref))["nonce_hex"].encode()
    holdout_service = ApplicationService(
        original.corpus,
        original.corpus,
        backend_factory=original.service._backend_factory,
        knowledge=original.service.knowledge,
        knowledge_version=original.service.knowledge_version,
        _binding=binding,
    )
    holdout_service._pricing_attestation = original.service._pricing_attestation
    executor = EvaluationExecutor(
        original.service,
        original.corpus,
        original.sources,
        holdout_service=holdout_service,
        holdout_controller=controller,
        holdout_batch=batch,
        _corpus_family=original._corpus_family,
        _schedule_verifier=original._schedule_verifier,
    )
    unrelated = original.corpus.root / ("f" * 32)
    unrelated.mkdir(mode=0o700)
    result = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=1,
        max_unit_cost_usd=1,
        random_seed=7,
        holdout_controller=controller,
        holdout_batch=batch,
    ).run("E", "holdout", 3)
    assert result.executed_units == 1
    assert result.records[0].usage["physical_calls"] > 0
    assert result.records[0].lineage.candidate_hash is not None
    assert result.records[0].lineage.verification_hash is not None
    assert result.records[0].diagnosis == {}
    public_bytes = b"\n".join(
        path.read_bytes()
        for path in sorted(executor.service.store.root.rglob("*"))
        if path.is_file()
    )
    evaluator_bytes = b"\n".join(
        path.read_bytes() for path in sorted(executor.corpus.root.rglob("*")) if path.is_file()
    )

    def assert_evaluator_only(label, secret):
        if secret not in evaluator_bytes:
            pytest.fail(f"{label} canary missing from evaluator storage", pytrace=False)
        if secret in public_bytes:
            pytest.fail(f"{label} canary crossed the public boundary", pytrace=False)

    for label, secret in (
        ("source", source_canary),
        ("provider", provider_canary),
        ("candidate", candidate_canary),
        ("verification", verification_canary),
        ("label", label_canary),
        ("nonce", nonce_canary),
        ("path", path_canary),
        ("evaluator path", str(executor.corpus.root).encode()),
    ):
        assert_evaluator_only(label, secret)


@pytest.mark.parametrize("native_evaluation_executor", ["private_split"], indirect=True)
def test_holdout_mode_e_provider_limitation_is_evaluator_only(
    native_evaluation_executor, monkeypatch
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.benchmark.holdout import HoldoutController
    from gpu_agent.service import ApplicationService

    original = native_evaluation_executor
    limitation_canary = b"LIMITATION_SECRET_CANARY_TASK3"
    original.service._provider.limitation_canary = limitation_canary.decode()
    original.service._provider.force_limitation = True
    binding, _ = _configure_responses_provider(original, monkeypatch, full_script=True)
    original.service.evaluator_store = original.corpus
    controller = HoldoutController(
        original.service.store,
        original.corpus,
        binding=binding,
        _schedule_verifier=original._schedule_verifier,
    )
    batch = controller.prepare()
    holdout_service = ApplicationService(
        original.corpus,
        original.corpus,
        backend_factory=original.service._backend_factory,
        knowledge=original.service.knowledge,
        knowledge_version=original.service.knowledge_version,
        _binding=binding,
    )
    holdout_service._pricing_attestation = original.service._pricing_attestation
    executor = EvaluationExecutor(
        original.service,
        original.corpus,
        original.sources,
        holdout_service=holdout_service,
        holdout_controller=controller,
        holdout_batch=batch,
        _corpus_family=original._corpus_family,
        _schedule_verifier=original._schedule_verifier,
    )
    result = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash or "",
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=1,
        max_unit_cost_usd=1,
        random_seed=7,
        holdout_controller=controller,
        holdout_batch=batch,
    ).run("E", "holdout", 3)
    assert result.executed_units == 1
    assert result.records[0].usage["physical_calls"] > 0
    assert result.records[0].failure_reason is None
    public_bytes = b"\n".join(
        path.read_bytes()
        for path in sorted(executor.service.store.root.rglob("*"))
        if path.is_file()
    )
    evaluator_bytes = b"\n".join(
        path.read_bytes() for path in sorted(executor.corpus.root.rglob("*")) if path.is_file()
    )
    if limitation_canary in public_bytes:
        pytest.fail("limitation canary crossed the public boundary", pytrace=False)
    if limitation_canary not in evaluator_bytes:
        pytest.fail("limitation canary missing from evaluator storage", pytrace=False)


@pytest.mark.parametrize("native_evaluation_executor", ["private_split"], indirect=True)
def test_holdout_mode_e_started_provider_call_is_not_reexecuted_on_resume(
    native_evaluation_executor, monkeypatch
):
    import gpu_agent.service as service_module
    from gpu_agent.agent.provider import Invocation
    from gpu_agent.benchmark.evaluation import EvaluationAttempt, EvaluationRunner
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.benchmark.holdout import HoldoutController
    from gpu_agent.service import ApplicationService

    original = native_evaluation_executor
    binding, _ = _configure_responses_provider(original, monkeypatch, full_script=True)
    original.service.evaluator_store = original.corpus
    controller = HoldoutController(
        original.service.store,
        original.corpus,
        binding=binding,
        _schedule_verifier=original._schedule_verifier,
    )
    batch = controller.prepare()
    holdout_service = ApplicationService(
        original.corpus,
        original.corpus,
        backend_factory=original.service._backend_factory,
        knowledge=original.service.knowledge,
        knowledge_version=original.service.knowledge_version,
        _binding=binding,
    )
    holdout_service._pricing_attestation = original.service._pricing_attestation
    executor = EvaluationExecutor(
        original.service,
        original.corpus,
        original.sources,
        holdout_service=holdout_service,
        holdout_controller=controller,
        holdout_batch=batch,
        _corpus_family=original._corpus_family,
        _schedule_verifier=original._schedule_verifier,
    )
    provider_factory = service_module.OpenAIResponsesProvider
    physical_calls = []

    def interrupting_factory(*args, **kwargs):
        provider = provider_factory(*args, **kwargs)

        def interrupt(request):
            physical_calls.append(request.client_request_id)
            raise KeyboardInterrupt

        provider._port.call = interrupt
        return provider

    monkeypatch.setattr(service_module, "OpenAIResponsesProvider", interrupting_factory)
    reserved_calls = []
    diagnose_reserved = ApplicationService._diagnose_reserved

    def capture_reserved(service, *args, **kwargs):
        reserved_calls.append((service, args, dict(kwargs)))
        return diagnose_reserved(service, *args, **kwargs)

    monkeypatch.setattr(ApplicationService, "_diagnose_reserved", capture_reserved)

    def runner():
        return EvaluationRunner(
            executor.service.store,
            executor,
            schedule_client=schedule_client_for_test(executor),
            commit=binding.repository.commit,
            prompt_version=binding.prompt_version or "",
            toolchain_hash=binding.toolchain_lock_hash or "",
            model_config_hash=binding.model_config_hash or "",
            binding=binding,
            max_cost_usd=1,
            max_unit_cost_usd=1,
            random_seed=7,
            holdout_controller=controller,
            holdout_batch=batch,
        )

    with pytest.raises(KeyboardInterrupt):
        runner().run("E", "holdout", 3)
    assert len(reserved_calls) == 1
    called_service, replay_args, replay_kwargs = reserved_calls[0]
    before_replay = list(physical_calls)
    with pytest.raises(ValueError, match="startable"):
        diagnose_reserved(called_service, *replay_args, **replay_kwargs)
    assert physical_calls == before_replay
    run_id = executor.service.store.recoverable_runs()[0].id
    evaluator_started = [
        Invocation.model_validate_json(holdout_service.store.read(ref))
        for run in holdout_service.store.recoverable_runs()
        if run.kind == "diagnosis"
        for ref in run.artifact_refs
        if ref.name.endswith("/STARTED.json")
    ]
    assert len(evaluator_started) == 1
    public_run = executor.service.store.load(run_id)
    attempt_ref = next(
        ref for ref in public_run.artifact_refs if ref.name == "evaluation/attempts/0.json"
    )
    attempt = EvaluationAttempt.model_validate_json(executor.service.store.read(attempt_ref))
    expected_request_id = hashlib.sha256(
        f"{attempt.idempotency_key}:0:plan:0".encode()
    ).hexdigest()[:32]
    assert evaluator_started[0].client_request_id == expected_request_id
    before = list(physical_calls)
    resumed = runner().resume(run_id, "E", "holdout", 3)
    assert resumed.stopped_reason == "AMBIGUOUS_STARTED_ATTEMPT"
    assert resumed.executed_units == 0
    assert physical_calls == before and len(before) == 1


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("input_hash", "f" * 64),
        ("status", "COMPLETED"),
        ("usage", {"physical_calls": 99}),
        ("latency_ms", 999.0),
        ("cost_usd", 99.0),
        ("failure_reason", "FORGED_FAILURE"),
        ("verdict", "FORGED_VERDICT"),
        ("regression_detected", True),
    ],
)
@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_task2_holdout_completion_rejects_forged_native_summary(
    prepared_holdout_execution, native_evaluation_executor, field, replacement
):
    case = prepared_holdout_execution
    forged = case.native.model_copy(update={field: replacement})
    with pytest.raises(ValueError):
        case.controller.complete_execution(case.prepared, forged)


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_task2_holdout_completion_rejects_dangling_provider_started(
    prepared_holdout_execution, native_evaluation_executor
):
    from gpu_agent.agent.provider import Invocation

    case = prepared_holdout_execution
    invocation = Invocation(
        invocation_id="f" * 32,
        run_id=case.prepared.diagnosis_run_id,
        kind="plan",
        attempt=0,
        state="STARTED",
        started_at=case.evaluator.load(case.prepared.diagnosis_run_id).events[0].at,
        configured_model="forged-model",
        endpoint_host="example.invalid",
        client_request_id="e" * 32,
    )
    _inject_terminal_artifact(
        case.evaluator,
        case.prepared.diagnosis_run_id,
        f"provider/{'f' * 32}/STARTED.json",
        invocation.model_dump_json().encode(),
    )
    with pytest.raises(ValueError):
        case.controller.complete_execution(case.prepared, case.native)


@pytest.mark.parametrize(
    "fault,error",
    [
        ("candidate_payload", "evaluation candidate lineage is invalid"),
        ("candidate_status", "evaluation candidate lineage is invalid"),
        ("candidate_binding", "evaluation candidate lineage is invalid"),
        ("candidate_origin", "evaluation candidate topology is invalid"),
        ("candidate_parent", "evaluation candidate provenance is invalid"),
        ("candidate_base_source_hash", "evaluation candidate provenance is invalid"),
        ("verification_payload", "evaluation verification lineage is invalid"),
        ("verification_status", "evaluation verification lineage is invalid"),
        ("verification_binding", "evaluation verification lineage is invalid"),
        ("verification_origin", "evaluation verification topology is invalid"),
    ],
)
@pytest.mark.parametrize("native_evaluation_executor", ["private_exact"], indirect=True)
def test_task2_holdout_completion_rejects_untrusted_child_artifacts(
    prepared_holdout_repair_execution, native_evaluation_executor, fault, error
):
    case = prepared_holdout_repair_execution
    candidate_id = case.native.lineage.candidate_run_id
    verification_id = case.native.lineage.verification_run_id
    assert candidate_id is not None and verification_id is not None
    native = case.native
    if fault == "candidate_payload":
        _rewrite_artifact(
            case.evaluator,
            candidate_id,
            "candidate.json",
            lambda value: value.update(provider="forged-provider"),
        )
    elif fault == "candidate_base_source_hash":
        _rewrite_artifact(
            case.evaluator,
            candidate_id,
            "candidate.json",
            lambda value: value.update(base_source_hash="f" * 64),
        )
    elif fault == "candidate_parent":
        _rewrite_artifact(
            case.evaluator,
            candidate_id,
            "candidate.json",
            lambda value: value.update(parent_run_id="f" * 32),
        )
    elif fault == "verification_payload":
        ref = _rewrite_artifact(
            case.evaluator,
            verification_id,
            "verification/result.json",
            lambda value: value.update(candidate_hash="f" * 64),
        )
        native = native.model_copy(
            update={
                "lineage": native.lineage.model_copy(
                    update={"public_verification_hash": ref.sha256}
                )
            }
        )
    if fault == "candidate_binding":
        _rewrite_manifest(case.evaluator, candidate_id, lambda value: value.update(binding=None))
    elif fault == "candidate_status":
        _rewrite_manifest(
            case.evaluator,
            candidate_id,
            lambda value: value.update(status="RUNNING", current_phase="FINALIZING"),
        )
    elif fault == "candidate_origin":
        _rewrite_manifest(
            case.evaluator, candidate_id, lambda value: value.update(external_origin=None)
        )
    if fault == "verification_binding":
        _rewrite_manifest(case.evaluator, verification_id, lambda value: value.update(binding=None))
    elif fault == "verification_status":
        _rewrite_manifest(
            case.evaluator,
            verification_id,
            lambda value: value.update(status="RUNNING", current_phase="FINALIZING"),
        )
    elif fault == "verification_origin":
        _rewrite_manifest(
            case.evaluator, verification_id, lambda value: value.update(external_origin=None)
        )
    with pytest.raises(ValueError, match=error):
        case.controller.complete_execution(case.prepared, native)


def _rewrite_manifest(store, run_id, change):
    path = store.root / run_id / "manifest.json"
    value = json.loads(path.read_bytes())
    change(value)
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _rewrite_artifact(store, run_id, name, change):
    run = store.load(run_id)
    ref = next(value for value in run.artifact_refs if value.name == name)
    path = store.root / ref.relative_path
    mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode | stat.S_IWUSR)
    value = json.loads(path.read_bytes())
    change(value)
    content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(content)
    path.chmod(mode)
    manifest_path = store.root / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for item in manifest["artifact_refs"]:
        if item["id"] == ref.id:
            item["sha256"] = hashlib.sha256(content).hexdigest()
            item["byte_count"] = len(content)
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    return store.load(run_id).artifact_refs[run.artifact_refs.index(ref)]


def _inject_terminal_artifact(store, run_id, name, content):
    artifact_id = "e" * 32
    artifact_path = store.root / run_id / "artifacts" / artifact_id
    artifact_path.write_bytes(content)
    artifact_path.chmod(0o400)
    manifest_path = store.root / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifact_refs"].append(
        {
            "id": artifact_id,
            "run_id": run_id,
            "name": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "visibility": store.visibility,
            "relative_path": f"{run_id}/artifacts/{artifact_id}",
            "byte_count": len(content),
        }
    )
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_task2_recovery_rejects_orphan_diagnosis(
    prepared_holdout_execution, native_evaluation_executor
):
    case = prepared_holdout_execution
    import shutil

    shutil.rmtree(case.evaluator.root / case.prepared.execution_run_id)
    assert (case.evaluator.root / case.prepared.diagnosis_run_id).exists()
    with pytest.raises(ValueError):
        case.controller.recover_execution(case.batch, case.item, case.attempt)


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_task2_recovery_rejects_broken_execution_symlink(
    prepared_holdout_execution, native_evaluation_executor
):
    case = prepared_holdout_execution
    import shutil

    execution_path = case.evaluator.root / case.prepared.execution_run_id
    shutil.rmtree(execution_path)
    execution_path.symlink_to(execution_path.with_name("missing-execution"))
    with pytest.raises(ValueError):
        case.controller.recover_execution(case.batch, case.item, case.attempt)


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_task2_reservation_does_not_repair_terminal_incomplete_execution(
    prepared_holdout_execution, native_evaluation_executor
):
    case = prepared_holdout_execution
    import shutil

    shutil.rmtree(case.evaluator.root / case.prepared.diagnosis_run_id)
    case.evaluator.transition(case.prepared.execution_run_id, "RUNNING", "FINALIZING")
    case.evaluator.transition(case.prepared.execution_run_id, "COMPLETED", None)
    with pytest.raises(ValueError):
        case.controller.reserve_execution(
            case.batch,
            evaluation_run_id=case.attempt.run_id,
            item=case.item,
            attempt=case.attempt,
        )
    assert not (case.evaluator.root / case.prepared.diagnosis_run_id).exists()


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_task2_reservation_finishes_exact_partial_running_reservation(
    prepared_holdout_execution, native_evaluation_executor
):
    case = prepared_holdout_execution
    import shutil

    shutil.rmtree(case.evaluator.root / case.prepared.diagnosis_run_id)
    assert case.evaluator.load(case.prepared.execution_run_id).status == "RUNNING"
    repeated = case.controller.reserve_execution(
        case.batch,
        evaluation_run_id=case.attempt.run_id,
        item=case.item,
        attempt=case.attempt,
    )
    assert repeated == case.prepared
    diagnosis = case.evaluator.load(case.prepared.diagnosis_run_id)
    assert diagnosis.parent_run_id == case.prepared.execution_run_id


@pytest.mark.parametrize("terminal", ["FAILED", "CANCELLED"])
@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_task2_reservation_rejects_failed_or_cancelled_parent_without_mutation(
    prepared_holdout_execution, native_evaluation_executor, terminal
):
    case = prepared_holdout_execution
    import shutil

    shutil.rmtree(case.evaluator.root / case.prepared.diagnosis_run_id)
    case.evaluator.transition(case.prepared.execution_run_id, terminal, None)
    with pytest.raises(ValueError):
        case.controller.reserve_execution(
            case.batch,
            evaluation_run_id=case.attempt.run_id,
            item=case.item,
            attempt=case.attempt,
        )
    assert not (case.evaluator.root / case.prepared.diagnosis_run_id).exists()


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_task2_recovery_rejects_non_directory_execution_path(
    prepared_holdout_execution, native_evaluation_executor
):
    case = prepared_holdout_execution
    import shutil

    execution_path = case.evaluator.root / case.prepared.execution_run_id
    shutil.rmtree(execution_path)
    execution_path.write_bytes(b"unsafe")
    with pytest.raises(ValueError):
        case.controller.recover_execution(case.batch, case.item, case.attempt)


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_task2_concurrent_same_attempt_reservation_returns_exact_winner(
    prepared_holdout_execution, native_evaluation_executor, monkeypatch
):
    import shutil
    import threading

    case = prepared_holdout_execution
    shutil.rmtree(case.evaluator.root / case.prepared.diagnosis_run_id)
    shutil.rmtree(case.evaluator.root / case.prepared.execution_run_id)
    original = case.controller._prepared_execution
    barrier = threading.Barrier(2)

    def synchronized_preparation(*args, **kwargs):
        prepared = original(*args, **kwargs)
        barrier.wait(timeout=5)
        return prepared

    monkeypatch.setattr(case.controller, "_prepared_execution", synchronized_preparation)

    def reserve():
        return case.controller.reserve_execution(
            case.batch,
            evaluation_run_id=case.attempt.run_id,
            item=case.item,
            attempt=case.attempt,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result() for future in [pool.submit(reserve), pool.submit(reserve)]]
    assert results == [case.prepared, case.prepared]


@pytest.mark.parametrize("operation", ["reserve", "recover"])
@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_task2_execution_directory_swap_between_probe_and_use_fails_closed(
    request,
    native_evaluation_executor,
    monkeypatch,
    operation,
):
    import shutil

    from gpu_agent.store import RunDirectorySetLease

    fixture = (
        "completed_holdout_execution" if operation == "recover" else "prepared_holdout_execution"
    )
    case = request.getfixturevalue(fixture)
    execution_path = case.evaluator.root / case.prepared.execution_run_id
    displaced = execution_path.with_name(f".{execution_path.name}-displaced")
    original = RunDirectorySetLease.load_optional
    swapped = False

    def swapping_load(lease, run_id):
        nonlocal swapped
        run = original(lease, run_id)
        if (
            lease.store is case.evaluator
            and run_id == case.prepared.execution_run_id
            and run is not None
            and not swapped
        ):
            swapped = True
            execution_path.rename(displaced)
            shutil.copytree(displaced, execution_path)
        return run

    monkeypatch.setattr(RunDirectorySetLease, "load_optional", swapping_load)
    with pytest.raises(ValueError):
        if operation == "reserve":
            case.controller.reserve_execution(
                case.batch,
                evaluation_run_id=case.attempt.run_id,
                item=case.item,
                attempt=case.attempt,
            )
        else:
            case.controller.recover_execution(case.batch, case.item, case.attempt)


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_task2_execution_claim_rejects_path_replacement_while_waiting(
    prepared_holdout_execution, native_evaluation_executor, monkeypatch
):
    import fcntl
    import os
    import threading

    import gpu_agent.benchmark.holdout as holdout_module

    case = prepared_holdout_execution
    execution_id = case.prepared.execution_run_id
    lock_path = case.evaluator.root / f".holdout-execution-{execution_id}.lock"
    original_flock = holdout_module.fcntl.flock
    old_fd = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
    original_flock(old_fd, fcntl.LOCK_EX)
    waiting = threading.Event()
    body_entered = threading.Event()
    replacement_fd = -1

    def observed_flock(fd, operation):
        waiting.set()
        return original_flock(fd, operation)

    def claim():
        with case.controller._execution_claim(execution_id):
            body_entered.set()

    monkeypatch.setattr(holdout_module.fcntl, "flock", observed_flock)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(claim)
            assert waiting.wait(timeout=5)
            lock_path.unlink()
            replacement_fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
            )
            original_flock(replacement_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            original_flock(old_fd, fcntl.LOCK_UN)
            with pytest.raises(ValueError):
                future.result(timeout=5)
        assert not body_entered.is_set()
    finally:
        original_flock(old_fd, fcntl.LOCK_UN)
        os.close(old_fd)
        if replacement_fd >= 0:
            original_flock(replacement_fd, fcntl.LOCK_UN)
            os.close(replacement_fd)


@pytest.mark.parametrize("mutation", ["chmod", "hardlink"])
@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_task2_execution_claim_rejects_post_acquire_metadata_mutation(
    prepared_holdout_execution, native_evaluation_executor, mutation
):
    import os

    case = prepared_holdout_execution
    execution_id = case.prepared.execution_run_id
    lock_path = case.evaluator.root / f".holdout-execution-{execution_id}.lock"
    linked_path = lock_path.with_suffix(".linked")
    body_entered = False

    with pytest.raises(ValueError):
        with case.controller._execution_claim(execution_id):
            body_entered = True
            if mutation == "chmod":
                lock_path.chmod(0o640)
            else:
                os.link(lock_path, linked_path)
    assert body_entered


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_holdout_public_lineage_contains_commitments_not_evaluator_run_ids(
    completed_holdout_execution,
    native_evaluation_executor,
):
    case = completed_holdout_execution
    wire = case.public.model_dump_json().encode()
    assert case.public.record_id == "6c89c3d549613dfa36f2dd40afc978cd"
    assert case.public.lineage.kind == "holdout_commitment"
    assert case.prepared.execution_run_id.encode() not in wire
    assert case.native.lineage.diagnosis_run_id.encode() not in wire
    assert b"case_0100" not in wire and b"vector-add" not in wire
    run = case.evaluator.load(case.prepared.execution_run_id)
    assert run.parent_run_id is None
    assert run.external_origin.run_id == case.attempt.run_id
    assert run.external_origin.visibility == "public"


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_holdout_execution_reservation_is_exact_and_attempt_scoped(
    completed_holdout_execution,
    native_evaluation_executor,
):
    case = completed_holdout_execution
    repeated = case.controller.reserve_execution(
        case.batch,
        evaluation_run_id=case.attempt.run_id,
        item=case.item,
        attempt=case.attempt,
    )
    other_attempt = case.attempt.model_copy(update={"idempotency_key": "d" * 64})
    different = case.controller.reserve_execution(
        case.batch,
        evaluation_run_id=case.attempt.run_id,
        item=case.item,
        attempt=other_attempt,
    )
    assert repeated == case.prepared
    assert different.execution_run_id != case.prepared.execution_run_id
    assert different.diagnosis_run_id != case.prepared.diagnosis_run_id
    with pytest.raises(ValueError):
        case.controller.recover_execution(case.batch, case.item, other_attempt)
    absent_attempt = case.attempt.model_copy(update={"idempotency_key": "e" * 64})
    assert case.controller.recover_execution(case.batch, case.item, absent_attempt) is None


@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_holdout_complete_execution_is_exact_byte_idempotent(
    completed_holdout_execution, native_evaluation_executor
):
    case = completed_holdout_execution
    repeated = case.controller.complete_execution(case.prepared, case.native)
    assert repeated.model_dump_json().encode() == case.public.model_dump_json().encode()
    with pytest.raises(ValueError):
        case.controller.complete_execution(
            case.prepared, case.native.model_copy(update={"latency_ms": 1})
        )


def _overwrite_holdout_binding(evaluator, run_id, field, replacement):
    run = evaluator.load(run_id)
    ref = next(item for item in run.artifact_refs if item.name == "holdout/execution-binding.json")
    artifact_path = evaluator.root / ref.relative_path
    manifest_path = evaluator.root / run_id / "manifest.json"
    artifact_mode = stat.S_IMODE(artifact_path.stat().st_mode)
    artifact_path.chmod(artifact_mode | stat.S_IWUSR)
    payload = json.loads(artifact_path.read_bytes())
    payload[field] = replacement
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    artifact_path.write_bytes(content)
    artifact_path.chmod(artifact_mode)
    manifest = json.loads(manifest_path.read_bytes())
    for item in manifest["artifact_refs"]:
        if item["id"] == ref.id:
            item["sha256"] = hashlib.sha256(content).hexdigest()
            item["byte_count"] = len(content)
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("ordinal", 2),
        ("schedule_hash", "f" * 64),
        ("attempt_hash", "e" * 64),
        ("corpus_cutoff", 99),
        ("public_record_hash", "d" * 64),
    ],
)
@pytest.mark.parametrize("native_evaluation_executor", ["private"], indirect=True)
def test_holdout_execution_binding_rejects_cross_store_tampering(
    completed_holdout_execution, field, replacement, native_evaluation_executor
):
    case = completed_holdout_execution
    _overwrite_holdout_binding(case.evaluator, case.prepared.execution_run_id, field, replacement)
    with pytest.raises(ValueError):
        case.controller.recover_execution(case.batch, case.item, case.attempt)
