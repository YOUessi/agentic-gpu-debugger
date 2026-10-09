"""Public candidate investigations retain one task's limits and fresh evidence."""

import json

import pytest

from gpu_agent.agent.models import (
    AgentBudget,
    DiagnosisResult,
    EvidenceClaim,
    FinishAction,
    InconclusiveAction,
    InspectSourceAction,
    MemcheckAction,
    PublicEvidence,
    PublicFinding,
    PublicRepairContext,
    PublicSource,
    RetrieveDocsAction,
)
from gpu_agent.agent.orchestrator import AgentOrchestrator, public_evidence
from gpu_agent.agent.policy import (
    BudgetLedger,
    LLMCallGate,
    action_policy_for_prompt,
    decide_action,
    missing_evidence,
)
from gpu_agent.agent.provider import FakeProvider, ProviderError
from gpu_agent.contracts import CurrentPhase
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import BuildRequest, WorkspaceRequest
from gpu_agent.store import RunStore


def _workspace(store, directory, run, *, source=b"int original;\n", factory=IsolatedGPUBackend):
    root = directory / run.id
    root.mkdir()
    (root / "kernel.cu").write_bytes(source)
    source_ref = store.put(run.id, "sources/kernel.cu", source, store.visibility)
    backend = factory(store, root, root / "tasks")
    handle = backend.prepare(
        WorkspaceRequest(
            run_id=run.id,
            source_manifest={"kernel.cu": source_ref.sha256},
            trust_level="UNTRUSTED",
        )
    )
    stdin_ref = store.put(run.id, "public-input.json", b"{}", store.visibility)
    store.put(
        run.id,
        "agent/acquisition-policy.json",
        b'{"mode":"E","required_tools":["memcheck"]}',
        store.visibility,
    )
    return backend, handle, stdin_ref


@pytest.fixture
def continuation_root(store, tmp_path):
    run = store.create_run("diagnosis")
    backend, handle, stdin_ref = _workspace(store, tmp_path, run)
    return AgentOrchestrator(
        store,
        FakeProvider([], DiagnosisResult.inconclusive("TEST"), ""),
        backend,
        handle,
        stdin_ref,
        None,
        "cuda=13.0;compute-sanitizer=13.0",
    )


def test_repair_cycle_reserves_final_calls_without_resetting_total_usage():
    gate = LLMCallGate(AgentBudget(max_llm_calls=6))
    gate.reserve("diagnose")
    gate.reserve("patch")
    gate.reserve("plan")
    gate.reserve("plan")

    gate.begin_repair_cycle()

    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        gate.reserve("plan")
    gate.reserve("diagnose")
    gate.reserve("patch")
    assert gate.snapshot().llm_calls == 6
    gate.begin_repair_cycle()
    for kind in ("plan", "diagnose", "patch"):
        with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
            gate.reserve(kind)
    assert gate.snapshot().llm_calls == 6


@pytest.mark.parametrize("kind", ["plan", "diagnose", "patch"])
def test_repair_cycle_keeps_used_format_retries(kind):
    gate = LLMCallGate()
    gate.reserve(kind)
    gate.reserve(kind, attempt=1)

    gate.begin_repair_cycle()

    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        gate.reserve(kind, attempt=1)
    gate.reserve(kind)
    assert gate.snapshot().llm_calls == 3


def test_repair_cycle_keeps_the_original_deadline():
    now = [100.0]
    gate = LLMCallGate(AgentBudget(max_wall_time_seconds=10), clock=lambda: now[0])
    gate.reserve("diagnose")
    gate.reserve("patch")
    now[0] = 109.0

    gate.begin_repair_cycle()

    assert gate.timeout(120) == 1
    now[0] = 110.0
    gate.begin_repair_cycle()
    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        gate.reserve("diagnose")
    assert gate.snapshot().remaining_seconds == 0
    assert gate.snapshot().llm_calls == 2


def test_repair_prompt_replay_enforces_full_source_action_policy():
    evidence = PublicEvidence(sources=[PublicSource(source_id="a" * 32, content="int x;\n")])
    action = InspectSourceAction(
        typed_arguments={"source_id": "a" * 32, "start_line": 1, "end_line": 1}
    )

    decision = decide_action(
        action,
        evidence,
        AgentBudget(),
        CurrentPhase.DIAGNOSING,
        set(),
        policy_version=action_policy_for_prompt("public-repair-v3-2026-10-08-v1"),
    )

    assert not decision.allowed
    assert decision.reason_codes == ["SOURCE_ALREADY_AVAILABLE"]


def test_continuation_repeats_acquisition_on_fresh_evidence_with_cumulative_usage(
    oob_service, tmp_path
):
    service, _, _ = oob_service
    store = service.store
    run = store.create_run("diagnosis")
    backend, handle, stdin_ref = _workspace(store, tmp_path, run, factory=service._backend_factory)
    backend.build(BuildRequest(workspace_id=handle.id))
    actions = [
        MemcheckAction(),
        RetrieveDocsAction(typed_arguments={"query": "out of bounds", "k": 3}),
        InconclusiveAction(),
    ]
    provider = FakeProvider([*actions, *actions], DiagnosisResult.inconclusive("TEST"), "")
    root = AgentOrchestrator(
        store, provider, backend, handle, stdin_ref, service.knowledge, service.knowledge_version
    )
    root.investigate(run.id)
    assert root.budget.agent_steps == 3
    assert root.budget.llm_calls == 3
    assert root.budget.sanitizer_calls == root.acquisition_usage.sanitizer_calls == 1
    assert root.budget.rag_calls == root.acquisition_usage.retrieval_calls == 1

    child_run = store.create_run("repair-investigation", parent_run_id=run.id)
    child_backend, child_handle, child_stdin = _workspace(
        store,
        tmp_path,
        child_run,
        source=b"int candidate;\n",
        factory=service._backend_factory,
    )
    child_backend.build(BuildRequest(workspace_id=child_handle.id))
    child = root.continue_in_workspace(child_backend, child_handle, child_stdin)
    assert child.ledger is root.ledger
    assert child.provider is root.provider
    assert child.budget == root.budget
    assert child.budget is not root.budget
    assert child.acquisition_usage == root.acquisition_usage
    assert child.seen == set() and child.executed == []
    assert len(root.seen) == len(root.executed) == 3

    result = child.investigate(child_run.id)

    assert result.limitations == ["MODEL_DECLARED_INCONCLUSIVE"]
    assert child.budget.agent_steps == 6
    assert child.budget.llm_calls == 6
    assert child.budget.sanitizer_calls == child.acquisition_usage.sanitizer_calls == 2
    assert child.budget.rag_calls == child.acquisition_usage.retrieval_calls == 2
    initial_child_payload = provider.inputs[3]
    assert initial_child_payload["evidence"]["sources"][0]["content"] == "int candidate;\n"
    assert initial_child_payload["evidence"]["tool_findings"] == []
    assert initial_child_payload["evidence"]["documentation"] == []
    assert initial_child_payload["controller_state"]["executed_actions"] == []
    parent_ids = {s.source_id for s in public_evidence(store, run.id).sources}
    assert not parent_ids & {s.source_id for s in public_evidence(store, child_run.id).sources}
    final_ref = next(
        ref for ref in store.load(child_run.id).artifact_refs if ref.name == "agent/budget.json"
    )
    assert json.loads(store.read(final_ref))["llm_calls"] == 6


def test_continuation_reuses_physical_tool_reservations(continuation_root, tmp_path):
    root = continuation_root
    root.budget = AgentBudget(max_sanitizer_calls=1)
    root.ledger = BudgetLedger(root.budget)
    reservation = root.ledger.reserve("run_memcheck")
    root.ledger.settle(reservation, "FAILED")
    run = root.store.create_run("repair-investigation", parent_run_id=root.handle.run_id)
    backend, handle, stdin_ref = _workspace(root.store, tmp_path, run)

    child = root.continue_in_workspace(backend, handle, stdin_ref)

    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        child._reserve("run_memcheck")
    assert root.ledger.audit[-1]["reason"] == "SANITIZER_BUDGET_EXHAUSTED"


def test_continuation_includes_patch_calls_since_the_previous_investigation(
    continuation_root, tmp_path
):
    root = continuation_root
    root.provider.gate.reserve("diagnose")
    root.provider.gate.reserve("patch")
    run = root.store.create_run("repair-investigation", parent_run_id=root.handle.run_id)

    child = root.continue_in_workspace(*_workspace(root.store, tmp_path, run))

    assert child.budget.llm_calls == 2
    assert child.provider.gate.snapshot().llm_calls == 2


def test_continuation_keeps_ledger_deadline(continuation_root, tmp_path):
    root = continuation_root
    now = [10.0]
    root.ledger = BudgetLedger(AgentBudget(max_wall_time_seconds=10), clock=lambda: now[0])
    run = root.store.create_run("repair-investigation", parent_run_id=root.handle.run_id)
    backend, handle, stdin_ref = _workspace(root.store, tmp_path, run)
    now[0] = 19.0

    child = root.continue_in_workspace(backend, handle, stdin_ref)
    now[0] = 20.0

    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        child._reserve("retrieve_official_docs")
    assert root.ledger.audit[-1]["reason"] == "WALL_TIME_EXHAUSTED"


def test_continuation_cannot_escape_agent_step_limit(continuation_root, tmp_path):
    root = continuation_root
    root.budget = AgentBudget(max_agent_steps=2, agent_steps=2)
    run = root.store.create_run("repair-investigation", parent_run_id=root.handle.run_id)
    backend, handle, stdin_ref = _workspace(root.store, tmp_path, run)

    child = root.continue_in_workspace(backend, handle, stdin_ref)
    result = child.investigate(run.id)

    assert result.limitations == ["AGENT_BUDGET_EXHAUSTED"]
    assert child.budget.agent_steps == 2
    assert child.provider.gate.snapshot().llm_calls == 0


def test_continuations_remain_siblings_of_the_original_run(continuation_root, tmp_path):
    root = continuation_root
    first_run = root.store.create_run("repair-investigation", parent_run_id=root.handle.run_id)
    first = root.continue_in_workspace(*_workspace(root.store, tmp_path, first_run))
    second_run = root.store.create_run("repair-investigation", parent_run_id=root.handle.run_id)

    second = first.continue_in_workspace(*_workspace(root.store, tmp_path, second_run))

    assert second.ledger is first.ledger is root.ledger
    with pytest.raises(ValueError, match="already used|same run"):
        root.continue_in_workspace(first.backend, first.handle, first.stdin_ref)
    with pytest.raises(ValueError, match="already used|same run"):
        first.continue_in_workspace(second.backend, second.handle, second.stdin_ref)


def test_continuation_rejects_original_run_reuse(continuation_root):
    root = continuation_root
    with pytest.raises(ValueError, match="already used|same run"):
        root.continue_in_workspace(root.backend, root.handle, root.stdin_ref)


@pytest.mark.parametrize("parent", ["none", "unrelated"])
def test_continuation_requires_original_root_lineage(continuation_root, tmp_path, parent):
    root = continuation_root
    parent_id = None if parent == "none" else root.store.create_run("diagnosis").id
    run = root.store.create_run("repair-investigation", parent_run_id=parent_id)
    backend, handle, stdin_ref = _workspace(root.store, tmp_path, run)
    root.provider.gate = LLMCallGate(AgentBudget(max_llm_calls=3))
    root.provider.gate.reserve("diagnose")
    root.provider.gate.reserve("patch")

    with pytest.raises(ValueError, match="parent|root"):
        root.continue_in_workspace(backend, handle, stdin_ref)

    # A rejected continuation must not reserve new final calls on the live gate.
    root.provider.gate.reserve("plan")
    assert root.provider.gate.snapshot().llm_calls == 3


def test_continuation_rejects_cross_run_input(continuation_root, tmp_path):
    root = continuation_root
    run = root.store.create_run("repair-investigation", parent_run_id=root.handle.run_id)
    backend, handle, _ = _workspace(root.store, tmp_path, run)

    with pytest.raises(ValueError, match="input|stdin|another run"):
        root.continue_in_workspace(backend, handle, root.stdin_ref)


def test_continuation_rejects_unregistered_input(continuation_root, tmp_path):
    root = continuation_root
    run = root.store.create_run("repair-investigation", parent_run_id=root.handle.run_id)
    backend, handle, stdin_ref = _workspace(root.store, tmp_path, run)
    forged = stdin_ref.model_copy(
        update={"id": "f" * 32, "relative_path": f"{run.id}/artifacts/{'f' * 32}"}
    )

    with pytest.raises(ValueError, match="unregistered"):
        root.continue_in_workspace(backend, handle, forged)


def test_continuation_rejects_backend_from_another_store(continuation_root, tmp_path):
    root = continuation_root
    run = root.store.create_run("repair-investigation", parent_run_id=root.handle.run_id)
    _, handle, stdin_ref = _workspace(root.store, tmp_path, run)
    other_store = RunStore(tmp_path / "other-store")
    foreign_backend = IsolatedGPUBackend(other_store, tmp_path, tmp_path / "foreign-tasks")

    with pytest.raises(ValueError, match="same.*store|another store"):
        root.continue_in_workspace(foreign_backend, handle, stdin_ref)


def test_continuation_rejects_evaluator_store(tmp_path):
    store = RunStore(tmp_path / "private-runs", visibility="evaluator")
    run = store.create_run("diagnosis")
    backend, handle, stdin_ref = _workspace(store, tmp_path, run)
    root = AgentOrchestrator(
        store,
        FakeProvider([], DiagnosisResult.inconclusive("TEST"), ""),
        backend,
        handle,
        stdin_ref,
        None,
        "cuda=13.0;compute-sanitizer=13.0",
    )

    with pytest.raises(ValueError, match="public store"):
        root.continue_in_workspace(backend, handle, stdin_ref)


@pytest.mark.parametrize(
    "functional_failure,memcheck,with_finding,expected_missing",
    [
        (None, "CLEAN", False, ["tool_finding"]),
        (False, "CLEAN", False, ["tool_finding"]),
        (True, "CLEAN", False, []),
        (True, None, False, ["memcheck_outcome"]),
        (True, "FINDING", True, ["documentation_for_finding"]),
    ],
)
def test_only_current_public_functional_failure_relaxes_the_tool_finding_requirement(
    functional_failure, memcheck, with_finding, expected_missing
):
    context = (
        None
        if functional_failure is None
        else PublicRepairContext(
            repair_round=1,
            original_source_sha256="a" * 64,
            candidate_source_sha256="b" * 64,
            previous_diagnosis_source_sha256="a" * 64,
            previous_diagnosis=DiagnosisResult.inconclusive("PRIOR"),
            public_checks={"functional_output": "FAILED"},
            public_feedback=[{"check": "functional_output", "reason": "value mismatch"}],
            public_functional_failure=functional_failure,
        )
    )
    evidence = PublicEvidence(
        sources=[PublicSource(source_id="a" * 32, content="int x;\n")],
        observed_facts=[
            EvidenceClaim(
                text="Current public output has incorrect values.", citation_ids=["b" * 32]
            )
        ],
        tool_findings=(
            [PublicFinding(artifact_id="c" * 32, category="Invalid __global__ read")]
            if with_finding
            else []
        ),
        sanitizer_outcomes={"memcheck": memcheck} if memcheck is not None else {},
        repair_context=context,
    )

    missing = missing_evidence(evidence)
    decision = decide_action(
        FinishAction(), evidence, AgentBudget(), CurrentPhase.DIAGNOSING, set()
    )

    assert missing == expected_missing
    assert decision.allowed is (not expected_missing)
    assert decision.reason_codes == ([] if not expected_missing else ["MANDATORY_EVIDENCE_MISSING"])
