import copy

import pytest

from tools.dev_failure_inventory import InventoryError
from tools.investigation_witness import transition


def bundle(tool="memcheck", outcome="CLEAN"):
    return {
        "sanitizer_results": [
            {
                "tool_result": {"typed_payload": {"tool": tool}},
                "completed": True,
                "check_outcome": outcome,
            }
        ],
        "retrieved_chunks": [],
    }


def test_selected_tool_requires_completed_native_result():
    before = bundle()
    after = copy.deepcopy(before)
    after["sanitizer_results"] += bundle("initcheck", "FINDING")["sanitizer_results"]
    assert transition({"action_type": "run_initcheck"}, before, after) == {
        "action": "run_initcheck",
        "actual_outcome": "FINDING",
    }


@pytest.mark.parametrize("change", ["absent", "wrong_tool", "incomplete", "changed_history"])
def test_allowed_proposal_alone_is_not_execution(change):
    before = bundle()
    after = copy.deepcopy(before)
    after["sanitizer_results"] += bundle("initcheck", "FINDING")["sanitizer_results"]
    if change == "absent":
        after = before
    elif change == "wrong_tool":
        after["sanitizer_results"][-1]["tool_result"]["typed_payload"]["tool"] = "racecheck"
    elif change == "incomplete":
        after["sanitizer_results"][-1]["completed"] = False
    else:
        after["sanitizer_results"][0]["check_outcome"] = "FINDING"
    with pytest.raises(InventoryError):
        transition({"action_type": "run_initcheck"}, before, after)


def test_empty_retrieval_cannot_be_claimed_successful():
    with pytest.raises(InventoryError):
        transition({"action_type": "retrieve_official_docs"}, bundle(), bundle())


def test_e_can_stop_without_fabricating_diagnosis(oob_service):
    from gpu_agent.agent.models import InconclusiveAction

    service, provider, source = oob_service
    provider.actions = [InconclusiveAction()]
    run = service.diagnose(source)
    result = service.diagnosis(run.id)
    assert result.diagnostic_outcome == "INCONCLUSIVE"
    assert result.limitations == ["MODEL_DECLARED_INCONCLUSIVE"]
    assert provider.kinds == ["plan"]
    assert not service.candidates(run.id)


@pytest.mark.parametrize("clean_first", [False, True])
def test_e_dispatch_depends_on_planner_evidence_not_rule_router(
    oob_service, monkeypatch, clean_first
):
    from gpu_agent.agent.models import (
        FinishAction,
        InitcheckAction,
        MemcheckAction,
        RetrieveDocsAction,
    )
    from gpu_agent.agent.rule_router import RuleRouter
    from gpu_agent.execution.process import ProcessCapture

    service, provider, source = oob_service
    calls = []

    class Backend(service._backend_factory):
        def _container(self, path, operation, timeout, *, stdin=b"", cancel=None):
            if operation in {"memcheck", "initcheck"}:
                calls.append(operation)
            if operation == "memcheck" and clean_first:
                return (
                    ProcessCapture(0, b"{}", b"", False),
                    b"",
                    b"========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n",
                )
            if operation == "initcheck":
                log = (
                    b"========= COMPUTE-SANITIZER\n"
                    b"========= Uninitialized __global__ memory read of size 4 bytes\n"
                    b"=========     at vector_add in /input/kernel.cu:9\n"
                    b"========= ERROR SUMMARY: 1 error\n"
                )
                return ProcessCapture(86, b"{}", b"", False), b"", log
            return super()._container(path, operation, timeout, stdin=stdin, cancel=cancel)

    service._backend_factory = Backend

    def plan(evidence, budget, feedback=None, state=None):
        provider._record("plan", {"evidence": evidence.model_dump(mode="json")})
        if "memcheck" not in evidence.sanitizer_outcomes:
            return MemcheckAction()
        if not evidence.tool_findings:
            return InitcheckAction()
        if not evidence.documentation:
            return RetrieveDocsAction(typed_arguments={"query": "memory read", "k": 3})
        return FinishAction()

    monkeypatch.setattr(provider, "plan", plan)
    monkeypatch.setattr(RuleRouter, "next_action", lambda *a: pytest.fail("E used fixed router"))
    run = service.diagnose(source, mode="E")
    assert calls == (["memcheck", "initcheck"] if clean_first else ["memcheck"])
    assert service.diagnosis(run.id).diagnostic_outcome == "DIAGNOSED"
