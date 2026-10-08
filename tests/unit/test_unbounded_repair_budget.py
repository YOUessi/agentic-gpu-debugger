"""Opt-in no separate sanitizer cap for public Repair v3, preserving normal limits."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from gpu_agent.agent.models import (
    AcquisitionUsage,
    AgentBudget,
    PublicEvidence,
    RacecheckAction,
)
from gpu_agent.agent.policy import BudgetExceeded, BudgetLedger, decide_action
from gpu_agent.cli import app
from gpu_agent.contracts import CurrentPhase
from gpu_agent.public_task import PublicTask
from gpu_agent.repair import RepairPolicy


def test_default_budget_unchanged_and_uncapped_usage_serializable():
    default = AgentBudget()
    assert default.max_sanitizer_calls == 4
    assert RepairPolicy().unbounded_sanitizer_calls is False
    assert AgentBudget.model_validate_json(default.model_dump_json()) == default
    unbounded = AgentBudget(max_sanitizer_calls=None, sanitizer_calls=7)
    assert AgentBudget.model_validate_json(unbounded.model_dump_json()) == unbounded
    assert AcquisitionUsage(sanitizer_calls=7, retrieval_calls=1).sanitizer_calls == 7


def test_no_separate_sanitizer_cap_still_keeps_other_budgets():
    ledger = BudgetLedger(AgentBudget(max_sanitizer_calls=None))
    for _ in range(7):
        reservation = ledger.reserve("run_racecheck")
        ledger.settle(reservation)
    assert sum(
        item["state"] == "COMPLETED" and item["action"] == "run_racecheck"
        for item in ledger.audit
    ) == 7
    bounded = BudgetLedger()
    for _ in range(4):
        bounded.settle(bounded.reserve("run_racecheck"))
    with pytest.raises(BudgetExceeded, match="SANITIZER_BUDGET_EXHAUSTED"):
        bounded.reserve("run_racecheck")


def test_policy_keeps_default_cap_and_unbounded_action_allowed():
    evidence = PublicEvidence(sanitizer_outcomes={"memcheck": "CLEAN"})
    action = RacecheckAction()
    bounded = decide_action(
        action, evidence,
        AgentBudget(sanitizer_calls=4),
        CurrentPhase.DIAGNOSING, set(),
    )
    unbounded = decide_action(
        action, evidence,
        AgentBudget(max_sanitizer_calls=None, sanitizer_calls=7),
        CurrentPhase.DIAGNOSING, set(),
    )
    assert bounded.reason_codes == ["AGENT_BUDGET_EXHAUSTED"]
    assert unbounded.allowed


@pytest.mark.parametrize("unbounded", [False, True])
def test_service_propagates_budget_only_when_opted_in(
    oob_service, monkeypatch, unbounded
):
    from gpu_agent import service as service_module

    service, _, source = oob_service
    kernel = (source / "kernel.cu").read_bytes()
    (source / "task.json").write_text(
        PublicTask(
            source_sha256=hashlib.sha256(kernel).hexdigest(),
            algorithm="vector-add-cpu-v1",
        ).model_dump_json()
    )
    original = service_module.AgentOrchestrator
    budgets = []

    def spy(*args, **kwargs):
        budgets.append(kwargs.get("budget"))
        return original(*args, **kwargs)

    monkeypatch.setattr(service_module, "AgentOrchestrator", spy)
    monkeypatch.setattr(
        service_module, "repair_candidates",
        lambda _store, _snapshot, first, *_args, **_kwargs: first,
    )
    service.repair(
        source,
        policy=RepairPolicy(
            version="public-repair-v3",
            unbounded_sanitizer_calls=unbounded,
        ),
    )
    assert len(budgets) == 1
    if unbounded:
        assert budgets[0] is not None and budgets[0].max_sanitizer_calls is None
    else:
        assert budgets[0] is None


def test_cli_explicitly_requires_v3_for_unbounded_mode(monkeypatch):
    from gpu_agent.service import ApplicationService

    policies = []

    class Stub:
        def repair(self, source, policy):
            policies.append(policy)
            return SimpleNamespace(id="a" * 32, artifact_refs=[]), SimpleNamespace(
                verdict="VERIFIED_FIXED", model_dump_json=lambda **_kwargs: "{}"
            )

    monkeypatch.setattr(ApplicationService, "configured", lambda: Stub())
    rejected = CliRunner().invoke(app, ["repair", "kernel.cu", "--unbounded-sanitizer-calls"])
    assert rejected.exit_code != 0
    assert not policies
    accepted = CliRunner().invoke(
        app,
        ["repair", "kernel.cu", "--reinvestigate", "--unbounded-sanitizer-calls"],
    )
    assert accepted.exit_code == 0, accepted.output
    assert len(policies) == 1
    assert policies[0].unbounded_sanitizer_calls is True


def test_default_v2_cli_still_has_original_policy(monkeypatch):
    from gpu_agent.service import ApplicationService
    policies = []

    class Stub:
        def repair(self, source, policy):
            policies.append(policy)
            return SimpleNamespace(id="a" * 32, artifact_refs=[]), SimpleNamespace(
                verdict="VERIFIED_FIXED", model_dump_json=lambda **_kwargs: "{}"
            )

    monkeypatch.setattr(ApplicationService, "configured", lambda: Stub())
    result = CliRunner().invoke(app, ["repair", "kernel.cu"])
    assert result.exit_code == 0, result.output
    assert policies[0].version == "public-repair-v2"
    assert policies[0].unbounded_sanitizer_calls is False
