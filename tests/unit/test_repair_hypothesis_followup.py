"""Source-scoped public hypothesis checks prevent premature V3 diagnosis."""

import pytest

from gpu_agent.agent.models import (
    AgentBudget,
    DiagnosisResult,
    FinishAction,
    PublicEvidence,
    PublicRepairContext,
)
from gpu_agent.agent.policy import decide_action, missing_evidence
from gpu_agent.agent.rule_router import RuleRouter
from gpu_agent.contracts import CurrentPhase


def evidence_for_family(
    family: str, *,
    functional_failure: bool = True,
    memcheck: str = "CLEAN",
    existing_tool: str | None = None,
) -> PublicEvidence:
    context = PublicRepairContext(
        repair_round=1,
        original_source_sha256="a" * 64,
        candidate_source_sha256="b" * 64,
        previous_diagnosis_source_sha256="a" * 64,
        previous_diagnosis=DiagnosisResult(
            diagnostic_outcome="DIAGNOSED",
            failure_family=family,
            root_cause="Prior public observation detected a potential hazard.",
        ),
        public_checks={"functional": "NUMERIC_MISMATCH"},
        public_feedback=[],
        public_functional_failure=functional_failure,
    )
    outcomes = {"memcheck": memcheck}
    if existing_tool:
        outcomes[existing_tool] = "CLEAN"
    return PublicEvidence(
        sources=[],
        repair_context=context,
        sanitizer_outcomes=outcomes,
    )


@pytest.mark.parametrize(
    "failure_family,expected_tool,missing_marker",
    [
        ("shared_memory_race", "run_racecheck", "racecheck_outcome"),
        ("uninitialized_memory_read", "run_initcheck", "initcheck_outcome"),
        ("barrier_misuse", "run_synccheck", "synccheck_outcome"),
    ],
)
def test_current_candidate_hazard_must_be_checked_before_diagnosis(
    failure_family, expected_tool, missing_marker
):
    evidence = evidence_for_family(failure_family)
    missing = missing_evidence(evidence)
    assert missing_marker in missing
    decision = decide_action(
        FinishAction(), evidence, AgentBudget(), CurrentPhase.DIAGNOSING, set()
    )
    assert not decision.allowed
    assert decision.reason_codes == ["MANDATORY_EVIDENCE_MISSING"]
    assert expected_tool in decision.mandatory_actions
    assert RuleRouter().next_action(evidence, AgentBudget()).action_type == expected_tool


@pytest.mark.parametrize(
    "family,tool",
    [
        ("shared_memory_race", "racecheck"),
        ("uninitialized_memory_read", "initcheck"),
        ("barrier_misuse", "synccheck"),
    ],
)
def test_completed_matching_check_allows_public_functional_diagnosis(family, tool):
    evidence = evidence_for_family(family, existing_tool=tool)
    assert missing_evidence(evidence) == []
    decision = decide_action(
        FinishAction(), evidence, AgentBudget(), CurrentPhase.DIAGNOSING, set()
    )
    assert decision.allowed


def test_clean_memcheck_alone_cannot_refute_shared_race():
    evidence = evidence_for_family("shared_memory_race")
    assert evidence.sanitizer_outcomes["memcheck"] == "CLEAN"
    assert "racecheck_outcome" in missing_evidence(evidence)


@pytest.mark.parametrize(
    "family,functional_failure,memcheck",
    [
        ("other", True, "CLEAN"),
        ("shared_memory_race", False, "CLEAN"),
        ("shared_memory_race", True, "FINDING"),
    ],
)
def test_no_hazard_specific_check_when_not_warranted(
    family, functional_failure, memcheck
):
    evidence = evidence_for_family(
        family,
        functional_failure=functional_failure,
        memcheck=memcheck,
    )
    assert not any(
        check in missing_evidence(evidence)
        for check in ("racecheck_outcome", "initcheck_outcome", "synccheck_outcome")
    )


def test_bounded_rule_router_does_not_bypass_hazard_budget():
    evidence = evidence_for_family("shared_memory_race")
    budget = AgentBudget(max_sanitizer_calls=1, sanitizer_calls=1)
    assert RuleRouter().next_action(evidence, budget).action_type == "declare_inconclusive"


def test_live_source_hypothesis_is_not_reused_as_current_citation():
    evidence = evidence_for_family("shared_memory_race")
    assert evidence.tool_findings == []
    assert evidence.documentation == []
    assert evidence.repair_context is not None
    assert evidence.repair_context.previous_diagnosis.root_cause
    assert "racecheck_outcome" in missing_evidence(evidence)
