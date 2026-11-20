"""Real LangGraph repair-cycle regression with typed fake decisions, no model or CUDA."""

from __future__ import annotations

import hashlib

import pytest

from gpu_agent.benchmark.hardset_v2_agent import run_repair_graph
from gpu_agent.benchmark.hardset_v2_agent_models import (
    HardsetAgentContext,
    HardsetDiagnosis,
    HardsetPublicCheck,
)
from gpu_agent.benchmark.hardset_v2_graph import FinishProbeAction
from gpu_agent.benchmark.hardset_v2_probe import (
    FunctionalProbeAction,
    FunctionalProbeObservation,
)


class Probe:
    def __init__(self):
        self.seen: list[int] = []

    @property
    def calls(self) -> int:
        return len(self.seen)

    @property
    def allowed_sizes(self) -> tuple[int, ...]:
        return (32, 33, 65)

    def invoke(self, action: FunctionalProbeAction) -> FunctionalProbeObservation:
        if action.size in self.seen:
            raise ValueError("duplicate")
        self.seen.append(action.size)
        return FunctionalProbeObservation(
            case_id="h001",
            size=action.size,
            outcome="PASS" if action.size == 32 else "PROGRAM_FAILED",
            input_sha256="a" * 64,
            evidence_ref_id=f"{action.size:032x}",
        )


class FakeLLM:
    def __init__(self, diffs: list[str]):
        self.diffs = diffs
        self.phases: list[str] = []
        self.repair_feedback: list[list[HardsetPublicCheck]] = []

    def plan_hardset_v2(self, snapshot):
        self.phases.append("plan")
        return FinishProbeAction() if snapshot.observations else FunctionalProbeAction(size=65)

    def diagnose_hardset_v2(self, context: HardsetAgentContext) -> HardsetDiagnosis:
        self.phases.append("diagnose")
        return HardsetDiagnosis(
            root_cause="Exponentiation is numerically unstable on extreme logits",
            suspected_lines=[1],
            cited_public_probe_ids=[context.observations[-1].evidence_ref_id],
            repair_strategy="Subtract group maximum before exponentiation",
            confidence="medium",
        )

    def propose_patch_hardset_v2(
        self, context, diagnosis, public_feedback, previous_candidate_hashes
    ):
        self.phases.append("patch")
        self.repair_feedback.append(list(public_feedback))
        assert diagnosis.cited_public_probe_ids
        return self.diffs.pop(0)


def validate(diff: str) -> tuple[str, bytes]:
    if diff == "ILLEGAL":
        raise ValueError("PATCH_SCOPE_REJECTED")
    return hashlib.sha256(diff.encode()).hexdigest(), diff.encode()


def check(code: bytes, attempt: int) -> HardsetPublicCheck:
    passed = code == b"FIXED"
    return HardsetPublicCheck(
        candidate_sha256=hashlib.sha256(code).hexdigest(),
        passed=passed,
        failure_code="PUBLIC_CHECKS_PASSED" if passed else "NUMERIC_MISMATCH",
        first_failing_size=None if passed else 65,
        public_run_id=f"{attempt:032x}",
    )


def run(mode: str, llm: FakeLLM, *, checker=check, validator=validate, limit=3):
    return run_repair_graph(
        probe=Probe(),
        mode=mode,
        case_id="h001",
        public_source="__global__ void softmax() {}\n",
        functional_requirement="stable softmax over valid group of 32",
        provider=llm,
        validate_candidate=validator,
        public_check=checker,
        max_probes=3,
        max_candidates=limit,
    )


def test_real_graph_uses_failure_evidence_and_revises_until_public_gate_passes():
    llm = FakeLLM(["WRONG", "FIXED"])
    result = run("E", llm)
    assert llm.phases == ["plan", "plan", "diagnose", "patch", "patch"]
    assert result["attempts"] == 2
    assert result["accepted_diff"] == "FIXED"
    assert [x["failure_code"] for x in result["feedback"]] == [
        "NUMERIC_MISMATCH",
        "PUBLIC_CHECKS_PASSED",
    ]
    assert llm.repair_feedback[1][0].first_failing_size == 65


def test_rule_mode_uses_same_diagnose_patch_and_public_check():
    llm = FakeLLM(["FIXED"])
    result = run("D", llm)
    assert llm.phases == ["diagnose", "patch"]
    assert len(result["observations"]) == 3
    assert result["stop_reason"] == "PUBLIC_CHECKS_PASSED"


def test_scope_rejected_without_cuda_then_public_patch_is_accepted():
    llm = FakeLLM(["ILLEGAL", "FIXED"])
    calls = []

    def instrument(code: bytes, attempt: int) -> HardsetPublicCheck:
        calls.append(attempt)
        return check(code, attempt)

    result = run("E", llm, checker=instrument, limit=2)
    assert calls == [2]
    assert result["feedback"][0]["failure_code"] == "PATCH_SCOPE_REJECTED"
    assert result["accepted_diff"] == "FIXED"


def test_repeated_candidate_stops_instead_of_private_or_unbounded_retries():
    result = run("E", FakeLLM(["WRONG", "WRONG"]))
    assert result["stop_reason"] == "REPEATED_CANDIDATE"
    assert result["attempts"] == 2
    assert not result.get("accepted_diff")


def test_checker_candidate_provenance_cannot_be_forged():
    def forged(code: bytes, attempt: int) -> HardsetPublicCheck:
        return HardsetPublicCheck(
            candidate_sha256="f" * 64,
            passed=True,
            failure_code="PUBLIC_CHECKS_PASSED",
        )

    with pytest.raises(ValueError, match="provenance mismatch"):
        run("D", FakeLLM(["FIXED"]), checker=forged)


def test_inconclusive_when_agent_finishes_before_any_real_evidence():
    class NoProbe(FakeLLM):
        def plan_hardset_v2(self, snapshot):
            return FinishProbeAction()

    llm = NoProbe(["FIXED"])
    result = run("E", llm)
    assert result["stop_reason"] == "NO_PUBLIC_EVIDENCE"
    assert llm.phases == []


@pytest.mark.parametrize("max_candidates", [0, 4, 99])
def test_controller_rejects_unbounded_repair_loops(max_candidates):
    with pytest.raises(ValueError, match="unbounded"):
        run("D", FakeLLM(["FIXED"]), limit=max_candidates)
