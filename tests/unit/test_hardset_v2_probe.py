"""HardSet V2: equal D/E typed functional-probe authority and leakage guards."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gpu_agent.benchmark.hardset_v2 import SIZES, expected_output, verify_numeric
from gpu_agent.benchmark.hardset_v2_graph import (
    FinishProbeAction,
    PlannerSnapshot,
    run_hardset_probe_graph,
)
from gpu_agent.benchmark.hardset_v2_probe import (
    FunctionalProbeAction,
    FunctionalProbeObservation,
    HardsetFunctionalProbe,
    ProbeBudgetError,
)
from gpu_agent.benchmark.hardset_v2_verifier import (
    HIDDEN_SIZES,
    HardsetPrivateProjection,
    hidden_input,
    secret_seed,
)

REPOSITORY = Path(__file__).resolve().parents[2]
SHA = "a" * 64
REF = "b" * 32


class FakeControlledProbe:
    def __init__(self, *, limit=3):
        self.limit = limit
        self.seen = []

    @property
    def calls(self):
        return len(self.seen)

    @property
    def allowed_sizes(self):
        return SIZES

    def invoke(self, action):
        n = action.size
        if n not in SIZES:
            raise ValueError("unsupported size")
        if n in self.seen:
            raise ProbeBudgetError("DUPLICATE_PROBE")
        if len(self.seen) >= self.limit:
            raise ProbeBudgetError("PROBE_BUDGET_EXHAUSTED")
        self.seen.append(n)
        return FunctionalProbeObservation(
            case_id="h001",
            size=n,
            outcome="NUMERIC_MISMATCH" if n in (33, 65) else "PASS",
            input_sha256=SHA,
            evidence_ref_id=REF,
        )


def test_rule_and_agent_share_exact_same_typed_probe_tool():
    baseline = FakeControlledProbe()
    agent_probe = FakeControlledProbe()
    d_state = run_hardset_probe_graph(baseline, case_id="h001", mode="D", max_probes=3)

    def model(snapshot: PlannerSnapshot):
        assert snapshot.remaining_probes >= 1
        assert all(x.case_id == "h001" for x in snapshot.observations)
        if snapshot.observations:
            return FinishProbeAction()
        return FunctionalProbeAction(size=65)

    e_state = run_hardset_probe_graph(
        agent_probe, case_id="h001", mode="E", max_probes=3, planner=model
    )
    assert [x["size"] for x in d_state["observations"]] == [32, 33, 65]
    assert [x["size"] for x in e_state["observations"]] == [65]
    assert baseline.calls == 3 and agent_probe.calls == 1
    assert d_state["observations"][1]["outcome"] == "NUMERIC_MISMATCH"
    assert e_state["observations"][0]["outcome"] == "NUMERIC_MISMATCH"
    serialized = json.dumps([d_state, e_state]).lower()
    for forbidden in ("expected_output", "private_seed", "oracle_values", "shell", "secret"):
        assert forbidden not in serialized


def test_agent_planner_cannot_introduce_new_tool_primitive():
    fake = FakeControlledProbe()
    with pytest.raises(ValueError, match="unauthorized"):
        run_hardset_probe_graph(
            fake,
            case_id="h001",
            mode="E",
            planner=lambda _: {"action_type": "shell", "command": "cat /tmp/secret"},
        )
    assert fake.calls == 0
    with pytest.raises(ValueError, match="requires"):
        run_hardset_probe_graph(fake, case_id="h001", mode="E")
    with pytest.raises(ValueError, match="does not accept"):
        run_hardset_probe_graph(
            fake,
            case_id="h001",
            mode="D",
            planner=lambda _: FinishProbeAction(),
        )


def test_invalid_or_duplicate_probe_is_rejected_without_continuing():
    fake = FakeControlledProbe()
    with pytest.raises(ValueError, match="unsupported"):
        run_hardset_probe_graph(
            fake,
            case_id="h001",
            mode="E",
            planner=lambda _: FunctionalProbeAction(size=34),
        )
    assert fake.calls == 0
    with pytest.raises(ProbeBudgetError, match="DUPLICATE"):
        run_hardset_probe_graph(
            fake,
            case_id="h001",
            mode="E",
            planner=lambda _: FunctionalProbeAction(size=32),
        )
    assert fake.calls == 1
    with pytest.raises(ValidationError):
        FunctionalProbeAction.model_validate(
            {"action_type": "run_functional_probe", "size": 33, "shell": "pwd"}
        )
    with pytest.raises(ValidationError):
        FunctionalProbeAction(size=514)


def test_case_preflight_rejects_any_output_reuse_or_unknown_case(tmp_path):
    o = tmp_path / "probe"
    with pytest.raises(ValueError, match="unregistered"):
        HardsetFunctionalProbe(REPOSITORY, o, "h999")
    o.mkdir()
    with pytest.raises(ValueError, match="exists"):
        HardsetFunctionalProbe(REPOSITORY, o, "h001")
    with pytest.raises(ValueError, match="bounded"):
        HardsetFunctionalProbe(REPOSITORY, tmp_path / "new", "h001", max_probes=0)


@pytest.mark.parametrize("n", HIDDEN_SIZES)
@pytest.mark.parametrize("case_id", ["h001", "h002", "h003"])
def test_private_inputs_use_separate_domain_and_host_reference(case_id, n):
    secret = bytes(range(32))
    raw = hidden_input(case_id, n, secret)
    values = json.loads(raw)
    assert values["n"] == n
    assert len(values["a"]) == n and len(values["b"]) == n
    assert len(expected_output(case_id, raw)) == n
    correct = expected_output(case_id, raw)
    encoded = json.dumps({"dtype": "float32", "shape": [n], "values": correct}).encode()
    assert verify_numeric(case_id, raw, encoded)
    assert n not in SIZES
    assert raw == hidden_input(case_id, n, secret)
    assert raw != hidden_input(case_id, n, b"a" * 32)


def test_evaluator_requires_secret_and_rejects_unregistered_domain():
    with pytest.raises(ValueError, match="at least 32"):
        secret_seed(b"too_short", case_id="h001", n=34)
    with pytest.raises(ValueError, match="invalid hidden"):
        secret_seed(b"a" * 32, case_id="h001", n=33)
    with pytest.raises(ValueError, match="invalid hidden"):
        secret_seed(b"a" * 32, case_id="h999", n=34)


def test_strict_public_projection_cannot_include_hidden_evidence():
    allowed = HardsetPrivateProjection(
        case_id="h001",
        candidate_sha256=SHA,
        verdict="NOT_VERIFIED",
        reason_code="HIDDEN_NUMERIC_FAILED",
    )
    assert "seed" not in allowed.model_dump_json()
    with pytest.raises(ValidationError):
        HardsetPrivateProjection.model_validate({**allowed.model_dump(), "hidden_input": [1, 2, 3]})


def test_scoped_patch_must_preflight_before_evaluator_directory(tmp_path):
    from gpu_agent.benchmark.hardset_v2_verifier import run_strict_hardset_verification

    dest = tmp_path / "private-failure"
    with pytest.raises(ValueError):
        run_strict_hardset_verification(
            REPOSITORY,
            dest,
            case_id="h001",
            unified_diff="--- a/kernel.cu\n+++ b/kernel.cu\n",
            evaluator_secret=b"x" * 32,
        )
    assert not dest.exists()
