"""Public diagnostic categories and independently reported sanitizer stage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_agent.benchmark.hardset_v2 import expected_output, sample_input
from gpu_agent.benchmark.hardset_v2_selfcheck import check_candidate_publicly
from gpu_agent.benchmark.public_numeric_feedback import classify_public_numeric
from gpu_agent.store import RunStore


def _output(values):
    return json.dumps({"dtype": "float32", "shape": [len(values)], "values": values}).encode()


def test_numeric_diagnostic_categories_are_bounded_and_match_checker() -> None:
    payload = sample_input("h002", 65)
    correct = expected_output("h002", payload)
    assert classify_public_numeric("h002", payload, _output(correct)).category == "PASS"
    tail = correct.copy()
    tail[-1] += 1.0
    d = classify_public_numeric("h002", payload, _output(tail))
    assert d.category == "GROUP_TAIL_MISMATCH"
    assert d.first_failing_group == 2 and d.failing_group_is_tail
    multi = correct.copy()
    multi[0] += 1.0
    multi[32] += 1.0
    assert (
        classify_public_numeric("h002", payload, _output(multi)).category == "MULTI_GROUP_MISMATCH"
    )
    assert (
        classify_public_numeric("h002", payload, _output([0.0] * 65)).category == "ALL_ZERO_OUTPUT"
    )
    nonfinite = correct.copy()
    nonfinite[0] = float("inf")
    d = classify_public_numeric("h002", payload, _output(nonfinite))
    assert d.category == "NON_FINITE_OUTPUT" and d.output_finite is False
    assert (
        classify_public_numeric("h002", payload, _output(correct[:-1])).category == "SHAPE_MISMATCH"
    )
    assert (
        classify_public_numeric("h002", payload, b'{"secret":"unexpected"}').category
        == "OUTPUT_INVALID"
    )
    assert set(d.model_dump()) == {
        "category",
        "input_size",
        "output_finite",
        "all_zero",
        "first_failing_group",
        "failing_group_is_tail",
    }


@pytest.mark.parametrize("diagnostic_mode", [False, True])
def test_numeric_failure_does_not_mask_sanitizer_in_diagnostic_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostic_mode: bool
) -> None:
    repo = Path(__file__).resolve().parents[2]
    store = RunStore(tmp_path / "public")
    parent = store.create_run("gpu_repository_agent")
    store.transition(parent.id, "RUNNING", "DIAGNOSING")
    calls = []

    class Backend:
        def __init__(self, store, source, output):
            self.store = store

        @staticmethod
        def _write_snapshot(path, raw):
            path.write_bytes(raw)

        def availability(self):
            return SimpleNamespace(ready=True)

        def prepare(self, request):
            self.run_id = request.run_id
            return SimpleNamespace(id="fixture")

        def build(self, request):
            return SimpleNamespace(success=True)

        def run(self, request):
            raw = self.store.read(request.stdin_ref)
            n = json.loads(raw)["n"]
            ref = self.store.put(self.run_id, "fixture/numeric.json", _output([0.0] * n), "public")
            return SimpleNamespace(
                output_ref=ref,
                runtime_status="SUCCESS",
                tool_result=SimpleNamespace(
                    tool_error=None, timed_out=False, cancelled=False, truncated=False
                ),
            )

        def run_sanitizer(self, request):
            calls.append(request.tool)
            return SimpleNamespace(completed=True, check_outcome="FINDING", program_output_ref=None)

        def cleanup(self, handle):
            return None

    monkeypatch.setattr("gpu_agent.benchmark.hardset_v2_selfcheck.IsolatedGPUBackend", Backend)
    result = check_candidate_publicly(
        store=store,
        parent_run_id=parent.id,
        repository=repo,
        case_id="h002",
        candidate=(repo / "benchmarks/hardset_v2/h002/mutant/kernel.cu").read_bytes(),
        output=tmp_path / "check",
        diagnostic_mode=diagnostic_mode,
    )
    assert result.failure_code == "NUMERIC_MISMATCH" and not result.passed
    if diagnostic_mode:
        assert result.numeric_diagnostic.category == "ALL_ZERO_OUTPUT"
        assert len(calls) == 4
        assert all(d.status == "FINDINGS" for d in result.sanitizer_diagnostics)
    else:
        assert not calls
        assert result.numeric_diagnostic is None
