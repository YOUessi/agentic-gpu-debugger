"""Algorithm semantics and anti-bypass checks, independent of model behavior."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_agent.benchmark.diversity import (
    CLEAN_NAMES,
    _exercise,
    checked_sources,
    remove_launches,
    run_diversity,
)
from gpu_agent.benchmark.models import AuthoritativeCaseRegistry
from gpu_agent.benchmark.validation import derive_oracle
from gpu_agent.store import RunStore
from gpu_agent.verification.oracle import ORACLE_IDS, reference_output
from gpu_agent.verification.truth import reference_source, resolve_truth

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks"
SPECS = AuthoritativeCaseRegistry.model_validate_json(
    (BENCH / "diverse-registry.json").read_bytes()
).cases


@pytest.mark.parametrize(
    "oracle,expected",
    [
        ("vector-add-cpu-v1", [5, 5, 5, 5]),
        ("rotate-add-cpu-v1", [6, 6, 6, 2]),
        ("stencil-cpu-v1", [7, 9, 11, 8]),
        ("histogram-cpu-v1", [2, 5, 4, 3]),
        ("warp-reduce-cpu-v1", [20, 20, 20, 20]),
        ("stencil2d-cpu-v1", [7, 9, 11, 8]),
        ("segment-scan-cpu-v1", [5, 10, 15, 20]),
    ],
)
def test_hand_computed_references(oracle, expected):
    assert reference_output(oracle, [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == expected


@pytest.mark.parametrize("oracle", ORACLE_IDS)
def test_oracle_rejects_missing_output_and_wrong_values(oracle):
    payload = json.dumps({"n": 4, "a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]}).encode()
    wrong = json.dumps({"dtype": "float32", "shape": [4], "values": [0.0, 0.0, 0.0, 0.0]}).encode()
    assert not derive_oracle(payload, wrong, oracle).passed
    assert not derive_oracle(payload, b"", oracle).passed


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.case_id)
def test_new_sources_are_pinned_and_resolve_real_algorithm(spec):
    manifest = checked_sources(BENCH, spec, "mutant")
    checked_sources(BENCH, spec, "clean")
    hashes = {Path(p).name: v for p, v in manifest.items()}
    truth = resolve_truth(hashes)
    assert truth is not None and truth.case_id == spec.case_id and truth.oracle == spec.oracle_id
    assert hashlib.sha256(reference_source(truth.oracle)).hexdigest() == spec.clean_source_hash
    assert resolve_truth({**hashes, "kernel.cu": "0" * 64}) is None
    assert resolve_truth({**hashes, "vector_api.h": "0" * 64}) is None
    text = reference_source(truth.oracle).decode()
    assert "<<<" in text and "<<<" not in remove_launches(text)
    assert (
        "Intentional" not in (BENCH / f"public/{spec.case_id}/public_input/kernel.cu").read_text()
    )


def test_unknown_checker_cannot_be_requested():
    with pytest.raises(ValueError, match="not registered"):
        reference_output("model-supplied-checker", [1.0], [2.0])
    with pytest.raises(ValueError):
        reference_source("../../private")


def test_new_algorithms_are_not_vector_add_with_renamed_kernels():
    assert len(CLEAN_NAMES) == 6
    a = [float(i + 1) for i in range(129)]
    b = [1.0] * 129
    outputs = [tuple(reference_output(s.oracle_id, a, b)) for s in SPECS]
    assert len(set(outputs)) == 6
    assert tuple(reference_output("vector-add-cpu-v1", a, b)) not in outputs


def test_old_registry_and_denominator_remain_separate():
    original = json.loads((BENCH / "corpus-registry.json").read_bytes())
    assert len(original["cases"]) == 16
    assert {s.case_id for s in SPECS}.isdisjoint(c["case_id"] for c in original["cases"])


def test_2d_stencil_row_edges_and_partial_last_row():
    values = reference_output("stencil2d-cpu-v1", [1.0] * 65, [0.0] * 65)
    assert [values[i] for i in (0, 1, 31, 32, 33, 63, 64)] == [3, 4, 3, 4, 4, 3, 2]
    assert reference_output("stencil2d-cpu-v1", [2], [3]) == [5]


def test_scan_resets_at_segment_boundary_and_keeps_tail():
    values = reference_output("segment-scan-cpu-v1", [1.0] * 257, [0.0] * 257)
    assert values == list(range(1, 129)) + list(range(1, 129)) + [1]
    assert reference_output("segment-scan-cpu-v1", [2], [3]) == [5]


def test_mutation_metadata_matches_all_manifest_provenance_hashes():
    metadata = json.loads((BENCH / "diverse-mutations.json").read_bytes())["mutations"]
    assert len(metadata) == len(SPECS)
    for spec, item in zip(SPECS, metadata, strict=True):
        assert spec.case_id == f"case_{item['number']:04}"
        assert spec.mutation_id == item["mutation"]
        assert spec.expected_finding == item["finding"]
        assert hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest() == (
            spec.mutation_provenance_hash
        )


def test_runner_refuses_overwrite_or_unknown_case_without_gpu(tmp_path):
    with pytest.raises(ValueError, match="Output exists"):
        run_diversity(ROOT, tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        run_diversity(ROOT, tmp_path / "uncreated", ("case_0001",))
    assert not (tmp_path / "uncreated").exists()


@pytest.mark.parametrize(
    "role,finding,error,expected",
    [
        ("mutant", True, None, True),
        ("mutant", False, None, False),
        ("mutant", True, "infrastructure failure", False),
        ("clean", True, None, False),
        ("ablation", True, None, False),
    ],
)
def test_timeout_is_only_a_symptom_not_acceptance(tmp_path, role, finding, error, expected):
    class Backend:
        def prepare(self, request):
            return SimpleNamespace(id="workspace")

        def build(self, request):
            return SimpleNamespace(success=True)

        def run(self, request):
            return SimpleNamespace(
                runtime_status="TIMEOUT",
                tool_result=SimpleNamespace(
                    tool_error=error, timed_out=True, cancelled=False, truncated=False
                ),
            )

        def run_sanitizer(self, request):
            return SimpleNamespace(
                completed=True,
                tool_result=None,
                check_outcome="FINDING" if finding else "CLEAN",
                findings=[SimpleNamespace(category=SPECS[-1].expected_finding)] if finding else [],
            )

        def cleanup(self, handle):
            pass

    result = _exercise(RunStore(tmp_path / "store"), Backend(), SPECS[-1], {}, role, b"{}")
    assert result["passed"] is expected
    if expected:
        assert result["checks"][0]["runtime_status"] == "TIMEOUT"
        assert result["checks"][0]["oracle_passed"] is False
        assert len(result["checks"]) == 1 + SPECS[-1].sanitizer_repetitions


@pytest.mark.parametrize("roles", [(), ("mutant", "mutant"), ("invented",)])
def test_role_selection_rejects_invalid_requests(tmp_path, roles):
    with pytest.raises(ValueError, match="role selection"):
        run_diversity(ROOT, tmp_path / "uncreated", roles=roles)
    assert not (tmp_path / "uncreated").exists()
