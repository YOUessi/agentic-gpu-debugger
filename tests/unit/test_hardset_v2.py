"""Offline contract tests for self-authored numeric + multi-kernel hard cases."""

import json
import math
from pathlib import Path

import pytest

from gpu_agent.benchmark.hardset_v2 import (
    HARDSET_CASES,
    SIZES,
    HardsetRegistry,
    expected_output,
    load_registry,
    manifest_for,
    sample_input,
    verify_numeric,
)

ROOT = Path(__file__).resolve().parents[2] / "benchmarks"


def test_real_checked_in_sources_have_distinct_trusted_hashes():
    registry, sha = load_registry(ROOT)
    assert len(sha) == 64 and set(HARDSET_CASES) == {x.case_id for x in registry.cases}
    assert registry.status == "NOT_YET_GPU_VALIDATED"
    for case in registry.cases:
        for role in ("clean", "mutant"):
            manifest = manifest_for(ROOT, case, role)
            kernel = f"hardset_v2/{case.case_id}/{role}/kernel.cu"
            assert manifest[kernel] == getattr(case, f"{role}_source_sha256")
            assert set(Path(name).name for name in manifest) == {
                "kernel.cu",
                "vector_api.h",
                "vector_io.cpp",
                "json.hpp",
            }
        assert case.clean_source_sha256 != case.mutant_source_sha256


@pytest.mark.parametrize("case", sorted(HARDSET_CASES))
@pytest.mark.parametrize("n", SIZES)
def test_cpu_reference_is_finite_and_has_expected_shape(case, n):
    output = expected_output(case, sample_input(case, n))
    assert len(output) == n
    assert all(math.isfinite(x) for x in output)


@pytest.mark.parametrize("n", SIZES)
def test_softmax_is_stable_under_large_logits_and_segment_normalized(n):
    actual = expected_output("h001", sample_input("h001", n))
    for start in range(0, n, 32):
        group = actual[start : start + 32]
        assert all(0 <= x <= 1 for x in group)
        assert abs(sum(group) - 1) < 1e-5


@pytest.mark.parametrize("n", SIZES)
def test_group_center_has_zero_mean_even_with_tail(n):
    actual = expected_output("h003", sample_input("h003", n))
    for start in range(0, n, 32):
        group = actual[start : start + 32]
        assert abs(sum(group)) < 1e-5


def test_predeclared_easy_and_stress_inputs_are_distinct():
    low = json.loads(sample_input("h001", 32))
    extreme = json.loads(sample_input("h001", 33))
    assert max(low["a"]) < 20
    assert min(extreme["a"]) > 90
    centered = json.loads(sample_input("h003", 32))
    assert sum(centered["b"]) == 0
    assert len(SIZES) >= 15


def test_rmsnorm_tail_divisor_error_is_detectable():
    payload = sample_input("h002", 33)
    actual = expected_output("h002", payload)
    # A wrong divide-by-32 for a length-one tail is not a harmless FP error.
    tail = actual[32]
    assert abs(tail) > 0.9
    assert abs(tail) < 1.1


@pytest.mark.parametrize("case", sorted(HARDSET_CASES))
def test_numeric_checker_detects_perturbed_valid_json(case):
    payload = sample_input(case, 65)
    expected = expected_output(case, payload)
    good = json.dumps({"dtype": "float32", "shape": [65], "values": expected}).encode()
    assert verify_numeric(case, payload, good)
    wrong = list(expected)
    wrong[-1] += 0.1
    corrupted = json.dumps({"dtype": "float32", "shape": [65], "values": wrong}).encode()
    assert not verify_numeric(case, payload, corrupted)
    assert not verify_numeric(case, payload, b'{"values":[null]}')


def test_tampered_mutant_fails_preflight_before_gpu_or_model(tmp_path):
    root = tmp_path / "benchmarks"
    origin = ROOT / "hardset_v2"
    for p in origin.rglob("*"):
        if p.is_file():
            relative = p.relative_to(origin)
            target = root / "hardset_v2" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
    registry, _ = load_registry(root)
    mutant = root / "hardset_v2" / "h003" / "mutant" / "kernel.cu"
    mutant.write_bytes(mutant.read_bytes() + b"\n// accidental change\n")
    with pytest.raises(ValueError, match="pinned registry"):
        load_registry(root)
    # The untouched manifest is still intrinsically valid, but cannot validate
    # altered bytes. Do not silently re-hash or accept a substitute.
    assert (
        HardsetRegistry.model_validate_json(
            (root / "hardset_v2" / "manifest.json").read_bytes()
        ).cases
        == registry.cases
    )


def test_reference_rejects_wrong_sizes_nonfinite_and_unknown_cases():
    with pytest.raises(ValueError):
        sample_input("h999", 65)
    with pytest.raises(ValueError):
        sample_input("h001", 34)
    with pytest.raises(ValueError):
        expected_output(
            "h003",
            json.dumps({"n": 2, "a": [math.inf, 1], "b": [0, 0]}).encode(),
        )
    with pytest.raises(ValueError):
        expected_output("h003", b'{"n":2,"a":[1],"b":[0,0]}')
