"""Static authority checks for the 16 public GPU-validation candidates."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"


def _input_hash(recipe: dict[str, object]) -> str:
    n = int(recipe["n"])
    payload = json.dumps(
        {
            "n": n,
            "a": [float(recipe["a_value"])] * n,
            "b": [float(recipe["b_value"])] * n,
        }
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _harness_hash() -> str:
    paths = {
        "vector_io.cpp": BENCHMARKS / "harness/vector_io.cpp",
        "vector_api.h": BENCHMARKS / "harness/vector_api.h",
        "json.hpp": BENCHMARKS / "harness/vendor/json.hpp",
    }
    manifest = sorted(
        (name, hashlib.sha256(path.read_bytes()).hexdigest()) for name, path in paths.items()
    )
    return hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()


def _without_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    source = re.sub(r"//[^\n]*", "", source)
    return "".join(source.split())


def test_public_registry_has_four_unique_cases_per_failure_family():
    registry = json.loads((BENCHMARKS / "corpus-registry.json").read_text(encoding="utf-8"))
    cases = registry["cases"]
    assert [case["case_id"] for case in cases] == [f"case_{number:04d}" for number in range(1, 17)]
    assert Counter(case["target_tool"] for case in cases) == {
        "memcheck": 4,
        "racecheck": 4,
        "initcheck": 4,
        "synccheck": 4,
    }
    assert len({case["template_id"] for case in cases}) == 16
    assert len({case["mutation_id"] for case in cases}) == 16
    assert all(case["split"] == "public" for case in cases)


def test_recipe_registry_mutations_and_all_hashes_are_exact():
    registry = json.loads((BENCHMARKS / "corpus-registry.json").read_text(encoding="utf-8"))
    recipes = json.loads((BENCHMARKS / "seed-batch.json").read_text(encoding="utf-8"))["cases"]
    operators = json.loads((BENCHMARKS / "templates/mutations.json").read_text(encoding="utf-8"))[
        "operators"
    ]
    specs = {case["case_id"]: case for case in registry["cases"]}
    assert [recipe["case_id"] for recipe in recipes] == list(specs)
    assert {(item["template_id"], item["mutation_id"]) for item in operators} == {
        (case["template_id"], case["mutation_id"]) for case in specs.values()
    }

    provenance_hash = hashlib.sha256(
        (BENCHMARKS / "templates/mutations.json").read_bytes()
    ).hexdigest()
    clean_hash = hashlib.sha256(
        (BENCHMARKS / "public/case_0000/public_input/kernel.cu").read_bytes()
    ).hexdigest()
    harness_hash = _harness_hash()
    for recipe in recipes:
        spec = specs[recipe["case_id"]]
        mutant_path = BENCHMARKS / recipe["mutant_source"]
        assert recipe["clean_source"] == "public/case_0000/public_input/kernel.cu"
        assert mutant_path == BENCHMARKS / f"public/{recipe['case_id']}/public_input/kernel.cu"
        assert spec["clean_source_hash"] == clean_hash
        assert spec["mutant_source_hash"] == hashlib.sha256(mutant_path.read_bytes()).hexdigest()
        assert spec["harness_hash"] == harness_hash
        assert spec["input_set_hash"] == _input_hash(recipe)
        assert spec["mutation_provenance_hash"] == provenance_hash


def test_mutants_are_code_distinct_and_keep_the_trusted_vector_contract():
    normalized: list[str] = []
    for number in range(1, 17):
        path = BENCHMARKS / f"public/case_{number:04d}/public_input/kernel.cu"
        source = path.read_text(encoding="utf-8")
        code = _without_comments(source)
        assert 'extern"C"' not in code
        assert "intrun_vector_add(constfloat*a,constfloat*b,float*out,std::size_tn)" in code
        assert "vector_add<<<" in code
        normalized.append(code)
    assert len(set(normalized)) == 16


def test_new_cases_are_explicitly_candidates_not_live_evidence():
    protocol = (ROOT / "docs/benchmark-protocol.md").read_text(encoding="utf-8")
    assert "case_0005" in protocol and "case_0016" in protocol
    assert "candidate" in protocol.lower()
    assert "GPU" in protocol
