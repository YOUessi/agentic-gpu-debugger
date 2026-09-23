"""Evaluator-owned verification truth, resolved per case instead of hard-wired to case_0001.

Every public development case is a mutation of the same vector-add template, so they share
one oracle, harness and reference. What differs per case is the registered mutant source,
the sanitizer that must observe the defect (`target_tool`) and the registered finding
category (`expected_finding`). Those come from the corpus registry; the shared numeric
parameters come from the development truth file. Nothing here reads agent evidence, so the
verdict for a candidate is independent of the mode that produced it.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from gpu_agent._resources import runtime_resource
from gpu_agent.execution.models import ExecutionModel, SanitizerTool
from gpu_agent.store import RunStore, read_regular

BASE_TRUTH = runtime_resource("benchmarks/development_truth/case_0001/case.json")
REFERENCE = runtime_resource("benchmarks/development_truth/case_0001/reference.cu")
REGISTRY = runtime_resource("benchmarks/corpus-registry.json")


class VerificationTruth(ExecutionModel):
    schema_version: Literal[2] = 2
    case_id: str = Field(pattern=r"^case_[0-9]{4}$")
    oracle: Literal["vector-add-cpu-v1"]
    atol: float = Field(ge=0)
    rtol: float = Field(ge=0)
    private_seed: int
    boundary_sizes: list[int]
    random_cases: int = Field(ge=0)
    source_hashes: dict[str, str]
    target_tool: SanitizerTool
    expected_finding: str = Field(min_length=1)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(read_regular(path, 1024 * 1024))
    if not isinstance(value, dict):
        raise ValueError("verification truth resource is malformed")
    return value


def resolve_truth(
    source_hashes: dict[str, str],
    *,
    base_path: Path | None = None,
    registry_path: Path | None = None,
) -> VerificationTruth | None:
    """Return the truth for exactly one registered public case, else None (no oracle)."""
    base = _load(base_path or BASE_TRUTH)
    registry = _load(registry_path or REGISTRY)
    base_hashes = base.get("source_hashes")
    cases = registry.get("cases")
    if not isinstance(base_hashes, dict) or not isinstance(cases, list):
        raise ValueError("verification truth resource is malformed")
    harness = {name: value for name, value in base_hashes.items() if name != "kernel.cu"}
    kernel = source_hashes.get("kernel.cu")
    if kernel is None or {k: v for k, v in source_hashes.items() if k != "kernel.cu"} != harness:
        return None
    matches = [
        case
        for case in cases
        if isinstance(case, dict)
        and case.get("mutant_source_hash") == kernel
        and case.get("oracle_id") == base.get("oracle")
        and case.get("split") == "public"
    ]
    if len(matches) != 1:
        return None
    case = matches[0]
    case_id = str(case["case_id"])
    base_seed = base.get("private_seed")
    if not isinstance(base_seed, int):
        raise ValueError("verification truth resource is malformed")
    # Independent holdout draws per case, reproducible from the evaluator-owned base seed.
    seed = int(hashlib.sha256(f"{base_seed}:{case_id}".encode()).hexdigest()[:12], 16)
    return VerificationTruth(
        case_id=case_id,
        oracle=base["oracle"],  # type: ignore[arg-type]
        atol=base["atol"],  # type: ignore[arg-type]
        rtol=base["rtol"],  # type: ignore[arg-type]
        private_seed=seed,
        boundary_sizes=base["boundary_sizes"],  # type: ignore[arg-type]
        random_cases=base["random_cases"],  # type: ignore[arg-type]
        source_hashes=dict(source_hashes),
        target_tool=SanitizerTool(case["target_tool"]),
        expected_finding=str(case["expected_finding"]),
    )


def reference_source() -> bytes:
    return read_regular(REFERENCE, 65536)


def resolve_run_truth(
    store: RunStore, run_id: str, source_hashes: dict[str, str]
) -> VerificationTruth | None:
    """Resolve private cases only through the evaluator's committed corpus cutoff.

    Private descriptors never enter a public store or a provider payload. A caller
    cannot supply a replacement truth object; replay resolves the same ledger again.
    """
    if store.visibility != "evaluator":
        return resolve_truth(source_hashes)
    run = store.load(run_id)
    units = [ref for ref in run.artifact_refs if ref.name == "evaluation/unit.json"]
    if not units:
        return resolve_truth(source_hashes)
    if len(units) != 1 or run.binding is None:
        raise ValueError("private verification unit is ambiguous or unbound")
    from gpu_agent.benchmark.evaluation import EvaluationUnitBinding
    from gpu_agent.benchmark.executor import registered_cases
    from gpu_agent.benchmark.ledger import CorpusFamily

    unit = EvaluationUnitBinding.model_validate_json(store.read(units[0]))
    if unit.split != "holdout" or unit.holdout_proof is None:
        raise ValueError("private verification requires holdout authority")
    family = CorpusFamily.configured(store)
    case = registered_cases(store, run.binding, family, cutoff=unit.corpus_cutoff).get(unit.case_id)
    if (
        case is None
        or case.split != "private"
        or case.source_hash != source_hashes.get("kernel.cu")
    ):
        return None
    base = _load(BASE_TRUTH)
    hashes = base.get("source_hashes")
    if not isinstance(hashes, dict) or case.oracle_id != base.get("oracle"):
        return None
    if {k: v for k, v in source_hashes.items() if k != "kernel.cu"} != {
        k: v for k, v in hashes.items() if k != "kernel.cu"
    }:
        return None
    seed = int(hashlib.sha256(f"{base['private_seed']}:{case.id}".encode()).hexdigest()[:12], 16)
    return VerificationTruth.model_validate(
        {
            "case_id": case.id,
            "oracle": case.oracle_id,
            "atol": base["atol"],
            "rtol": base["rtol"],
            "private_seed": seed,
            "boundary_sizes": base["boundary_sizes"],
            "random_cases": base["random_cases"],
            "source_hashes": source_hashes,
            "target_tool": case.target_tool,
            "expected_finding": case.expected_finding,
        }
    )


def required_tools(
    target: SanitizerTool, mode: Literal["standard", "full"], memcheck_clean: bool
) -> list[SanitizerTool]:
    """Deterministic sanitizer order for one verification input.

    Memcheck always runs first and the case's target tool always runs, so a candidate that
    leaves a race, uninitialized read or barrier error in place cannot pass on the numeric
    oracle alone. Full mode adds the remaining tools once memcheck is clean.
    """
    tools = [SanitizerTool.MEMCHECK]
    if target != SanitizerTool.MEMCHECK:
        tools.append(target)
    if mode == "full" and memcheck_clean:
        tools.extend(tool for tool in SanitizerTool if tool not in tools)
    return tools
