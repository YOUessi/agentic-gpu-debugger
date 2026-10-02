"""Public functional requirements; never resolve evaluator truth or execute user code."""

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field

from gpu_agent.execution.models import ExecutionModel
from gpu_agent.verification.oracle import NumericOracle, OracleId, parse_output, reference_output

DESCRIPTIONS: dict[str, str] = {
    "stencil2d-cpu-v1": (
        "Treat a as a row-major 2D grid of width 32 and ceil(n/32) rows. The last row "
        "may be partial. output[i] is center, left, right, up, down, then b[i], added "
        "in that order. Neighbors outside the logical row or valid n elements are zero."
    ),
    "segment-scan-cpu-v1": (
        "Partition inputs into consecutive segments of at most 128 elements. Each output "
        "is the inclusive prefix sum of a[j]+b[j] from its segment start through that index. "
        "Reset the sum at each segment. Float32 scan uses doubling offsets 1,2,4,...,64, "
        "adding the previous stage's value at i-offset to its value at i when available."
    ),
    "vector-add-cpu-v1": "For every i in [0,n), output[i] = a[i] + b[i].",
    "rotate-add-cpu-v1": (
        "For every i in [0,n), output[i] = a[(i+1) mod n] + b[i]. The shift wraps around."
    ),
    "stencil-cpu-v1": (
        "output[i] = ((a[i-1] + a[i]) + a[i+1]) + b[i]; out-of-range a elements are zero."
    ),
    "histogram-cpu-v1": (
        "Output has n initially-zero bins. Each input contributes weight "
        "floor(abs(b[i]) mod 5)+1 to bin floor(abs(a[i]) mod min(n,32))."
    ),
    "warp-reduce-cpu-v1": (
        "Partition indices into consecutive groups of at most 32. Sum a[i]+b[i] over "
        "each group's valid indices, and output that sum at every index in the group. "
        "No elements outside the group contribute."
    ),
}


class PublicTask(ExecutionModel):
    version: Literal["public-task-v1"] = "public-task-v1"
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    algorithm: OracleId

    @property
    def requirement(self) -> str:
        return (
            DESCRIPTIONS[self.algorithm] + " Inputs and outputs are float32; atol=1e-5, rtol=1e-5."
        )


class PublicRepairInputError(ValueError):
    """Value-free CLI diagnostics for checks performed before external work."""

    def __init__(
        self,
        code: Literal[
            "PUBLIC_TASK_UNAVAILABLE",
            "PUBLIC_TASK_INVALID",
            "PUBLIC_INPUT_INVALID",
            "PUBLIC_INTERFACE_UNSUPPORTED",
        ],
    ) -> None:
        self.code = code
        super().__init__(code)


def load_public_task(kernel: Path, source: bytes) -> PublicTask | None:
    from gpu_agent.store import read_regular

    path = kernel.parent / "task.json"
    if not path.exists() and not path.is_symlink():
        return None
    task = PublicTask.model_validate_json(read_regular(path.absolute(), 16384))
    if task.source_sha256 != hashlib.sha256(source).hexdigest():
        raise ValueError("public task source binding mismatch")
    return task


def public_expected_output(task: PublicTask, stdin: bytes) -> list[float]:
    """Validate public input before any external execution, using the same checker."""
    data = json.loads(stdin)
    if not isinstance(data, dict) or set(data) != {"n", "a", "b"}:
        raise ValueError("invalid public functional input")
    n, a, b = data["n"], data["a"], data["b"]
    if (
        type(n) is not int
        or not 1 <= n <= 65536
        or not isinstance(a, list)
        or not isinstance(b, list)
        or len(a) != n
        or len(b) != n
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in [*a, *b])
    ):
        raise ValueError("invalid public functional input")
    expected = reference_output(task.algorithm, a, b)
    if any(not math.isfinite(x) for x in expected):
        raise ValueError("public reference output must be finite")
    return expected


def check_public_output(task: PublicTask, stdin: bytes, output: bytes) -> str:
    """Only caller-supplied public input; no registry, hidden seed, or candidate checker."""
    expected = public_expected_output(task, stdin)
    try:
        actual = parse_output(output)
    except ValueError:
        return "INVALID_NUMERIC_OUTPUT"
    result = NumericOracle(1e-5, 1e-5, False, False).check(actual, expected)
    return "PASSED" if result.passed else result.failure_reason or "NUMERIC_MISMATCH"
