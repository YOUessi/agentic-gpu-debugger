"""Bounded public mismatch categories; reference values never leave the controller."""
from __future__ import annotations

import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.benchmark.hardset_v2 import expected_output
from gpu_agent.verification.oracle import NumericOracle, parse_output


class PublicNumericDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    category: Literal[
        "PASS", "OUTPUT_INVALID", "NON_FINITE_OUTPUT", "SHAPE_MISMATCH",
        "ALL_ZERO_OUTPUT", "GROUP_TAIL_MISMATCH", "MULTI_GROUP_MISMATCH",
        "FINITE_NUMERIC_MISMATCH",
    ]
    input_size: int = Field(ge=1, le=65536)
    output_finite: bool | None = None
    all_zero: bool | None = None
    first_failing_group: int | None = Field(default=None, ge=0, le=2047)
    failing_group_is_tail: bool | None = None


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    obj: dict[str, object] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate output field")
        obj[key] = value
    return obj


def _contains_nonfinite_numeric_values(raw: bytes) -> bool:
    """Only classify explicit nonfinite values in a structurally valid envelope."""
    if not raw or len(raw) > 2 * 1024 * 1024:
        return False
    try:
        obj = json.loads(raw, object_pairs_hook=_unique_pairs)
        if not isinstance(obj, dict) or set(obj) != {"dtype", "shape", "values"}:
            return False
        values, shape = obj["values"], obj["shape"]
        if (
            obj["dtype"] != "float32" or not isinstance(shape, list) or len(shape) != 1
            or type(shape[0]) is not int or not 1 <= shape[0] <= 65536
            or not isinstance(values, list) or len(values) != shape[0]
            or any(type(v) not in (int, float) for v in values)
        ):
            return False
        return any(not math.isfinite(v) for v in values)
    except (ValueError, UnicodeError, RecursionError, OverflowError, TypeError):
        return False


def classify_public_numeric(
    case_id: str, public_input: bytes, observed: bytes,
) -> PublicNumericDiagnostic:
    """Use exactly the registered numeric tolerance, exposing only bounded labels.

    Group indices are zero-based, width 32. Multi-group mismatch is descriptive,
    not a claim about the root cause or actual hardware warp synchronization.
    """
    reference = expected_output(case_id, public_input)
    n = len(reference)
    try:
        actual = parse_output(observed)
    except ValueError:
        nonfinite = _contains_nonfinite_numeric_values(observed)
        return PublicNumericDiagnostic(
            category="NON_FINITE_OUTPUT" if nonfinite else "OUTPUT_INVALID",
            input_size=n, output_finite=False if nonfinite else None,
        )
    if len(actual) != n:
        return PublicNumericDiagnostic(
            category="SHAPE_MISMATCH", input_size=n, output_finite=True
        )
    oracle = NumericOracle(3e-5, 3e-5, False, False)
    groups = [
        offset // 32
        for offset in range(0, n, 32)
        if not oracle.check(actual[offset:offset+32], reference[offset:offset+32]).passed
    ]
    all_zero = all(value == 0.0 for value in actual)
    if not groups:
        return PublicNumericDiagnostic(
            category="PASS", input_size=n, output_finite=True, all_zero=all_zero
        )
    tail_group = n // 32 if n % 32 else None
    first = groups[0]
    category: Literal[
        "ALL_ZERO_OUTPUT", "GROUP_TAIL_MISMATCH", "MULTI_GROUP_MISMATCH",
        "FINITE_NUMERIC_MISMATCH",
    ] = (
        "ALL_ZERO_OUTPUT" if all_zero else "GROUP_TAIL_MISMATCH"
        if tail_group is not None and groups == [tail_group] else
        "MULTI_GROUP_MISMATCH" if len(groups) > 1 else "FINITE_NUMERIC_MISMATCH"
    )
    return PublicNumericDiagnostic(
        category=category, input_size=n, output_finite=True, all_zero=all_zero,
        first_failing_group=first, failing_group_is_tail=first == tail_group,
    )
