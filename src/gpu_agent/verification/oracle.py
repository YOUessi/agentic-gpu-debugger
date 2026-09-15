"""Strict numeric stdout protocol and trusted host-side float32 reference."""

import json
import math
import struct
from collections.abc import Callable, Sequence
from typing import Protocol

from gpu_agent.verification.models import OracleResult


def _number(value: object) -> bool:
    return type(value) in {int, float}


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate output key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("nonfinite JSON number")


def parse_output(output: bytes) -> list[float]:
    if not output or len(output) > 2 * 1024 * 1024:
        raise ValueError("missing or oversized output")
    try:
        data = json.loads(output, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid numeric output") from exc
    if not isinstance(data, dict) or set(data) != {"dtype", "shape", "values"}:
        raise ValueError("expected numeric output fields")
    shape, values = data["shape"], data["values"]
    if (
        data["dtype"] != "float32"
        or not isinstance(shape, list)
        or len(shape) != 1
        or type(shape[0]) is not int
        or not 1 <= shape[0] <= 65536
        or not isinstance(values, list)
        or len(values) != shape[0]
    ):
        raise ValueError("invalid output shape or dtype")
    try:
        if any(not _number(x) or not math.isfinite(x) for x in values):
            raise ValueError("output must contain finite numbers")
    except OverflowError as exc:
        raise ValueError("output number exceeds numeric range") from exc
    return [float(x) for x in values]


class NumericOracle:
    def __init__(self, atol: float, rtol: float, allow_nan: bool, allow_inf: bool) -> None:
        if any(not math.isfinite(x) or x < 0 for x in (atol, rtol)):
            raise ValueError("tolerances must be finite and nonnegative")
        self.atol, self.rtol, self.allow_nan, self.allow_inf = atol, rtol, allow_nan, allow_inf

    def check(self, actual: Sequence[object], expected: Sequence[object]) -> OracleResult:
        reason = None
        if len(actual) != len(expected):
            reason = "SHAPE_MISMATCH"
        else:
            for a, e in zip(actual, expected, strict=True):
                if not _number(a) or not _number(e):
                    reason = "NON_NUMERIC_OR_SHAPE"
                    break
                assert isinstance(a, (float, int)) and isinstance(e, (float, int))
                try:
                    a, e = float(a), float(e)
                except OverflowError:
                    reason = "NUMERIC_RANGE"
                    break
                if math.isnan(a) or math.isnan(e):
                    equal = self.allow_nan and math.isnan(a) and math.isnan(e)
                elif math.isinf(a) or math.isinf(e):
                    equal = self.allow_inf and a == e
                else:
                    equal = abs(a - e) <= self.atol + self.rtol * abs(e)
                if not equal:
                    reason = "NUMERIC_MISMATCH"
                    break
        return OracleResult(
            passed=reason is None,
            atol=self.atol,
            rtol=self.rtol,
            nan_policy="paired" if self.allow_nan else "reject",
            inf_policy="same-sign" if self.allow_inf else "reject",
            failure_reason=reason,
        )


def _float32(value: float) -> float:
    return float(struct.unpack("f", struct.pack("f", value))[0])


def reference_add(a: list[float], b: list[float]) -> list[float]:
    return [_float32(_float32(x) + _float32(y)) for x, y in zip(a, b, strict=True)]


class TrustedChecker(Protocol):
    def __call__(self, actual: Sequence[object], expected: Sequence[object]) -> OracleResult: ...


class ExpectedOutputChecker:
    def __init__(self, expected: Sequence[object], oracle: NumericOracle) -> None:
        self._expected, self._oracle = tuple(expected), oracle

    def __call__(self, actual: Sequence[object], expected: Sequence[object] = ()) -> OracleResult:
        return self._oracle.check(actual, self._expected)


class CPUReferenceChecker:
    def __init__(
        self,
        reference: Callable[[list[float], list[float]], list[float]],
        oracle: NumericOracle,
    ) -> None:
        self._reference, self._oracle = reference, oracle

    def check(self, actual: Sequence[object], a: list[float], b: list[float]) -> OracleResult:
        return self._oracle.check(actual, self._reference(a, b))


class TrustedCheckerRegistry:
    """Resolve only controller-registered callables; never import user paths."""

    def __init__(self, checkers: dict[str, TrustedChecker] | None = None) -> None:
        self._checkers = dict(checkers or {})

    def register(self, checker_id: str, checker: TrustedChecker) -> None:
        if not checker_id or checker_id in self._checkers or not callable(checker):
            raise ValueError("invalid or duplicate checker")
        self._checkers[checker_id] = checker

    def resolve(self, checker_id: str) -> TrustedChecker:
        try:
            return self._checkers[checker_id]
        except KeyError as error:
            raise ValueError("checker is not registered") from error
