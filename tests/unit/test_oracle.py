import json
import math

import pytest


@pytest.mark.parametrize(
    "actual,expected,atol,rtol,want",
    [
        ([1.001], [1.0], 0.002, 0, True),
        ([101.0], [100.0], 0, 0.02, True),
        ([1.1], [1.0], 0.001, 0.001, False),
        ([1.0, 2.0], [1.0], 0, 0, False),
        ([[1.0]], [1.0], 0, 0, False),
        ([math.nan], [math.nan], 0, 0, False),
        ([math.inf], [math.inf], 0, 0, False),
        ([True], [1.0], 0, 0, False),
    ],
)
def test_numeric_oracle(actual, expected, atol, rtol, want):
    from gpu_agent.verification.oracle import NumericOracle

    assert NumericOracle(atol, rtol, False, False).check(actual, expected).passed is want


def test_explicit_nonfinite_policy():
    from gpu_agent.verification.oracle import NumericOracle

    oracle = NumericOracle(0, 0, True, True)
    assert oracle.check([math.nan, math.inf], [math.nan, math.inf]).passed
    assert not oracle.check([math.inf], [-math.inf]).passed
    assert not oracle.check([math.nan], [1]).passed


@pytest.mark.parametrize(
    "output",
    [
        b"",
        b"PASS",
        b"{}",
        b'{"dtype":"float32","shape":[1],"values":[3]} PASS',
        b'{"dtype":"float32","shape":[1],"values":[0],"values":[3]}',
        b'{"dtype":"float32","shape":[1],"values":[NaN]}',
        b'{"dtype":"float32","shape":[true],"values":[3]}',
        b'{"dtype":"float32","shape":[2],"values":[3]}',
        b'{"dtype":"float32","shape":[1],"values":[3],"PASS":true}',
    ],
)
def test_untrusted_output_is_strict_json(output):
    from gpu_agent.verification.oracle import parse_output

    with pytest.raises(ValueError):
        parse_output(output)


def test_numeric_protocol_and_host_reference():
    from gpu_agent.verification.oracle import parse_output, reference_add

    assert parse_output(
        json.dumps({"dtype": "float32", "shape": [2], "values": [3, -1.5]}).encode()
    ) == [3.0, -1.5]
    assert reference_add([1, -2], [2, 0.5]) == [3.0, -1.5]


def test_huge_integer_output_is_rejected_without_overflow():
    from gpu_agent.verification.oracle import NumericOracle, parse_output

    output = json.dumps({"dtype": "float32", "shape": [1], "values": [10**400]}).encode()
    with pytest.raises(ValueError):
        parse_output(output)
    assert not NumericOracle(0, 0, False, False).check([10**400], [1]).passed
