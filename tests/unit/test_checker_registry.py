"""Only trusted, pre-registered oracle implementations are resolvable."""

import pytest


def test_expected_output_and_cpu_reference_checkers():
    from gpu_agent.verification.oracle import (
        CPUReferenceChecker,
        ExpectedOutputChecker,
        NumericOracle,
        reference_add,
    )

    oracle = NumericOracle(1e-6, 1e-6, False, False)
    expected = ExpectedOutputChecker([3.0], oracle)
    assert expected([3.0]).passed
    cpu = CPUReferenceChecker(reference_add, oracle)
    assert cpu.check([3.0], [1.0], [2.0]).passed
    assert not cpu.check([4.0], [1.0], [2.0]).passed


def test_registry_never_dynamic_imports_user_checker():
    from gpu_agent.verification.oracle import (
        ExpectedOutputChecker,
        NumericOracle,
        TrustedCheckerRegistry,
    )

    checker = ExpectedOutputChecker([3.0], NumericOracle(0, 0, False, False))
    registry = TrustedCheckerRegistry({"expected-v1": checker})
    assert registry.resolve("expected-v1") is checker
    with pytest.raises(ValueError, match="not registered"):
        registry.resolve("user.module:checker")
    with pytest.raises(ValueError, match="duplicate"):
        registry.register("expected-v1", checker)
