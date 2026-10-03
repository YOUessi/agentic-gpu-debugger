"""GPU acceptance with a unique module name for pytest's default import mode."""

from pathlib import Path

import pytest

from gpu_agent.benchmark.diversity import run_diversity


@pytest.mark.gpu
@pytest.mark.container
@pytest.mark.parametrize("case", [f"case_{n:04d}" for n in range(17, 23)])
def test_diverse_core_computation(tmp_path, case):
    result = run_diversity(Path(__file__).resolve().parents[2], tmp_path / "evidence", (case,))
    assert result["gpu_ready"], result
    assert result["passed"], result
