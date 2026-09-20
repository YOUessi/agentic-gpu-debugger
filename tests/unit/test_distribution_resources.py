from pathlib import Path

import pytest

from gpu_agent._resources import runtime_resource
from gpu_agent.environment import load_toolchain_lock

REQUIRED_RUNTIME_FILES = (
    "containers/Dockerfile",
    "containers/runner.py",
    "containers/toolchain.lock.json",
    "benchmarks/corpus-registry.json",
    "benchmarks/seed-batch.json",
    "benchmarks/development_truth/case_0001/case.json",
    "benchmarks/development_truth/case_0001/reference.cu",
    "benchmarks/harness/vector_api.h",
    "benchmarks/harness/vector_io.cpp",
    "benchmarks/harness/vendor/LICENSE.MIT",
    "benchmarks/harness/vendor/json.hpp",
    "benchmarks/public/case_0000/public_input/kernel.cu",
    "benchmarks/public/case_0001/public_input/kernel.cu",
    "benchmarks/public/case_0002/public_input/kernel.cu",
    "benchmarks/public/case_0003/public_input/kernel.cu",
    "benchmarks/public/case_0004/public_input/kernel.cu",
    "benchmarks/templates/mutations.json",
    "evaluation/modes.json",
    "evaluation/protocol.md",
    "evaluation/rubric.md",
    "knowledge/retrieval-eval.json",
    "knowledge/sources.json",
)


def test_required_runtime_resources_are_present_and_regular() -> None:
    for name in REQUIRED_RUNTIME_FILES:
        path = runtime_resource(name)
        assert path.is_file(), name
        assert not path.is_symlink(), name

    lock = runtime_resource("containers/toolchain.lock.json")
    assert load_toolchain_lock(lock).lock_hash


@pytest.mark.parametrize("name", ["../LICENSE", "/etc/passwd", "containers/../LICENSE"])
def test_runtime_resource_rejects_paths_outside_distribution(name: str) -> None:
    with pytest.raises(ValueError, match="invalid runtime resource path"):
        runtime_resource(name)


def test_runtime_resource_uses_distribution_namespace() -> None:
    path = runtime_resource("evaluation/modes.json")
    assert path.parts[-3:] == ("agentic-gpu-debugger", "evaluation", "modes.json") or path == (
        Path(__file__).resolve().parents[2] / "evaluation/modes.json"
    )
