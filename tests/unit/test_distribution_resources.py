import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from rich.text import Text
from typer.main import get_command
from typer.testing import CliRunner

from gpu_agent._resources import runtime_resource
from gpu_agent.cli import app
from gpu_agent.environment import load_toolchain_lock

runner = CliRunner()

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
    "docs/v2-operator-runbook.md",
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


@pytest.mark.parametrize("color", [False, True])
def test_operator_commands_are_exposed(color: bool, monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    scoring = runner.invoke(app, ["benchmark", "score-holdout", "--help"], color=color)
    freezing = runner.invoke(app, ["release", "freeze-selection", "--help"], color=color)

    # Verify real registered options independently of Rich's terminal rendering.
    root = get_command(app)
    for group, command, expected in (
        ("benchmark", "score-holdout", {"--labels", "--private-binding-run-id"}),
        ("release", "freeze-selection", {"--release-test-run-id", "--output"}),
    ):
        params = root.commands[group].commands[command].params
        assert expected <= {option for param in params for option in param.opts}

    assert scoring.exit_code == 0
    scoring_text = Text.from_ansi(scoring.output).plain
    assert "--labels" in scoring_text and "--private-binding-run-id" in scoring_text
    assert freezing.exit_code == 0
    freezing_text = Text.from_ansi(freezing.output).plain
    assert "--release-test-run-id" in freezing_text and "--output" in freezing_text


def test_operator_runbook_is_in_built_distributions(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    output = tmp_path / "dist"
    build = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output)],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, f"Distribution build failed:\n{build.stdout}\n{build.stderr}"

    wheel = next(output.glob("*.whl"))
    source = next(output.glob("*.tar.gz"))
    expected = "share/agentic-gpu-debugger/docs/v2-operator-runbook.md"
    with zipfile.ZipFile(wheel) as archive:
        assert any(name.endswith(expected) for name in archive.namelist())
        for number in range(23):
            task = (
                f"share/agentic-gpu-debugger/benchmarks/public/case_{number:04d}/"
                "public_input/task.json"
            )
            assert any(name.endswith(task) for name in archive.namelist())
    with tarfile.open(source, "r:gz") as archive:
        assert any(member.name.endswith("docs/v2-operator-runbook.md") for member in archive)
