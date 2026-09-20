import subprocess


def test_real_controller_runs_tests_from_exact_commit_archive(tmp_path):
    from gpu_agent.contracts import RunBinding
    from gpu_agent.provenance import capture_repository_snapshot
    from gpu_agent.release_controller import ReleaseEvidenceController
    from gpu_agent.store import RunStore

    repository = tmp_path / "repository"
    (repository / "evaluation").mkdir(parents=True)
    (repository / "src/gpu_agent").mkdir(parents=True)
    (repository / "tests").mkdir()
    (repository / "src/gpu_agent/__init__.py").write_text("")
    (repository / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nmarkers=["release_evidence: controller evidence"]\n'
    )
    node_id = "tests/test_positive.py::test_positive"
    (repository / "evaluation/release-test-allowlist.json").write_text(
        """{
  "schema_version": 1,
  "nodes": [{
    "node_id": "tests/test_positive.py::test_positive",
    "required_markers": ["release_evidence"]
  }]
}
"""
    )
    (repository / "tests/test_positive.py").write_text(
        "import pytest\n\n@pytest.mark.release_evidence\ndef test_positive():\n    assert True\n"
    )
    (repository / "tests/conftest.py").write_text(
        """import json
from pathlib import Path
import pytest

items = {}
passed = []

def pytest_addoption(parser):
    parser.addoption("--require-live", action="store_true")
    parser.addoption("--release-evidence-report")

def pytest_collection_finish(session):
    for item in session.items:
        items[item.nodeid] = sorted({marker.name for marker in item.iter_markers()})

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.passed:
        passed.append(item.nodeid)

def pytest_sessionfinish(session, exitstatus):
    import gpu_agent
    payload = {
        "schema_version": 1,
        "collected_node_ids": sorted(items),
        "markers": {key: items[key] for key in sorted(items)},
        "passed_node_ids": sorted(passed),
        "skipped_node_ids": [],
        "failed_node_ids": [],
        "exit_status": int(exitstatus),
        "gpu_agent_origin": str(Path(gpu_agent.__file__).resolve()),
    }
    Path(session.config.getoption("--release-evidence-report")).write_text(
        json.dumps(payload)
    )
"""
    )
    for arguments in (
        ["init", "-q"],
        ["config", "user.name", "Test"],
        ["config", "user.email", "test@example.invalid"],
        ["add", "."],
        ["commit", "-qm", "release fixture"],
    ):
        subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True)
    snapshot = capture_repository_snapshot(repository)
    binding = RunBinding(
        repository=snapshot,
        purpose="release_acceptance",
        toolchain_lock_hash="1" * 64,
        prompt_version="v2",
        model_config_hash="2" * 64,
        corpus_ledger_namespace_hash="3" * 64,
    )
    store = RunStore(tmp_path / "runs")
    run_id = ReleaseEvidenceController(store, repository, binding).collect(1)
    run = store.load(run_id)
    assert run.status.value == "COMPLETED"
    assert node_id.encode() in store.read(
        next(ref for ref in run.artifact_refs if ref.name == "release/pytest-evidence.json")
    )
