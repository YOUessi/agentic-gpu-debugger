import json
from pathlib import Path

import pytest


def _snapshot():
    from gpu_agent.contracts import RepositorySnapshot

    return RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True)


def _binding():
    from gpu_agent.contracts import RunBinding

    return RunBinding(
        repository=_snapshot(),
        purpose="release_acceptance",
        toolchain_lock_hash="c" * 64,
        prompt_version="v2",
        model_config_hash="d" * 64,
        corpus_ledger_namespace_hash="e" * 64,
    )


def _repository(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    root = tmp_path / "repository"
    (root / "evaluation").mkdir(parents=True)
    nodes = [
        {
            "node_id": "tests/test_live.py::test_positive",
            "required_markers": ["gpu", "release_evidence"],
        }
    ]
    (root / "evaluation/release-test-allowlist.json").write_text(
        json.dumps({"schema_version": 1, "nodes": nodes})
    )
    return root, nodes


class FakeReleaseProcess:
    def __init__(self, *, outcome: str = "passed", replace_node: str | None = None):
        self.outcome = outcome
        self.replace_node = replace_node
        self.argv: list[str] = []

    def execute(self, argv, cwd, timeout_seconds, max_log_bytes, *, env=None):
        from gpu_agent.execution.process import ProcessCapture

        self.argv = list(argv)
        report_path = Path(
            next(
                item.split("=", 1)[1]
                for item in argv
                if item.startswith("--release-evidence-report=")
            )
        )
        junit_path = Path(
            next(item.split("=", 1)[1] for item in argv if item.startswith("--junitxml="))
        )
        node = self.replace_node or "tests/test_live.py::test_positive"
        outcomes = {"passed": [], "skipped": [], "failed": []}
        outcomes[self.outcome].append(node)
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "collected_node_ids": [node],
                    "markers": {node: ["gpu", "release_evidence"]},
                    "passed_node_ids": outcomes["passed"],
                    "skipped_node_ids": outcomes["skipped"],
                    "failed_node_ids": outcomes["failed"],
                    "exit_status": 0 if self.outcome == "passed" else 1,
                }
            )
        )
        junit_path.write_text('<testsuite tests="1" failures="0" skipped="0"></testsuite>')
        return ProcessCapture(
            0 if self.outcome == "passed" else 1,
            b"one passed",
            b"",
            False,
        )


def _controller(tmp_path: Path, process: FakeReleaseProcess):
    from gpu_agent.release_controller import ReleaseEvidenceController
    from gpu_agent.store import RunStore

    repository, _ = _repository(tmp_path)
    store = RunStore(tmp_path / "runs")
    controller = ReleaseEvidenceController(
        store,
        repository,
        _binding(),
        process=process,
        snapshot_capture=lambda *_args, **_kwargs: _snapshot(),
    )
    return controller, store


def test_collects_fixed_same_commit_release_evidence(tmp_path):
    from gpu_agent.benchmark.release import ReleaseTestEvidence

    process = FakeReleaseProcess()
    controller, store = _controller(tmp_path, process)
    run_id = controller.collect(24)

    run = store.load(run_id)
    assert run.status.value == "COMPLETED"
    assert process.argv[1:7] == ["-I", "-m", "pytest", "-m", "release_evidence", "--require-live"]
    evidence_ref = next(
        ref for ref in run.artifact_refs if ref.name == "release/test-evidence.json"
    )
    evidence = ReleaseTestEvidence.model_validate_json(store.read(evidence_ref))
    assert evidence.corpus_cutoff == 24
    assert evidence.test_counts.model_dump() == {
        "expected": 1,
        "executed": 1,
        "skipped_required": 0,
        "failed": 0,
    }
    invocation_ref = next(
        ref for ref in run.artifact_refs if ref.name == "release/test-invocation.json"
    )
    assert b"<controller-temp>" in store.read(invocation_ref)


@pytest.mark.parametrize(
    "process",
    [
        FakeReleaseProcess(outcome="skipped"),
        FakeReleaseProcess(outcome="failed"),
        FakeReleaseProcess(replace_node="tests/test_live.py::test_negative"),
    ],
)
def test_skips_failures_and_collection_drift_fail_closed(tmp_path, process):
    controller, store = _controller(tmp_path, process)
    with pytest.raises(ValueError):
        controller.collect(24)
    runs = [path.name for path in store.root.iterdir() if len(path.name) == 32]
    assert len(runs) == 1
    run = store.load(runs[0])
    assert run.status.value == "FAILED"
    assert not any(ref.name == "release/test-evidence.json" for ref in run.artifact_refs)


def test_rejects_unbound_or_non_public_controller(tmp_path):
    from gpu_agent.contracts import RunBinding
    from gpu_agent.release_controller import ReleaseEvidenceController
    from gpu_agent.store import RunStore

    repository, _ = _repository(tmp_path)
    incomplete = RunBinding(repository=_snapshot(), purpose="release_acceptance")
    with pytest.raises(ValueError, match="incomplete"):
        ReleaseEvidenceController(
            RunStore(tmp_path / "runs"), repository, incomplete, process=FakeReleaseProcess()
        )
    with pytest.raises(ValueError, match="public"):
        ReleaseEvidenceController(
            RunStore(tmp_path / "evaluator", visibility="evaluator"),
            repository,
            _binding(),
            process=FakeReleaseProcess(),
        )
