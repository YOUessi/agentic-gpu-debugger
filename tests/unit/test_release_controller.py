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
    module = root / "src/gpu_agent/__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
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
    def __init__(
        self,
        *,
        outcome: str = "passed",
        replace_node: str | None = None,
        valid_junit: bool = True,
        error: BaseException | None = None,
    ):
        self.outcome = outcome
        self.replace_node = replace_node
        self.valid_junit = valid_junit
        self.error = error
        self.argv: list[str] = []

    def execute(self, argv, cwd, timeout_seconds, max_log_bytes, *, env=None):
        from gpu_agent.execution.process import ProcessCapture

        if self.error is not None:
            raise self.error
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
                    "gpu_agent_origin": str(cwd / "src/gpu_agent/__init__.py"),
                }
            )
        )
        junit_path.write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase classname="tests.test_live" name="test_positive" />'
            "</testsuite>"
            if self.valid_junit
            else '<testsuite tests="1" failures="0" skipped="0"></testsuite>'
        )
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
        source_snapshot=lambda _root, _snapshot: (repository, "f" * 64),
    )
    return controller, store


def _collector_release_resolver(tmp_path: Path):
    from gpu_agent.benchmark.release import ReleaseEvidenceRoots, _ReleaseEvidenceResolver
    from gpu_agent.store import RunStore

    controller, public = _controller(tmp_path, FakeReleaseProcess())
    run_id = controller.collect(24)
    roots = ReleaseEvidenceRoots(
        development_evaluation_run_id="1" * 32,
        holdout_evaluation_run_id="2" * 32,
        private_binding_run_id="3" * 32,
        release_test_run_id=run_id,
    )
    resolver = _ReleaseEvidenceResolver(
        roots,
        public,
        RunStore(tmp_path / "evaluator", visibility="evaluator"),
        object(),
        controller.repository,
        _snapshot(),
    )
    evaluation_binding = _binding().model_copy(update={"purpose": "evaluation"})
    return resolver, public, evaluation_binding, run_id


def test_collector_inventory_is_accepted_by_release_resolver(tmp_path):
    resolver, _public, evaluation_binding, _run_id = _collector_release_resolver(tmp_path)

    counts = resolver._release_tests(evaluation_binding, 24)

    assert counts.model_dump() == {
        "expected": 1,
        "executed": 1,
        "skipped_required": 0,
        "failed": 0,
    }


def test_release_resolver_rejects_extra_artifact_in_collector_inventory(tmp_path):
    from gpu_agent.benchmark.release import _ReleaseRootError
    from gpu_agent.contracts import ArtifactRef

    resolver, public, evaluation_binding, run_id = _collector_release_resolver(tmp_path)
    manifest_path = public.root / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    artifact_id = "f" * 32
    manifest["artifact_refs"].append(
        ArtifactRef(
            id=artifact_id,
            run_id=run_id,
            name="release/unrelated.json",
            sha256="0" * 64,
            visibility="public",
            relative_path=f"{run_id}/artifacts/{artifact_id}",
            byte_count=2,
        ).model_dump(mode="json")
    )
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())

    with pytest.raises(_ReleaseRootError):
        resolver._release_tests(evaluation_binding, 24)


def test_collects_fixed_same_commit_release_evidence(tmp_path):
    from gpu_agent.benchmark.release import ReleaseTestEvidence
    from gpu_agent.release_controller import verify_persisted_release_artifacts

    process = FakeReleaseProcess()
    controller, store = _controller(tmp_path, process)
    run_id = controller.collect(24)

    run = store.load(run_id)
    assert run.status.value == "COMPLETED"
    assert process.argv[1:3] == ["-I", "-c"]
    assert process.argv[4:7] == ["-m", "release_evidence", "--require-live"]
    evidence_ref = next(
        ref for ref in run.artifact_refs if ref.name == "release/test-evidence.json"
    )
    evidence = ReleaseTestEvidence.model_validate_json(store.read(evidence_ref))
    assert evidence.corpus_cutoff == 24
    assert evidence.prompt_version == "v2"
    assert evidence.test_counts.model_dump() == {
        "expected": 1,
        "executed": 1,
        "skipped_required": 0,
        "failed": 0,
    }
    invocation_ref = next(
        ref for ref in run.artifact_refs if ref.name == "release/test-invocation.json"
    )
    invocation = store.read(invocation_ref)
    assert b"<controller-temp>" in invocation and b"<controller-bootstrap>" in invocation
    assert {
        "release/test-allowlist.json",
        "release/pytest-evidence.json",
        "release/pytest-junit.xml",
        "release/pytest-stdout.log",
        "release/pytest-stderr.log",
    }.issubset({ref.name for ref in run.artifact_refs})
    verify_persisted_release_artifacts(store, run)


@pytest.mark.parametrize(
    "process",
    [
        FakeReleaseProcess(outcome="skipped"),
        FakeReleaseProcess(outcome="failed"),
        FakeReleaseProcess(replace_node="tests/test_live.py::test_negative"),
        FakeReleaseProcess(valid_junit=False),
        FakeReleaseProcess(error=RuntimeError("controller bug")),
    ],
)
def test_skips_failures_and_collection_drift_fail_closed(tmp_path, process):
    controller, store = _controller(tmp_path, process)
    with pytest.raises((ValueError, RuntimeError)):
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
