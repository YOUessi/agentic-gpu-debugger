import pytest

pytestmark = pytest.mark.release


def test_current_release_is_truthfully_closed_without_frozen_evidence(tmp_path, request):
    """Offline proves fail-closed; --require-live requires real same-commit stores."""
    import os
    import subprocess
    from pathlib import Path

    from gpu_agent.benchmark.release import ReleaseEvidenceIndex, ReleaseGate, ReleaseManifest
    from gpu_agent.store import RunStore

    repository = Path(__file__).resolve().parents[2]
    path = repository / "evaluation/release-manifest.json"
    require_live = request.config.getoption("--require-live")
    if not path.exists():
        if require_live:
            pytest.fail("release manifest is not frozen")
        assert not path.exists()
        return
    public_root = os.environ.get("GPU_AGENT_RELEASE_PUBLIC_ROOT")
    evaluator_root = os.environ.get("GPU_AGENT_RELEASE_EVALUATOR_ROOT")
    if public_root is None or evaluator_root is None:
        if require_live:
            pytest.fail("release evidence roots are not configured")
        public_root = str(tmp_path / "empty-public")
        evaluator_root = str(tmp_path / "empty-evaluator")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    index = ReleaseEvidenceIndex.derive(
        RunStore(Path(public_root)),
        RunStore(Path(evaluator_root), visibility="evaluator"),
        current_commit=commit,
    )
    result = ReleaseGate().check(ReleaseManifest.model_validate_json(path.read_bytes()), index)
    if require_live:
        assert result.passed, result.reason_codes
    else:
        assert not result.passed
