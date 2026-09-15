import pytest

pytestmark = pytest.mark.release


def test_current_release_requires_frozen_manifest():
    """Intentionally fails until the paid evaluation and 16+8 corpus exist."""
    from pathlib import Path

    from gpu_agent.benchmark.release import ReleaseGate, ReleaseManifest

    path = Path(__file__).resolve().parents[2] / "evaluation/release-manifest.json"
    assert path.exists(), "release manifest is not frozen"
    manifest = ReleaseManifest.model_validate_json(path.read_bytes())
    result = ReleaseGate().check(manifest)
    assert result.passed, result.reason_codes
