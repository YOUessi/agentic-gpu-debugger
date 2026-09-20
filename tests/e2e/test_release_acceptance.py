import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.release


def test_current_release_requires_frozen_native_evidence():
    """Pass only when frozen claims match the selected same-commit evidence graph."""
    from gpu_agent.benchmark.release import ReleaseGate, ReleaseManifest
    from gpu_agent.cli import _derive_release_evidence
    from gpu_agent.store import read_regular

    repository = Path(__file__).resolve().parents[2]
    manifest_path = Path(
        os.environ.get(
            "GPU_AGENT_RELEASE_MANIFEST",
            repository / "evaluation/release-manifest.json",
        )
    )
    selection_path = Path(
        os.environ.get(
            "GPU_AGENT_RELEASE_SELECTION",
            repository / "evaluation/release-selection.json",
        )
    )
    assert manifest_path.is_file(), "release manifest is not frozen"
    assert selection_path.is_file(), "release evidence selection is not frozen"
    manifest = ReleaseManifest.model_validate_json(
        read_regular(manifest_path.absolute(), 1024 * 1024)
    )
    evidence = _derive_release_evidence(selection_path, repository)
    result = ReleaseGate().check(manifest, evidence)
    assert result.passed, result.reason_codes
