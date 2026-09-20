from datetime import UTC, datetime

import pytest


def test_reviewed_pricing_is_bound_to_observed_provider_and_commit(tmp_path, monkeypatch):
    from gpu_agent.benchmark.controller_config import reviewed_pricing_attestation
    from gpu_agent.contracts import RepositorySnapshot

    snapshot = RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True)
    monkeypatch.setattr(
        "gpu_agent.benchmark.controller_config.capture_repository_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")
    monkeypatch.setenv("GPU_AGENT_STORE_FALSE_SUPPORTED", "1")

    result = reviewed_pricing_attestation(
        repository=tmp_path,
        expected_commit="a" * 40,
        input_usd_per_million=0.28,
        output_usd_per_million=0.42,
        source_uri="https://api-docs.deepseek.com/quick_start/pricing",
        reviewed_at=datetime(2026, 9, 20, tzinfo=UTC),
        source_content_hash="c" * 64,
    )

    assert result.source == "REVIEWED"
    assert result.provider == "deepseek-responses"
    assert result.model == "deepseek-chat"
    assert result.repository_commit == "a" * 40
    assert result.model_config_hash != "0" * 64


def test_reviewed_pricing_rejects_provider_without_store_false(tmp_path, monkeypatch):
    from gpu_agent.benchmark.controller_config import reviewed_pricing_attestation
    from gpu_agent.contracts import RepositorySnapshot

    monkeypatch.setattr(
        "gpu_agent.benchmark.controller_config.capture_repository_snapshot",
        lambda *_args, **_kwargs: RepositorySnapshot(
            commit="a" * 40, tracked_tree_hash="b" * 64, clean=True
        ),
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://third-party.example")
    monkeypatch.setenv("OPENAI_MODEL", "model")
    monkeypatch.delenv("GPU_AGENT_STORE_FALSE_SUPPORTED", raising=False)
    with pytest.raises(ValueError, match="store=false"):
        reviewed_pricing_attestation(
            repository=tmp_path,
            expected_commit="a" * 40,
            input_usd_per_million=1,
            output_usd_per_million=1,
            source_uri="https://third-party.example/pricing",
            reviewed_at=datetime.now(UTC),
            source_content_hash="c" * 64,
        )


def test_private_controller_file_is_exclusive_and_owner_only(tmp_path):
    from gpu_agent.benchmark.controller_config import write_private_new

    path = tmp_path / "controller" / "pricing.json"
    write_private_new(path, b"first")
    assert path.read_bytes() == b"first"
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        write_private_new(path, b"replacement")
    assert path.read_bytes() == b"first"
