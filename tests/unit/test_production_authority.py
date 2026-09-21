import hashlib
from datetime import UTC, datetime

import pytest


def test_production_family_stores_only_schedule_public_key(tmp_path):
    import json

    from gpu_agent.benchmark.ledger import CorpusFamily

    public_key = b"-----BEGIN PUBLIC KEY-----\nreviewed\n-----END PUBLIC KEY-----\n"
    family = CorpusFamily.provision_production(
        tmp_path / "controller",
        public_store=tmp_path / "public",
        evaluator_store=tmp_path / "evaluator",
        repository=tmp_path / "repository",
        schedule_public_key=public_key,
    )

    assert family.schedule_authority_profile == "PRODUCTION"
    assert family.schedule_public_key_hash == hashlib.sha256(public_key).hexdigest()
    assert family.schedule_public_key_path.read_bytes() == public_key
    assert not list(family.root.rglob("*private*"))
    config = json.loads((family.root / "family.json").read_text())
    assert config["schema_version"] == 3
    assert config["public_store_pin"]["visibility"] == "public"
    assert config["evaluator_store_pin"]["visibility"] == "evaluator"
    assert config["public_store_pin"]["device"] >= 0
    assert config["public_store_pin"]["inode"] > 0


def _signing_request():
    from gpu_agent.benchmark.schedule_authority import (
        CorpusUniverseEntry,
        EvaluationScheduleSigningRequest,
    )
    from gpu_agent.contracts import RepositorySnapshot, RunBinding

    binding = RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="evaluation",
        toolchain_lock_hash="c" * 64,
        prompt_version="prompt-v1",
        model_config_hash="d" * 64,
        corpus_ledger_namespace_hash="e" * 64,
    )
    universe = CorpusUniverseEntry(
        commit_sequence=1,
        transaction_id="1" * 32,
        registration_run_id="2" * 32,
        manifest_hash="3" * 64,
        identity_hash="4" * 64,
        template_hash="5" * 64,
    )
    return EvaluationScheduleSigningRequest(
        transaction_id="6" * 32,
        evaluation_run_id="7" * 32,
        target_store_hash="8" * 64,
        schedule_hash="9" * 64,
        binding=binding,
        selection="all",
        modes=["A", "B", "C", "D", "E"],
        split="development",
        repeats=3,
        random_seed=42,
        max_cost_usd=10,
        max_unit_cost_usd=1,
        case_templates={"case_0001": "template-1"},
        holdout_aliases=[],
        corpus_namespace_hash="e" * 64,
        corpus_visibility="public",
        corpus_cutoff=1,
        cutoff_reservation_hash="a" * 64,
        corpus_universe=[universe],
        corpus_universe_hash="b" * 64,
        authority_profile="PRODUCTION",
        authority_key_hash="c" * 64,
        queued_manifest_hash="d" * 64,
        event_prefix_hash="e" * 64,
        event_prefix_count=1,
        artifact_prefix_hash="f" * 64,
        artifact_prefix_count=1,
        child_inventory_hash="0" * 64,
    )


def test_external_schedule_client_is_bounded_and_returns_exact_request(tmp_path, monkeypatch):
    from gpu_agent.benchmark.schedule_authority import (
        EvaluationScheduleReceipt,
        ExternalCommandScheduleCommitClient,
    )
    from gpu_agent.execution.process import ProcessCapture, ProcessExecutor

    command = tmp_path / "schedule-authority"
    command.write_text("#!/bin/sh\nexit 1\n")
    command.chmod(0o700)
    request = _signing_request()
    receipt = EvaluationScheduleReceipt(
        request=request,
        payload_hash=hashlib.sha256(request.signed_bytes()).hexdigest(),
        signature_hex="1" * 128,
    )
    observed = {}

    def execute(self, argv, cwd, timeout_seconds, max_log_bytes, **kwargs):
        observed.update(
            argv=argv,
            cwd=cwd,
            timeout=timeout_seconds,
            limit=max_log_bytes,
            stdin=kwargs["stdin"],
            env=kwargs["env"],
        )
        return ProcessCapture(0, receipt.model_dump_json().encode(), b"", False)

    monkeypatch.setattr(ProcessExecutor, "execute", execute)
    result = ExternalCommandScheduleCommitClient(command).commit(request)

    assert result == receipt
    assert observed["argv"] == [str(command)]
    assert observed["stdin"] == request.model_dump_json().encode()
    assert observed["limit"] == 256 * 1024
    assert "OPENAI_API_KEY" not in observed["env"]


def test_external_schedule_client_rejects_mutable_command(tmp_path):
    from gpu_agent.benchmark.schedule_authority import ExternalCommandScheduleCommitClient

    command = tmp_path / "schedule-authority"
    command.write_text("#!/bin/sh\n")
    command.chmod(0o722)
    with pytest.raises(ValueError, match="unsafe"):
        ExternalCommandScheduleCommitClient(command)


def test_reviewed_pricing_attestation_is_source_bound():
    from gpu_agent.benchmark.evaluation import PricingAttestation

    attestation = PricingAttestation.reviewed(
        provider="openai-compatible-responses",
        model="deepseek-chat",
        commit="a" * 40,
        model_config_hash="b" * 64,
        input_usd_per_million=0.28,
        output_usd_per_million=0.42,
        source_uri="https://api-docs.deepseek.com/quick_start/pricing",
        reviewed_at=datetime(2026, 9, 20, tzinfo=UTC),
        source_content_hash="c" * 64,
    )

    assert attestation.source == "REVIEWED"
    assert attestation.cost(1_000_000, 1_000_000) == pytest.approx(0.70)
    assert attestation.rate_card_hash != attestation.model_config_hash


def test_reviewed_pricing_requires_https_source():
    from gpu_agent.benchmark.evaluation import PricingAttestation

    with pytest.raises(ValueError, match="HTTPS"):
        PricingAttestation.reviewed(
            provider="provider",
            model="model",
            commit="a" * 40,
            model_config_hash="b" * 64,
            input_usd_per_million=1,
            output_usd_per_million=1,
            source_uri="file:///tmp/pricing",
            reviewed_at=datetime.now(UTC),
            source_content_hash="c" * 64,
        )
