"""One-time controller configuration; no private signing or provider keys are persisted."""

from __future__ import annotations

import os
import stat
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from gpu_agent.agent.prompts import PROMPT_VERSION
from gpu_agent.agent.provider import OpenAIProviderSettings
from gpu_agent.benchmark.evaluation import (
    EvaluationProviderPolicy,
    PricingAttestation,
)
from gpu_agent.benchmark.ledger import CorpusFamily
from gpu_agent.provenance import capture_repository_snapshot
from gpu_agent.store import read_regular, reject_symlinks, sync_directory


def provision_production_family(
    *,
    controller_root: Path,
    public_store: Path,
    evaluator_store: Path,
    repository: Path,
    schedule_public_key: Path,
) -> CorpusFamily:
    key = read_regular(schedule_public_key.absolute(), 64 * 1024)
    return CorpusFamily.provision_production(
        controller_root.absolute(),
        public_store=public_store.absolute(),
        evaluator_store=evaluator_store.absolute(),
        repository=repository.absolute(),
        schedule_public_key=key,
    )


def reviewed_pricing_attestation(
    *,
    repository: Path,
    expected_commit: str,
    input_usd_per_million: float,
    output_usd_per_million: float,
    source_uri: str,
    reviewed_at: datetime,
    source_content_hash: str,
) -> PricingAttestation:
    snapshot = capture_repository_snapshot(repository, expected_commit=expected_commit)
    settings = OpenAIProviderSettings.from_environment()
    endpoint_host = urlsplit(settings.endpoint or "").hostname or ""
    if not endpoint_host or not settings.model:
        raise ValueError("provider endpoint and model must be configured")
    if endpoint_host != "api.openai.com" and not settings.supports_store_false:
        raise ValueError("provider must support store=false")
    provider = "deepseek-responses" if endpoint_host == "api.deepseek.com" else "openai-responses"
    preliminary = PricingAttestation.reviewed(
        provider=provider,
        model=settings.model,
        commit=snapshot.commit,
        model_config_hash="0" * 64,
        input_usd_per_million=input_usd_per_million,
        output_usd_per_million=output_usd_per_million,
        source_uri=source_uri,
        reviewed_at=reviewed_at,
        source_content_hash=source_content_hash,
    )
    policy = EvaluationProviderPolicy(
        provider=provider,
        endpoint_host=endpoint_host,
        configured_model=settings.model,
        allowed_response_models=[settings.model],
        prompt_version=PROMPT_VERSION,
        pricing_hash=preliminary.rate_card_hash,
    )
    return preliminary.model_copy(update={"model_config_hash": policy.sha256})


def write_private_new(path: Path, content: bytes) -> None:
    """Create one owner-only controller file without following links or overwriting."""
    target = path.absolute()
    reject_symlinks(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target.parent, 0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = target.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ValueError("controller file permissions are unsafe")
        sync_directory(target.parent)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
