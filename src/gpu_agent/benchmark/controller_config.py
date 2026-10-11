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
from gpu_agent.store import RunStore, read_regular, reject_symlinks, sync_directory


def _overlap(first: Path, second: Path) -> bool:
    left, right = first.resolve(strict=True), second.resolve(strict=True)
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _owner_only_directory(path: Path) -> None:
    reject_symlinks(path)
    info = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("production store directory is unavailable or unsafe")


def validate_production_store_configuration(
    family: CorpusFamily,
    repository: Path,
) -> tuple[RunStore, RunStore]:
    """Open the two exact pinned production stores without creating or repairing paths."""
    public_value = os.environ.get("GPU_AGENT_RUN_ROOT")
    evaluator_value = os.environ.get("GPU_AGENT_EVALUATOR_ROOT")
    if not public_value or not evaluator_value:
        raise ValueError("production store environment is incomplete")
    public_path = Path(public_value)
    evaluator_parent = Path(evaluator_value)
    if not public_path.is_absolute() or not evaluator_parent.is_absolute():
        raise ValueError("production store paths must be absolute")
    evaluator_path = evaluator_parent / "runs"
    repo = repository.absolute()
    for path in (public_path, evaluator_parent, evaluator_path, family.root):
        _owner_only_directory(path)
    reject_symlinks(repo)
    if not repo.is_dir():
        raise ValueError("repository is unavailable")
    if any(
        _overlap(left, right)
        for left, right in (
            (public_path, evaluator_parent),
            (public_path, family.root),
            (public_path, repo),
            (evaluator_parent, family.root),
            (evaluator_parent, repo),
            (family.root, repo),
        )
    ):
        raise ValueError("production controller paths overlap")
    public = RunStore(public_path, visibility="public")
    evaluator = RunStore(evaluator_path, visibility="evaluator")
    family.require_store(public)
    family.require_store(evaluator)
    if public.identity == evaluator.identity:
        raise ValueError("production stores must be distinct")
    return public, evaluator


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
