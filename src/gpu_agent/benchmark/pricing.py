"""Controller-owned loader for reviewed provider pricing attestations."""

import hashlib
import hmac
import json
import os
from pathlib import Path

from gpu_agent.benchmark.evaluation import PricingAttestation
from gpu_agent.contracts import RunBinding
from gpu_agent.store import read_regular, reject_symlinks


class ReviewedPricingRegistry:
    """Read-only signed registry; this package intentionally has no signing API."""

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        reject_symlinks(self.root)
        if not self.root.is_dir() or self.root.stat().st_mode & 0o077:
            raise ValueError("reviewed pricing registry is unavailable or unsafe")
        self._key_path = self.root / "registry.key"
        self._records_path = self.root / "attestations.json"
        reject_symlinks(self._key_path)
        reject_symlinks(self._records_path)
        self._key = read_regular(self._key_path, 32)
        if len(self._key) != 32 or self._key_path.stat().st_mode & 0o077:
            raise ValueError("reviewed pricing registry key is unavailable or unsafe")

    @classmethod
    def configured(cls) -> "ReviewedPricingRegistry":
        raw = os.environ.get("GPU_AGENT_PRICING_REGISTRY_ROOT")
        if not raw:
            raise ValueError("reviewed pricing registry is required")
        return cls(Path(raw))

    def load(
        self,
        *,
        provider: str,
        model: str,
        binding: RunBinding,
    ) -> PricingAttestation:
        payload = json.loads(read_regular(self._records_path, 1024 * 1024))
        if set(payload) != {"schema_version", "attestations"} or payload["schema_version"] != 1:
            raise ValueError("reviewed pricing registry is malformed")
        raw_records = payload["attestations"]
        if not isinstance(raw_records, list):
            raise ValueError("reviewed pricing registry is malformed")
        matches: list[PricingAttestation] = []
        for raw in raw_records:
            record = PricingAttestation.model_validate(raw)
            signature = record.registry_signature
            expected = hmac.new(self._key, record.signing_content(), hashlib.sha256).hexdigest()
            if (
                record.source != "REVIEWED_REGISTRY"
                or signature is None
                or not hmac.compare_digest(signature, expected)
            ):
                raise ValueError("reviewed pricing attestation signature is invalid")
            if (
                record.provider == provider
                and record.model == model
                and record.repository_commit == binding.repository.commit
                and record.model_config_hash == binding.model_config_hash
            ):
                matches.append(record)
        if len(matches) != 1:
            raise ValueError("reviewed pricing attestation is unavailable or ambiguous")
        return matches[0]
