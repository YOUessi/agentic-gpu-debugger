"""Test-only external schedule signer.

The production package deliberately has no signing-key API or implementation.
"""

import hashlib
import json
import subprocess
import threading
from pathlib import Path

from gpu_agent.benchmark.schedule_authority import (
    EvaluationScheduleReceipt,
    EvaluationScheduleSigningRequest,
)

_CLIENTS: dict[str, "TestScheduleCommitClient"] = {}


class TestScheduleCommitClient:
    __test__ = False

    def __init__(self, private_key: Path, public_key: bytes) -> None:
        self._private_key = private_key
        self.public_key = public_key
        self._lock = threading.Lock()

    @classmethod
    def create(cls, root: Path) -> "TestScheduleCommitClient":
        root.mkdir(parents=True, exist_ok=True)
        private_key, public_key = root / "schedule-private.pem", root / "schedule-public.pem"
        subprocess.run(
            ["/usr/bin/openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private_key)],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        subprocess.run(
            [
                "/usr/bin/openssl",
                "pkey",
                "-in",
                str(private_key),
                "-pubout",
                "-out",
                str(public_key),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        private_key.chmod(0o600)
        return cls(private_key, public_key.read_bytes())

    def commit(self, request: EvaluationScheduleSigningRequest) -> EvaluationScheduleReceipt:
        with self._lock:
            payload = request.signed_bytes()
            payload_path = self._private_key.parent / f"{request.transaction_id}.payload"
            signature_path = self._private_key.parent / f"{request.transaction_id}.signature"
            payload_path.write_bytes(payload)
            subprocess.run(
                [
                    "/usr/bin/openssl",
                    "pkeyutl",
                    "-sign",
                    "-inkey",
                    str(self._private_key),
                    "-rawin",
                    "-in",
                    str(payload_path),
                    "-out",
                    str(signature_path),
                ],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            signature = signature_path.read_bytes()
            payload_path.unlink()
            signature_path.unlink()
        return EvaluationScheduleReceipt(
            request=request,
            payload_hash=hashlib.sha256(payload).hexdigest(),
            signature_hex=signature.hex(),
        )


def register_test_schedule_client(client: TestScheduleCommitClient) -> None:
    _CLIENTS[hashlib.sha256(client.public_key).hexdigest()] = client


def schedule_client_for_test(executor) -> TestScheduleCommitClient:
    key_hash = executor._corpus_family.schedule_public_key_hash
    if key_hash not in _CLIENTS:
        raise ValueError("test schedule authority is not registered")
    return _CLIENTS[key_hash]


def reserve_schedule_for_test(executor, runner, run_id, schedule) -> None:
    """Mirror the controller reservation step for low-level schedule tests."""
    from gpu_agent.benchmark.schedule_authority import (
        bind_reserved_schedule,
        reserve_evaluation_cutoff,
    )

    binding = executor.service.binding
    if binding is None:
        raise ValueError("test evaluation binding is unavailable")
    holdout_aliases = []
    if schedule.holdout_proof is not None:
        alias_run = runner.store.load(schedule.holdout_proof.public_run_id)
        alias_ref = next(
            ref for ref in alias_run.artifact_refs if ref.name == "holdout/aliases.json"
        )
        holdout_aliases = json.loads(runner.store.read(alias_ref))["aliases"]
    reservation = reserve_evaluation_cutoff(
        executor._corpus_family,
        runner.store,
        run_id,
        binding,
        selection=schedule.selection,
        modes=schedule.modes,
        split=schedule.split,
        repeats=schedule.repeats,
        random_seed=schedule.random_seed,
        max_cost_usd=schedule.bindings.max_cost_usd,
        max_unit_cost_usd=schedule.bindings.max_unit_cost_usd,
        holdout_proof=schedule.holdout_proof,
        holdout_aliases=holdout_aliases,
    )
    if reservation.corpus_cutoff != schedule.corpus_cutoff:
        raise ValueError("test schedule cutoff differs from reservation")
    bind_reserved_schedule(executor._corpus_family, runner.store, run_id, schedule, binding)
