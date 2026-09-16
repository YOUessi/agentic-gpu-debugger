"""Externally signed authority for immutable, globally complete schedules.

Production contains only an external-client protocol and Ed25519 verification.  It
never generates, loads, or persists a schedule signing key.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.benchmark.evaluation import EvaluationSchedule, HoldoutScheduleProof
from gpu_agent.benchmark.ledger import CorpusFamily, CorpusTransaction
from gpu_agent.benchmark.models import CaseManifest
from gpu_agent.contracts import RunBinding, RunStatus
from gpu_agent.store import RunStore, reject_symlinks

_OPENSSL = Path("/usr/bin/openssl")
_SIGNING_DOMAIN = b"gpu-agent-evaluation-schedule-v2\0"


class CorpusUniverseEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    commit_sequence: int = Field(ge=1)
    transaction_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    registration_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    identity_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    template_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class EvaluationScheduleSigningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[2] = 2
    transaction_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    target_store_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    schedule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    binding: RunBinding
    selection: str
    modes: list[str]
    split: str
    repeats: int
    random_seed: int
    max_cost_usd: float | None
    max_unit_cost_usd: float | None
    case_templates: dict[str, str]
    holdout_proof: HoldoutScheduleProof | None = None
    holdout_aliases: list[str]
    corpus_namespace_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_visibility: Literal["public", "evaluator"]
    corpus_cutoff: int = Field(ge=1)
    corpus_universe: list[CorpusUniverseEntry]
    corpus_universe_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    authority_profile: Literal["PRODUCTION", "TEST_ONLY"]
    authority_key_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    def signed_bytes(self) -> bytes:
        return _SIGNING_DOMAIN + _canonical(self)


class EvaluationScheduleReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    state: Literal["COMMITTED"] = "COMMITTED"
    request: EvaluationScheduleSigningRequest
    algorithm: Literal["Ed25519"] = "Ed25519"
    payload_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    signature_hex: str = Field(pattern=r"^[a-f0-9]{128}$")


class ScheduleCommitClient(Protocol):
    """External trust boundary; no implementation exists in production code."""

    def commit(self, request: EvaluationScheduleSigningRequest) -> EvaluationScheduleReceipt: ...


def _canonical(value: BaseModel) -> bytes:
    return json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()


def _schedule_hash(schedule: EvaluationSchedule) -> str:
    return hashlib.sha256(_canonical(schedule)).hexdigest()


def _store_hash(namespace_hash: str, store: RunStore) -> str:
    content = b"gpu-agent-evaluation-store-v2\0" + bytes.fromhex(namespace_hash)
    return hashlib.sha256(content + b"\0" + str(store.root).encode()).hexdigest()


def _case_templates(schedule: EvaluationSchedule) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in schedule.items:
        previous = result.setdefault(item.case_id, item.template_id)
        if previous != item.template_id:
            raise ValueError("evaluation case has multiple templates")
    return result


def _validate_coverage(schedule: EvaluationSchedule) -> dict[str, str]:
    expected_modes = (
        ["A", "B", "C", "D", "E"] if schedule.selection == "all" else [schedule.selection]
    )
    if schedule.modes != expected_modes or schedule.repeats < 3:
        raise ValueError("evaluation schedule mode coverage is invalid")
    if [item.ordinal for item in schedule.items] != list(range(len(schedule.items))):
        raise ValueError("evaluation schedule ordinals are not canonical")
    pairs = _case_templates(schedule)
    expected = Counter(
        (case_id, template_id, mode, repeat, schedule.split)
        for repeat in range(schedule.repeats)
        for case_id, template_id in pairs.items()
        for mode in schedule.modes
    )
    observed = Counter(
        (item.case_id, item.template_id, item.mode, item.repeat, item.split)
        for item in schedule.items
    )
    if not pairs or observed != expected:
        raise ValueError("evaluation schedule does not have exact global coverage")
    if schedule.split == "holdout":
        aliases = set(pairs)
        if (
            schedule.holdout_proof is None
            or any(case_id != template_id for case_id, template_id in pairs.items())
            or any(item.holdout_proof != schedule.holdout_proof for item in schedule.items)
            or aliases != {item.case_id for item in schedule.items}
        ):
            raise ValueError("holdout schedule authority is incomplete")
    elif schedule.holdout_proof is not None or any(
        item.holdout_proof is not None for item in schedule.items
    ):
        raise ValueError("development schedule carries holdout authority")
    return pairs


def _validated_holdout_aliases(
    schedule: EvaluationSchedule, store: RunStore, binding: RunBinding
) -> list[str]:
    if schedule.split != "holdout":
        return []
    proof = schedule.holdout_proof
    if proof is None:
        raise ValueError("holdout schedule proof is missing")
    run = store.load(proof.public_run_id)
    refs = [ref for ref in run.artifact_refs if ref.name == "holdout/aliases.json"]
    if (
        run.kind != "holdout_aliases"
        or run.status != RunStatus.COMPLETED
        or run.binding != binding
        or len(refs) != 1
        or refs[0].sha256 != proof.aliases_hash
    ):
        raise ValueError("holdout schedule aliases are not authoritative")
    payload = json.loads(store.read(refs[0]))
    aliases = payload.get("aliases") if set(payload) == {"schema_version", "aliases"} else None
    if (
        payload.get("schema_version") != 1
        or not isinstance(aliases, list)
        or not aliases
        or len(aliases) != len(set(aliases))
        or set(aliases) != {item.case_id for item in schedule.items}
    ):
        raise ValueError("holdout schedule does not cover the complete alias batch")
    return aliases


def _entry(transaction: CorpusTransaction) -> CorpusUniverseEntry:
    if transaction.commit_sequence is None:
        raise ValueError("committed corpus transaction has no sequence")
    return CorpusUniverseEntry(
        commit_sequence=transaction.commit_sequence,
        transaction_id=transaction.transaction_id,
        registration_run_id=transaction.run_id,
        manifest_hash=transaction.manifest_hash,
        identity_hash=transaction.case_hash,
        template_hash=transaction.template_hash,
    )


def _universe(
    family: CorpusFamily,
    binding: RunBinding,
    visibility: Literal["public", "evaluator"],
    cutoff: int | None = None,
) -> tuple[list[CorpusUniverseEntry], dict[str, CaseManifest]]:
    """Revalidate native registrations selected by an immutable ledger snapshot."""
    from gpu_agent.benchmark.executor import registered_cases

    store = family.corpus_store(visibility)
    cases = registered_cases(store, binding, family)
    selected = [
        transaction
        for transaction in family.ledger.committed_through(cutoff)
        if transaction.visibility == visibility
        and transaction.target_store_hash == family.ledger.target_store_hash(store)
    ]
    by_identity = {case.case_identity_hash: case for case in cases.values()}
    if None in by_identity or len(by_identity) != len(cases):
        raise ValueError("corpus universe identities are invalid")
    selected_cases: dict[str, CaseManifest] = {}
    for transaction in selected:
        case = by_identity.get(transaction.case_hash)
        if (
            case is None
            or case.template_identity_hash != transaction.template_hash
            or case.source_pair_hash != transaction.source_pair_hash
        ):
            raise ValueError("committed corpus universe differs from native registrations")
        selected_cases[case.id] = case
    if cutoff is None and set(selected_cases) != set(cases):
        raise ValueError("native corpus registration is absent from committed universe")
    entries = [_entry(transaction) for transaction in selected]
    if not entries:
        raise ValueError("committed corpus universe is empty")
    return entries, selected_cases


def _universe_hash(entries: list[CorpusUniverseEntry]) -> str:
    content = json.dumps(
        [item.model_dump(mode="json") for item in entries],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(b"gpu-agent-corpus-universe-v1\0" + content).hexdigest()


def build_signing_request(
    family: CorpusFamily,
    store: RunStore,
    run_id: str,
    schedule: EvaluationSchedule,
    binding: RunBinding,
    *,
    corpus_cutoff: int | None = None,
) -> EvaluationScheduleSigningRequest:
    profile = family.schedule_authority_profile
    key_hash = family.schedule_public_key_hash
    if profile == "UNCONFIGURED" or key_hash is None:
        raise ValueError("external schedule authority is not configured")
    run = store.load(run_id)
    pairs = _validate_coverage(schedule)
    if (
        run.kind != "evaluation"
        or run.status
        not in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.FAILED}
        or run.binding != binding
        or binding.purpose != "evaluation"
        or binding.corpus_ledger_namespace_hash != family.namespace_hash
        or schedule.bindings.commit != binding.repository.commit
        or schedule.bindings.prompt_version != binding.prompt_version
        or schedule.bindings.toolchain_hash != binding.toolchain_lock_hash
        or schedule.bindings.model_config_hash != binding.model_config_hash
    ):
        raise ValueError("evaluation schedule differs from its immutable binding")
    visibility: Literal["public", "evaluator"] = (
        "public" if schedule.split == "development" else "evaluator"
    )
    committed = family.ledger.committed_through()
    effective_cutoff = len(committed) if corpus_cutoff is None else corpus_cutoff
    entries, cases = _universe(family, binding, visibility, effective_cutoff)
    aliases = _validated_holdout_aliases(schedule, store, binding)
    expected_pairs = (
        {case.id: case.template_id for case in cases.values()}
        if visibility == "public"
        else {alias: alias for alias in aliases}
    )
    if pairs != expected_pairs or (visibility == "evaluator" and len(aliases) != len(cases)):
        raise ValueError("evaluation schedule is not the authoritative corpus universe")
    digest = _schedule_hash(schedule)
    target = _store_hash(family.namespace_hash, store)
    transaction_id = hashlib.sha256(
        f"evaluation-schedule-v2:{run_id}:{target}:{digest}".encode()
    ).hexdigest()[:32]
    return EvaluationScheduleSigningRequest(
        transaction_id=transaction_id,
        evaluation_run_id=run_id,
        target_store_hash=target,
        schedule_hash=digest,
        binding=binding,
        selection=schedule.selection,
        modes=list(schedule.modes),
        split=schedule.split,
        repeats=schedule.repeats,
        random_seed=schedule.random_seed,
        max_cost_usd=schedule.bindings.max_cost_usd,
        max_unit_cost_usd=schedule.bindings.max_unit_cost_usd,
        case_templates=pairs,
        holdout_proof=schedule.holdout_proof,
        holdout_aliases=aliases,
        corpus_namespace_hash=family.namespace_hash,
        corpus_visibility=visibility,
        corpus_cutoff=effective_cutoff,
        corpus_universe=entries,
        corpus_universe_hash=_universe_hash(entries),
        authority_profile=profile,
        authority_key_hash=key_hash,
    )


def _verify_signature(public_key: Path, receipt: EvaluationScheduleReceipt) -> None:
    payload = receipt.request.signed_bytes()
    if receipt.payload_hash != hashlib.sha256(payload).hexdigest():
        raise ValueError("schedule receipt payload hash differs")
    reject_symlinks(public_key)
    if not _OPENSSL.is_file():
        raise ValueError("schedule signature verifier is unavailable")
    with tempfile.TemporaryDirectory(prefix="gpu-agent-schedule-verify-") as raw:
        root = Path(raw)
        payload_path, signature_path = root / "payload", root / "signature"
        payload_path.write_bytes(payload)
        signature_path.write_bytes(bytes.fromhex(receipt.signature_hex))
        result = subprocess.run(
            [
                str(_OPENSSL),
                "pkeyutl",
                "-verify",
                "-pubin",
                "-inkey",
                str(public_key),
                "-rawin",
                "-in",
                str(payload_path),
                "-sigfile",
                str(signature_path),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=5,
            check=False,
        )
    if result.returncode != 0 or len(result.stdout) > 4096 or len(result.stderr) > 4096:
        raise ValueError("schedule receipt signature is invalid")


class EvaluationScheduleVerifier:
    """Verifier-only consumer retaining no writer, client, or private key."""

    def __init__(self, family: CorpusFamily, store: RunStore, *, allow_test: bool) -> None:
        profile = family.schedule_authority_profile
        if profile == "UNCONFIGURED" or (profile == "TEST_ONLY" and not allow_test):
            raise ValueError("production schedule authority is not configured")
        self.__family_root = family.root
        self.__store = store
        self.__profile = profile
        self.__key_hash = family.schedule_public_key_hash
        self.__public_key = family.schedule_public_key_path

    @classmethod
    def for_family(cls, family: CorpusFamily, store: RunStore) -> EvaluationScheduleVerifier:
        return cls(family, store, allow_test=False)

    @classmethod
    def _for_test(cls, family: CorpusFamily, store: RunStore) -> EvaluationScheduleVerifier:
        return cls(family, store, allow_test=True)

    def verify(self, run_id: str) -> EvaluationScheduleReceipt:
        family = CorpusFamily.open(self.__family_root)
        run = self.__store.load(run_id)
        schedule_refs = [ref for ref in run.artifact_refs if ref.name == "evaluation/schedule.json"]
        receipt_refs = [
            ref for ref in run.artifact_refs if ref.name == "evaluation/schedule-receipt.json"
        ]
        if len(schedule_refs) != 1 or len(receipt_refs) != 1 or run.binding is None:
            raise ValueError("evaluation schedule authority artifacts are missing or ambiguous")
        schedule = EvaluationSchedule.model_validate_json(self.__store.read(schedule_refs[0]))
        receipt = EvaluationScheduleReceipt.model_validate_json(self.__store.read(receipt_refs[0]))
        if (
            receipt.request.authority_profile != self.__profile
            or receipt.request.authority_key_hash != self.__key_hash
        ):
            raise ValueError("schedule receipt uses another authority")
        _verify_signature(self.__public_key, receipt)
        expected = build_signing_request(
            family,
            self.__store,
            run_id,
            schedule,
            run.binding,
            corpus_cutoff=receipt.request.corpus_cutoff,
        )
        if receipt.request != expected:
            raise ValueError("schedule receipt differs from native authority inputs")
        return receipt


def seal_schedule(
    family: CorpusFamily,
    store: RunStore,
    run_id: str,
    schedule: EvaluationSchedule,
    binding: RunBinding,
    client: ScheduleCommitClient | None,
    verifier: EvaluationScheduleVerifier,
) -> EvaluationScheduleReceipt:
    if client is None:
        raise ValueError("external schedule authority is required")
    request = build_signing_request(family, store, run_id, schedule, binding)
    receipt = client.commit(request)
    if receipt.request != request or receipt.state != "COMMITTED":
        raise ValueError("external schedule authority returned another transaction")
    _verify_signature(family.schedule_public_key_path, receipt)
    store.put_if_absent_exact(
        run_id,
        "evaluation/schedule-receipt.json",
        receipt.model_dump_json().encode(),
        "public",
    )
    return EvaluationScheduleVerifier.verify(verifier, run_id)
