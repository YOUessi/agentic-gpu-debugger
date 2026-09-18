"""Externally signed authority for immutable, globally complete schedules.

Production contains only an external-client protocol and Ed25519 verification.  It
never generates, loads, or persists a schedule signing key.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.benchmark.evaluation import (
    EvaluationAttempt,
    EvaluationBindings,
    EvaluationExecutionClaim,
    EvaluationSchedule,
    EvaluationScheduleItem,
    EvaluationUnitBinding,
    HoldoutScheduleProof,
)
from gpu_agent.benchmark.ledger import (
    CorpusFamily,
    CorpusLedger,
    CorpusTransaction,
    EvaluationCutoffReservation,
    EvaluationModeName,
    EvaluationSelectionName,
)
from gpu_agent.benchmark.models import CaseManifest
from gpu_agent.contracts import ArtifactRef, CurrentPhase, RunBinding, RunManifest, RunStatus
from gpu_agent.store import EvaluationRunLease, RunStore, RunStoreIdentity, reject_symlinks

_OPENSSL = Path("/usr/bin/openssl")
_SIGNING_DOMAIN = b"gpu-agent-evaluation-schedule-v3\0"


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
    schema_version: Literal[4] = 4
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
    cutoff_reservation_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_universe: list[CorpusUniverseEntry]
    corpus_universe_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    authority_profile: Literal["PRODUCTION", "TEST_ONLY"]
    authority_key_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    queued_manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_prefix_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_prefix_count: int = Field(ge=1)
    artifact_prefix_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_prefix_count: int = Field(ge=1)
    child_inventory_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    child_count: Literal[0] = 0

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


def _sequence_hash(domain: bytes, values: Sequence[BaseModel]) -> str:
    content = json.dumps(
        [value.model_dump(mode="json") for value in values],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(domain + content).hexdigest()


def _queued_manifest_hash(run: RunManifest) -> str:
    return hashlib.sha256(
        b"gpu-agent-evaluation-queued-manifest-v1\0" + _canonical(run)
    ).hexdigest()


_EMPTY_CHILD_INVENTORY_HASH = hashlib.sha256(
    b"gpu-agent-evaluation-child-inventory-v1\0[]"
).hexdigest()


def _store_hash(namespace_hash: str, store: RunStore) -> str:
    content = b"gpu-agent-evaluation-store-v2\0" + bytes.fromhex(namespace_hash)
    return hashlib.sha256(content + b"\0" + str(store.root).encode()).hexdigest()


def _reservation_hash(reservation: EvaluationCutoffReservation) -> str:
    return hashlib.sha256(
        b"gpu-agent-evaluation-cutoff-reservation-v1\0" + _canonical(reservation)
    ).hexdigest()


def reserve_evaluation_cutoff(
    family: CorpusFamily,
    store: RunStore,
    run_id: str,
    binding: RunBinding,
    *,
    selection: EvaluationSelectionName,
    modes: Sequence[EvaluationModeName],
    split: Literal["development", "holdout"],
    repeats: int,
    random_seed: int,
    max_cost_usd: float | None,
    max_unit_cost_usd: float | None,
    holdout_proof: HoldoutScheduleProof | None = None,
    holdout_aliases: Sequence[str] = (),
) -> EvaluationCutoffReservation:
    if store.visibility != "public":
        raise ValueError("evaluation cutoff reservation requires the public evaluation store")
    with store.evaluation_run_lease(run_id) as lease:
        return family.ledger.reserve_evaluation_cutoff(
            lease=lease,
            target_store_hash=_store_hash(family.namespace_hash, store),
            binding=binding,
            selection=selection,
            modes=list(modes),
            split=split,
            repeats=repeats,
            random_seed=random_seed,
            max_cost_usd=max_cost_usd,
            max_unit_cost_usd=max_unit_cost_usd,
            holdout_public_run_id=(holdout_proof.public_run_id if holdout_proof else None),
            holdout_aliases_hash=(holdout_proof.aliases_hash if holdout_proof else None),
            holdout_aliases=list(holdout_aliases),
        )


def _validate_reservation(
    family: CorpusFamily,
    lease: EvaluationRunLease,
    schedule: EvaluationSchedule,
    binding: RunBinding,
    reservation: EvaluationCutoffReservation,
    *,
    allow_unbound_schedule: bool = False,
) -> None:
    family.ledger.validate_evaluation_reservation_prestate(reservation, lease, binding)
    expected = _authority_schedule(family, lease, reservation, binding)
    if (
        reservation.evaluation_run_id != lease.run_id
        or reservation.target_store_hash != _store_hash(family.namespace_hash, lease.store)
        or reservation.target_visibility != lease.store.visibility
        or reservation.binding != binding
        or reservation.selection != schedule.selection
        or reservation.modes != schedule.modes
        or reservation.split != schedule.split
        or reservation.repeats != schedule.repeats
        or reservation.random_seed != schedule.random_seed
        or reservation.max_cost_usd != schedule.bindings.max_cost_usd
        or reservation.max_unit_cost_usd != schedule.bindings.max_unit_cost_usd
        or reservation.corpus_cutoff != schedule.corpus_cutoff
        or reservation.schedule_hash
        not in (
            {None, _schedule_hash(schedule)}
            if allow_unbound_schedule
            else {_schedule_hash(schedule)}
        )
        or schedule != expected
        or _canonical(schedule) != _canonical(expected)
    ):
        raise ValueError("evaluation schedule differs from cutoff reservation")


def bind_reserved_schedule(
    family: CorpusFamily,
    store: RunStore,
    run_id: str,
    schedule: EvaluationSchedule,
    binding: RunBinding,
) -> EvaluationCutoffReservation:
    with store.evaluation_run_lease(run_id) as lease:
        return _bind_reserved_schedule_leased(family, lease, schedule, binding)


def _bind_reserved_schedule_leased(
    family: CorpusFamily,
    lease: EvaluationRunLease,
    schedule: EvaluationSchedule,
    binding: RunBinding,
) -> EvaluationCutoffReservation:
    run = lease.load()
    if (
        run.kind != "evaluation"
        or run.binding != binding
        or run.status != RunStatus.QUEUED
        or run.current_phase is not None
        or run.last_completed_phase is not None
    ):
        raise ValueError("schedule binding requires the exact QUEUED evaluation state")
    reservation = family.ledger.evaluation_cutoff_reservation(lease, binding)
    _validate_reservation(
        family,
        lease,
        schedule,
        binding,
        reservation,
        allow_unbound_schedule=True,
    )
    family.ledger.validate_evaluation_reservation_prestate(
        reservation,
        lease,
        binding,
        require_pristine=reservation.schedule_hash is None,
    )
    if reservation.schedule_hash is not None:
        return reservation
    # All schedule semantics were checked above. The exact signed save is the final
    # authority action; after it linearizes this call chain must only return.
    return CorpusLedger.bind_evaluation_schedule(
        family.ledger,
        lease,
        binding,
        reservation.preparation_id,
        _schedule_hash(schedule),
    )


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
            or schedule.holdout_proof.corpus_cutoff != schedule.corpus_cutoff
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
    aliases = (
        payload.get("aliases")
        if set(payload) == {"schema_version", "corpus_cutoff", "aliases"}
        else None
    )
    if (
        payload.get("schema_version") != 2
        or payload.get("corpus_cutoff") != schedule.corpus_cutoff
        or proof.corpus_cutoff != schedule.corpus_cutoff
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
    cases = registered_cases(store, binding, family, cutoff=cutoff)
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
    if set(selected_cases) != set(cases):
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


def _authority_schedule(
    family: CorpusFamily,
    lease: EvaluationRunLease,
    reservation: EvaluationCutoffReservation,
    binding: RunBinding,
) -> EvaluationSchedule:
    """Deterministically derive the only schedule eligible for the hash CAS."""
    if (
        reservation.binding != binding
        or binding.purpose != "evaluation"
        or binding.corpus_ledger_namespace_hash != family.namespace_hash
        or binding.prompt_version is None
        or binding.toolchain_lock_hash is None
        or binding.model_config_hash is None
    ):
        raise ValueError("evaluation reservation binding is incomplete")
    modes: list[EvaluationModeName] = (
        ["A", "B", "C", "D", "E"] if reservation.selection == "all" else [reservation.selection]
    )
    if reservation.modes != modes:
        raise ValueError("evaluation reservation modes are invalid")
    visibility: Literal["public", "evaluator"] = (
        "public" if reservation.split == "development" else "evaluator"
    )
    _, cases = _universe(family, binding, visibility, reservation.corpus_cutoff)
    holdout_proof: HoldoutScheduleProof | None = None
    if reservation.split == "development":
        if (
            reservation.holdout_public_run_id is not None
            or reservation.holdout_aliases_hash is not None
            or reservation.holdout_aliases
        ):
            raise ValueError("development reservation carries holdout authority")
        case_pairs = sorted((case.id, case.template_id) for case in cases.values())
    else:
        if (
            reservation.holdout_public_run_id is None
            or reservation.holdout_aliases_hash is None
            or not reservation.holdout_aliases
            or len(reservation.holdout_aliases) != len(set(reservation.holdout_aliases))
            or len(reservation.holdout_aliases) != len(cases)
        ):
            raise ValueError("holdout reservation authority is incomplete")
        holdout_proof = HoldoutScheduleProof(
            public_run_id=reservation.holdout_public_run_id,
            aliases_hash=reservation.holdout_aliases_hash,
            corpus_cutoff=reservation.corpus_cutoff,
        )
        case_pairs = [(alias, alias) for alias in sorted(reservation.holdout_aliases)]
    units = [
        (case_id, template_id, mode, repeat)
        for repeat in range(reservation.repeats)
        for case_id, template_id in case_pairs
        for mode in modes
    ]
    random.Random(reservation.random_seed).shuffle(units)
    schedule = EvaluationSchedule(
        selection=reservation.selection,
        modes=modes,
        split=reservation.split,
        repeats=reservation.repeats,
        random_seed=reservation.random_seed,
        corpus_cutoff=reservation.corpus_cutoff,
        bindings=EvaluationBindings(
            commit=binding.repository.commit,
            prompt_version=binding.prompt_version,
            toolchain_hash=binding.toolchain_lock_hash,
            model_config_hash=binding.model_config_hash,
            max_cost_usd=reservation.max_cost_usd,
            max_unit_cost_usd=reservation.max_unit_cost_usd,
        ),
        items=[
            EvaluationScheduleItem(
                ordinal=ordinal,
                case_id=case_id,
                template_id=template_id,
                mode=mode,
                repeat=repeat,
                split=reservation.split,
                holdout_proof=holdout_proof,
            )
            for ordinal, (case_id, template_id, mode, repeat) in enumerate(units)
        ],
        holdout_proof=holdout_proof,
    )
    _validate_coverage(schedule)
    if reservation.split == "holdout":
        aliases = _validated_holdout_aliases(schedule, lease.store, binding)
        if aliases != reservation.holdout_aliases:
            raise ValueError("holdout alias order differs from cutoff reservation")
    return schedule


def rebuild_reserved_schedule(
    family: CorpusFamily,
    store: RunStore,
    run_id: str,
    binding: RunBinding,
) -> EvaluationSchedule:
    """Rebuild the canonical schedule without accepting caller-selected order."""
    with store.evaluation_run_lease(run_id) as lease:
        return _rebuild_reserved_schedule_leased(family, lease, binding)


def _rebuild_reserved_schedule_leased(
    family: CorpusFamily,
    lease: EvaluationRunLease,
    binding: RunBinding,
) -> EvaluationSchedule:
    reservation = family.ledger.evaluation_cutoff_reservation(lease, binding)
    return _authority_schedule(family, lease, reservation, binding)


def _assemble_signing_request(
    family: CorpusFamily,
    store: RunStore,
    run_id: str,
    schedule: EvaluationSchedule,
    binding: RunBinding,
    *,
    corpus_cutoff: int,
    queued_manifest_hash: str,
    event_prefix_hash: str,
    event_prefix_count: int,
    artifact_prefix_hash: str,
    artifact_prefix_count: int,
    reservation: EvaluationCutoffReservation,
) -> EvaluationScheduleSigningRequest:
    profile = family.schedule_authority_profile
    key_hash = family.schedule_public_key_hash
    if profile == "UNCONFIGURED" or key_hash is None:
        raise ValueError("external schedule authority is not configured")
    pairs = _validate_coverage(schedule)
    if (
        binding.purpose != "evaluation"
        or binding.corpus_ledger_namespace_hash != family.namespace_hash
        or schedule.bindings.commit != binding.repository.commit
        or schedule.bindings.prompt_version != binding.prompt_version
        or schedule.bindings.toolchain_hash != binding.toolchain_lock_hash
        or schedule.bindings.model_config_hash != binding.model_config_hash
        or schedule.corpus_cutoff != corpus_cutoff
    ):
        raise ValueError("evaluation schedule differs from its immutable binding")
    visibility: Literal["public", "evaluator"] = (
        "public" if schedule.split == "development" else "evaluator"
    )
    entries, cases = _universe(family, binding, visibility, corpus_cutoff)
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
        f"evaluation-schedule-v4:{run_id}:{target}:{digest}:{queued_manifest_hash}".encode()
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
        corpus_cutoff=corpus_cutoff,
        cutoff_reservation_hash=_reservation_hash(reservation),
        corpus_universe=entries,
        corpus_universe_hash=_universe_hash(entries),
        authority_profile=profile,
        authority_key_hash=key_hash,
        queued_manifest_hash=queued_manifest_hash,
        event_prefix_hash=event_prefix_hash,
        event_prefix_count=event_prefix_count,
        artifact_prefix_hash=artifact_prefix_hash,
        artifact_prefix_count=artifact_prefix_count,
        child_inventory_hash=_EMPTY_CHILD_INVENTORY_HASH,
    )


def build_signing_request(
    family: CorpusFamily,
    store: RunStore,
    run_id: str,
    schedule: EvaluationSchedule,
    binding: RunBinding,
    *,
    corpus_cutoff: int | None = None,
) -> EvaluationScheduleSigningRequest:
    """Build the one signable snapshot before any evaluation child can exist."""
    with store.evaluation_run_lease(run_id) as lease:
        return _build_signing_request_leased(
            family, lease, schedule, binding, corpus_cutoff=corpus_cutoff
        )


def _build_signing_request_leased(
    family: CorpusFamily,
    lease: EvaluationRunLease,
    schedule: EvaluationSchedule,
    binding: RunBinding,
    *,
    corpus_cutoff: int | None = None,
) -> EvaluationScheduleSigningRequest:
    run = lease.load()
    schedule_refs = [ref for ref in run.artifact_refs if ref.name == "evaluation/schedule.json"]
    if (
        run.kind != "evaluation"
        or run.status != RunStatus.QUEUED
        or run.current_phase is not None
        or run.last_completed_phase is not None
        or run.binding != binding
        or len(run.events) != 1
        or run.events[0].status != RunStatus.QUEUED
        or run.events[0].phase is not None
        or len(run.artifact_refs) != 1
        or len(schedule_refs) != 1
        or EvaluationSchedule.model_validate_json(lease.read(schedule_refs[0])) != schedule
    ):
        raise ValueError("signing requires the exact QUEUED evaluation pre-state")
    if lease.children():
        raise ValueError("signing requires a zero-child evaluation pre-state")
    reservation = family.ledger.evaluation_cutoff_reservation(lease, binding)
    _validate_reservation(family, lease, schedule, binding, reservation)
    cutoff = reservation.corpus_cutoff if corpus_cutoff is None else corpus_cutoff
    if cutoff != reservation.corpus_cutoff:
        raise ValueError("requested corpus cutoff differs from controller reservation")
    family.ledger.committed_through(cutoff)
    return _assemble_signing_request(
        family,
        lease.store,
        lease.run_id,
        schedule,
        binding,
        corpus_cutoff=cutoff,
        queued_manifest_hash=_queued_manifest_hash(run),
        event_prefix_hash=_sequence_hash(b"gpu-agent-evaluation-event-prefix-v1\0", run.events),
        event_prefix_count=len(run.events),
        artifact_prefix_hash=_sequence_hash(
            b"gpu-agent-evaluation-artifact-prefix-v1\0", run.artifact_refs
        ),
        artifact_prefix_count=len(run.artifact_refs),
        reservation=reservation,
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


def _validate_signed_prestate(
    lease: EvaluationRunLease,
    run: RunManifest,
    receipt: EvaluationScheduleReceipt,
) -> None:
    request = receipt.request
    if (
        run.kind != "evaluation"
        or run.binding != request.binding
        or request.child_count != 0
        or request.child_inventory_hash != _EMPTY_CHILD_INVENTORY_HASH
        or request.event_prefix_count > len(run.events)
        or request.artifact_prefix_count >= len(run.artifact_refs)
    ):
        raise ValueError("schedule receipt does not bind the evaluation pre-state")
    events = run.events[: request.event_prefix_count]
    artifacts = run.artifact_refs[: request.artifact_prefix_count]
    receipt_ref = run.artifact_refs[request.artifact_prefix_count]
    queued = run.model_copy(
        update={
            "status": RunStatus.QUEUED,
            "current_phase": None,
            "last_completed_phase": None,
            "events": events,
            "artifact_refs": artifacts,
        }
    )
    if (
        request.event_prefix_count != 1
        or events[0].status != RunStatus.QUEUED
        or events[0].phase is not None
        or request.event_prefix_hash
        != _sequence_hash(b"gpu-agent-evaluation-event-prefix-v1\0", events)
        or request.artifact_prefix_hash
        != _sequence_hash(b"gpu-agent-evaluation-artifact-prefix-v1\0", artifacts)
        or request.queued_manifest_hash != _queued_manifest_hash(queued)
        or receipt_ref.name != "evaluation/schedule-receipt.json"
    ):
        raise ValueError("evaluation prefix differs from the signed QUEUED pre-state")
    if run.status == RunStatus.QUEUED:
        if (
            len(run.events) != request.event_prefix_count
            or len(run.artifact_refs) != request.artifact_prefix_count + 1
            or lease.children()
        ):
            raise ValueError("QUEUED evaluation changed after schedule signing")
    else:
        first_active = run.events[request.event_prefix_count]
        if first_active.status != RunStatus.RUNNING or first_active.phase != CurrentPhase.EXECUTING:
            raise ValueError("evaluation did not activate from the signed pre-state")


class EvaluationScheduleVerifier:
    """Verifier-only consumer retaining no writer, client, or private key."""

    def __init__(self, family: CorpusFamily, store: RunStore, *, allow_test: bool) -> None:
        profile = family.schedule_authority_profile
        if profile == "UNCONFIGURED" or (profile == "TEST_ONLY" and not allow_test):
            raise ValueError("production schedule authority is not configured")
        self.__family_root = family.root
        self.__store = store
        self.__store_identity = store.identity
        self.__profile = profile
        self.__key_hash = family.schedule_public_key_hash
        self.__public_key = family.schedule_public_key_path

    @classmethod
    def for_family(cls, family: CorpusFamily, store: RunStore) -> EvaluationScheduleVerifier:
        return cls(family, store, allow_test=False)

    @classmethod
    def _for_test(cls, family: CorpusFamily, store: RunStore) -> EvaluationScheduleVerifier:
        return cls(family, store, allow_test=True)

    @property
    def store_identity(self) -> RunStoreIdentity:
        return self.__store_identity

    def require_store(self, store: RunStore) -> None:
        if type(store) is not RunStore or store.identity != self.__store_identity:
            raise ValueError("evaluation verifier store identity differs")

    def verify(self, run_id: str) -> EvaluationScheduleReceipt:
        EvaluationScheduleVerifier.require_store(self, self.__store)
        with self.__store.evaluation_run_lease(run_id) as lease:
            return EvaluationScheduleVerifier._verify_leased(self, lease)

    def verify_queued(self, run_id: str) -> EvaluationScheduleReceipt:
        """Verify status and receipt atomically for a still-QUEUED evaluation."""
        EvaluationScheduleVerifier.require_store(self, self.__store)
        with self.__store.evaluation_run_lease(run_id) as lease:
            run = lease.load()
            if (
                run.status != RunStatus.QUEUED
                or run.current_phase is not None
                or run.last_completed_phase is not None
            ):
                raise ValueError("schedule recovery requires the exact QUEUED state")
            return EvaluationScheduleVerifier._verify_leased(self, lease)

    def _verify_leased(self, lease: EvaluationRunLease) -> EvaluationScheduleReceipt:
        EvaluationScheduleVerifier.require_store(self, lease.store)
        family = CorpusFamily.open(self.__family_root)
        run = lease.load()
        schedule_refs = [ref for ref in run.artifact_refs if ref.name == "evaluation/schedule.json"]
        receipt_refs = [
            ref for ref in run.artifact_refs if ref.name == "evaluation/schedule-receipt.json"
        ]
        if len(schedule_refs) != 1 or len(receipt_refs) != 1 or run.binding is None:
            raise ValueError("evaluation schedule authority artifacts are missing or ambiguous")
        schedule = EvaluationSchedule.model_validate_json(lease.read(schedule_refs[0]))
        receipt = EvaluationScheduleReceipt.model_validate_json(lease.read(receipt_refs[0]))
        if (
            receipt.request.authority_profile != self.__profile
            or receipt.request.authority_key_hash != self.__key_hash
        ):
            raise ValueError("schedule receipt uses another authority")
        _verify_signature(self.__public_key, receipt)
        _validate_signed_prestate(lease, run, receipt)
        reservation = family.ledger.evaluation_cutoff_reservation(lease, run.binding)
        _validate_reservation(family, lease, schedule, run.binding, reservation)
        expected = _assemble_signing_request(
            family,
            lease.store,
            lease.run_id,
            schedule,
            run.binding,
            corpus_cutoff=receipt.request.corpus_cutoff,
            queued_manifest_hash=receipt.request.queued_manifest_hash,
            event_prefix_hash=receipt.request.event_prefix_hash,
            event_prefix_count=receipt.request.event_prefix_count,
            artifact_prefix_hash=receipt.request.artifact_prefix_hash,
            artifact_prefix_count=receipt.request.artifact_prefix_count,
            reservation=reservation,
        )
        if receipt.request != expected:
            raise ValueError("schedule receipt differs from native authority inputs")
        lease.validate()
        return receipt

    def validate_unit(self, store: RunStore, unit: EvaluationUnitBinding) -> None:
        with store.evaluation_run_lease(unit.evaluation_run_id) as lease:
            _validate_evaluation_unit(lease, self, unit)

    def _validate_unit_leased(self, lease: EvaluationRunLease, unit: EvaluationUnitBinding) -> None:
        _validate_evaluation_unit(lease, self, unit)


def _one_ref(run: RunManifest, name: str) -> ArtifactRef:
    refs = [ref for ref in run.artifact_refs if ref.name == name]
    if len(refs) != 1:
        raise ValueError(f"evaluation artifact {name} is missing or ambiguous")
    return refs[0]


def _ordinal_inventory(run: RunManifest, prefix: str) -> list[int]:
    result: list[int] = []
    for ref in run.artifact_refs:
        if not ref.name.startswith(prefix):
            continue
        suffix = ref.name.removeprefix(prefix)
        if not suffix.endswith(".json") or not suffix[:-5].isdigit():
            raise ValueError("evaluation ordinal artifact name is invalid")
        result.append(int(suffix[:-5]))
    if len(result) != len(set(result)):
        raise ValueError("evaluation ordinal artifact is duplicated")
    return sorted(result)


def _validate_evaluation_unit(
    lease: EvaluationRunLease,
    verifier: EvaluationScheduleVerifier,
    unit: EvaluationUnitBinding,
) -> None:
    """Revalidate one claimed unit immediately before creating physical work."""
    store = lease.store
    EvaluationScheduleVerifier.require_store(verifier, store)
    EvaluationScheduleVerifier._verify_leased(verifier, lease)
    parent = lease.load()
    if parent.status != RunStatus.RUNNING or parent.current_phase != CurrentPhase.EXECUTING:
        raise ValueError("evaluation parent is not RUNNING")
    schedule = EvaluationSchedule.model_validate_json(
        lease.read(_one_ref(parent, "evaluation/schedule.json"))
    )
    if unit.ordinal >= len(schedule.items):
        raise ValueError("evaluation unit ordinal is outside the signed schedule")
    schedule_hash = _schedule_hash(schedule)
    item = schedule.items[unit.ordinal]
    if schedule.bindings.max_unit_cost_usd is None:
        raise ValueError("evaluation unit has no frozen reservation")
    attempt = EvaluationAttempt.model_validate_json(
        lease.read(_one_ref(parent, f"evaluation/attempts/{unit.ordinal}.json"))
    )
    expected_attempt = EvaluationAttempt(
        run_id=parent.id,
        ordinal=unit.ordinal,
        schedule_hash=schedule_hash,
        corpus_cutoff=schedule.corpus_cutoff,
        idempotency_key=hashlib.sha256(
            f"{parent.id}:{schedule_hash}:{unit.ordinal}".encode()
        ).hexdigest(),
        reserved_cost_usd=schedule.bindings.max_unit_cost_usd,
    )
    expected_unit = EvaluationUnitBinding(
        evaluation_run_id=parent.id,
        ordinal=item.ordinal,
        schedule_hash=schedule_hash,
        corpus_cutoff=schedule.corpus_cutoff,
        idempotency_key=expected_attempt.idempotency_key,
        reserved_cost_usd=expected_attempt.reserved_cost_usd,
        case_id=item.case_id,
        template_id=item.template_id,
        mode=item.mode,
        repeat=item.repeat,
        split=item.split,
        holdout_proof=item.holdout_proof,
    )
    attempt_content = expected_attempt.model_dump_json().encode()
    expected_claim = EvaluationExecutionClaim(
        run_id=parent.id,
        ordinal=item.ordinal,
        schedule_hash=schedule_hash,
        corpus_cutoff=schedule.corpus_cutoff,
        attempt_hash=hashlib.sha256(attempt_content).hexdigest(),
    )
    claim = EvaluationExecutionClaim.model_validate_json(
        lease.read(_one_ref(parent, f"evaluation/claims/{unit.ordinal}.json"))
    )
    if unit != expected_unit or attempt != expected_attempt or claim != expected_claim:
        raise ValueError("evaluation unit is not the claimed signed ordinal")
    expected_previous = list(range(unit.ordinal))
    if (
        _ordinal_inventory(parent, "evaluation/attempts/") != list(range(unit.ordinal + 1))
        or _ordinal_inventory(parent, "evaluation/claims/") != list(range(unit.ordinal + 1))
        or _ordinal_inventory(parent, "evaluation/records/") != expected_previous
    ):
        raise ValueError("evaluation unit is not the canonical claimed ordinal")
    children = lease.children()
    child_ordinals: list[int] = []
    for child in children:
        if child.kind != "diagnosis" or child.status != RunStatus.COMPLETED:
            raise ValueError("evaluation parent has an unexpected child")
        child_unit = EvaluationUnitBinding.model_validate_json(
            store.read(_one_ref(child, "evaluation/unit.json"))
        )
        child_ordinals.append(child_unit.ordinal)
    if sorted(child_ordinals) != expected_previous:
        raise ValueError("evaluation parent has extra or missing diagnosis children")


def activate_schedule(
    store: RunStore,
    verifier: EvaluationScheduleVerifier,
    run_id: str,
) -> EvaluationScheduleReceipt:
    """Verify the committed receipt, atomically activate, then verify the active prefix."""
    EvaluationScheduleVerifier.require_store(verifier, store)
    receipt = EvaluationScheduleVerifier.verify(verifier, run_id)
    store.activate_evaluation(verifier, run_id)
    active = EvaluationScheduleVerifier.verify(verifier, run_id)
    if active != receipt:
        raise ValueError("evaluation receipt changed during activation")
    return active


def verify_existing_schedule_binding(
    family: CorpusFamily,
    store: RunStore,
    run_id: str,
    schedule: EvaluationSchedule,
    binding: RunBinding,
    verifier: EvaluationScheduleVerifier,
) -> EvaluationScheduleReceipt:
    """Read-only validation for resuming an already active evaluation."""
    EvaluationScheduleVerifier.require_store(verifier, store)
    with store.evaluation_run_lease(run_id) as lease:
        run = lease.load()
        if (
            run.kind != "evaluation"
            or run.binding != binding
            or run.status != RunStatus.RUNNING
            or run.current_phase != CurrentPhase.EXECUTING
        ):
            raise ValueError("existing schedule validation requires a RUNNING evaluation")
        schedule_refs = [ref for ref in run.artifact_refs if ref.name == "evaluation/schedule.json"]
        if (
            len(schedule_refs) != 1
            or EvaluationSchedule.model_validate_json(lease.read(schedule_refs[0])) != schedule
        ):
            raise ValueError("active evaluation schedule differs from recovery input")
        receipt = EvaluationScheduleVerifier._verify_leased(verifier, lease)
        reservation = family.ledger.evaluation_cutoff_reservation(lease, binding)
        _validate_reservation(family, lease, schedule, binding, reservation)
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
    EvaluationScheduleVerifier.require_store(verifier, store)
    bind_reserved_schedule(family, store, run_id, schedule, binding)
    try:
        request = build_signing_request(family, store, run_id, schedule, binding)
    except ValueError as build_error:
        # A concurrent exact seal may have installed the receipt after binding but
        # before this signing snapshot acquired the run lease. Only a fully verified
        # receipt for this exact caller input makes that race idempotent.
        try:
            committed = EvaluationScheduleVerifier.verify_queued(verifier, run_id)
        except ValueError:
            raise build_error from None
        if (
            committed.request.schedule_hash != _schedule_hash(schedule)
            or committed.request.binding != binding
        ):
            raise build_error
        return committed
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
    return EvaluationScheduleVerifier.verify_queued(verifier, run_id)
