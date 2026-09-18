"""Trusted corpus-family configuration and crash-safe opaque uniqueness ledger."""

import fcntl
import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.contracts import RunBinding, RunManifest, RunStatus, StateEvent, Visibility, new_id
from gpu_agent.store import (
    EvaluationRunLease,
    RunStore,
    read_regular,
    reject_symlinks,
    sync_directory,
)


class _FamilyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[2] = 2
    public_store: str
    evaluator_store: str
    schedule_authority_profile: Literal["UNCONFIGURED", "PRODUCTION", "TEST_ONLY"] = "UNCONFIGURED"
    schedule_public_key_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class CorpusTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    transaction_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    owner_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    state: Literal["PREPARED", "COMMITTED"]
    case_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    template_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_pair_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    target_store_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    visibility: Visibility
    manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    commit_sequence: int | None = Field(default=None, ge=1)


EvaluationModeName = Literal["A", "B", "C", "D", "E"]
EvaluationSelectionName = EvaluationModeName | Literal["all"]


class _EvaluationReservationPrestate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target_store_root: str
    target_store_device: int = Field(ge=0)
    target_store_inode: int = Field(ge=1)
    target_visibility: Visibility
    target_run_path: str
    target_run_device: int = Field(ge=0)
    target_run_inode: int = Field(ge=1)
    target_run_lock_device: int = Field(ge=0)
    target_run_lock_inode: int = Field(ge=1)
    queued_manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_prefix_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_prefix_count: Literal[1] = 1
    artifact_prefix_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_prefix_count: Literal[0] = 0
    child_inventory_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    child_count: Literal[0] = 0


class EvaluationCutoffReservation(BaseModel):
    """Controller-owned snapshot of the corpus head for one evaluation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[4] = 4
    state: Literal["PREPARED", "COMMITTED"]
    evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    target_store_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    target_store_root: str
    target_store_device: int = Field(ge=0)
    target_store_inode: int = Field(ge=1)
    target_visibility: Visibility
    target_run_path: str
    target_run_device: int = Field(ge=0)
    target_run_inode: int = Field(ge=1)
    target_run_lock_device: int = Field(ge=0)
    target_run_lock_inode: int = Field(ge=1)
    binding: RunBinding
    selection: EvaluationSelectionName
    modes: list[EvaluationModeName]
    split: Literal["development", "holdout"]
    repeats: int = Field(ge=3)
    random_seed: int
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_unit_cost_usd: float | None = Field(default=None, ge=0)
    holdout_public_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    holdout_aliases_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    holdout_aliases: list[str] = Field(default_factory=list)
    corpus_cutoff: int = Field(ge=1)
    queued_manifest_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_prefix_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_prefix_count: Literal[1] = 1
    artifact_prefix_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_prefix_count: Literal[0] = 0
    child_inventory_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    child_count: Literal[0] = 0
    schedule_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


_RESERVATION_MANIFEST_DOMAIN = b"gpu-agent-evaluation-reservation-manifest-v1\0"
_RESERVATION_EVENT_DOMAIN = b"gpu-agent-evaluation-reservation-event-prefix-v1\0"
_RESERVATION_ARTIFACT_DOMAIN = b"gpu-agent-evaluation-reservation-artifact-prefix-v1\0"
_RESERVATION_CHILD_DOMAIN = b"gpu-agent-evaluation-reservation-child-inventory-v1\0"


def _canonical_model(value: BaseModel) -> bytes:
    return json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()


def _reservation_sequence_hash(domain: bytes, values: Sequence[BaseModel]) -> str:
    content = json.dumps(
        [value.model_dump(mode="json") for value in values],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(domain + content).hexdigest()


def _reservation_manifest_hash(run: RunManifest) -> str:
    return hashlib.sha256(_RESERVATION_MANIFEST_DOMAIN + _canonical_model(run)).hexdigest()


def _empty_reservation_child_hash() -> str:
    return hashlib.sha256(_RESERVATION_CHILD_DOMAIN + b"[]").hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _atomic_create(path: Path, content: bytes, mode: int) -> None:
    """Install a complete durable file without ever exposing a partial destination."""
    fd, temporary = tempfile.mkstemp(prefix=".create-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            return
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


class CorpusFamily:
    """Controller-private config shared by public and evaluator corpus stores."""

    def __init__(self, root: Path, config: _FamilyConfig, ledger: "CorpusLedger") -> None:
        self.root = root
        self._config = config
        self.ledger = ledger

    @property
    def namespace_hash(self) -> str:
        return self.ledger.namespace_hash

    def _marker_bytes(self) -> bytes:
        return json.dumps(
            {
                "schema_version": 2,
                "ledger_namespace_hash": self.namespace_hash,
                "schedule_authority_profile": self._config.schedule_authority_profile,
                "schedule_public_key_hash": self._config.schedule_public_key_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def _pin_store(self, path: Path) -> None:
        reject_symlinks(path)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        marker = path / ".corpus-family.json"
        expected = self._marker_bytes()
        _atomic_create(marker, expected, 0o600)
        if read_regular(marker, 64 * 1024) != expected:
            raise ValueError("corpus store is already pinned to another family")

    def _verify_store_pins(self) -> None:
        for path in (
            Path(self._config.public_store),
            Path(self._config.evaluator_store),
        ):
            marker = path / ".corpus-family.json"
            if read_regular(marker, 64 * 1024) != self._marker_bytes():
                raise ValueError("corpus store family pin is missing or changed")

    @classmethod
    def provision(
        cls,
        root: Path,
        *,
        public_store: Path,
        evaluator_store: Path,
        repository: Path,
    ) -> "CorpusFamily":
        return cls._provision(
            root,
            public_store=public_store,
            evaluator_store=evaluator_store,
            repository=repository,
            test_schedule_public_key=None,
        )

    @classmethod
    def _provision_for_test(
        cls,
        root: Path,
        *,
        public_store: Path,
        evaluator_store: Path,
        repository: Path,
        schedule_public_key: bytes,
    ) -> "CorpusFamily":
        """Provision a family whose schedule authority is explicitly non-production."""
        return cls._provision(
            root,
            public_store=public_store,
            evaluator_store=evaluator_store,
            repository=repository,
            test_schedule_public_key=schedule_public_key,
        )

    @classmethod
    def _provision(
        cls,
        root: Path,
        *,
        public_store: Path,
        evaluator_store: Path,
        repository: Path,
        test_schedule_public_key: bytes | None,
    ) -> "CorpusFamily":
        controller_root = root.absolute()
        public = public_store.absolute()
        evaluator = evaluator_store.absolute()
        repo = repository.absolute()
        if public == evaluator or any(
            _is_within(controller_root, store) or _is_within(store, controller_root)
            for store in (public, evaluator, repo)
        ):
            raise ValueError("corpus controller state must be separate from stores and repository")
        reject_symlinks(controller_root)
        controller_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(controller_root, 0o700)
        key_hash = (
            hashlib.sha256(test_schedule_public_key).hexdigest()
            if test_schedule_public_key is not None
            else None
        )
        config = _FamilyConfig(
            public_store=str(public),
            evaluator_store=str(evaluator),
            schedule_authority_profile=(
                "TEST_ONLY" if test_schedule_public_key is not None else "UNCONFIGURED"
            ),
            schedule_public_key_hash=key_hash,
        )
        config_path = controller_root / "family.json"
        _atomic_create(config_path, config.model_dump_json().encode(), 0o600)
        observed = _FamilyConfig.model_validate_json(read_regular(config_path, 64 * 1024))
        if observed != config:
            raise ValueError("corpus family is already configured for different stores")
        if test_schedule_public_key is not None:
            key_path = controller_root / "schedule-authority.pub"
            _atomic_create(key_path, test_schedule_public_key, 0o400)
            if read_regular(key_path, 64 * 1024) != test_schedule_public_key:
                raise ValueError("test schedule public key differs from family configuration")
        family = cls(controller_root, observed, CorpusLedger(controller_root / "ledger"))
        family._verify_schedule_key()
        family._pin_store(public)
        family._pin_store(evaluator)
        return family

    @classmethod
    def open(cls, root: Path) -> "CorpusFamily":
        controller_root = root.absolute()
        reject_symlinks(controller_root)
        if not controller_root.is_dir() or controller_root.stat().st_mode & 0o077:
            raise ValueError("corpus family controller root is unavailable or unsafe")
        config = _FamilyConfig.model_validate_json(
            read_regular(controller_root / "family.json", 64 * 1024)
        )
        public, evaluator = Path(config.public_store), Path(config.evaluator_store)
        if public == evaluator or any(
            _is_within(controller_root, store) or _is_within(store, controller_root)
            for store in (public, evaluator)
        ):
            raise ValueError("corpus family store boundaries are unsafe")
        family = cls(controller_root, config, CorpusLedger(controller_root / "ledger"))
        family._verify_schedule_key()
        family._verify_store_pins()
        return family

    @classmethod
    def configured(cls, store: RunStore) -> "CorpusFamily":
        raw = os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT")
        if not raw:
            raise ValueError("trusted corpus family configuration is required")
        family = cls.open(Path(raw))
        family.require_store(store)
        return family

    def require_store(self, store: RunStore) -> None:
        expected = (
            Path(self._config.public_store)
            if store.visibility == "public"
            else Path(self._config.evaluator_store)
        )
        if store.root != expected:
            raise ValueError("store does not belong to the configured corpus family")

    def _verify_schedule_key(self) -> None:
        if self._config.schedule_authority_profile == "UNCONFIGURED":
            if self._config.schedule_public_key_hash is not None:
                raise ValueError("unconfigured schedule authority has a public key")
            return
        content = read_regular(self.root / "schedule-authority.pub", 64 * 1024)
        if hashlib.sha256(content).hexdigest() != self._config.schedule_public_key_hash:
            raise ValueError("schedule authority public key is unavailable or changed")

    @property
    def schedule_authority_profile(self) -> Literal["UNCONFIGURED", "PRODUCTION", "TEST_ONLY"]:
        return self._config.schedule_authority_profile

    @property
    def schedule_public_key_hash(self) -> str | None:
        return self._config.schedule_public_key_hash

    @property
    def schedule_public_key_path(self) -> Path:
        self._verify_schedule_key()
        if self._config.schedule_authority_profile == "UNCONFIGURED":
            raise ValueError("external schedule authority is not configured")
        return self.root / "schedule-authority.pub"

    def corpus_store(self, visibility: Visibility) -> RunStore:
        """Open one store fixed by the controller-private family configuration."""
        path = (
            Path(self._config.public_store)
            if visibility == "public"
            else Path(self._config.evaluator_store)
        )
        store = RunStore(path, visibility=visibility)
        self.require_store(store)
        return store

    def reject_repository_overlap(self, repository: Path) -> None:
        repo = repository.absolute()
        if _is_within(self.root, repo) or _is_within(repo, self.root):
            raise ValueError("corpus controller secrets must not overlap the repository")


class CorpusLedger:
    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        reject_symlinks(self.root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.key_path = self.root / "identity.key"
        init_path = self.root / ".init-lock"
        init_fd = os.open(init_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(init_fd, fcntl.LOCK_EX)
            if not self.key_path.exists():
                _atomic_create(self.key_path, os.urandom(32), 0o600)
            self.__key = read_regular(self.key_path, 32)
        finally:
            os.close(init_fd)
        if len(self.__key) != 32 or self.key_path.stat().st_mode & 0o077:
            raise ValueError("corpus ledger key is unavailable or has unsafe permissions")
        self.namespace_hash = hashlib.sha256(
            b"gpu-agent-corpus-ledger-v2\0" + self.__key
        ).hexdigest()

    def _identity(self, domain: bytes, value: bytes) -> str:
        return hmac.new(self.__key, domain + b"\0" + value, hashlib.sha256).hexdigest()

    def identities(
        self, case_identity: bytes, template_identity: bytes, source_pair: bytes
    ) -> tuple[str, str, str]:
        return (
            self._identity(b"case", case_identity),
            self._identity(b"template", template_identity),
            self._identity(b"source-pair", source_pair),
        )

    def target_store_hash(self, store: RunStore) -> str:
        return self._identity(b"target-store", str(store.root).encode())

    def _locked_state(self) -> tuple[int, dict[str, object]]:
        lock_path = self.root / ".lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            state_path = self.root / "transactions.json"
            if state_path.exists():
                state = json.loads(read_regular(state_path, 16 * 1024 * 1024))
            else:
                state = {"schema_version": 3, "transactions": [], "evaluation_reservations": []}
            if state.get("schema_version") != 3 or not isinstance(state.get("transactions"), list):
                raise ValueError("corpus ledger is malformed")
            state.setdefault("evaluation_reservations", [])
            if not isinstance(state["evaluation_reservations"], list):
                raise ValueError("corpus ledger evaluation reservations are malformed")
        except BaseException:
            os.close(fd)
            raise
        return fd, state

    def _save(self, state: dict[str, object]) -> None:
        raw = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
        fd, temporary = tempfile.mkstemp(prefix=".ledger-", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.root / "transactions.json")
            sync_directory(self.root)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _transaction(value: object) -> CorpusTransaction:
        try:
            return CorpusTransaction.model_validate(value)
        except ValueError as exc:
            raise ValueError("corpus transaction is malformed") from exc

    def _get_transaction(self, transaction_id: str) -> CorpusTransaction:
        fd, state = self._locked_state()
        try:
            transactions = state["transactions"]
            assert isinstance(transactions, list)
            for raw in transactions:
                observed = self._transaction(raw)
                if observed.transaction_id == transaction_id:
                    return observed
            raise ValueError("corpus transaction is unavailable")
        finally:
            os.close(fd)

    def committed(self, transaction_id: str) -> CorpusTransaction:
        """Return one durably committed transaction; PREPARED is never evidence."""
        transaction = self._get_transaction(transaction_id)
        if transaction.state != "COMMITTED":
            raise ValueError("corpus transaction is not committed")
        return transaction

    def committed_through(self, cutoff: int | None = None) -> list[CorpusTransaction]:
        """Return the ordered, immutable COMMITTED corpus universe at ``cutoff``."""
        fd, state = self._locked_state()
        try:
            raw_transactions = state["transactions"]
            assert isinstance(raw_transactions, list)
            transactions = [self._transaction(item) for item in raw_transactions]
        finally:
            os.close(fd)
        committed = [item for item in transactions if item.state == "COMMITTED"]
        if any(item.commit_sequence is None for item in committed):
            raise ValueError("committed corpus transaction has no sequence")
        ordered = sorted(committed, key=lambda item: item.commit_sequence or 0)
        if [item.commit_sequence for item in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError("corpus commit sequence is not contiguous")
        if cutoff is None:
            return ordered
        if cutoff < 0 or cutoff > len(ordered):
            raise ValueError("corpus universe cutoff is invalid")
        return ordered[:cutoff]

    @staticmethod
    def _committed_in_state(state: dict[str, object]) -> list[CorpusTransaction]:
        raw_transactions = state["transactions"]
        assert isinstance(raw_transactions, list)
        transactions = [CorpusLedger._transaction(item) for item in raw_transactions]
        committed = [item for item in transactions if item.state == "COMMITTED"]
        if any(item.commit_sequence is None for item in committed):
            raise ValueError("committed corpus transaction has no sequence")
        ordered = sorted(committed, key=lambda item: item.commit_sequence or 0)
        if [item.commit_sequence for item in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError("corpus commit sequence is not contiguous")
        return ordered

    @staticmethod
    def _reservation(value: object) -> EvaluationCutoffReservation:
        try:
            return EvaluationCutoffReservation.model_validate(value)
        except ValueError as exc:
            raise ValueError("evaluation cutoff reservation is malformed") from exc

    @staticmethod
    def _reservation_prestate(
        lease: EvaluationRunLease,
        binding: RunBinding,
        *,
        require_pristine: bool,
    ) -> _EvaluationReservationPrestate:
        """Reconstruct the initial state solely through the pinned run lease."""
        lease.validate()
        run = lease.load()
        if (
            run.kind != "evaluation"
            or run.binding != binding
            or binding.purpose != "evaluation"
            or not run.events
            or run.events[0].status != RunStatus.QUEUED
            or run.events[0].phase is not None
        ):
            raise ValueError("evaluation reservation requires a bound QUEUED run")
        initial_events: list[StateEvent] = run.events[:1]
        initial = run.model_copy(
            update={
                "status": RunStatus.QUEUED,
                "current_phase": None,
                "last_completed_phase": None,
                "events": initial_events,
                "artifact_refs": [],
            }
        )
        children = lease.children() if require_pristine else []
        if require_pristine and (
            run != initial or len(run.events) != 1 or run.artifact_refs or children
        ):
            raise ValueError("evaluation reservation requires an untouched QUEUED run")
        identity = lease.identity
        prestate = _EvaluationReservationPrestate(
            target_store_root=str(Path(identity.resolved_path).parent),
            target_store_device=identity.root_device,
            target_store_inode=identity.root_inode,
            target_visibility=lease.store.visibility,
            target_run_path=identity.resolved_path,
            target_run_device=identity.device,
            target_run_inode=identity.inode,
            target_run_lock_device=identity.lock_device,
            target_run_lock_inode=identity.lock_inode,
            queued_manifest_hash=_reservation_manifest_hash(initial),
            event_prefix_hash=_reservation_sequence_hash(_RESERVATION_EVENT_DOMAIN, initial_events),
            artifact_prefix_hash=_reservation_sequence_hash(_RESERVATION_ARTIFACT_DOMAIN, []),
            child_inventory_hash=_empty_reservation_child_hash(),
        )
        lease.validate()
        return prestate

    @staticmethod
    def validate_evaluation_reservation_prestate(
        reservation: EvaluationCutoffReservation,
        lease: EvaluationRunLease,
        binding: RunBinding,
        *,
        require_pristine: bool = False,
    ) -> None:
        observed = CorpusLedger._reservation_prestate(
            lease,
            binding,
            require_pristine=require_pristine,
        )
        expected = _EvaluationReservationPrestate(
            target_store_root=reservation.target_store_root,
            target_store_device=reservation.target_store_device,
            target_store_inode=reservation.target_store_inode,
            target_visibility=reservation.target_visibility,
            target_run_path=reservation.target_run_path,
            target_run_device=reservation.target_run_device,
            target_run_inode=reservation.target_run_inode,
            target_run_lock_device=reservation.target_run_lock_device,
            target_run_lock_inode=reservation.target_run_lock_inode,
            queued_manifest_hash=reservation.queued_manifest_hash,
            event_prefix_hash=reservation.event_prefix_hash,
            artifact_prefix_hash=reservation.artifact_prefix_hash,
            child_inventory_hash=reservation.child_inventory_hash,
        )
        if observed != expected or reservation.evaluation_run_id != lease.run_id:
            raise ValueError("evaluation run prestate differs from cutoff reservation")

    def reserve_evaluation_cutoff(
        self,
        *,
        lease: EvaluationRunLease,
        target_store_hash: str,
        binding: RunBinding,
        selection: EvaluationSelectionName,
        modes: list[EvaluationModeName],
        split: Literal["development", "holdout"],
        repeats: int,
        random_seed: int,
        max_cost_usd: float | None,
        max_unit_cost_usd: float | None,
        holdout_public_run_id: str | None,
        holdout_aliases_hash: str | None,
        holdout_aliases: list[str],
    ) -> EvaluationCutoffReservation:
        """Two-phase reservation under the global lease -> ledger lock order."""
        lease.validate()
        fd, state = self._locked_state()
        try:
            prestate = self._reservation_prestate(lease, binding, require_pristine=True)
            raw_reservations = state["evaluation_reservations"]
            assert isinstance(raw_reservations, list)

            def candidate(
                cutoff: int, status: Literal["PREPARED", "COMMITTED"]
            ) -> EvaluationCutoffReservation:
                return EvaluationCutoffReservation(
                    state=status,
                    evaluation_run_id=lease.run_id,
                    target_store_hash=target_store_hash,
                    binding=binding,
                    selection=selection,
                    modes=modes,
                    split=split,
                    repeats=repeats,
                    random_seed=random_seed,
                    max_cost_usd=max_cost_usd,
                    max_unit_cost_usd=max_unit_cost_usd,
                    holdout_public_run_id=holdout_public_run_id,
                    holdout_aliases_hash=holdout_aliases_hash,
                    holdout_aliases=holdout_aliases,
                    corpus_cutoff=cutoff,
                    target_store_root=prestate.target_store_root,
                    target_store_device=prestate.target_store_device,
                    target_store_inode=prestate.target_store_inode,
                    target_visibility=prestate.target_visibility,
                    target_run_path=prestate.target_run_path,
                    target_run_device=prestate.target_run_device,
                    target_run_inode=prestate.target_run_inode,
                    target_run_lock_device=prestate.target_run_lock_device,
                    target_run_lock_inode=prestate.target_run_lock_inode,
                    queued_manifest_hash=prestate.queued_manifest_hash,
                    event_prefix_hash=prestate.event_prefix_hash,
                    artifact_prefix_hash=prestate.artifact_prefix_hash,
                    child_inventory_hash=prestate.child_inventory_hash,
                )

            match: tuple[int, EvaluationCutoffReservation] | None = None
            for index, raw in enumerate(raw_reservations):
                observed = self._reservation(raw)
                if observed.evaluation_run_id == lease.run_id:
                    if match is not None:
                        raise ValueError("evaluation cutoff reservation is ambiguous")
                    match = index, observed
            if match is None:
                cutoff = len(self._committed_in_state(state))
                if cutoff < 1:
                    raise ValueError("evaluation cutoff reservation requires a committed corpus")
                prepared = candidate(cutoff, "PREPARED")
                raw_reservations.append(prepared.model_dump(mode="json"))
                index = len(raw_reservations) - 1
                lease.validate()
                self._save(state)
                lease.validate()
            else:
                index, observed = match
                prepared = candidate(observed.corpus_cutoff, "PREPARED")
                if (
                    observed.model_copy(update={"state": "PREPARED", "schedule_hash": None})
                    != prepared
                ):
                    raise ValueError("evaluation cutoff reservation differs from run authority")
                if observed.state == "COMMITTED":
                    return observed
            committed = prepared.model_copy(update={"state": "COMMITTED"})
            lease.validate()
            raw_reservations[index] = committed.model_dump(mode="json")
            try:
                self._save(state)
                lease.validate()
            except BaseException:
                raw_reservations[index] = prepared.model_dump(mode="json")
                try:
                    self._save(state)
                except BaseException:
                    # Preserve the original error. A COMMITTED value is still unusable
                    # unless a reader can revalidate this exact leased prestate.
                    pass
                raise
            return committed
        finally:
            os.close(fd)

    def evaluation_cutoff_reservation(
        self, lease: EvaluationRunLease, binding: RunBinding
    ) -> EvaluationCutoffReservation:
        """Return one usable reservation; PREPARED state is never authority."""
        lease.validate()
        fd, state = self._locked_state()
        try:
            raw_reservations = state["evaluation_reservations"]
            assert isinstance(raw_reservations, list)
            observed = [self._reservation(raw) for raw in raw_reservations]
            matches = [item for item in observed if item.evaluation_run_id == lease.run_id]
            if len(matches) != 1 or matches[0].state != "COMMITTED":
                raise ValueError("evaluation cutoff reservation is unavailable or ambiguous")
            self.validate_evaluation_reservation_prestate(matches[0], lease, binding)
            lease.validate()
            return matches[0]
        finally:
            os.close(fd)

    def bind_evaluation_schedule(
        self,
        lease: EvaluationRunLease,
        binding: RunBinding,
        schedule_hash: str,
    ) -> EvaluationCutoffReservation:
        """CAS the authority-derived schedule hash without accepting caller prestate."""
        lease.validate()
        fd, state = self._locked_state()
        try:
            raw_reservations = state["evaluation_reservations"]
            assert isinstance(raw_reservations, list)
            match: tuple[int, EvaluationCutoffReservation] | None = None
            for index, raw in enumerate(raw_reservations):
                observed = self._reservation(raw)
                if observed.evaluation_run_id == lease.run_id:
                    if match is not None:
                        raise ValueError("evaluation cutoff reservation is ambiguous")
                    match = index, observed
            if match is None or match[1].state != "COMMITTED":
                raise ValueError("evaluation cutoff reservation is unavailable")
            index, observed = match
            self.validate_evaluation_reservation_prestate(
                observed,
                lease,
                binding,
                require_pristine=observed.schedule_hash is None,
            )
            if observed.schedule_hash not in {None, schedule_hash}:
                raise ValueError("evaluation schedule differs from cutoff reservation")
            bound = observed.model_copy(update={"schedule_hash": schedule_hash})
            if observed != bound:
                lease.validate()
                raw_reservations[index] = bound.model_dump(mode="json")
                self._save(state)
                try:
                    lease.validate()
                except BaseException:
                    raw_reservations[index] = observed.model_dump(mode="json")
                    self._save(state)
                    raise
            return bound
        finally:
            os.close(fd)

    @contextmanager
    def registration_lock(self, transaction: CorpusTransaction) -> Iterator[CorpusTransaction]:
        """Serialize one transaction's RunStore completion and ledger commit.

        The per-transaction flock is released by the kernel if a controller dies. The
        shared ledger lock is acquired only briefly to reload state, then released
        before any RunStore operation, so the lock order cannot invert.
        """
        lock_path = self.root / f".transaction-{transaction.transaction_id}.lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if os.fstat(fd).st_mode & 0o077:
                raise ValueError("corpus transaction lock has unsafe permissions")
            fcntl.flock(fd, fcntl.LOCK_EX)
            observed = self._get_transaction(transaction.transaction_id)
            if observed.owner_id != transaction.owner_id:
                raise ValueError("corpus transaction owner changed")
            exact = observed == transaction
            committed_while_waiting = (
                transaction.state == "PREPARED"
                and observed.state == "COMMITTED"
                and observed.model_copy(update={"state": "PREPARED", "commit_sequence": None})
                == transaction
            )
            if not exact and not committed_while_waiting:
                raise ValueError("corpus transaction changed before recovery")
            yield observed
        finally:
            os.close(fd)

    def prepare(
        self,
        case_identity: bytes,
        template_identity: bytes,
        source_pair: bytes,
        *,
        store: RunStore,
        manifest_hash: str,
    ) -> CorpusTransaction:
        case_hash, template_hash, pair_hash = self.identities(
            case_identity, template_identity, source_pair
        )
        target_hash = self.target_store_hash(store)
        fd, state = self._locked_state()
        try:
            transactions = state["transactions"]
            assert isinstance(transactions, list)
            for raw in transactions:
                existing = self._transaction(raw)
                collision = (
                    existing.case_hash == case_hash
                    or existing.template_hash == template_hash
                    or existing.source_pair_hash == pair_hash
                )
                exact = (
                    existing.case_hash == case_hash
                    and existing.template_hash == template_hash
                    and existing.source_pair_hash == pair_hash
                    and existing.target_store_hash == target_hash
                    and existing.visibility == store.visibility
                    and existing.manifest_hash == manifest_hash
                )
                if collision and not exact:
                    raise ValueError("corpus identity or source pair is already reserved")
                if exact:
                    return existing
            transaction = CorpusTransaction(
                transaction_id=new_id(),
                owner_id=new_id(),
                run_id=new_id(),
                state="PREPARED",
                case_hash=case_hash,
                template_hash=template_hash,
                source_pair_hash=pair_hash,
                target_store_hash=target_hash,
                visibility=store.visibility,
                manifest_hash=manifest_hash,
                commit_sequence=None,
            )
            transactions.append(transaction.model_dump(mode="json"))
            self._save(state)
            return transaction
        finally:
            os.close(fd)

    def commit(self, transaction: CorpusTransaction) -> CorpusTransaction:
        fd, state = self._locked_state()
        try:
            transactions = state["transactions"]
            assert isinstance(transactions, list)
            for index, raw in enumerate(transactions):
                observed = self._transaction(raw)
                if observed.transaction_id != transaction.transaction_id:
                    continue
                if observed != transaction:
                    raise ValueError("corpus transaction changed before commit")
                if observed.commit_sequence is not None:
                    raise ValueError("prepared corpus transaction already has a commit sequence")
                next_sequence = 1 + sum(
                    1 for item in transactions if self._transaction(item).state == "COMMITTED"
                )
                committed = observed.model_copy(
                    update={"state": "COMMITTED", "commit_sequence": next_sequence}
                )
                transactions[index] = committed.model_dump(mode="json")
                self._save(state)
                return committed
            raise ValueError("corpus transaction is unavailable")
        finally:
            os.close(fd)
