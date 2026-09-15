"""Trusted corpus-family configuration and crash-safe opaque uniqueness ledger."""

import fcntl
import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.contracts import Visibility, new_id
from gpu_agent.store import RunStore, read_regular, reject_symlinks, sync_directory


class _FamilyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    public_store: str
    evaluator_store: str


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
            {"schema_version": 1, "ledger_namespace_hash": self.namespace_hash},
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
        config = _FamilyConfig(public_store=str(public), evaluator_store=str(evaluator))
        config_path = controller_root / "family.json"
        _atomic_create(config_path, config.model_dump_json().encode(), 0o600)
        observed = _FamilyConfig.model_validate_json(read_regular(config_path, 64 * 1024))
        if observed != config:
            raise ValueError("corpus family is already configured for different stores")
        family = cls(controller_root, observed, CorpusLedger(controller_root / "ledger"))
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
            self.key = read_regular(self.key_path, 32)
        finally:
            os.close(init_fd)
        if len(self.key) != 32 or self.key_path.stat().st_mode & 0o077:
            raise ValueError("corpus ledger key is unavailable or has unsafe permissions")
        self.namespace_hash = hashlib.sha256(b"gpu-agent-corpus-ledger-v2\0" + self.key).hexdigest()

    def _identity(self, domain: bytes, value: bytes) -> str:
        return hmac.new(self.key, domain + b"\0" + value, hashlib.sha256).hexdigest()

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
                state = {"schema_version": 2, "transactions": []}
            if state.get("schema_version") != 2 or not isinstance(state.get("transactions"), list):
                raise ValueError("corpus ledger is malformed")
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
                and observed.model_copy(update={"state": "PREPARED"}) == transaction
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
                committed = observed.model_copy(update={"state": "COMMITTED"})
                transactions[index] = committed.model_dump(mode="json")
                self._save(state)
                return committed
            raise ValueError("corpus transaction is unavailable")
        finally:
            os.close(fd)
