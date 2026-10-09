"""Trusted corpus-family configuration and crash-safe opaque uniqueness ledger."""

import fcntl
import hashlib
import hmac
import json
import os
import re
import stat
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
    RunStoreIdentity,
    read_regular,
    reject_symlinks,
    sync_directory,
)


class _StorePin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    resolved_path: str
    device: int = Field(ge=0)
    inode: int = Field(ge=1)
    visibility: Visibility

    @classmethod
    def from_identity(cls, identity: RunStoreIdentity) -> "_StorePin":
        return cls(
            resolved_path=identity.resolved_root,
            device=identity.device,
            inode=identity.inode,
            visibility=identity.visibility,
        )

    def identity(self) -> RunStoreIdentity:
        return RunStoreIdentity(
            resolved_root=self.resolved_path,
            device=self.device,
            inode=self.inode,
            visibility=self.visibility,
        )


class _FamilyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[3] = 3
    public_store: str
    evaluator_store: str
    public_store_pin: _StorePin
    evaluator_store_pin: _StorePin
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
    schema_version: Literal[5] = 5
    state: Literal["PREPARED", "COMMITTED"]
    preparation_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    preparation_mac: str = Field(pattern=r"^[a-f0-9]{64}$")
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
_RESERVATION_PREPARATION_DOMAIN = b"gpu-agent-evaluation-reservation-preparation-v1\0"


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


def _require_directory(path: Path, *, owner_only: bool) -> os.stat_result:
    reject_symlinks(path)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        pinned = os.fstat(fd)
        named = path.stat(follow_symlinks=False)
        expected_mode = 0o700 if owner_only else stat.S_IMODE(pinned.st_mode)
        if (
            not stat.S_ISDIR(pinned.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or pinned.st_uid != os.geteuid()
            or named.st_uid != os.geteuid()
            or stat.S_IMODE(pinned.st_mode) != expected_mode
            or stat.S_IMODE(named.st_mode) != expected_mode
            or (pinned.st_dev, pinned.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError("production family path is unavailable or unsafe")
        reject_symlinks(path)
        confirmed = path.stat(follow_symlinks=False)
        if (confirmed.st_dev, confirmed.st_ino) != (pinned.st_dev, pinned.st_ino):
            raise ValueError("production family path identity changed")
        return pinned
    finally:
        os.close(fd)


def _read_owned_regular(path: Path, limit: int, *, mode: int) -> bytes:
    """Read one controller file through a no-follow fd and pin its pathname identity."""
    parent = path.parent
    reject_symlinks(parent)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    file_fd = -1
    try:
        parent_info = os.fstat(parent_fd)
        parent_named = parent.stat(follow_symlinks=False)
        if not stat.S_ISDIR(parent_info.st_mode) or (parent_info.st_dev, parent_info.st_ino) != (
            parent_named.st_dev,
            parent_named.st_ino,
        ):
            raise ValueError("controller file parent identity changed")
        file_fd = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        pinned = os.fstat(file_fd)
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)

        def validate(observed: os.stat_result) -> None:
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_uid != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != mode
                or observed.st_nlink != 1
                or (observed.st_dev, observed.st_ino) != (pinned.st_dev, pinned.st_ino)
                or observed.st_size > limit
            ):
                raise ValueError("controller file is unavailable or unsafe")

        validate(pinned)
        validate(named)
        data = b""
        while len(data) <= limit:
            chunk = os.read(file_fd, min(64 * 1024, limit + 1 - len(data)))
            if not chunk:
                break
            data += chunk
        if len(data) > limit:
            raise ValueError("controller file grew beyond limit")
        validate(os.fstat(file_fd))
        validate(os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False))
        parent_confirmed = parent.stat(follow_symlinks=False)
        if (parent_confirmed.st_dev, parent_confirmed.st_ino) != (
            parent_info.st_dev,
            parent_info.st_ino,
        ):
            raise ValueError("controller file parent identity changed")
        return data
    except OSError as exc:
        raise ValueError("controller file is unavailable or unsafe") from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


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

    def _store_pin(self, visibility: Visibility) -> _StorePin:
        return (
            self._config.public_store_pin
            if visibility == "public"
            else self._config.evaluator_store_pin
        )

    def _marker_bytes(self, visibility: Visibility) -> bytes:
        return json.dumps(
            {
                "schema_version": 3,
                "ledger_namespace_hash": self.namespace_hash,
                "schedule_authority_profile": self._config.schedule_authority_profile,
                "schedule_public_key_hash": self._config.schedule_public_key_hash,
                "store_pin": self._store_pin(visibility).model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def _pin_store(self, path: Path, visibility: Visibility) -> None:
        reject_symlinks(path)
        marker = path / ".corpus-family.json"
        expected = self._marker_bytes(visibility)
        _atomic_create(marker, expected, 0o600)
        if _read_owned_regular(marker, 64 * 1024, mode=0o600) != expected:
            raise ValueError("corpus store is already pinned to another family")

    def _verify_store_pins(self) -> None:
        stores: tuple[tuple[Path, Visibility], ...] = (
            (Path(self._config.public_store), "public"),
            (Path(self._config.evaluator_store), "evaluator"),
        )
        for path, visibility in stores:
            store = RunStore(path, visibility=visibility)
            if store.identity != self._store_pin(visibility).identity():
                raise ValueError("corpus store identity pin changed")
            marker = path / ".corpus-family.json"
            if _read_owned_regular(marker, 64 * 1024, mode=0o600) != self._marker_bytes(visibility):
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
            schedule_public_key=None,
            schedule_authority_profile="UNCONFIGURED",
        )

    @classmethod
    def provision_production(
        cls,
        root: Path,
        *,
        public_store: Path,
        evaluator_store: Path,
        repository: Path,
        schedule_public_key: bytes,
    ) -> "CorpusFamily":
        """Provision verification for an external production schedule signer."""
        if not schedule_public_key:
            raise ValueError("production schedule public key is required")
        return cls._provision(
            root,
            public_store=public_store,
            evaluator_store=evaluator_store,
            repository=repository,
            schedule_public_key=schedule_public_key,
            schedule_authority_profile="PRODUCTION",
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
            schedule_public_key=schedule_public_key,
            schedule_authority_profile="TEST_ONLY",
        )

    @classmethod
    def _provision(
        cls,
        root: Path,
        *,
        public_store: Path,
        evaluator_store: Path,
        repository: Path,
        schedule_public_key: bytes | None,
        schedule_authority_profile: Literal["UNCONFIGURED", "PRODUCTION", "TEST_ONLY"],
    ) -> "CorpusFamily":
        controller_root = root.absolute()
        public = public_store.absolute()
        evaluator = evaluator_store.absolute()
        repo = repository.absolute()
        public_capability: RunStore | None = None
        evaluator_capability: RunStore | None = None
        if schedule_authority_profile == "PRODUCTION":
            reject_symlinks(controller_root)
            if controller_root.exists():
                _require_directory(controller_root, owner_only=True)
            _require_directory(public, owner_only=True)
            _require_directory(evaluator, owner_only=True)
            _require_directory(repo, owner_only=False)
            resolved = tuple(path.resolve(strict=True) for path in (public, evaluator, repo))
            resolved_controller = controller_root.resolve(strict=False)
            if any(
                _is_within(left, right) or _is_within(right, left)
                for left, right in (
                    (resolved[0], resolved[1]),
                    (resolved[0], resolved[2]),
                    (resolved[1], resolved[2]),
                    (resolved_controller, resolved[0]),
                    (resolved_controller, resolved[1]),
                    (resolved_controller, resolved[2]),
                )
            ):
                raise ValueError("production stores and repository must not overlap")
            public_capability = RunStore(public, visibility="public")
            evaluator_capability = RunStore(evaluator, visibility="evaluator")
            if (
                public_capability.identity.device,
                public_capability.identity.inode,
            ) == (
                evaluator_capability.identity.device,
                evaluator_capability.identity.inode,
            ):
                raise ValueError("production stores must be distinct")
        if public == evaluator or any(
            _is_within(controller_root, store) or _is_within(store, controller_root)
            for store in (public, evaluator, repo)
        ):
            raise ValueError("corpus controller state must be separate from stores and repository")
        reject_symlinks(controller_root)
        controller_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(controller_root, 0o700)
        _require_directory(controller_root, owner_only=True)
        public_capability = public_capability or RunStore(public, visibility="public")
        evaluator_capability = evaluator_capability or RunStore(evaluator, visibility="evaluator")
        if (schedule_authority_profile == "UNCONFIGURED") != (schedule_public_key is None):
            raise ValueError("schedule authority profile and public key differ")
        key_hash = (
            hashlib.sha256(schedule_public_key).hexdigest()
            if schedule_public_key is not None
            else None
        )
        config = _FamilyConfig(
            public_store=str(public),
            evaluator_store=str(evaluator),
            public_store_pin=_StorePin.from_identity(public_capability.identity),
            evaluator_store_pin=_StorePin.from_identity(evaluator_capability.identity),
            schedule_authority_profile=schedule_authority_profile,
            schedule_public_key_hash=key_hash,
        )
        config_path = controller_root / "family.json"
        _atomic_create(config_path, config.model_dump_json().encode(), 0o600)
        observed = _FamilyConfig.model_validate_json(
            _read_owned_regular(config_path, 64 * 1024, mode=0o600)
        )
        if observed != config:
            raise ValueError("corpus family is already configured for different stores")
        if schedule_public_key is not None:
            key_path = controller_root / "schedule-authority.pub"
            _atomic_create(key_path, schedule_public_key, 0o400)
            if _read_owned_regular(key_path, 64 * 1024, mode=0o400) != schedule_public_key:
                raise ValueError("schedule public key differs from family configuration")
        family = cls(controller_root, observed, CorpusLedger(controller_root / "ledger"))
        family._verify_schedule_key()
        family._pin_store(public, "public")
        family._pin_store(evaluator, "evaluator")
        return family

    @classmethod
    def open(cls, root: Path) -> "CorpusFamily":
        controller_root = root.absolute()
        reject_symlinks(controller_root)
        _require_directory(controller_root, owner_only=True)
        config = _FamilyConfig.model_validate_json(
            _read_owned_regular(controller_root / "family.json", 64 * 1024, mode=0o600)
        )
        public, evaluator = Path(config.public_store), Path(config.evaluator_store)
        if public == evaluator or any(
            _is_within(controller_root, store) or _is_within(store, controller_root)
            for store in (public, evaluator)
        ):
            raise ValueError("corpus family store boundaries are unsafe")
        stores: tuple[tuple[Path, _StorePin, Visibility], ...] = (
            (public, config.public_store_pin, "public"),
            (evaluator, config.evaluator_store_pin, "evaluator"),
        )
        for path, pin, visibility in stores:
            info = _require_directory(path, owner_only=True)
            observed = RunStoreIdentity(
                resolved_root=str(path.resolve(strict=True)),
                device=info.st_dev,
                inode=info.st_ino,
                visibility=visibility,
            )
            if observed != pin.identity():
                raise ValueError("corpus store identity pin changed")
        family = cls(
            controller_root,
            config,
            CorpusLedger(controller_root / "ledger", create=False),
        )
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
        expected = self._store_pin(store.visibility).identity()
        if store.identity != expected:
            raise ValueError("store does not belong to the configured corpus family")

    def _verify_schedule_key(self) -> None:
        if self._config.schedule_authority_profile == "UNCONFIGURED":
            if self._config.schedule_public_key_hash is not None:
                raise ValueError("unconfigured schedule authority has a public key")
            return
        content = _read_owned_regular(self.root / "schedule-authority.pub", 64 * 1024, mode=0o400)
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
        path = Path(
            self._config.public_store if visibility == "public" else self._config.evaluator_store
        )
        store = RunStore(path, visibility=visibility)
        self.require_store(store)
        return store

    def reject_repository_overlap(self, repository: Path) -> None:
        repo = repository.absolute()
        if _is_within(self.root, repo) or _is_within(repo, self.root):
            raise ValueError("corpus controller secrets must not overlap the repository")


class CorpusLedger:
    def __init__(self, root: Path, *, create: bool = True) -> None:
        self.root = root.absolute()
        reject_symlinks(self.root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
        _require_directory(self.root, owner_only=True)
        self.key_path = self.root / "identity.key"
        init_path = self.root / ".init-lock"
        flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0)
        init_fd = os.open(init_path, flags, 0o600)
        try:
            fcntl.flock(init_fd, fcntl.LOCK_EX)
            init_info = os.fstat(init_fd)
            if (
                not stat.S_ISREG(init_info.st_mode)
                or init_info.st_uid != os.geteuid()
                or stat.S_IMODE(init_info.st_mode) != 0o600
                or init_info.st_nlink != 1
            ):
                raise ValueError("corpus ledger lock is unavailable or unsafe")
            init_named = init_path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(init_named.st_mode)
                or init_named.st_uid != os.geteuid()
                or stat.S_IMODE(init_named.st_mode) != 0o600
                or init_named.st_nlink != 1
                or (init_named.st_dev, init_named.st_ino) != (init_info.st_dev, init_info.st_ino)
            ):
                raise ValueError("corpus ledger lock is unavailable or unsafe")
            if create and not self.key_path.exists():
                _atomic_create(self.key_path, os.urandom(32), 0o600)
            self.__key = _read_owned_regular(self.key_path, 32, mode=0o600)
            confirmed = init_path.stat(follow_symlinks=False)
            if (confirmed.st_dev, confirmed.st_ino) != (
                init_info.st_dev,
                init_info.st_ino,
            ):
                raise ValueError("corpus ledger lock identity changed")
        finally:
            os.close(init_fd)
        key_info = self.key_path.stat(follow_symlinks=False)
        if (
            len(self.__key) != 32
            or not stat.S_ISREG(key_info.st_mode)
            or key_info.st_uid != os.geteuid()
            or stat.S_IMODE(key_info.st_mode) != 0o600
            or key_info.st_nlink != 1
        ):
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

    def _exact_reservation_is_durable(self, expected: EvaluationCutoffReservation) -> bool:
        """Resolve an uncertain atomic-save outcome without reacquiring the ledger lock."""
        try:
            state = json.loads(read_regular(self.root / "transactions.json", 16 * 1024 * 1024))
            raw_reservations = state.get("evaluation_reservations")
            if state.get("schema_version") != 3 or not isinstance(raw_reservations, list):
                return False
            matches = [
                self._reservation(raw)
                for raw in raw_reservations
                if isinstance(raw, dict)
                and raw.get("evaluation_run_id") == expected.evaluation_run_id
            ]
            return matches == [expected]
        except (OSError, ValueError, TypeError):
            return False

    def _commit_prepared_reservation(
        self,
        lease: EvaluationRunLease,
        preparation_id: str,
    ) -> EvaluationCutoffReservation:
        """Reload and commit one authenticated PREPARED record under a fresh ledger lock."""
        if type(self) is not CorpusLedger or type(lease) is not EvaluationRunLease:
            raise ValueError("cutoff authority commit requires native controller types")
        if not re.fullmatch(r"[a-f0-9]{32}", preparation_id):
            raise ValueError("evaluation preparation ID is invalid")
        fd, state = CorpusLedger._locked_state(self)
        finalized = False
        try:
            raw_reservations = state["evaluation_reservations"]
            assert isinstance(raw_reservations, list)
            matches: list[tuple[int, EvaluationCutoffReservation]] = []
            for index, raw in enumerate(raw_reservations):
                observed = CorpusLedger._reservation(self, raw)
                if (
                    observed.preparation_id == preparation_id
                    or observed.evaluation_run_id == lease.run_id
                ):
                    matches.append((index, observed))
            if len(matches) != 1:
                raise ValueError("evaluation cutoff preparation is unavailable or ambiguous")
            index, prepared = matches[0]
            if (
                prepared.preparation_id != preparation_id
                or prepared.evaluation_run_id != lease.run_id
                or prepared.state not in {"PREPARED", "COMMITTED"}
                or prepared.corpus_cutoff > len(CorpusLedger._committed_in_state(state))
            ):
                raise ValueError("evaluation cutoff preparation is invalid")
            CorpusLedger.validate_evaluation_reservation_prestate(
                prepared,
                lease,
                prepared.binding,
                require_pristine=prepared.schedule_hash is None,
            )
            if prepared.state == "COMMITTED":
                return prepared
            committed = prepared.model_copy(
                update={"state": "COMMITTED", "preparation_mac": "0" * 64}
            )
            committed = committed.model_copy(
                update={"preparation_mac": self.__reservation_mac(committed)}
            )
            raw_reservations[index] = committed.model_dump(mode="json")
            # Last fallible authority check. Exact COMMITTED persistence is the
            # linearization point; an uncertain save is resolved by locked read-back.
            EvaluationRunLease.validate(lease)
            try:
                CorpusLedger._save(self, state)
            except BaseException:
                if not CorpusLedger._exact_reservation_is_durable(self, committed):
                    raise
            finalized = True
            return committed
        finally:
            try:
                os.close(fd)
            except OSError:
                if not finalized:
                    raise

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

    def __reservation_mac(self, reservation: EvaluationCutoffReservation) -> str:
        normalized = reservation.model_copy(update={"preparation_mac": "0" * 64})
        return hmac.new(
            self.__key,
            _RESERVATION_PREPARATION_DOMAIN + _canonical_model(normalized),
            hashlib.sha256,
        ).hexdigest()

    def _reservation(self, value: object) -> EvaluationCutoffReservation:
        if isinstance(value, dict) and value.get("schema_version") == 4:
            raise ValueError(
                "unsupported legacy evaluation cutoff reservation schema; "
                "automatic authority migration is forbidden"
            )
        try:
            reservation = EvaluationCutoffReservation.model_validate(value)
        except ValueError as exc:
            raise ValueError("evaluation cutoff reservation is malformed") from exc
        if not hmac.compare_digest(
            reservation.preparation_mac,
            self.__reservation_mac(reservation),
        ):
            raise ValueError("evaluation cutoff reservation preparation is unauthenticated")
        return reservation

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
        finalized = False
        try:
            prestate = self._reservation_prestate(lease, binding, require_pristine=True)
            raw_reservations = state["evaluation_reservations"]
            assert isinstance(raw_reservations, list)

            def candidate(
                cutoff: int,
                preparation_id: str,
                preparation_mac: str,
            ) -> EvaluationCutoffReservation:
                return EvaluationCutoffReservation(
                    state="PREPARED",
                    preparation_id=preparation_id,
                    preparation_mac=preparation_mac,
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
                prepared = candidate(cutoff, new_id(), "0" * 64)
                prepared = prepared.model_copy(
                    update={"preparation_mac": self.__reservation_mac(prepared)}
                )
                raw_reservations.append(prepared.model_dump(mode="json"))
                lease.validate()
                self._save(state)
                lease.validate()
            else:
                _, observed = match
                prepared = candidate(
                    observed.corpus_cutoff,
                    observed.preparation_id,
                    observed.preparation_mac,
                )
                if (
                    observed.model_copy(update={"state": "PREPARED", "schedule_hash": None})
                    != prepared
                ):
                    raise ValueError("evaluation cutoff reservation differs from run authority")
                if observed.state == "COMMITTED":
                    return observed
            os.close(fd)
            fd = -1
            committed = EvaluationRunLease._commit_cutoff_reservation(
                lease, self, prepared.preparation_id
            )
            finalized = True
            return committed
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    if not finalized:
                        raise

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

    def _commit_schedule_binding(
        self,
        lease: EvaluationRunLease,
        binding: RunBinding,
        preparation_id: str,
        schedule_hash: str,
    ) -> EvaluationCutoffReservation:
        """Reload, validate, sign, and durably bind one schedule as one primitive."""
        if type(self) is not CorpusLedger or type(lease) is not EvaluationRunLease:
            raise ValueError("schedule authority commit requires native controller types")
        if not re.fullmatch(r"[a-f0-9]{32}", preparation_id):
            raise ValueError("evaluation preparation ID is invalid")
        if not re.fullmatch(r"[a-f0-9]{64}", schedule_hash):
            raise ValueError("evaluation schedule hash is invalid")
        lease.validate()
        fd, state = CorpusLedger._locked_state(self)
        finalized = False
        try:
            run = lease.load()
            if (
                run.kind != "evaluation"
                or run.binding != binding
                or run.status != RunStatus.QUEUED
                or run.current_phase is not None
                or run.last_completed_phase is not None
            ):
                raise ValueError("schedule binding requires the exact QUEUED evaluation state")
            raw_reservations = state["evaluation_reservations"]
            assert isinstance(raw_reservations, list)
            match: tuple[int, EvaluationCutoffReservation] | None = None
            for index, raw in enumerate(raw_reservations):
                observed = CorpusLedger._reservation(self, raw)
                if (
                    observed.preparation_id == preparation_id
                    or observed.evaluation_run_id == lease.run_id
                ):
                    if match is not None:
                        raise ValueError("evaluation cutoff reservation is ambiguous")
                    match = index, observed
            if (
                match is None
                or match[1].preparation_id != preparation_id
                or match[1].evaluation_run_id != lease.run_id
                or match[1].state != "COMMITTED"
                or match[1].binding != binding
            ):
                raise ValueError("evaluation cutoff reservation is unavailable")
            index, observed = match
            CorpusLedger.validate_evaluation_reservation_prestate(
                observed,
                lease,
                binding,
                require_pristine=observed.schedule_hash is None,
            )
            if observed.schedule_hash not in {None, schedule_hash}:
                raise ValueError("evaluation schedule differs from cutoff reservation")
            bound = observed.model_copy(
                update={"schedule_hash": schedule_hash, "preparation_mac": "0" * 64}
            )
            bound = bound.model_copy(
                update={"preparation_mac": CorpusLedger.__reservation_mac(self, bound)}
            )
            raw_reservations[index] = bound.model_dump(mode="json")
            # Every successful invocation installs the exact signed record. All
            # fallible lease and authority checks precede this linearization point.
            EvaluationRunLease.validate(lease)
            try:
                CorpusLedger._save(self, state)
            except BaseException:
                if not CorpusLedger._exact_reservation_is_durable(self, bound):
                    raise
            finalized = True
            return bound
        finally:
            try:
                os.close(fd)
            except OSError:
                if not finalized:
                    raise

    def bind_evaluation_schedule(
        self,
        lease: EvaluationRunLease,
        binding: RunBinding,
        preparation_id: str,
        schedule_hash: str,
    ) -> EvaluationCutoffReservation:
        """Linearize the authority-derived schedule hash from stable identifiers."""
        return EvaluationRunLease._bind_cutoff_schedule(
            lease,
            self,
            binding,
            preparation_id,
            schedule_hash,
        )

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
