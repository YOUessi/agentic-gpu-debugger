"""Controller-owned authority for immutable, globally complete evaluation schedules.

The writer and verifier deliberately have disjoint APIs.  Evaluation executors only
receive the verifier.  The writer is created transiently by ``EvaluationRunner`` and
stores transaction state below the already trusted corpus-family controller root.
"""

import fcntl
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.benchmark.evaluation import EvaluationSchedule, HoldoutScheduleProof
from gpu_agent.benchmark.ledger import CorpusFamily
from gpu_agent.contracts import RunBinding, RunStatus, new_id
from gpu_agent.store import RunStore, read_regular, reject_symlinks, sync_directory


class _ScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    public_store: str
    corpus_namespace_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    owner_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class EvaluationScheduleTransaction(BaseModel):
    """Exact transaction persisted in controller state and copied as a public receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    transaction_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    owner_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    state: Literal["PREPARED", "COMMITTED"]
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


def _canonical(value: BaseModel) -> bytes:
    return json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()


def _schedule_hash(schedule: EvaluationSchedule) -> str:
    return hashlib.sha256(_canonical(schedule)).hexdigest()


def _atomic_replace(path: Path, content: bytes, mode: int = 0o600) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".schedule-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _store_hash(namespace_hash: str, store: RunStore) -> str:
    content = b"gpu-agent-evaluation-store-v1\0" + bytes.fromhex(namespace_hash)
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


class _ScheduleState:
    def __init__(self, root: Path, store: RunStore, namespace_hash: str, *, create: bool) -> None:
        self.root = root.absolute()
        reject_symlinks(self.root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
        if not self.root.is_dir() or self.root.stat().st_mode & 0o077:
            raise ValueError("evaluation schedule authority root is unsafe")
        config_path = self.root / "config.json"
        if create:
            init_fd = os.open(
                self.root / ".init-lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            try:
                fcntl.flock(init_fd, fcntl.LOCK_EX)
                if not config_path.exists():
                    initial = _ScheduleConfig(
                        public_store=str(store.root),
                        corpus_namespace_hash=namespace_hash,
                        owner_id=new_id(),
                    )
                    fd = os.open(
                        config_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                    )
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(initial.model_dump_json().encode())
                        stream.flush()
                        os.fsync(stream.fileno())
                    sync_directory(self.root)
            finally:
                os.close(init_fd)
        self.config = _ScheduleConfig.model_validate_json(read_regular(config_path, 65536))
        if (
            self.config.public_store != str(store.root)
            or self.config.corpus_namespace_hash != namespace_hash
        ):
            raise ValueError("evaluation schedule authority is pinned to another target")

    def locked(self) -> tuple[int, list[EvaluationScheduleTransaction]]:
        path = self.root / ".lock"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            ledger_path = self.root / "transactions.json"
            if not ledger_path.exists():
                return fd, []
            raw = json.loads(read_regular(ledger_path, 16 * 1024 * 1024))
            if raw.get("schema_version") != 1 or not isinstance(raw.get("transactions"), list):
                raise ValueError("evaluation schedule ledger is malformed")
            return fd, [
                EvaluationScheduleTransaction.model_validate(item) for item in raw["transactions"]
            ]
        except BaseException:
            os.close(fd)
            raise

    def save(self, transactions: list[EvaluationScheduleTransaction]) -> None:
        content = json.dumps(
            {
                "schema_version": 1,
                "transactions": [item.model_dump(mode="json") for item in transactions],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        _atomic_replace(self.root / "transactions.json", content)


class EvaluationScheduleAuthority:
    """Controller writer.  Do not retain this object in an executor or agent."""

    def __init__(self, state: _ScheduleState, store: RunStore) -> None:
        self.__state = state
        self.__store = store

    @classmethod
    def for_family(cls, family: CorpusFamily, store: RunStore) -> "EvaluationScheduleAuthority":
        family._verify_store_pins()
        state = _ScheduleState(
            family.root / "evaluation-schedules", store, family.namespace_hash, create=True
        )
        return cls(state, store)

    def _transaction(
        self, run_id: str, schedule: EvaluationSchedule, binding: RunBinding
    ) -> EvaluationScheduleTransaction:
        run = self.__store.load(run_id)
        pairs = _validate_coverage(schedule)
        if (
            run.kind != "evaluation"
            or run.status not in {RunStatus.QUEUED, RunStatus.RUNNING}
            or run.binding != binding
            or binding.purpose != "evaluation"
            or binding.corpus_ledger_namespace_hash != self.__state.config.corpus_namespace_hash
            or schedule.bindings.commit != binding.repository.commit
            or schedule.bindings.prompt_version != binding.prompt_version
            or schedule.bindings.toolchain_hash != binding.toolchain_lock_hash
            or schedule.bindings.model_config_hash != binding.model_config_hash
        ):
            raise ValueError("evaluation schedule differs from its immutable binding")
        holdout_aliases = _validated_holdout_aliases(schedule, self.__store, binding)
        digest = _schedule_hash(schedule)
        target = _store_hash(self.__state.config.corpus_namespace_hash, self.__store)
        transaction_id = hashlib.sha256(
            f"evaluation-schedule-v1:{run_id}:{target}:{digest}".encode()
        ).hexdigest()[:32]
        return EvaluationScheduleTransaction(
            transaction_id=transaction_id,
            owner_id=self.__state.config.owner_id,
            state="PREPARED",
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
            holdout_aliases=holdout_aliases,
        )

    def prepare(
        self, run_id: str, schedule: EvaluationSchedule, binding: RunBinding
    ) -> EvaluationScheduleTransaction:
        expected = self._transaction(run_id, schedule, binding)
        fd, transactions = self.__state.locked()
        try:
            for observed in transactions:
                if observed.transaction_id != expected.transaction_id:
                    if observed.evaluation_run_id == run_id:
                        raise ValueError("evaluation run already has another schedule transaction")
                    continue
                if observed == expected or (
                    observed.state == "COMMITTED"
                    and observed.model_copy(update={"state": "PREPARED"}) == expected
                ):
                    return observed
                raise ValueError("evaluation schedule transaction differs")
            transactions.append(expected)
            self.__state.save(transactions)
            return expected
        finally:
            os.close(fd)

    def commit(self, prepared: EvaluationScheduleTransaction) -> EvaluationScheduleTransaction:
        fd, transactions = self.__state.locked()
        try:
            for index, observed in enumerate(transactions):
                if observed.transaction_id != prepared.transaction_id:
                    continue
                if observed.state == "COMMITTED" and observed.model_copy(
                    update={"state": "PREPARED"}
                ) == prepared.model_copy(update={"state": "PREPARED"}):
                    return observed
                if observed != prepared or prepared.state != "PREPARED":
                    raise ValueError("evaluation schedule transaction changed before commit")
                committed = observed.model_copy(update={"state": "COMMITTED"})
                transactions[index] = committed
                self.__state.save(transactions)
                return committed
            raise ValueError("evaluation schedule transaction is unavailable")
        finally:
            os.close(fd)

    def persist_receipt(self, store: RunStore, committed: EvaluationScheduleTransaction) -> None:
        if store.root != self.__store.root or committed.state != "COMMITTED":
            raise ValueError("only a committed transaction may be receipted")
        store.put_if_absent_exact(
            committed.evaluation_run_id,
            "evaluation/schedule-receipt.json",
            committed.model_dump_json().encode(),
            "public",
        )

    def seal(
        self, store: RunStore, run_id: str, schedule: EvaluationSchedule, binding: RunBinding
    ) -> EvaluationScheduleTransaction:
        prepared = self.prepare(run_id, schedule, binding)
        committed = self.commit(prepared)
        self.persist_receipt(store, committed)
        return committed


class EvaluationScheduleVerifier:
    """Read-only schedule verifier used independently by every downstream consumer."""

    def __init__(self, root: Path, config: _ScheduleConfig, store: RunStore) -> None:
        self.__root = root
        self.__config = config
        self._store = store

    @classmethod
    def for_family(cls, family: CorpusFamily, store: RunStore) -> "EvaluationScheduleVerifier":
        state = _ScheduleState(
            family.root / "evaluation-schedules", store, family.namespace_hash, create=False
        )
        return cls(state.root, state.config, store)

    def verify(self, run_id: str) -> EvaluationScheduleTransaction:
        run = self._store.load(run_id)
        schedule_refs = [r for r in run.artifact_refs if r.name == "evaluation/schedule.json"]
        if len(schedule_refs) != 1:
            raise ValueError("evaluation schedule artifact is missing or ambiguous")
        schedule = EvaluationSchedule.model_validate_json(self._store.read(schedule_refs[0]))
        pairs = _validate_coverage(schedule)
        if run.binding is None:
            raise ValueError("evaluation schedule run has no immutable binding")
        holdout_aliases = _validated_holdout_aliases(schedule, self._store, run.binding)
        digest = _schedule_hash(schedule)
        target = _store_hash(self.__config.corpus_namespace_hash, self._store)
        transaction_id = hashlib.sha256(
            f"evaluation-schedule-v1:{run_id}:{target}:{digest}".encode()
        ).hexdigest()[:32]
        state = _ScheduleState(
            self.__root,
            self._store,
            self.__config.corpus_namespace_hash,
            create=False,
        )
        fd, transactions = state.locked()
        try:
            selected = [item for item in transactions if item.transaction_id == transaction_id]
        finally:
            os.close(fd)
        if len(selected) != 1 or selected[0].state != "COMMITTED":
            raise ValueError("evaluation schedule transaction is not committed")
        transaction = selected[0]
        receipt_refs = [
            r for r in run.artifact_refs if r.name == "evaluation/schedule-receipt.json"
        ]
        if len(receipt_refs) != 1:
            raise ValueError("evaluation schedule receipt is missing or ambiguous")
        receipt = EvaluationScheduleTransaction.model_validate_json(
            self._store.read(receipt_refs[0])
        )
        if (
            receipt != transaction
            or transaction.evaluation_run_id != run_id
            or transaction.owner_id != self.__config.owner_id
            or transaction.target_store_hash != target
            or transaction.schedule_hash != digest
            or transaction.binding != run.binding
            or transaction.case_templates != pairs
            or transaction.selection != schedule.selection
            or transaction.modes != schedule.modes
            or transaction.split != schedule.split
            or transaction.repeats != schedule.repeats
            or transaction.random_seed != schedule.random_seed
            or transaction.max_cost_usd != schedule.bindings.max_cost_usd
            or transaction.max_unit_cost_usd != schedule.bindings.max_unit_cost_usd
            or transaction.holdout_proof != schedule.holdout_proof
            or transaction.holdout_aliases != holdout_aliases
        ):
            raise ValueError("evaluation schedule receipt differs from controller state")
        return transaction
