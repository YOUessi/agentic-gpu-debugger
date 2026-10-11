"""Controller-only execution for private holdout corpus candidates.

No CLI imports this module. Private identities, plans, run IDs, and evidence are
persisted only in the evaluator store; the returned projection is deliberately
opaque and contains no per-case fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.benchmark.batch_models import RoleSummary
from gpu_agent.benchmark.batch_security import BatchInputError, DirectoryPin, data_lock
from gpu_agent.benchmark.builder import BenchmarkBuilder
from gpu_agent.benchmark.ledger import CorpusFamily
from gpu_agent.benchmark.models import (
    AuthoritativeCaseRegistry,
    AuthoritativeCaseSpec,
    CaseExecutionPlan,
)
from gpu_agent.benchmark.validation import CaseValidationController, source_identities
from gpu_agent.contracts import RepositorySnapshot, RunBinding, RunStatus, now
from gpu_agent.environment import ExpectedToolchain, load_toolchain_lock
from gpu_agent.execution.isolated import Availability, IsolatedGPUBackend
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.provenance import capture_repository_snapshot
from gpu_agent.store import read_regular, reject_symlinks

_REGISTRY = "registry.json"
_SOURCE_MANIFESTS = "source-manifests.json"
_FILE_LIMIT = 4 * 1024 * 1024
_TREE_LIMIT = 32 * 1024 * 1024
_MAX_FILES = 64


class PrivateBatchInputError(BatchInputError):
    """Private controller input is unavailable or changed."""


class _PrivateSourceCase(ExecutionModel):
    case_id: str = Field(pattern=r"^case_[0-9]{4}$")
    clean_manifest: dict[str, str] = Field(min_length=4, max_length=4)
    mutant_manifest: dict[str, str] = Field(min_length=4, max_length=4)
    input_path: str = Field(min_length=1, max_length=256)


class _PrivateSourceRegistry(ExecutionModel):
    schema_version: Literal[1] = 1
    cases: list[_PrivateSourceCase] = Field(min_length=1, max_length=8)


@dataclass(frozen=True, repr=False)
class PreparedPrivateCase:
    spec: AuthoritativeCaseSpec
    clean_plan: CaseExecutionPlan
    mutant_plan: CaseExecutionPlan
    input_bytes: bytes


@dataclass(frozen=True, repr=False)
class PreparedPrivateBatch:
    repository: Path
    private_root: Path
    workspace_root: Path
    family_root: Path
    evaluator_root: Path
    repository_snapshot: RepositorySnapshot
    toolchain: ExpectedToolchain
    binding: RunBinding
    registry_bytes: bytes
    file_hashes: tuple[tuple[str, str], ...]
    private_batch_hash: str
    cases: tuple[PreparedPrivateCase, ...]


class PrivateBatchProjection(BaseModel):
    """The only value suitable for crossing out of the evaluator controller."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    batch_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    case_count: int = Field(ge=1, le=8)
    status: Literal["COMPLETED", "FAILED", "CANCELLED"]
    private_batch_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class _PrivateCaseAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    status: Literal["NOT_RUN", "RUNNING", "VALIDATED", "REGISTERED", "FAILED"] = "NOT_RUN"
    clean: RoleSummary | None = None
    mutant: RoleSummary | None = None
    reason_code: str | None = None
    error_type: str | None = None


class _PrivateBatchAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    batch_run_id: str
    status: RunStatus = RunStatus.RUNNING
    register_requested: bool
    private_batch_hash: str
    started_at: datetime = Field(default_factory=now)
    finished_at: datetime | None = None
    cases: list[_PrivateCaseAudit]
    stopped_reason: str | None = None


def _overlap(first: Path, second: Path) -> bool:
    left, right = first.resolve(), second.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or "\\" in value
        or any(part in {"", ".", "..", ".git"} for part in path.parts)
    ):
        raise PrivateBatchInputError(
            "PRIVATE_INPUT_INVALID", "Private manifest contains an unsafe relative path."
        )
    return path


def _private_file(info: os.stat_result) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.geteuid()
        and stat.S_IMODE(info.st_mode) == 0o600
        and info.st_nlink == 1
    )


def _scan_private_tree(root: Path) -> dict[str, bytes]:
    try:
        pin = DirectoryPin(root)
    except BatchInputError as exc:
        raise PrivateBatchInputError(
            "PRIVATE_INPUT_UNSAFE", "Private input root must be owner-only."
        ) from exc
    files: dict[str, bytes] = {}
    total = 0
    try:
        for path in sorted(root.rglob("*")):
            reject_symlinks(path)
            info = path.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                nested = DirectoryPin(path)
                nested.close()
                continue
            if not _private_file(info):
                raise ValueError("unsafe private input file")
            relative = path.relative_to(root).as_posix()
            _safe_relative(relative)
            content = read_regular(path, _FILE_LIMIT)
            files[relative] = content
            total += len(content)
            if len(files) > _MAX_FILES or total > _TREE_LIMIT:
                raise ValueError("private input tree exceeds bounds")
        pin.validate()
    except (OSError, ValueError, BatchInputError) as exc:
        raise PrivateBatchInputError(
            "PRIVATE_INPUT_UNSAFE", "Private input tree failed its integrity policy."
        ) from exc
    finally:
        pin.close()
    return files


def _tree_hash(files: dict[str, bytes]) -> tuple[tuple[tuple[str, str], ...], str]:
    hashes = tuple(
        sorted((name, hashlib.sha256(content).hexdigest()) for name, content in files.items())
    )
    canonical = json.dumps(hashes, separators=(",", ":")).encode()
    return hashes, hashlib.sha256(canonical).hexdigest()


def _plan(
    spec: AuthoritativeCaseSpec,
    role: Literal["clean", "mutant"],
    manifest: dict[str, str],
    registry_hash: str,
) -> CaseExecutionPlan:
    return CaseExecutionPlan(
        case_id=spec.case_id,
        template_id=spec.template_id,
        mutation_id="clean" if role == "clean" else spec.mutation_id,
        role=role,
        split="private",
        source_manifest=manifest,
        oracle_id=spec.oracle_id,
        target_tool=spec.target_tool,
        expected_finding=spec.expected_finding,
        sanitizer_repetitions=spec.sanitizer_repetitions,
        case_registry_hash=registry_hash,
        case_spec_hash=CaseValidationController.spec_hash(spec),
        mutation_provenance_hash=spec.mutation_provenance_hash,
    )


def _load_cases(files: dict[str, bytes]) -> tuple[bytes, tuple[PreparedPrivateCase, ...]]:
    try:
        registry_bytes = files[_REGISTRY]
        registry = AuthoritativeCaseRegistry.model_validate_json(registry_bytes)
        sources = _PrivateSourceRegistry.model_validate_json(files[_SOURCE_MANIFESTS])
        if not registry.cases or len(registry.cases) > 8:
            raise ValueError("private registry size is invalid")
        specs = {spec.case_id: spec for spec in registry.cases}
        recipes = {case.case_id: case for case in sources.cases}
        if (
            len(specs) != len(registry.cases)
            or len(recipes) != len(sources.cases)
            or specs.keys() != recipes.keys()
            or any(spec.split != "private" for spec in specs.values())
        ):
            raise ValueError("private registry identities are invalid")
        expected_files = {_REGISTRY, _SOURCE_MANIFESTS}
        registry_hash = hashlib.sha256(registry_bytes).hexdigest()
        prepared = []
        for spec in registry.cases:
            recipe = recipes[spec.case_id]
            input_name = _safe_relative(recipe.input_path).as_posix()
            input_bytes = files[input_name]
            if hashlib.sha256(input_bytes).hexdigest() != spec.input_set_hash:
                raise ValueError("private input hash differs from registry")
            expected_files.add(input_name)
            manifests: list[dict[str, str]] = []
            for role, manifest in (
                ("clean", recipe.clean_manifest),
                ("mutant", recipe.mutant_manifest),
            ):
                normalized: dict[str, str] = {}
                for raw_name, expected_hash in manifest.items():
                    name = _safe_relative(raw_name).as_posix()
                    content = files[name]
                    actual_hash = hashlib.sha256(content).hexdigest()
                    if actual_hash != expected_hash:
                        raise ValueError("private source manifest hash mismatch")
                    normalized[name] = expected_hash
                    expected_files.add(name)
                source_hash, harness_hash = source_identities(normalized)
                expected_source = (
                    spec.clean_source_hash if role == "clean" else spec.mutant_source_hash
                )
                if source_hash != expected_source or harness_hash != spec.harness_hash:
                    raise ValueError("private source identity differs from registry")
                manifests.append(normalized)
            if manifests[0] == manifests[1] or spec.clean_source_hash == spec.mutant_source_hash:
                raise ValueError("private clean and mutant sources must differ")
            prepared.append(
                PreparedPrivateCase(
                    spec=spec,
                    clean_plan=_plan(spec, "clean", manifests[0], registry_hash),
                    mutant_plan=_plan(spec, "mutant", manifests[1], registry_hash),
                    input_bytes=input_bytes,
                )
            )
        if set(files) != expected_files:
            raise ValueError("private input tree contains undeclared files")
        return registry_bytes, tuple(prepared)
    except (KeyError, OSError, ValueError, UnicodeError) as exc:
        if isinstance(exc, PrivateBatchInputError):
            raise
        raise PrivateBatchInputError(
            "PRIVATE_INPUT_INVALID", "Private registry or source manifests are invalid."
        ) from exc


def _configured_family() -> CorpusFamily:
    configured = os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT")
    if not configured:
        raise PrivateBatchInputError(
            "FAMILY_CONFIG_REQUIRED", "A trusted existing corpus family is required."
        )
    try:
        return CorpusFamily.open(Path(configured))
    except (OSError, ValueError) as exc:
        raise PrivateBatchInputError(
            "FAMILY_CONFIG_REQUIRED", "The trusted corpus family is unavailable."
        ) from exc


def prepare_private_batch(
    repository: Path,
    private_root: Path,
    workspace_root: Path,
) -> PreparedPrivateBatch:
    """Read and bind an external private tree without creating runs or authority."""
    repo = repository.absolute()
    private = private_root.absolute()
    workspace = workspace_root.absolute()
    try:
        reject_symlinks(repo)
        repo = repo.resolve()
        private_pin = DirectoryPin(private)
        private_pin.close()
        workspace_pin = DirectoryPin(workspace)
        workspace_pin.close()
        private, workspace = private.resolve(), workspace.resolve()
        before = capture_repository_snapshot(repo)
        toolchain = load_toolchain_lock(repo / "containers/toolchain.lock.json")
        family = _configured_family()
        evaluator = family.corpus_store("evaluator")
        boundaries = (repo, family.root, evaluator.root, family.corpus_store("public").root)
        if _overlap(private, workspace) or any(
            _overlap(candidate, protected)
            for candidate in (private, workspace)
            for protected in boundaries
        ):
            raise ValueError("private controller paths overlap a protected root")
        family.reject_repository_overlap(repo)
        files = _scan_private_tree(private)
        registry_bytes, cases = _load_cases(files)
        file_hashes, private_batch_hash = _tree_hash(files)
        confirmed = capture_repository_snapshot(repo, expected_commit=before.commit)
        if confirmed != before:
            raise ValueError("repository changed during private preflight")
        binding = RunBinding(
            repository=before,
            purpose="corpus_validation",
            toolchain_lock_hash=toolchain.lock_hash,
            case_registry_hash=hashlib.sha256(registry_bytes).hexdigest(),
            corpus_ledger_namespace_hash=family.namespace_hash,
        )
        return PreparedPrivateBatch(
            repository=repo,
            private_root=private,
            workspace_root=workspace,
            family_root=family.root,
            evaluator_root=evaluator.root,
            repository_snapshot=before,
            toolchain=toolchain,
            binding=binding,
            registry_bytes=registry_bytes,
            file_hashes=file_hashes,
            private_batch_hash=private_batch_hash,
            cases=cases,
        )
    except PrivateBatchInputError:
        raise
    except (OSError, ValueError) as exc:
        raise PrivateBatchInputError(
            "PRIVATE_PREFLIGHT_FAILED", "Private batch preflight failed closed."
        ) from exc


class _PrivateFilePin:
    def __init__(self, path: Path, expected_hash: str) -> None:
        self.path = path
        self.expected_hash = expected_hash
        self.fd = -1
        try:
            reject_symlinks(path)
            self.fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            info = os.fstat(self.fd)
            if not _private_file(info):
                raise ValueError("unsafe private file")
            self.identity = (info.st_dev, info.st_ino)
            self.validate()
        except (OSError, ValueError) as exc:
            self.close()
            raise PrivateBatchInputError(
                "PRIVATE_INPUT_CHANGED", "Private input identity or content changed."
            ) from exc

    def _content_hash(self) -> str:
        digest = hashlib.sha256()
        offset = 0
        while True:
            chunk = os.pread(self.fd, 65536, offset)
            if not chunk:
                break
            digest.update(chunk)
            offset += len(chunk)
            if offset > _FILE_LIMIT:
                raise ValueError("private file exceeds bound")
        return digest.hexdigest()

    def validate(self) -> None:
        try:
            reject_symlinks(self.path)
            pinned = os.fstat(self.fd)
            named = self.path.stat(follow_symlinks=False)
            if (
                not _private_file(pinned)
                or not _private_file(named)
                or (pinned.st_dev, pinned.st_ino) != self.identity
                or (named.st_dev, named.st_ino) != self.identity
                or self._content_hash() != self.expected_hash
            ):
                raise ValueError("private file changed")
        except (OSError, ValueError) as exc:
            raise PrivateBatchInputError(
                "PRIVATE_INPUT_CHANGED", "Private input identity or content changed."
            ) from exc

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class _PrivateTreeLease:
    def __init__(self, prepared: PreparedPrivateBatch) -> None:
        self.directories: list[DirectoryPin] = []
        self.files: list[_PrivateFilePin] = []
        try:
            directory_paths = {prepared.private_root}
            for name, _ in prepared.file_hashes:
                path = prepared.private_root / name
                directory_paths.update(path.parents)
            directory_paths = {
                path
                for path in directory_paths
                if path == prepared.private_root or path.is_relative_to(prepared.private_root)
            }
            for path in sorted(directory_paths, key=lambda item: len(item.parts)):
                self.directories.append(DirectoryPin(path))
            for name, expected in prepared.file_hashes:
                self.files.append(_PrivateFilePin(prepared.private_root / name, expected))
            self.validate()
        except BaseException:
            self.close()
            raise

    def validate(self) -> None:
        for directory in self.directories:
            directory.validate()
        for file in self.files:
            file.validate()

    def close(self) -> None:
        for file in reversed(self.files):
            file.close()
        for directory in reversed(self.directories):
            directory.close()


class _PrivateBatchRunner:
    def __init__(
        self,
        controller: CaseValidationController,
        builder: BenchmarkBuilder,
        availability: Callable[[], Availability],
    ) -> None:
        if controller.store is not builder.store or controller.store.visibility != "evaluator":
            raise ValueError("private batch requires one evaluator store")
        self.controller = controller
        self.builder = builder
        self.store = controller.store
        self.availability = availability
        self.used_recovery = False

    def _role(
        self,
        batch_id: str,
        plan: CaseExecutionPlan,
        input_bytes: bytes,
        checkpoint: Callable[[], None],
        integrity_check: Callable[[], None],
        recovery_check: Callable[[], None],
    ) -> tuple[RoleSummary, bool]:
        run_id: str | None = None
        error: Exception | KeyboardInterrupt | None = None

        def created(actual_id: str) -> None:
            nonlocal run_id
            run_id = actual_id
            checkpoint()

        try:
            run_id = self.controller.execute(
                plan,
                input_bytes,
                parent_run_id=batch_id,
                on_created=created,
                integrity_check=integrity_check,
                recovery_check=recovery_check,
            )
        except (Exception, KeyboardInterrupt) as exc:
            error = exc
        role = RoleSummary(run_id=run_id)
        if run_id is not None:
            try:
                role.run_status = self.store.load(run_id).status
            except (OSError, ValueError) as exc:
                role.failure_stage = "EVIDENCE"
                role.reason_code = "EVIDENCE_UNREADABLE"
                role.error_type = type(exc).__name__
        if error is not None:
            role.error_type = type(error).__name__
            if isinstance(error, KeyboardInterrupt):
                role.failure_stage, role.reason_code = "EXECUTION", "USER_CANCELLED"
            elif role.reason_code is None:
                role.failure_stage, role.reason_code = "EXECUTION", "EXECUTION_FAILED"
        elif role.run_status != RunStatus.COMPLETED and role.reason_code is None:
            role.failure_stage, role.reason_code = "EXECUTION", "EXECUTION_FAILED"
        return role, isinstance(error, KeyboardInterrupt)

    def run(
        self,
        cases: tuple[PreparedPrivateCase, ...],
        private_batch_hash: str,
        *,
        register: bool,
        integrity_check: Callable[[], None],
        recovery_check: Callable[[], None],
    ) -> PrivateBatchProjection:
        integrity_check()
        batch = self.store.create_run("private_seed_batch", binding=self.controller.binding)
        self.store.transition(batch.id, RunStatus.RUNNING, "PREPARING")
        audit = _PrivateBatchAudit(
            batch_run_id=batch.id,
            register_requested=register,
            private_batch_hash=private_batch_hash,
            cases=[_PrivateCaseAudit(case_id=case.spec.case_id) for case in cases],
        )
        serial = 0

        def checkpoint() -> None:
            nonlocal serial
            integrity_check()
            self.store.put(
                batch.id,
                f"private-batch/progress/{serial:04d}.json",
                audit.model_dump_json().encode(),
                "evaluator",
            )
            serial += 1

        checkpoint()
        try:
            availability = self.availability()
            integrity_check()
            if not availability.ready:
                audit.status = RunStatus.FAILED
                audit.stopped_reason = "BACKEND_UNAVAILABLE"
                self.store.put(
                    batch.id,
                    "private-batch/backend-availability.json",
                    json.dumps({"ready": False, "reason": availability.reason[:4096]}).encode(),
                    "evaluator",
                )
            else:
                self.store.transition(batch.id, RunStatus.RUNNING, "VERIFYING")
                for prepared, result in zip(cases, audit.cases, strict=True):
                    result.status = "RUNNING"
                    checkpoint()
                    result.clean, cancelled = self._role(
                        batch.id,
                        prepared.clean_plan,
                        prepared.input_bytes,
                        checkpoint,
                        integrity_check,
                        recovery_check,
                    )
                    checkpoint()
                    if result.clean.reason_code is None:
                        result.mutant, cancelled = self._role(
                            batch.id,
                            prepared.mutant_plan,
                            prepared.input_bytes,
                            checkpoint,
                            integrity_check,
                            recovery_check,
                        )
                        checkpoint()
                    failed = result.mutant if result.mutant is not None else result.clean
                    if failed.reason_code is not None:
                        result.status = "FAILED"
                        result.reason_code = failed.reason_code
                        result.error_type = failed.error_type
                    else:
                        assert result.clean.run_id is not None and result.mutant is not None
                        assert result.mutant.run_id is not None
                        try:
                            integrity_check()
                            validated = self.builder.validate(
                                result.clean.run_id, result.mutant.run_id
                            )
                            result.status = "VALIDATED"
                            if register:
                                integrity_check()
                                self.builder.register(validated)
                                result.status = "REGISTERED"
                        except (OSError, ValueError) as exc:
                            stage = result.status
                            result.status = "FAILED"
                            result.reason_code = (
                                "REGISTRATION_REJECTED"
                                if stage == "VALIDATED"
                                else "NATIVE_EVIDENCE_REJECTED"
                            )
                            result.error_type = type(exc).__name__
                    checkpoint()
                    if cancelled:
                        audit.status = RunStatus.CANCELLED
                        audit.stopped_reason = "USER_CANCELLED"
                        break
                if audit.status == RunStatus.RUNNING:
                    audit.status = (
                        RunStatus.COMPLETED
                        if all(item.status in {"VALIDATED", "REGISTERED"} for item in audit.cases)
                        else RunStatus.FAILED
                    )
        except PrivateBatchInputError:
            recovery_check()
            self.used_recovery = True
            audit.status = RunStatus.FAILED
            audit.stopped_reason = "PRIVATE_INPUT_INTEGRITY_FAILED"
        except KeyboardInterrupt:
            audit.status = RunStatus.CANCELLED
            audit.stopped_reason = "USER_CANCELLED"
        except Exception as exc:
            integrity_check()
            audit.status = RunStatus.FAILED
            audit.stopped_reason = "PRIVATE_BATCH_CONTROLLER_FAILED"
            self.store.put(
                batch.id,
                "private-batch/error.json",
                json.dumps(
                    {"reason_code": audit.stopped_reason, "error_type": type(exc).__name__}
                ).encode(),
                "evaluator",
            )
        audit.finished_at = now()
        for result in audit.cases:
            if result.status == "RUNNING":
                result.status = "FAILED"
                result.reason_code = audit.stopped_reason or "CONTROLLER_FAILED"
            elif result.status == "NOT_RUN":
                result.reason_code = audit.stopped_reason or "NOT_SCHEDULED"
        final_check = recovery_check if self.used_recovery else integrity_check
        final_check()
        self.store.transition(batch.id, RunStatus.RUNNING, "FINALIZING")
        self.store.put(
            batch.id,
            "private-batch/summary.json",
            audit.model_dump_json().encode(),
            "evaluator",
        )
        self.store.transition(batch.id, audit.status, None)
        projection_status: Literal["COMPLETED", "FAILED", "CANCELLED"] = (
            "COMPLETED"
            if audit.status == RunStatus.COMPLETED
            else ("CANCELLED" if audit.status == RunStatus.CANCELLED else "FAILED")
        )
        return PrivateBatchProjection(
            batch_id=batch.id,
            case_count=len(cases),
            status=projection_status,
            private_batch_hash=private_batch_hash,
        )


def run_private_batch(
    prepared: PreparedPrivateBatch,
    *,
    register: bool = False,
) -> PrivateBatchProjection:
    """Execute private clean/mutant pairs without touching the public store."""
    current = prepare_private_batch(
        prepared.repository,
        prepared.private_root,
        prepared.workspace_root,
    )
    if current != prepared:
        raise PrivateBatchInputError(
            "PRIVATE_PREFLIGHT_CHANGED", "Private inputs changed after preflight."
        )
    if os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT") != str(prepared.family_root):
        raise PrivateBatchInputError(
            "FAMILY_CONFIG_CHANGED", "Corpus family configuration changed after preflight."
        )
    family = CorpusFamily.open(prepared.family_root)
    store = family.corpus_store("evaluator")
    if store.root != prepared.evaluator_root:
        raise PrivateBatchInputError(
            "FAMILY_CONFIG_CHANGED", "Evaluator store changed after preflight."
        )
    with ExitStack() as stack:
        lock = stack.enter_context(data_lock(store.root))
        tree = _PrivateTreeLease(prepared)
        stack.callback(tree.close)
        workspace_pin = DirectoryPin(prepared.workspace_root)
        stack.callback(workspace_pin.close)
        family_pin = DirectoryPin(prepared.family_root)
        stack.callback(family_pin.close)
        ledger_pin = DirectoryPin(prepared.family_root / "ledger")
        stack.callback(ledger_pin.close)

        def recovery_check() -> None:
            lock.validate()
            workspace_pin.validate()
            family_pin.validate()
            ledger_pin.validate()
            if os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT") != str(prepared.family_root):
                raise PrivateBatchInputError(
                    "FAMILY_CONFIG_CHANGED", "Corpus family configuration changed."
                )
            reopened = CorpusFamily.open(prepared.family_root)
            reopened.require_store(store)
            if reopened.namespace_hash != prepared.binding.corpus_ledger_namespace_hash:
                raise PrivateBatchInputError(
                    "FAMILY_CONFIG_CHANGED", "Corpus family authority changed."
                )

        def integrity_check() -> None:
            recovery_check()
            tree.validate()
            files = _scan_private_tree(prepared.private_root)
            if _tree_hash(files) != (prepared.file_hashes, prepared.private_batch_hash):
                raise PrivateBatchInputError(
                    "PRIVATE_INPUT_CHANGED", "Private input tree changed during execution."
                )

        integrity_check()
        backend = IsolatedGPUBackend(store, prepared.private_root, prepared.workspace_root)
        controller = CaseValidationController._for_private_batch(
            store,
            backend,
            prepared.binding,
            prepared.repository,
            prepared.registry_bytes,
        )
        runner = _PrivateBatchRunner(controller, BenchmarkBuilder(store), backend.availability)
        projection = runner.run(
            prepared.cases,
            prepared.private_batch_hash,
            register=register,
            integrity_check=integrity_check,
            recovery_check=recovery_check,
        )
        (recovery_check if runner.used_recovery else integrity_check)()
        if (
            capture_repository_snapshot(
                prepared.repository, expected_commit=prepared.repository_snapshot.commit
            )
            != prepared.repository_snapshot
        ):
            raise PrivateBatchInputError(
                "REPOSITORY_CHANGED", "Repository changed during private execution."
            )
        return projection
