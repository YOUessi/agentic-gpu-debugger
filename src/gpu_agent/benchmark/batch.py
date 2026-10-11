"""Serial public seed validation using the native controller and registration gate.

This is a GPU-only operations entry point, not an evaluation runner. It never calls
an LLM or bypasses native source/toolchain/ledger validation.
"""

import hashlib
import json
import os
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Literal, cast

from gpu_agent.benchmark.batch_models import (
    BatchSummary,
    PreflightReport,
    PreparedBatch,
    PreparedSeed,
    RoleSummary,
    SeedDescription,
    SeedRecipes,
    SeedResult,
)
from gpu_agent.benchmark.batch_report import render_batch, summarize_role
from gpu_agent.benchmark.batch_security import BatchInputError, DirectoryPin, public_store_path
from gpu_agent.benchmark.batch_security import data_lock as _data_lock
from gpu_agent.benchmark.builder import BenchmarkBuilder
from gpu_agent.benchmark.ledger import CorpusFamily
from gpu_agent.benchmark.models import AuthoritativeCaseRegistry, CaseExecutionPlan
from gpu_agent.benchmark.validation import CaseValidationController, source_identities
from gpu_agent.contracts import RunBinding, RunStatus, now
from gpu_agent.environment import load_toolchain_lock
from gpu_agent.execution.isolated import Availability, IsolatedGPUBackend
from gpu_agent.provenance import capture_repository_snapshot
from gpu_agent.store import RunStore, read_regular, reject_symlinks

HARNESS_PATHS = ("harness/vector_io.cpp", "harness/vector_api.h", "harness/vendor/json.hpp")
PACKAGE_REPOSITORY = Path(__file__).resolve().parents[3]


def _safe_roots(repository: Path, data_root: Path) -> tuple[Path, Path]:
    repo, data = repository.absolute(), data_root.absolute()
    try:
        reject_symlinks(repo)
        reject_symlinks(data)
    except ValueError as exc:
        raise BatchInputError(
            "UNSAFE_PATH", "Repository/data paths must not contain symlinks."
        ) from exc
    # Normalize '..' only after checking each lexical path component for links.
    repo, data = repo.resolve(), data.resolve()
    if data.is_relative_to(repo) or repo.is_relative_to(data):
        raise BatchInputError(
            "DATA_ROOT_OVERLAP", "Use a dedicated directory outside the repository."
        )
    if data.exists() and not data.is_dir():
        raise BatchInputError("DATA_ROOT_INVALID", "The data root is not a directory.")
    if os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT"):
        public_store_path(data)
    return repo, data


def prepare_seed_batch(
    repository: Path, data_root: Path, case_ids: tuple[str, ...] = ()
) -> PreparedBatch:
    """Static preflight: no Docker probe, no RunStore creation, no registration."""
    repo, data = _safe_roots(repository, data_root)
    try:
        before = capture_repository_snapshot(repo)
    except (OSError, ValueError) as exc:
        raise BatchInputError(
            "REPOSITORY_NOT_READY", "Use a clean committed Git checkout (ZIP alone has no HEAD)."
        ) from exc
    try:
        lock = load_toolchain_lock(repo / "containers/toolchain.lock.json")
    except (OSError, ValueError) as exc:
        raise BatchInputError(
            "TOOLCHAIN_LOCK_INVALID", "Lock or hashed container inputs differ."
        ) from exc
    try:
        registry_bytes = read_regular(repo / "benchmarks/corpus-registry.json", 1024 * 1024)
        registry = AuthoritativeCaseRegistry.model_validate_json(registry_bytes)
        recipe_bytes = read_regular(repo / "benchmarks/seed-batch.json", 65536)
        recipes = SeedRecipes.model_validate_json(recipe_bytes)
    except (OSError, ValueError) as exc:
        raise BatchInputError(
            "SEED_CONFIG_INVALID", "Registry or seed recipe is missing/invalid."
        ) from exc
    specs = {spec.case_id: spec for spec in registry.cases}
    available = {recipe.case_id for recipe in recipes.cases}
    if len(specs) != len(registry.cases) or len(available) != len(recipes.cases):
        raise BatchInputError("DUPLICATE_CASE", "Registry and recipe identities must be unique.")
    if len(set(case_ids)) != len(case_ids) or not set(case_ids).issubset(available):
        raise BatchInputError("CASE_SELECTION_INVALID", "Unknown or repeated --case selection.")
    selected = set(case_ids) if case_ids else available
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()
    seeds, descriptions = [], []
    for recipe in recipes.cases:
        if recipe.case_id not in selected:
            continue
        spec = specs.get(recipe.case_id)
        if spec is None or spec.split != "public":
            raise BatchInputError(
                "PUBLIC_SEED_REQUIRED", "Only registered-spec public seeds are supported."
            )
        input_bytes = json.dumps(
            {
                "n": recipe.n,
                "a": [recipe.a_value] * recipe.n,
                "b": [recipe.b_value] * recipe.n,
            }
        ).encode()
        if hashlib.sha256(input_bytes).hexdigest() != spec.input_set_hash:
            raise BatchInputError(
                "INPUT_IDENTITY_MISMATCH",
                f"{recipe.case_id}: do not rewrite hashes to bypass this gate.",
            )
        plans = []
        for role, source in (("clean", recipe.clean_source), ("mutant", recipe.mutant_source)):
            try:
                manifest = {
                    name: hashlib.sha256(
                        read_regular(repo / "benchmarks" / name, 4 * 1024 * 1024)
                    ).hexdigest()
                    for name in (source, *HARNESS_PATHS)
                }
                source_hash, harness_hash = source_identities(manifest)
            except (OSError, ValueError) as exc:
                raise BatchInputError(
                    "SOURCE_UNAVAILABLE", f"{recipe.case_id}: bounded source/harness missing."
                ) from exc
            expected = spec.clean_source_hash if role == "clean" else spec.mutant_source_hash
            if source_hash != expected or harness_hash != spec.harness_hash:
                raise BatchInputError(
                    "SOURCE_IDENTITY_MISMATCH",
                    f"{recipe.case_id}: source/harness differs from registry.",
                )
            plans.append(
                CaseExecutionPlan(
                    case_id=spec.case_id,
                    template_id=spec.template_id,
                    mutation_id="clean" if role == "clean" else spec.mutation_id,
                    role=cast(Literal["clean", "mutant"], role),
                    split="public",
                    source_manifest=manifest,
                    oracle_id=spec.oracle_id,
                    target_tool=spec.target_tool,
                    expected_finding=spec.expected_finding,
                    sanitizer_repetitions=spec.sanitizer_repetitions,
                    case_registry_hash=registry_hash,
                    case_spec_hash=CaseValidationController.spec_hash(spec),
                    mutation_provenance_hash=spec.mutation_provenance_hash,
                )
            )
        seeds.append(PreparedSeed(spec, plans[0], plans[1], input_bytes))
        descriptions.append(
            SeedDescription(
                case_id=spec.case_id,
                target_tool=spec.target_tool,
                n=recipe.n,
                repetitions=spec.sanitizer_repetitions,
                source_hash=spec.mutant_source_hash,
                clean_source_hash=spec.clean_source_hash,
                harness_hash=spec.harness_hash,
                input_set_hash=spec.input_set_hash,
            )
        )
    if capture_repository_snapshot(repo, expected_commit=before.commit) != before:
        raise BatchInputError("REPOSITORY_CHANGED", "Repository changed during preflight.")
    return PreparedBatch(
        PreflightReport(
            repository=str(repo),
            data_root=str(data),
            repository_snapshot=before,
            toolchain=lock,
            registry_hash=registry_hash,
            recipe_hash=hashlib.sha256(recipe_bytes).hexdigest(),
            cases=descriptions,
        ),
        tuple(seeds),
    )


class SeedBatchRunner:
    """Sequential native case runs; reports are diagnostics, never authority inputs."""

    def __init__(self, controller: CaseValidationController, builder: BenchmarkBuilder) -> None:
        if builder.store is not controller.store:
            raise ValueError("batch controller and builder must share one RunStore")
        self.controller, self.builder, self.store = controller, builder, controller.store

    def _role(
        self,
        batch_id: str,
        plan: CaseExecutionPlan,
        input_bytes: bytes,
        on_created: Callable[[str], None] | None = None,
        integrity_check: Callable[[], None] | None = None,
    ) -> tuple[RoleSummary, bool]:
        error: Exception | KeyboardInterrupt | None = None
        run_id: str | None = None

        def created(actual_id: str) -> None:
            nonlocal run_id
            run_id = actual_id
            if on_created is not None:
                on_created(actual_id)

        try:
            run_id = self.controller.execute(
                plan,
                input_bytes,
                parent_run_id=batch_id,
                on_created=created,
                integrity_check=integrity_check,
            )
        except (Exception, KeyboardInterrupt) as exc:
            error = exc
        try:
            role = summarize_role(self.store, run_id, plan)
        except (OSError, ValueError) as exc:
            role = RoleSummary(
                run_id=run_id,
                failure_stage="EVIDENCE",
                reason_code="EVIDENCE_UNREADABLE",
                error_type=type(exc).__name__,
            )
        if error is not None:
            role.error_type = type(error).__name__
            if isinstance(error, KeyboardInterrupt):
                role.failure_stage, role.reason_code = "EXECUTION", "USER_CANCELLED"
            elif role.reason_code is None:
                role.failure_stage, role.reason_code = "EXECUTION", "EXECUTION_FAILED"
        return role, isinstance(error, KeyboardInterrupt)

    def run(
        self,
        seeds: tuple[PreparedSeed, ...],
        *,
        register: bool = False,
        preflight: PreflightReport | None = None,
        availability: Callable[[], Availability] | None = None,
        progress: Callable[[str], None] | None = None,
        integrity_check: Callable[[], None] | None = None,
    ) -> BatchSummary:
        if self.store.visibility != "public":
            raise ValueError("seed batches support the public store only")
        if integrity_check is not None:
            integrity_check()
        if not seeds or len({seed.spec.case_id for seed in seeds}) != len(seeds):
            raise ValueError("batch requires distinct seeds")
        batch = self.store.create_run("seed_batch", binding=self.controller.binding)
        self.store.transition(batch.id, "RUNNING", "PREPARING")
        summary = BatchSummary(
            batch_run_id=batch.id,
            register_requested=register,
            cases=[
                SeedResult(
                    case_id=seed.spec.case_id,
                    target_tool=seed.spec.target_tool,
                    repetitions=seed.spec.sanitizer_repetitions,
                )
                for seed in seeds
            ],
        )
        serial = 0

        def checkpoint() -> None:
            nonlocal serial
            if integrity_check is not None:
                integrity_check()
            self.store.put(
                batch.id,
                f"batch/progress/{serial:04d}.json",
                summary.model_dump_json().encode(),
                "public",
            )
            serial += 1

        checkpoint()
        try:
            if progress:
                progress(f"batch_run_id {batch.id}")
            if preflight is not None:
                self.store.put(
                    batch.id, "batch/preflight.json", preflight.model_dump_json().encode(), "public"
                )
            if availability is not None:
                observed = availability()
                if not observed.ready:
                    summary.status = RunStatus.FAILED
                    summary.stopped_reason = "BACKEND_UNAVAILABLE"
                    for case in summary.cases:
                        case.reason_code = "BACKEND_UNAVAILABLE"
                        case.failure_stage = "PREFLIGHT"
                    self.store.put(
                        batch.id,
                        "batch/backend-availability.json",
                        json.dumps({"ready": False, "reason": observed.reason[:4096]}).encode(),
                        "public",
                    )
            if summary.stopped_reason is None:
                self.store.transition(batch.id, "RUNNING", "VERIFYING")
                for seed, case in zip(seeds, summary.cases, strict=True):
                    case.status = "RUNNING"
                    checkpoint()
                    if progress:
                        progress(f"{case.case_id}: clean -> mutant -> native validation")

                    def record_created(
                        role_name: str, created_id: str, entry: SeedResult = case
                    ) -> None:
                        setattr(entry, role_name, RoleSummary(run_id=created_id))
                        checkpoint()

                    case.clean, cancelled = self._role(
                        batch.id,
                        seed.clean_plan,
                        seed.input_bytes,
                        on_created=partial(record_created, "clean"),
                        integrity_check=integrity_check,
                    )
                    checkpoint()
                    if case.clean.reason_code is None:
                        case.mutant, cancelled = self._role(
                            batch.id,
                            seed.mutant_plan,
                            seed.input_bytes,
                            on_created=partial(record_created, "mutant"),
                            integrity_check=integrity_check,
                        )
                        checkpoint()
                    failed_role = case.mutant if case.mutant is not None else case.clean
                    if failed_role.reason_code is not None:
                        case.status = "FAILED"
                        case.failure_stage = failed_role.failure_stage
                        case.reason_code = failed_role.reason_code
                        case.error_type = failed_role.error_type
                    else:
                        assert case.clean.run_id is not None and case.mutant is not None
                        assert case.mutant.run_id is not None
                        try:
                            if integrity_check is not None:
                                integrity_check()
                            validated = self.builder.validate(case.clean.run_id, case.mutant.run_id)
                            case.status = "VALIDATED"
                            if register:
                                if integrity_check is not None:
                                    integrity_check()
                                self.builder.register(validated)
                                case.status = "REGISTERED"
                        except (OSError, ValueError) as exc:
                            stage = "REGISTER" if case.status == "VALIDATED" else "VALIDATE"
                            case.status, case.failure_stage = "FAILED", stage
                            case.reason_code = (
                                "REGISTRATION_REJECTED"
                                if stage == "REGISTER"
                                else "NATIVE_EVIDENCE_REJECTED"
                            )
                            case.error_type = type(exc).__name__
                    checkpoint()
                    if progress:
                        progress(f"{case.case_id}: {case.status} {case.reason_code or ''}".rstrip())
                    if cancelled:
                        summary.status = RunStatus.CANCELLED
                        summary.stopped_reason = "USER_CANCELLED"
                        break
                if summary.status == RunStatus.RUNNING:
                    summary.status = RunStatus.COMPLETED
        except KeyboardInterrupt:
            summary.status, summary.stopped_reason = RunStatus.CANCELLED, "USER_CANCELLED"
        except Exception as exc:
            if integrity_check is not None:
                integrity_check()  # Do not write into a replaced/unsafe root.
            summary.status, summary.stopped_reason = RunStatus.FAILED, "BATCH_CONTROLLER_FAILED"
            self.store.put(
                batch.id,
                "batch/error.json",
                json.dumps(
                    {"reason_code": summary.stopped_reason, "error_type": type(exc).__name__}
                ).encode(),
                "public",
            )
        summary.finished_at = now()
        for case in summary.cases:
            if case.status == "RUNNING":
                case.status = "FAILED"
                case.failure_stage, case.reason_code = "CONTROLLER", summary.stopped_reason
            elif case.status == "NOT_RUN" and case.reason_code is None:
                case.reason_code = summary.stopped_reason or "NOT_SCHEDULED"
        if integrity_check is not None:
            integrity_check()
        self.store.transition(batch.id, "RUNNING", "FINALIZING")
        self.store.put(
            batch.id, "batch/summary.json", summary.model_dump_json(indent=2).encode(), "public"
        )
        self.store.put(batch.id, "batch/report.md", render_batch(summary).encode(), "public")
        self.store.transition(batch.id, summary.status, None)
        return summary


def run_public_seeds(
    prepared: PreparedBatch,
    *,
    register: bool = False,
    progress: Callable[[str], None] | None = None,
) -> BatchSummary:
    repo, data = _safe_roots(Path(prepared.report.repository), Path(prepared.report.data_root))
    if repo != PACKAGE_REPOSITORY:
        raise BatchInputError(
            "CHECKOUT_MISMATCH", "Install this checkout with pip install -e . first."
        )
    current = prepare_seed_batch(repo, data, tuple(seed.spec.case_id for seed in prepared.seeds))
    if current != prepared:
        raise BatchInputError("PREFLIGHT_CHANGED", "Source/configuration changed after preflight.")
    configured = os.environ.get("GPU_AGENT_CORPUS_FAMILY_ROOT")
    if not configured:
        raise BatchInputError(
            "FAMILY_CONFIG_REQUIRED", "Configure a trusted existing corpus family before execution."
        )
    family_root = Path(configured).absolute()
    # A missing family/key is never provisioned by a batch entry point.
    for path in (family_root / "family.json", family_root / "ledger/identity.key"):
        try:
            read_regular(path, 65536)
        except (OSError, ValueError) as exc:
            raise BatchInputError(
                "FAMILY_CONFIG_REQUIRED", "Existing family/key is required."
            ) from exc
    public_root = public_store_path(data)
    with _data_lock(data) as lease, ExitStack() as pins:
        watched = []
        for path in (family_root, family_root / "ledger", public_root):
            pin = DirectoryPin(path)
            pins.callback(pin.close)
            watched.append(pin)
        family = CorpusFamily.open(family_root)
        family.reject_repository_overlap(repo)
        store = RunStore(public_root)
        family.require_store(store)
        workspace = data / "workspaces"
        reject_symlinks(workspace)
        workspace.mkdir(mode=0o700, exist_ok=True)
        pin = DirectoryPin(workspace)
        pins.callback(pin.close)
        watched.append(pin)
        namespace = family.namespace_hash

        def check_integrity() -> None:
            lease.validate()
            for watched_pin in watched:
                watched_pin.validate()
            reopened = CorpusFamily.configured(store)
            if reopened.root != family_root or reopened.namespace_hash != namespace:
                raise BatchInputError(
                    "FAMILY_CHANGED", "Configured family changed during the batch."
                )

        check_integrity()
        binding = RunBinding(
            repository=prepared.report.repository_snapshot,
            purpose="corpus_validation",
            toolchain_lock_hash=prepared.report.toolchain.lock_hash,
            case_registry_hash=prepared.report.registry_hash,
            corpus_ledger_namespace_hash=namespace,
        )
        backend = IsolatedGPUBackend(store, repo / "benchmarks", workspace)
        controller = CaseValidationController(store, backend, binding, repo)
        result = SeedBatchRunner(controller, BenchmarkBuilder(store)).run(
            prepared.seeds,
            register=register,
            preflight=prepared.report,
            availability=backend.availability,
            progress=progress,
            integrity_check=check_integrity,
        )
        check_integrity()
        return result
