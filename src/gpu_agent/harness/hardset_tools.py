"""A workspace-scoped, auditable CUDA tool host (no arbitrary host shell).

The model can choose tool ORDER. It cannot choose arbitrary host file paths,
compiler flags, Docker commands, hidden verifier calls or reference outputs.
All writes are an immutable original-source diff in a disposable CUDA workspace.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Literal, cast

from gpu_agent.benchmark.hardset_v2 import HARDSET_CASES, PUBLIC_REQUIREMENTS, load_registry
from gpu_agent.benchmark.hardset_v2_probe import (
    FunctionalProbeAction,
    HardsetFunctionalProbe,
)
from gpu_agent.benchmark.hardset_v2_selfcheck import check_candidate_publicly
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.harness.models import (
    ApplyPatch,
    FinishTask,
    FunctionalProbe,
    HarnessAction,
    ListFiles,
    ReadFile,
    RunPublicChecks,
    SearchCode,
    SessionSnapshot,
    ToolObservation,
)
from gpu_agent.patching import (
    SourceSnapshot,
    apply_generated_candidate,
    materialize_candidate,
)
from gpu_agent.store import RunStore, read_regular, reject_symlinks


class HardsetWorkspaceTools:
    """Typed tool adapter over the existing verified GPU/RunStore implementation."""

    ALLOWED_TOOLS = (
        "list_files",
        "read_file",
        "search_code",
        "functional_probe",
        "apply_patch",
        "run_public_checks",
        "finish",
    )

    def __init__(
        self,
        repository: Path,
        output: Path,
        case_id: str,
        *,
        max_probes: int = 6,
        max_checks: int = 3,
    ) -> None:
        if case_id not in HARDSET_CASES:
            raise ValueError("unknown HardSet problem")
        if (
            type(max_probes) is not int
            or type(max_checks) is not int
            or not 1 <= max_probes <= 15
            or not 1 <= max_checks <= 3
        ):
            raise ValueError("unbounded workspace tool budget")
        reject_symlinks(repository)
        reject_symlinks(output)
        if output.exists():
            raise ValueError("workspace output already exists")
        root = repository.resolve() / "benchmarks"
        registry, _ = load_registry(root)
        case = next(c for c in registry.cases if c.case_id == case_id)
        original = read_regular(root / "hardset_v2" / case_id / "mutant" / "kernel.cu", 4 << 20)
        if hashlib.sha256(original).hexdigest() != case.mutant_source_sha256:
            raise ValueError("mutant changed since manifest freeze")
        self.root = repository.resolve()
        self.output = output
        self.case_id = case_id
        self.original = original
        self.source_hash = case.mutant_source_sha256
        self.functional_requirement = PUBLIC_REQUIREMENTS[case_id]
        self.max_probes, self.max_checks = max_probes, max_checks
        self._probe = HardsetFunctionalProbe(
            self.root, self.output / "workspace", case_id, max_probes=max_probes
        )
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._snapshot: SourceSnapshot | None = None
        self._store: RunStore | None = None
        self._run_id: str | None = None
        self._candidate: bytes | None = None
        self._patch_diff: str | None = None
        self._candidate_hash: str | None = None
        self._public_passed = False
        self._finished = False
        self._checks = 0
        self._events = 0
        self._attempted_hashes: set[str] = set()
        self._read_cache: set[tuple[str, int, int, str | None]] = set()
        self._search_cache: set[tuple[str, str | None]] = set()

    @property
    def run_id(self) -> str:
        if self._run_id is None:
            raise ValueError("workspace not active")
        return self._run_id

    @property
    def accepted_diff(self) -> str | None:
        return self._patch_diff if self._public_passed and self._finished else None

    @property
    def tool_calls(self) -> int:
        return self._events

    def __enter__(self) -> HardsetWorkspaceTools:
        self._probe.__enter__()
        self._store = self._probe.public_store
        self._run_id = self._probe.run_id
        self._tmp = tempfile.TemporaryDirectory(prefix="gpu-harness-v1-source-")
        original_file = Path(self._tmp.name) / "kernel.cu"
        IsolatedGPUBackend._write_snapshot(original_file, self.original)
        self._snapshot = SourceSnapshot(
            parent_run_id=self.run_id,
            root=Path(self._tmp.name),
            hashes={"kernel.cu": self.source_hash},
        )
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        try:
            if self._tmp is not None:
                self._tmp.cleanup()
        finally:
            self._probe.__exit__(exc_type, exc_val, exc_tb)

    def _file_bytes(self, name: str) -> bytes:
        if name == "kernel.cu":
            return self._candidate if self._candidate is not None else self.original
        if name in {"vector_io.cpp", "vector_api.h"}:
            return read_regular(self.root / "benchmarks" / "harness" / name, 4 << 20)
        raise ValueError("read permission denied")

    def snapshot(self, goal: str, history: list[dict[str, str]], steps: int) -> SessionSnapshot:
        return SessionSnapshot(
            goal=goal,
            case_id=cast(Literal["h001", "h002", "h003"], self.case_id),
            source_hash=self.source_hash,
            functional_requirement=self.functional_requirement,
            current_candidate_sha256=self._candidate_hash,
            public_checks_passed=self._public_passed,
            allowed_tools=self.ALLOWED_TOOLS,
            allowed_probe_sizes=self._probe.allowed_sizes,
            remaining_steps=steps,
            history=tuple(history[-30:]),
        )

    def _observation(
        self, tool: str, status: Literal["OK", "REJECTED", "FAILED"], text: str
    ) -> ToolObservation:
        return ToolObservation(
            tool=tool,
            status=status,
            content=text[:4000],
        )

    def invoke(self, action: HarnessAction) -> ToolObservation:
        if self._snapshot is None or self._store is None or self._finished:
            raise ValueError("workspace not active")
        self._events += 1
        try:
            result = self._execute(action)
        except (OSError, ValueError) as exc:
            # Only controller-owned bounded reason categories are model-visible.
            msg = type(exc).__name__ + ": " + str(exc)[:300]
            result = self._observation(action.tool, "REJECTED", msg)
        self._store.put(
            self.run_id,
            f"harness/v1/event-{self._events:03d}.json",
            json.dumps(
                {
                    "tool": action.tool,
                    "status": result.status,
                    "output_hash": hashlib.sha256(result.content.encode()).hexdigest(),
                    "public_result_excerpt": result.content[:700],
                    "candidate_sha256": self._candidate_hash,
                    "checks": self._checks,
                },
                sort_keys=True,
            ).encode(),
            "public",
        )
        return result

    def _execute(self, action: HarnessAction) -> ToolObservation:
        if self._store is None:
            raise ValueError("RunStore has not been initialized")
        if isinstance(action, ListFiles):
            return self._observation(
                action.tool,
                "OK",
                "kernel.cu [editable]\nvector_api.h [read-only]\nvector_io.cpp [read-only]",
            )
        if isinstance(action, ReadFile):
            read_key = (action.path, action.start_line, action.lines, self._candidate_hash)
            if read_key in self._read_cache:
                return self._observation(
                    action.tool,
                    "REJECTED",
                    "Identical file region was already read; use the earlier observation. "
                    "Choose an informative new tool (probe, patch or check) instead.",
                )
            self._read_cache.add(read_key)
            lines = self._file_bytes(action.path).decode("utf-8").splitlines()
            start = action.start_line - 1
            selected = lines[start : start + action.lines]
            return self._observation(
                action.tool,
                "OK",
                "\n".join(f"{start + i + 1:4d} {line}" for i, line in enumerate(selected))
                or "No lines at this position.",
            )
        if isinstance(action, SearchCode):
            needle = action.needle
            search_key = (needle, self._candidate_hash)
            if search_key in self._search_cache:
                return self._observation(
                    action.tool,
                    "REJECTED",
                    "Identical source search already performed; use prior result.",
                )
            self._search_cache.add(search_key)
            lines = self._file_bytes("kernel.cu").decode("utf-8").splitlines()
            hits = [f"{n}: {line}" for n, line in enumerate(lines, 1) if needle in line]
            return self._observation(action.tool, "OK", "\n".join(hits[:20]) or "No matches")
        if isinstance(action, FunctionalProbe):
            evidence = self._probe.invoke(FunctionalProbeAction(size=action.size))
            return self._observation(action.tool, "OK", evidence.model_dump_json())
        if isinstance(action, ApplyPatch):
            if self._snapshot is None:
                raise ValueError("source snapshot missing")
            patch = apply_generated_candidate(self._snapshot, action.unified_diff)
            new_source = materialize_candidate(self._snapshot, patch)["kernel.cu"]
            digest = hashlib.sha256(new_source).hexdigest()
            if digest in self._attempted_hashes:
                raise ValueError("candidate was previously proposed; revise it")
            self._attempted_hashes.add(digest)
            self._candidate = new_source
            self._candidate_hash = digest
            self._patch_diff = action.unified_diff
            self._public_passed = False
            return self._observation(
                action.tool,
                "OK",
                f"Patch accepted to isolated workspace; candidate sha256={digest}. "
                "Run run_public_checks before finish.",
            )
        if isinstance(action, RunPublicChecks):
            if self._candidate is None:
                raise ValueError("no candidate: apply a patch first")
            if self._checks >= self.max_checks:
                raise ValueError("public CUDA check budget exhausted")
            self._checks += 1
            result = check_candidate_publicly(
                store=self._store,
                parent_run_id=self.run_id,
                repository=self.root,
                case_id=self.case_id,
                candidate=self._candidate,
                output=self.output / f"candidate-{self._checks:02d}",
            )
            if result.candidate_sha256 != self._candidate_hash:
                raise ValueError("candidate provenance mismatch")
            self._public_passed = result.passed
            return self._observation(
                action.tool,
                "OK" if result.passed else "FAILED",
                result.model_dump_json(),
            )
        if isinstance(action, FinishTask):
            if not self._public_passed or self._candidate is None:
                raise ValueError(
                    "cannot finish: current candidate has not passed public GPU checks"
                )
            self._finished = True
            return self._observation(
                action.tool,
                "OK",
                "Public repair validated; private verifier will run OUTSIDE the agent loop.",
            )
        raise ValueError("unknown tool")
