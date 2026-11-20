"""Controller-owned, budgeted Functional Probe for versioned HardSet V2.

Only pinned CUDA fixtures and fixed input sizes are executable. The public
observation reveals result/shape hashes and pass/fail categories, never the
CPU expected output or evaluator-only test inputs. This is a new V2 tool,
NOT an addition to the frozen A-E v1 action protocol.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.benchmark.hardset_v2 import (
    SIZES,
    HardsetCase,
    load_registry,
    manifest_for,
    sample_input,
    verify_numeric,
)
from gpu_agent.contracts import ArtifactRef
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import (
    BuildRequest,
    ExecutionRequest,
    WorkspaceHandle,
    WorkspaceRequest,
)
from gpu_agent.store import RunStore, reject_symlinks

ProbeVerdict = Literal["PASS", "NUMERIC_MISMATCH", "PROGRAM_FAILED", "OUTPUT_INVALID"]


class FunctionalProbeAction(BaseModel):
    """Minimal model-facing tool request: no path, source, script, or code."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    action_type: Literal["run_functional_probe"] = "run_functional_probe"
    size: int = Field(ge=1, le=513)


class FunctionalProbeObservation(BaseModel):
    """Public projection safe for D and E alike."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["hardset-functional-probe-v2"] = "hardset-functional-probe-v2"
    case_id: str
    size: int
    outcome: ProbeVerdict
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_ref_id: str = Field(pattern=r"^[a-f0-9]{32}$")

    @property
    def passed(self) -> bool:
        return self.outcome == "PASS"


class ProbeBudgetError(ValueError):
    pass


class HardsetFunctionalProbe:
    """A fixed-case public investigation session shared by D/E researchers.

    The operator chooses the case/role before the LLM is run. The LLM can
    propose ONLY a fixed-size typed action; the controller owns the Docker
    session, policy, allowed sizes, budget, source hash and CPU Oracle.
    """

    def __init__(
        self,
        repository: Path,
        output: Path,
        case_id: str,
        *,
        max_probes: int = 4,
    ) -> None:
        if type(max_probes) is not int or not 1 <= max_probes <= 15:
            raise ValueError("max_probes must be a bounded integer")
        reject_symlinks(repository)
        reject_symlinks(output)
        if output.exists():
            raise ValueError("public probe output exists; overwrite refused")
        root = repository.resolve() / "benchmarks"
        registry, registry_sha = load_registry(root)
        case = next((entry for entry in registry.cases if entry.case_id == case_id), None)
        if case is None:
            raise ValueError("unregistered hardset case")
        # Before directory creation ensure source is the registered mutant, not
        # a human/model-selected replacement or clean oracle.
        manifest = manifest_for(root, case, "mutant")
        self._root = root
        self._manifest = manifest
        self._case: HardsetCase = case
        self._registry_sha = registry_sha
        self._output = output
        self._max_probes = max_probes
        self._seen: set[int] = set()
        self._closed = False
        self._store: RunStore | None = None
        self._backend: IsolatedGPUBackend | None = None
        self._handle: WorkspaceHandle | None = None
        self.run_id: str | None = None

    @property
    def calls(self) -> int:
        return len(self._seen)

    @property
    def allowed_sizes(self) -> tuple[int, ...]:
        return SIZES

    def __enter__(self) -> HardsetFunctionalProbe:
        if self._closed or self.run_id is not None:
            raise ValueError("probe session cannot be restarted")
        self._output.mkdir(parents=True, exist_ok=False, mode=0o700)
        self._store = RunStore(self._output / "public")
        self._backend = IsolatedGPUBackend(self._store, self._root, self._output / "workspaces")
        availability = self._backend.availability()
        if not availability.ready:
            self._closed = True
            raise ValueError("native CUDA backend is unavailable")
        run = self._store.create_run("hardset_v2_functional_probe")
        self.run_id = run.id
        self._store.transition(run.id, "RUNNING", "PREPARING")
        self._store.put(
            run.id,
            "hardset/probe-contract.json",
            json.dumps(
                {
                    "schema_version": "functional-probe-v2",
                    "case_id": self._case.case_id,
                    "registry_sha256": self._registry_sha,
                    "source_sha256": self._case.mutant_source_sha256,
                    "max_probes": self._max_probes,
                    "allowed_sizes": list(SIZES),
                },
                sort_keys=True,
            ).encode(),
            "public",
        )
        try:
            self._handle = self._backend.prepare(
                WorkspaceRequest(
                    run_id=run.id,
                    source_manifest=self._manifest,
                    trust_level="UNTRUSTED",
                )
            )
            built = self._backend.build(BuildRequest(workspace_id=self._handle.id))
            if not built.success:
                raise ValueError("registered hardset mutant failed to build")
            self._store.transition(run.id, "RUNNING", "DIAGNOSING")
        except BaseException:
            self.__exit__(None, None, None, force_failure=True)
            raise
        return self

    def invoke(self, action: FunctionalProbeAction) -> FunctionalProbeObservation:
        if self._closed or self._store is None or self._backend is None or self._handle is None:
            raise ValueError("probe session is not active")
        if action.action_type != "run_functional_probe":
            raise ValueError("unsupported probe action")
        n = action.size
        if n not in SIZES:
            raise ValueError("probe size is not an authorized HardSet size")
        if n in self._seen:
            raise ProbeBudgetError("DUPLICATE_PROBE")
        if len(self._seen) >= self._max_probes:
            raise ProbeBudgetError("PROBE_BUDGET_EXHAUSTED")
        self._seen.add(n)  # Failed GPU requests are never silently retried.
        assert self.run_id is not None
        raw = sample_input(self._case.case_id, n)
        input_ref = self._store.put(
            self.run_id, f"hardset/probe-{self.calls:02d}-input.json", raw, "public"
        )
        result = self._backend.run(
            ExecutionRequest(
                workspace_id=self._handle.id,
                stdin_ref=input_ref,
                timeout_seconds=120,
            )
        )
        tool = result.tool_result
        if tool.tool_error or tool.timed_out or tool.cancelled or tool.truncated:
            # An infrastructure failure is never converted into a numeric failure.
            raise ValueError("FUNCTIONAL_PROBE_INCOMPLETE")
        if result.runtime_status == "SUCCESS":
            payload = self._store.read(result.output_ref)
            passed = verify_numeric(self._case.case_id, raw, payload)
            if passed:
                verdict: ProbeVerdict = "PASS"
            else:
                try:
                    from gpu_agent.verification.oracle import parse_output

                    parse_output(payload)
                except ValueError:
                    verdict = "OUTPUT_INVALID"
                else:
                    verdict = "NUMERIC_MISMATCH"
        else:
            verdict = "PROGRAM_FAILED"
        body = {
            "schema_version": "hardset-functional-probe-v2",
            "case_id": self._case.case_id,
            "size": n,
            "outcome": verdict,
            "input_sha256": hashlib.sha256(raw).hexdigest(),
        }
        ref: ArtifactRef = self._store.put(
            self.run_id,
            f"hardset/probe-{self.calls:02d}-observation.json",
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
            "public",
        )
        return FunctionalProbeObservation.model_validate({**body, "evidence_ref_id": ref.id})

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
        *,
        force_failure: bool = False,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        failed = bool(exc_type is not None or force_failure)
        if self._backend is not None and self._handle is not None:
            try:
                self._backend.cleanup(self._handle)
            except Exception:
                failed = True
        if self._store is not None and self.run_id is not None:
            self._store.put(
                self.run_id,
                "hardset/probe-summary.json",
                json.dumps(
                    {"probes_executed": self.calls, "status": "FAILED" if failed else "COMPLETED"}
                ).encode(),
                "public",
            )
            run = self._store.load(self.run_id)
            if run.status.value == "RUNNING":
                self._store.transition(self.run_id, "COMPLETED" if not failed else "FAILED", None)
