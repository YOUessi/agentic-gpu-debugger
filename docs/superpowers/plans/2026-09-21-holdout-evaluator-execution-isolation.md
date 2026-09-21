# Holdout Evaluator Execution Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute every private holdout unit and retain its source/model/native evidence only in evaluator storage while keeping signed scheduling and blind result persistence in the public store.

**Architecture:** `EvaluationRunner` remains the public coordinator. `EvaluationExecutor` receives a public development service and, only for holdout, an evaluator service plus a `HoldoutController` that owns deterministic evaluator execution transactions. The evaluator transaction stores full native lineage and produces one commitment-only `PublicEvaluationRecord`; interrupted public persistence can recover that exact projection without another GPU or provider call.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, Typer, immutable `RunStore`, signed Ed25519 evaluation schedules, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-21-holdout-evaluator-execution-isolation-design.md`

## Global Constraints

- Never write holdout source bytes, private case/template/operator identity, evaluator paths, evaluator run IDs, labels, Oracle truth, provider payloads, or private artifact references to the public store, Git, stdout, or stderr.
- Development scheduling, diagnosis, native lineage, and public record behavior remain public and backward-compatible.
- Mode E remains the only provider-calling mode; total and unit cost caps, pricing attestation, response-model policy, `store=false`, and invocation idempotency remain mandatory.
- A completed evaluator transaction is recoverable without another GPU/provider call; a `STARTED` provider invocation remains ambiguous and is never replayed.
- The public and evaluator services use exact `CorpusFamily` stores; paths, visibility, device/inode identity, and non-overlap fail before mutation.
- Every production change follows red-green-refactor. Run only the focused affected tests per task, then one necessary complete offline regression after all tasks.
- Do not call DeepSeek, run live GPU acceptance, regenerate corpus evidence, push, or publish during implementation.
- Any subagent used for implementation or review must use `gpt-5.6-sol`; `gpt-6-astra` requires fresh explicit user approval.

## Review Focus

- A holdout source/provider canary must be absent from every byte of the public store even when a unit succeeds and generates a candidate.
- Recovery after evaluator completion but before public record persistence must install the same record without another provider/GPU invocation.
- A mismatched alias, ordinal, attempt hash, schedule hash, cutoff, projection hash, or evaluator run state must fail closed.
- Development units must continue to persist and validate native public run IDs, source evidence, candidates, and verification results.
- Wrong/overlapping/symlinked public or evaluator paths must fail before alias, schedule, run, GPU, or provider side effects.

---

### Task 1: Make the diagnosis, agent, provider, candidate, and verification workflow visibility-preserving

**Files:**
- Modify: `src/gpu_agent/evidence/repository.py`
- Modify: `src/gpu_agent/agent/orchestrator.py`
- Modify: `src/gpu_agent/agent/provider.py`
- Modify: `src/gpu_agent/service.py`
- Modify: `src/gpu_agent/verification/engine.py`
- Modify: `src/gpu_agent/verification/derivation.py`
- Modify: `tests/conftest.py`
- Modify: `tests/unit/test_evidence.py`
- Modify: `tests/unit/test_agent_loop.py`
- Modify: `tests/unit/test_service.py`
- Modify: `tests/unit/test_verification_engine.py`

**Interfaces:**
- Consumes: `RunStore.visibility`, `EvidenceRepository.view(run_id)`, existing `RunBinding`, provider invocation and verification schemas.
- Produces: one complete workflow that writes every artifact to `service.store.visibility`; `ApplicationService(..., evaluator_store: RunStore, ...)`; evaluator-safe `diagnose`, candidate registration, and verification without weakening parent/run binding checks.

- [ ] **Step 1: Write the failing evaluator-visibility tests**

Add a fixture that reuses the existing real fake backend/provider but supplies an evaluator
`RunStore`. Name the break each test catches:

```python
def test_evaluator_diagnosis_keeps_all_artifacts_in_evaluator_store(
    evaluator_oob_service, tmp_path
):
    service, _, source, public = evaluator_oob_service
    run = service.diagnose(source, mode="E")
    assert run.status == RunStatus.COMPLETED
    assert service.store.visibility == "evaluator"
    for path in service.store.root.rglob("manifest.json"):
        manifest = service.store.load(path.parent.name)
        assert all(ref.visibility == "evaluator" for ref in manifest.artifact_refs)
    assert not any(b"PRIVATE-SOURCE-CANARY" in p.read_bytes() for p in public.root.rglob("*") if p.is_file())


def test_evaluator_candidate_and_verification_remain_parent_scoped(evaluator_oob_service):
    service, _, source, _ = evaluator_oob_service
    diagnosis = service.diagnose(source, mode="E")
    candidate = service.candidates(diagnosis.id)
    assert len(candidate) == 1
    result = service.verify(diagnosis.id, candidate[0])
    assert isinstance(result.verdict, VerificationVerdict)
    assert all(
        ref.visibility == "evaluator"
        for run in (service.store.load(diagnosis.id), service.store.load(candidate[0]))
        for ref in run.artifact_refs
    )
```

Keep the existing public-service assertions and add one literal assertion that a public diagnosis
still stores `sources/kernel.cu` with public visibility.

- [ ] **Step 2: Run the tests and verify the expected red failure**

Run:

```bash
pytest -q \
  tests/unit/test_evidence.py \
  tests/unit/test_agent_loop.py \
  tests/unit/test_service.py \
  tests/unit/test_verification_engine.py
```

Expected: the new evaluator tests fail at current public-only constructors/hard-coded artifact
visibility; existing public tests remain green.

- [ ] **Step 3: Implement one store-derived visibility path**

Use `store.visibility` as the single source of artifact visibility. Do not add visibility
arguments to individual writes.

```python
def _evidence(store: RunStore) -> EvidenceRepository:
    return EvidenceRepository(store, evaluator=store.visibility == "evaluator")
```

Required changes:

- replace workflow calls to `EvidenceRepository(store).public_view(...)` with controller-only
  `view(...)` where the data is not an export;
- keep `public_view()` public-only for reporting/export security;
- replace hard-coded `"public"` in `ApplicationService`, `AgentOrchestrator`,
  `OpenAIResponsesProvider`, candidate registration, and native verification result writes with
  the owning store's visibility;
- allow candidate/verification workflows on public or evaluator stores, while preserving exact
  parent ID, binding, source hash, single-candidate, and artifact-ref validation;
- update persisted verification derivation to accept exactly the existing public-diagnosis /
  evaluator-audit topology or the new evaluator-local topology, and reject every mixed topology;
- pass an exact evaluator `RunStore` into `ApplicationService`/`VerificationEngine` instead of
  reconstructing one from a loosely interpreted path;
- when the diagnosis store is already evaluator, keep its verification audit in that evaluator
  store and parent-scope every lookup so unrelated private corpus/label runs cannot be read;
- keep report/export entry points dependent on `public_view()` so evaluator evidence cannot be
  exported through existing public APIs.

- [ ] **Step 4: Run the focused tests and refactor only after green**

Run the command from Step 2. Expected: all selected tests pass with no warnings. Then run:

```bash
ruff check src/gpu_agent/evidence/repository.py src/gpu_agent/agent/orchestrator.py \
  src/gpu_agent/agent/provider.py src/gpu_agent/service.py \
  src/gpu_agent/verification/engine.py src/gpu_agent/verification/derivation.py \
  tests/conftest.py \
  tests/unit/test_evidence.py tests/unit/test_agent_loop.py \
  tests/unit/test_service.py tests/unit/test_verification_engine.py
mypy src/gpu_agent/evidence/repository.py src/gpu_agent/agent/orchestrator.py \
  src/gpu_agent/agent/provider.py src/gpu_agent/service.py \
  src/gpu_agent/verification/engine.py src/gpu_agent/verification/derivation.py
```

- [ ] **Step 5: Commit**

```bash
git add src/gpu_agent/evidence/repository.py src/gpu_agent/agent/orchestrator.py \
  src/gpu_agent/agent/provider.py src/gpu_agent/service.py \
  src/gpu_agent/verification/engine.py src/gpu_agent/verification/derivation.py \
  tests/conftest.py \
  tests/unit/test_evidence.py tests/unit/test_agent_loop.py \
  tests/unit/test_service.py tests/unit/test_verification_engine.py
git commit -m "refactor: preserve diagnosis store visibility"
```

### Task 2: Add commitment-only public lineage and evaluator holdout execution transactions

**Files:**
- Modify: `src/gpu_agent/benchmark/evaluation.py`
- Modify: `src/gpu_agent/benchmark/holdout.py`
- Modify: `tests/unit/test_evaluation_modes.py`
- Modify: `tests/unit/test_holdout_scoring.py`

**Interfaces:**
- Consumes: `EvaluationAttempt`, `EvaluationScheduleItem`, `HoldoutBatch`, private alias mapping, `ExternalRunOrigin`, terminal native `EvaluationRecord`.
- Produces: `NativeEvaluationLineage`, `HoldoutEvaluationLineage`, `HoldoutExecutionBinding`, `PreparedHoldoutExecution`, `HoldoutController.reserve_execution()`, `complete_execution()`, `recover_execution()`.

- [ ] **Step 1: Write failing model and transaction tests**

Add literal, hand-derived tests for these contracts:

```python
def test_holdout_public_lineage_contains_commitments_not_evaluator_run_ids(private_executor):
    prepared = private_executor.holdout_controller.reserve_execution(
        private_executor.holdout_batch, evaluation_run_id="a" * 32,
        item=scheduled_item, attempt=attempt,
    )
    public = private_executor.holdout_controller.complete_execution(
        prepared, native_record
    )
    wire = public.model_dump_json().encode()
    assert public.record_id == expected_blind_record_id
    assert public.lineage.kind == "holdout_commitment"
    assert prepared.execution_run_id.encode() not in wire
    assert native_record.lineage.diagnosis_run_id.encode() not in wire
    assert b"case_0100" not in wire and b"vector-add" not in wire


@pytest.mark.parametrize(
    "field,replacement",
    [("ordinal", 2), ("schedule_hash", "f" * 64), ("attempt_hash", "e" * 64),
     ("corpus_cutoff", 99), ("public_record_hash", "d" * 64)],
)
def test_holdout_execution_binding_rejects_cross_store_tampering(
    completed_execution, field, replacement
):
    tamper_evaluator_binding(completed_execution, field, replacement)
    with pytest.raises(ValueError):
        completed_execution.controller.recover_execution(
            completed_execution.batch, completed_execution.item,
            completed_execution.attempt,
        )
```

Also prove reserving the same exact attempt returns the same evaluator run, a different attempt
cannot reuse it, and `complete_execution()` is exact-byte idempotent.

- [ ] **Step 2: Run the focused tests and verify red**

```bash
pytest -q \
  tests/unit/test_evaluation_modes.py \
  tests/unit/test_holdout_scoring.py \
  -k 'holdout and (lineage or execution or tamper or identity)'
```

Expected: failures because holdout records currently expose native evaluator IDs and no durable
execution transaction exists.

- [ ] **Step 3: Implement discriminated lineage models**

In `evaluation.py`, replace the single lineage shape with a discriminated union:

```python
class NativeEvaluationLineage(ExecutionModel):
    kind: Literal["native"] = "native"
    corpus_cutoff: int = Field(ge=1)
    diagnosis_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    diagnosis_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    provider_invocation_hashes: list[str]
    candidate_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    verification_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    public_verification_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class HoldoutEvaluationLineage(ExecutionModel):
    kind: Literal["holdout_commitment"] = "holdout_commitment"
    corpus_cutoff: int = Field(ge=1)
    execution_commitment: str = Field(pattern=r"^[a-f0-9]{64}$")
    diagnosis_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    provider_invocation_hashes: list[str]
    candidate_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    verification_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


EvaluationLineage = Annotated[
    NativeEvaluationLineage | HoldoutEvaluationLineage,
    Field(discriminator="kind"),
]
```

Development constructors use `NativeEvaluationLineage`; holdout public records use only
`HoldoutEvaluationLineage`. A private evaluator `EvaluationRecord` retains native lineage.

- [ ] **Step 4: Implement the evaluator execution transaction**

Add internal frozen models in `holdout.py` with exact bindings:

```python
class HoldoutExecutionBinding(ExecutionModel):
    schema_version: Literal[1] = 1
    public_evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    alias_mapping_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    ordinal: int = Field(ge=0)
    schedule_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    attempt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    corpus_cutoff: int = Field(ge=1)
    alias: str = Field(pattern=r"^[a-f0-9]{64}$")
    private_case_id: str
    private_template_id: str
    diagnosis_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    native_record_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    public_record_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
```

`reserve_execution()` derives one deterministic evaluator `holdout_execution` run ID from the
alias-mapping run, public evaluation run, schedule hash, ordinal, and attempt hash; creates or
validates the exact **top-level** evaluator transaction with a public `ExternalRunOrigin`; resolves
private identities only into its evaluator binding; and creates an evaluator `diagnosis` child
reservation with the same `RunBinding`. The execution transaction must not be a child of the
alias-mapping run because that run's direct-child inventory is reserved for `holdout_score`.

`complete_execution()` must:

1. validate the native record entirely against evaluator artifacts;
2. create a deterministic blind public `record_id` from public evaluation run, schedule hash,
   and ordinal;
3. replace private IDs with the scheduled alias and native run IDs with content commitments;
4. persist `holdout/native-record.json`, `holdout/public-record.json`, and the final exact binding
   in the evaluator transaction;
5. terminalize the transaction and return the public record in memory.

`recover_execution()` returns the exact persisted public record only when every evaluator binding,
artifact hash, terminal state, attempt, item, proof, and alias mapping revalidates; otherwise it
returns `None` only for a genuinely absent transaction and raises for partial/tampered state.

- [ ] **Step 5: Run focused tests, static checks, and commit**

```bash
pytest -q tests/unit/test_evaluation_modes.py tests/unit/test_holdout_scoring.py \
  -k 'holdout and (lineage or execution or tamper or identity)'
ruff check src/gpu_agent/benchmark/evaluation.py src/gpu_agent/benchmark/holdout.py \
  tests/unit/test_evaluation_modes.py tests/unit/test_holdout_scoring.py
mypy src/gpu_agent/benchmark/evaluation.py src/gpu_agent/benchmark/holdout.py
git add src/gpu_agent/benchmark/evaluation.py src/gpu_agent/benchmark/holdout.py \
  tests/unit/test_evaluation_modes.py tests/unit/test_holdout_scoring.py
git commit -m "feat: bind holdout execution in evaluator storage"
```

### Task 3: Route holdout execution to the evaluator worker and recover completed projections

**Files:**
- Modify: `src/gpu_agent/service.py`
- Modify: `src/gpu_agent/benchmark/executor.py`
- Modify: `src/gpu_agent/benchmark/evaluation.py`
- Modify: `tests/conftest.py`
- Modify: `tests/unit/test_evaluation_modes.py`
- Modify: `tests/unit/test_evaluation_persistence.py`
- Modify: `tests/integration/test_private_visibility.py`

**Interfaces:**
- Consumes: Task 1 evaluator-capable `ApplicationService`; Task 2 holdout execution transaction and lineage models.
- Produces: `EvaluationExecutor(public_service, ..., holdout_service=...)`, split-selected worker, `recover_scheduled()`, public-runner recovery of completed evaluator work.

- [ ] **Step 1: Write the full leak reproduction and recovery tests**

Use real RunStores, real controller schemas, and the existing fake backend/provider boundary.
Place unique literals in the holdout source, private IDs, provider response, evaluator path, labels,
and alias nonce. After three D-mode holdout repeats, byte-scan every public file:

```python
def test_holdout_execution_has_zero_private_bytes_in_public_store(private_split_executor):
    result = runner(private_split_executor).run("D", "holdout", 3)
    assert result.executed_units == 3
    public_bytes = b"\n".join(
        path.read_bytes()
        for path in sorted(private_split_executor.service.store.root.rglob("*"))
        if path.is_file()
    )
    for canary in PRIVATE_CANARIES:
        assert canary not in public_bytes
    evaluator_bytes = b"\n".join(
        path.read_bytes()
        for path in sorted(private_split_executor.holdout_service.store.root.rglob("*"))
        if path.is_file()
    )
    assert PRIVATE_SOURCE_CANARY in evaluator_bytes
    assert PROVIDER_PAYLOAD_CANARY in evaluator_bytes
```

Recovery test:

```python
def test_resume_persists_completed_evaluator_projection_without_reexecution(
    private_split_executor, monkeypatch
):
    calls = count_backend_and_provider_calls(private_split_executor)
    interrupt_once_after_evaluator_completion_before_public_record(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        runner(private_split_executor).run("D", "holdout", 3)
    before = calls.snapshot()
    recovered = runner(private_split_executor).resume(active_run_id, "D", "holdout", 3)
    assert recovered.executed_units == 3
    assert calls.snapshot() == before
```

Add negative cases for a partial evaluator run and `STARTED` provider invocation: both retain the
existing `AMBIGUOUS_STARTED_ATTEMPT` behavior and never call the provider again. Add a development
test proving native public lineage and source persistence are unchanged.

- [ ] **Step 2: Run tests and verify red**

```bash
pytest -q \
  tests/unit/test_evaluation_modes.py \
  tests/unit/test_evaluation_persistence.py \
  tests/integration/test_private_visibility.py \
  -k 'holdout or evaluator_projection or development_lineage'
```

Expected: current executor writes private source/provider bytes publicly and cannot recover a
completed evaluator transaction.

- [ ] **Step 3: Implement split-service executor construction**

Change the constructor to require paired holdout capabilities:

```python
def __init__(
    self,
    service: ApplicationService,
    corpus: RunStore,
    sources: Mapping[str, Path],
    *,
    holdout_service: ApplicationService | None = None,
    holdout_controller: HoldoutController | None = None,
    holdout_batch: HoldoutBatch | None = None,
    ...,
) -> None:
```

Invariants:

- `service.store` is exact family public store and remains the runner/coordinator store;
- development corpus/sources are public and no holdout arguments exist;
- holdout corpus and `holdout_service.store` are the exact family evaluator store;
- controller, batch, and evaluator service are all present or all absent;
- both services carry the same immutable evaluation `RunBinding`;
- the schedule verifier remains bound to the public coordinator; evaluator diagnosis starts only
  through the controller-created reservation from Task 2.

Refactor native record derivation into a helper accepting the selected service/store so both
splits share one evidence parser without copying logic.

- [ ] **Step 4: Route and validate scheduled units by split**

For development, preserve the existing path and return `EvaluationRecord`.

For holdout:

1. persist the public claim;
2. reserve the evaluator execution transaction;
3. call the evaluator service against the controller-reserved evaluator diagnosis child;
4. derive and validate the full native record from evaluator artifacts;
5. call `complete_execution()` and return only `PublicEvaluationRecord`;
6. validate the public record through `HoldoutController`, never by loading its lineage from the
   public store.

Do not attach evaluator run IDs or `ArtifactRef` values to the returned object.

- [ ] **Step 5: Recover exact evaluator completions before applying ambiguity stop**

Add:

```python
def recover_scheduled(
    self, evaluation_run_id: str, ordinal: int
) -> PublicEvaluationRecord | None:
    ...
```

In `EvaluationRunner._execute()`, inspect an incomplete persisted attempt before the generic
ambiguous-attempt failure. Only a terminal, exact holdout evaluator transaction may yield a
record; persist and re-load that record with the normal validation path, then continue. Missing,
partial, development, or ambiguous-provider work retains the closed failure behavior.

- [ ] **Step 6: Run focused tests, static checks, and commit**

```bash
pytest -q tests/unit/test_evaluation_modes.py tests/unit/test_evaluation_persistence.py \
  tests/integration/test_private_visibility.py \
  -k 'holdout or evaluator_projection or development_lineage'
ruff check src/gpu_agent/service.py src/gpu_agent/benchmark/executor.py \
  src/gpu_agent/benchmark/evaluation.py tests/conftest.py \
  tests/unit/test_evaluation_modes.py tests/unit/test_evaluation_persistence.py \
  tests/integration/test_private_visibility.py
mypy src/gpu_agent/service.py src/gpu_agent/benchmark/executor.py \
  src/gpu_agent/benchmark/evaluation.py
git add src/gpu_agent/service.py src/gpu_agent/benchmark/executor.py \
  src/gpu_agent/benchmark/evaluation.py tests/conftest.py \
  tests/unit/test_evaluation_modes.py tests/unit/test_evaluation_persistence.py \
  tests/integration/test_private_visibility.py
git commit -m "fix: isolate holdout execution evidence"
```

### Task 4: Enforce exact production store configuration before side effects

**Files:**
- Modify: `src/gpu_agent/benchmark/ledger.py`
- Modify: `src/gpu_agent/benchmark/controller_config.py`
- Modify: `src/gpu_agent/cli.py`
- Modify: `src/gpu_agent/service.py`
- Modify: `.env.example`
- Modify: `tests/unit/test_controller_config.py`
- Modify: `tests/unit/test_corpus_registration.py`
- Modify: `tests/unit/test_production_authority.py`
- Modify: `tests/unit/test_service.py`
- Modify: `tests/integration/test_benchmark_cli.py`
- Modify: `docs/v2-operator-runbook.md`
- Modify: `docs/v2-release-status.md`

**Interfaces:**
- Consumes: exact public/evaluator `CorpusFamily` stores and Task 3 split-service constructor.
- Produces: production CLI builds public coordinator and evaluator worker from family stores; `GPU_AGENT_RUN_ROOT` and `GPU_AGENT_EVALUATOR_ROOT/runs` exact-path gate.

- [ ] **Step 1: Write fail-before-mutation CLI tests**

Parameterize wrong public path, wrong evaluator parent, symlink, overlap, wrong visibility, and a
family store with a different device/inode. Capture directory trees before invocation and assert
they are byte-identical afterward. Also assert the provider fake and GPU backend counters remain
zero.

```python
@pytest.mark.parametrize("fault", STORE_CONFIGURATION_FAULTS)
def test_production_evaluate_rejects_store_fault_before_side_effect(
    production_cli_fixture, fault
):
    before = production_cli_fixture.snapshot()
    result = production_cli_fixture.invoke_evaluate(fault=fault)
    assert result.exit_code != 0
    assert "COST_BOUND_UNAVAILABLE" in result.output
    assert production_cli_fixture.snapshot() == before
    assert production_cli_fixture.provider_calls == 0
    assert production_cli_fixture.gpu_calls == 0
```

Add a successful construction test proving development receives only the public service and
holdout receives a public coordinator plus exact evaluator service.

- [ ] **Step 2: Run tests and verify red**

```bash
pytest -q tests/unit/test_service.py tests/integration/test_benchmark_cli.py \
  -k 'store or evaluator_root or configured_evaluation or side_effect'
```

- [ ] **Step 3: Implement exact construction and path checks**

In `_configured_evaluation_runner()`:

- open the family first and obtain exact `public` and `evaluator` stores;
- require `GPU_AGENT_RUN_ROOT` to resolve to public store;
- require `(Path(GPU_AGENT_EVALUATOR_ROOT) / "runs")` to resolve to evaluator store;
- validate real owner-only directories, visibility, non-overlap, and family pinning before
  `OpenAIResponsesProvider.ensure_available()`, alias preparation, schedule reservation, or run
  creation;
- construct the public service from the exact public store;
- only for holdout, construct the evaluator service from the exact evaluator store and pass it to
  `EvaluationExecutor`;
- retain provider/pricing/model-policy validation on both services without making a provider call.

Upgrade the family store pin schema to persist and revalidate resolved path, device, inode, and
visibility. Reject the old production family schema explicitly; do not silently migrate it or
reuse old corpus evidence. Put the shared path/ownership/non-overlap validation in
`controller_config.py` so CLI and service construction cannot disagree.

Update the runbook to provision:

```bash
export GPU_AGENT_RUN_ROOT=/srv/gpu-agent-data/v2-public
export GPU_AGENT_EVALUATOR_ROOT=/srv/gpu-agent-private/v2-evaluator

gpu-agent benchmark provision-family \
  --public-store "$GPU_AGENT_RUN_ROOT" \
  --evaluator-store "$GPU_AGENT_EVALUATOR_ROOT/runs" \
  ...
```

Explicitly state that the previous direct evaluator path is invalid and requires a new family;
never mutate an existing family config.

- [ ] **Step 4: Run focused tests, static checks, and commit**

```bash
pytest -q tests/unit/test_service.py tests/unit/test_controller_config.py \
  tests/unit/test_corpus_registration.py tests/unit/test_production_authority.py \
  tests/integration/test_benchmark_cli.py \
  -k 'store or evaluator_root or configured_evaluation or side_effect'
ruff check src/gpu_agent/benchmark/ledger.py src/gpu_agent/benchmark/controller_config.py \
  src/gpu_agent/cli.py src/gpu_agent/service.py tests/unit/test_service.py \
  tests/unit/test_controller_config.py tests/unit/test_corpus_registration.py \
  tests/unit/test_production_authority.py \
  tests/integration/test_benchmark_cli.py
mypy src/gpu_agent/benchmark/ledger.py src/gpu_agent/benchmark/controller_config.py \
  src/gpu_agent/cli.py src/gpu_agent/service.py
git add src/gpu_agent/benchmark/ledger.py src/gpu_agent/benchmark/controller_config.py \
  src/gpu_agent/cli.py src/gpu_agent/service.py .env.example \
  tests/unit/test_service.py tests/unit/test_controller_config.py \
  tests/unit/test_corpus_registration.py tests/unit/test_production_authority.py \
  tests/integration/test_benchmark_cli.py \
  docs/v2-operator-runbook.md docs/v2-release-status.md
git commit -m "fix: pin production evaluation stores"
```

### Task 5: Reconcile scoring/release derivation and verify the repaired boundary

**Files:**
- Modify: `src/gpu_agent/benchmark/executor.py`
- Modify: `src/gpu_agent/benchmark/holdout.py`
- Modify: `src/gpu_agent/benchmark/holdout_scoring.py`
- Modify: `src/gpu_agent/benchmark/metrics.py`
- Modify: `src/gpu_agent/benchmark/release.py`
- Modify: `tests/unit/test_holdout_scoring.py`
- Modify: `tests/unit/test_release_gate.py`
- Modify: `tests/integration/test_evaluation_views.py`
- Modify: `tests/integration/test_private_visibility.py`

**Interfaces:**
- Consumes: blind public holdout lineage and evaluator `HoldoutExecutionBinding` from Tasks 2–3.
- Produces: scoring and release validation that privately resolve evaluator commitments while public exports remain blind.

- [ ] **Step 1: Write failing scoring/release privacy tests**

Add one complete 120-record synthetic holdout fixture using real stores/models but no GPU/provider.
Assert:

- scoring joins each blind public record to exactly one terminal evaluator execution and label;
- duplicate/missing/reordered projection, wrong commitment, wrong ordinal, wrong private identity,
  or evaluator transaction in non-terminal state fails;
- metrics use evaluator native records/labels but serialize no private field publicly;
- release selection/manifest may retain allowed opaque controller IDs only in controller-owned
  output, never public store/report;
- byte scans cover source, private IDs, alias nonce, labels, provider payload, evaluator absolute
  path, and evaluator run IDs.

In `test_private_visibility.py`, parameterize modes `A`, `B`, `C`, `D`, and `E`. Mode E uses the
existing local fake provider and must persist its request/response canary only in evaluator
storage; the test must not make a network request. Add a development control proving the same
public source/native lineage behavior remains unchanged.

- [ ] **Step 2: Run tests and verify red**

```bash
pytest -q \
  tests/unit/test_holdout_scoring.py \
  tests/unit/test_release_gate.py \
  tests/integration/test_evaluation_views.py \
  tests/integration/test_private_visibility.py
```

Expected: existing native-lineage assumptions reject commitment-only records or attempt to load
evaluator run IDs from the public store.

- [ ] **Step 3: Implement private commitment resolution**

Centralize one `HoldoutController` loader that takes public evaluation run + ordinal, derives the
deterministic evaluator transaction, and validates its exact binding/native/public artifacts.
Use it from:

- `EvaluationExecutor.validate_scheduled_record()` for holdout;
- `HoldoutController.validated_evaluation()` and scoring preparation;
- persisted record metrics loading;
- release gate native-lineage validation.

Keep `holdout_execution` transactions top-level with public external origins. Preserve the
alias-mapping run's direct children exclusively for deterministic `holdout_score` runs so existing
score inventory and recovery cannot confuse execution with adjudication.

Do not duplicate cross-store validation logic and do not scan for the newest matching evaluator
run. Every lookup is derived from the selected evaluation, schedule, ordinal, attempt, mapping
run, and corpus cutoff.

- [ ] **Step 4: Run affected regression and static checks**

```bash
pytest -q \
  tests/unit/test_holdout_scoring.py \
  tests/unit/test_release_gate.py \
  tests/integration/test_evaluation_views.py \
  tests/integration/test_private_visibility.py
ruff check src/gpu_agent/benchmark/executor.py src/gpu_agent/benchmark/holdout.py \
  src/gpu_agent/benchmark/holdout_scoring.py src/gpu_agent/benchmark/metrics.py \
  src/gpu_agent/benchmark/release.py tests/unit/test_holdout_scoring.py \
  tests/unit/test_release_gate.py tests/integration/test_evaluation_views.py \
  tests/integration/test_private_visibility.py
mypy src/gpu_agent/benchmark/executor.py src/gpu_agent/benchmark/holdout.py \
  src/gpu_agent/benchmark/holdout_scoring.py src/gpu_agent/benchmark/metrics.py \
  src/gpu_agent/benchmark/release.py
```

- [ ] **Step 5: Run one complete offline regression and packaging check**

This single broad run is necessary because the store/lineage types cross service, agent,
evaluation, scoring, and release boundaries. Do not repeat it after already-passing intermediate
tasks unless a later patch touches those boundaries.

```bash
pytest -q -m 'not gpu and not container and not live_llm and not release'
ruff check .
ruff format --check .
mypy src
python -m build
python -m twine check dist/*
git diff --check
```

Record exact pass/fail/skip counts; do not convert partial verification into a completion claim.

- [ ] **Step 6: Commit**

```bash
git add src/gpu_agent/benchmark/executor.py src/gpu_agent/benchmark/holdout.py \
  src/gpu_agent/benchmark/holdout_scoring.py src/gpu_agent/benchmark/metrics.py \
  src/gpu_agent/benchmark/release.py tests/unit/test_holdout_scoring.py \
  tests/unit/test_release_gate.py tests/integration/test_evaluation_views.py \
  tests/integration/test_private_visibility.py
git commit -m "fix: validate blind holdout release lineage"
```

## Post-implementation live sequence (not part of the code tasks)

After independent review accepts the implementation and the checkout is clean:

1. create a fresh release commit and release checkout;
2. provision a new corpus family whose evaluator store is exactly
   `$GPU_AGENT_EVALUATOR_ROOT/runs`;
3. recreate pricing/model attestation bound to the new commit;
4. GPU validate/register the 16 public and 8 private cases on that commit;
5. execute 240 development and 120 holdout units once, serially, under the already approved unit
   and split caps;
6. generate the external 120-record label package, score, collect zero-skip release evidence,
   freeze selection, derive the manifest, and run the final release check;
7. push only after the user asks or confirms the reviewed branch is ready.
