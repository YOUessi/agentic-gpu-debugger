# V2 Operator Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed, resumable 120-record holdout scoring workflow and derive a canonical release selection from four explicit native evidence roots.

**Architecture:** A shared controller-artifact module protects external inputs and no-replace outputs. A holdout scoring controller performs full read-only preflight, claims one deterministic evaluator-only session, persists exact per-record decisions through the existing `HoldoutController`, and writes metrics only after 120/120 records. A shared release evidence resolver then validates that scoring session and derives the same canonical selection used by the existing release gate.

**Tech Stack:** Python 3.11/3.12, Pydantic v2 models, Typer, pytest, Linux `renameat2(RENAME_NOREPLACE)`, existing `RunStore`, `CorpusFamily`, `HoldoutController`, and `ReleaseGate`.

**Spec:** `docs/superpowers/specs/2026-09-20-v2-operator-workflow-design.md`

## Global Constraints

- Both commands derive public and evaluator stores only from `GPU_AGENT_CORPUS_FAMILY_ROOT`; no caller-supplied store roots.
- Both commands perform zero GPU and zero provider/model calls.
- External inputs and outputs must be absolute and outside the Git checkout and both RunStores.
- Private labels, scores, aliases, identities, nonce, and evaluator artifact paths never enter the public store, repository, or CLI output.
- One holdout evaluation/mapping pair accepts exactly one canonical label package through one deterministic scoring session.
- All existing score children are compared with the package before the first new session or score write.
- Holdout coverage is exactly eight aliases × five modes × three repeats = 120 records.
- Release selection accepts exactly four explicit roots and never scans for a latest run.
- Freezing requires a completed exact scoring session and a fully passing existing release gate; there is no incomplete override.
- Production schedule signing remains external; repository code never creates or stores a production private key.
- No-replace publication requires Linux `renameat2(RENAME_NOREPLACE)` and fails closed if unavailable.
- All feature and bug-fix implementation follows red-green-refactor TDD.

## Review Focus

- Existing 1–119 score children from another package must be rejected before any new write; Task 2 pins this with a zero-mutation conflict test.
- A directory containing 120 manually created score children but no completed scoring session must not freeze; Task 4 pins this in resolver tests.
- A crash after session input binding but before terminal metrics must resume only with the identical package; Task 3 covers every persistence boundary.
- Symlinked paths and repository/RunStore descendants reached after path resolution must be rejected; Task 1 covers lexical and resolved containment.
- Duplicate Mode-E diagnosis lineage must fail rather than be hidden by set conversion; Task 4 asserts record count equals unique diagnosis-run count.

---

### Task 1: Secure controller artifact I/O

**Files:**
- Create: `src/gpu_agent/benchmark/controller_artifacts.py`
- Modify: `src/gpu_agent/benchmark/release.py`
- Modify: `src/gpu_agent/cli.py`
- Test: `tests/unit/test_controller_artifacts.py`
- Test: `tests/unit/test_release_gate.py`
- Test: `tests/integration/test_benchmark_cli.py`

**Interfaces:**
- Consumes: `gpu_agent.store.read_regular`, `gpu_agent.store.reject_symlinks`, `gpu_agent.store.sync_directory`.
- Produces: `validate_external_artifact_path(path: Path, *, repository: Path, forbidden_roots: Sequence[Path] = ()) -> Path`.
- Produces: `read_private_external(path: Path, *, repository: Path, forbidden_roots: Sequence[Path], limit: int) -> bytes`.
- Produces: `write_private_atomic_new(path: Path, content: bytes, *, repository: Path, forbidden_roots: Sequence[Path] = ()) -> str`, returning the SHA-256 hex digest.
- Preserves: `external_release_artifact_path()` and `validate_external_release_artifact_path()` as compatibility wrappers in `release.py`, adding `forbidden_roots` so existing release commands exclude both configured RunStores.

- [ ] **Step 1: Write failing path and reader tests**

Create tests with literal expected outcomes:

```python
def test_external_artifact_rejects_resolved_forbidden_descendant(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    link = tmp_path / "controller-link"
    link.symlink_to(repository, target_is_directory=True)
    with pytest.raises(ValueError, match="external artifact path is unsafe"):
        validate_external_artifact_path(
            link / "labels.json",
            repository=repository,
            forbidden_roots=(),
        )


@pytest.mark.parametrize("mode", [0o644, 0o660, 0o606])
def test_private_reader_rejects_group_or_other_permissions(tmp_path, mode):
    path = tmp_path / "labels.json"
    path.write_bytes(b"{}")
    path.chmod(mode)
    with pytest.raises(ValueError, match="private external artifact is unsafe"):
        read_private_external(
            path,
            repository=tmp_path / "repo",
            forbidden_roots=(),
            limit=1024,
        )
```

Also cover relative paths, exact forbidden roots, public/evaluator descendants, a symlink into the repository, a symlink to an otherwise safe external file, symlinked output parents, non-regular files, wrong owner through a patched `fstat`, `st_nlink != 1`, path/fd device-inode mismatch, empty/oversize reads, and a valid `0600` file. Add CLI regressions proving `derive-manifest` and `check` reject selection/manifest paths inside either configured RunStore before reading bytes.

- [ ] **Step 2: Run the reader tests and verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_controller_artifacts.py -k 'external_artifact or private_reader'
```

Expected: collection/import failure because `controller_artifacts` and its functions do not exist.

- [ ] **Step 3: Implement path validation and fd-bound private reads**

Reject symlinks on the caller's lexical path before resolving it, then perform containment after `resolve(strict=False)` and open the parent and target with no-follow semantics. Compare `lstat` and `fstat` device/inode/type/owner/mode/link count before reading a bounded byte count. Use one stable error message for rejected metadata and never include file content in exceptions.

```python
def validate_external_artifact_path(
    path: Path,
    *,
    repository: Path,
    forbidden_roots: Sequence[Path] = (),
) -> Path:
    if not path.is_absolute():
        raise ValueError("external artifact path is unsafe")
    reject_symlinks(path)
    resolved = path.resolve(strict=False)
    roots = (repository, *forbidden_roots)
    if any(_within(resolved, root.resolve(strict=False)) for root in roots):
        raise ValueError("external artifact path is unsafe")
    return resolved
```

- [ ] **Step 4: Run reader tests and verify GREEN**

Run the Step 2 command.

Expected: all selected tests pass.

- [ ] **Step 5: Write failing atomic publication tests**

Cover new publication, byte-identical retry, different-content no-overwrite, `0600` target mode, missing/unsafe parent, simulated `ENOSYS`, simulated `EEXIST`, crash before rename leaving only a dot-prefixed temporary file, and a target that becomes unsafe before exact retry.

```python
def test_atomic_writer_never_overwrites_different_content(tmp_path):
    output = _secure_parent(tmp_path) / "selection.json"
    output.write_bytes(b"first")
    output.chmod(0o600)
    with pytest.raises(ValueError, match="external artifact output conflicts"):
        write_private_atomic_new(
            output,
            b"second",
            repository=tmp_path / "repo",
        )
    assert output.read_bytes() == b"first"
```

- [ ] **Step 6: Run writer tests and verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_controller_artifacts.py -k atomic_writer
```

Expected: FAIL because `write_private_atomic_new()` is missing.

- [ ] **Step 7: Implement Linux no-replace publication**

Load `renameat2` from libc with `ctypes.CDLL(None, use_errno=True)`, set its signature, and call it with the same securely opened parent fd for source and destination:

```python
_RENAME_NOREPLACE = 1

def _rename_noreplace(parent_fd: int, temporary: str, target: str) -> None:
    result = _LIBC.renameat2(
        parent_fd,
        os.fsencode(temporary),
        parent_fd,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(target)
        raise OSError(error, os.strerror(error))
```

Create a random dot-prefixed temp with `dir_fd`, mode `0600`, write in a loop, `fsync`, call the no-replace primitive, and `fsync` the parent. On `EEXIST`, validate and compare the target through the private reader. On unavailable `renameat2`, raise a stable unsafe-output error. Delete only the temp name created by this invocation. Update existing release wrappers and CLI helpers so they open `CorpusFamily`, derive public/evaluator roots, and pass both as `forbidden_roots` before reading selection or manifest bytes.

- [ ] **Step 8: Run Task 1 tests and compatibility tests**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_controller_artifacts.py tests/unit/test_release_gate.py \
  tests/integration/test_benchmark_cli.py
```

Expected: PASS. Existing release CLI path behavior remains fail-closed.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/gpu_agent/benchmark/controller_artifacts.py \
  src/gpu_agent/benchmark/release.py src/gpu_agent/cli.py \
  tests/unit/test_controller_artifacts.py tests/unit/test_release_gate.py
git add tests/integration/test_benchmark_cli.py
git commit -m "feat: secure external controller artifacts"
```

### Task 2: Holdout package models and zero-mutation preflight

**Files:**
- Create: `src/gpu_agent/benchmark/holdout_scoring.py`
- Modify: `src/gpu_agent/benchmark/holdout.py`
- Test: `tests/unit/test_holdout_scoring.py`
- Test: `tests/unit/test_evaluation_modes.py`

**Interfaces:**
- Consumes: Task 1 `read_private_external()` and `validate_external_artifact_path()`.
- Produces: immutable `HoldoutJudgment`, `HoldoutLabelPackage`, `PreparedHoldoutScore`, `HoldoutScoringPlanItem`, and `HoldoutScoringPlan` models.
- Produces: `HoldoutController.validated_evaluation(batch: HoldoutBatch, evaluation_run_id: str) -> ValidatedHoldoutEvaluation`.
- Produces: `HoldoutController.prepare_score(batch, alias, public_record_ref, *, labels, score, should_be_inconclusive, private_holdout_passed) -> PreparedHoldoutScore`.
- Produces: `HoldoutScoringController.preflight(evaluation_run_id: str, private_binding_run_id: str, labels_path: Path, repository: Path) -> HoldoutScoringPlan`, which does not mutate either store.

- [ ] **Step 1: Write failing package, evaluation-root, and exact-coverage tests**

Build a native eight-alias, A-E, three-repeat signed test evaluation using existing schedule-authority fixtures. Write each package to a secure external `0600` file. Assert rejection for 119 judgments, 121 judgments, duplicate blind IDs, wrong evaluation/mapping IDs, schedule hash, alias hash, cutoff, record hash, blind-payload hash, record-set hash, rubric hash, and noncanonical extra fields.

Mutate native evidence one invariant at a time and assert rejection before any write: evaluation root not `COMPLETED`; schedule `selection != all`; non-null manifest `stopped_reason`; 119 or 121 attempts; 119 or 121 records; noncanonical/missing/duplicate ordinals; duplicate record IDs; manifest counts or ordered records not exact; modes not exactly A-E; repeats not three; aliases not exactly eight; or the Cartesian product not 8 × 5 × 3.

```python
def test_preflight_requires_exact_blind_record_set(scoring_fixture):
    package_path = scoring_fixture.write_package(
        scoring_fixture.package.model_copy(
            update={"judgments": scoring_fixture.package.judgments[:-1]}
        )
    )
    with pytest.raises(ValueError, match="holdout label package is incomplete"):
        scoring_fixture.controller.preflight(
            scoring_fixture.evaluation_run_id,
            scoring_fixture.mapping_run_id,
            package_path,
            scoring_fixture.repository,
        )
    assert scoring_fixture.snapshot_stores() == scoring_fixture.before_store_snapshot
```

- [ ] **Step 2: Write failing zero-mutation existing-child tests**

Seed an exact subset of 1, 37, or 119 score children and then, in separate parametrized cases, a conflicting score, unexpected child kind, duplicate public-record binding, wrong deterministic ID, wrong parent/binding, nonterminal child, or altered private-score bytes. Snapshot both RunStores byte-for-byte before `preflight()`. Assert exact subsets are accepted into the returned plan, every conflict is rejected, and both snapshots remain identical.

```python
def test_conflicting_existing_score_fails_before_any_mutation(scoring_fixture):
    scoring_fixture.persist_conflicting_score(ordinal=99)
    before = scoring_fixture.snapshot_stores()
    with pytest.raises(ValueError, match="existing holdout score conflicts"):
        scoring_fixture.controller.preflight(
            scoring_fixture.evaluation_run_id,
            scoring_fixture.mapping_run_id,
            scoring_fixture.labels_path,
            scoring_fixture.repository,
        )
    assert scoring_fixture.snapshot_stores() == before
```

- [ ] **Step 3: Run every preflight test and verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_holdout_scoring.py \
  -k 'package or evaluation_root or exact or cartesian or existing_score or zero_mutation'
```

Expected: import/attribute failure because scoring models and controller do not exist.

- [ ] **Step 4: Add validated holdout evaluation and pure typed score preparation**

Refactor the existing `_resolve_public_record()` validation into one evaluation-wide loader. Tighten it to require a `COMPLETED` evaluation and preserve every signed schedule, attempt, terminal manifest, holdout proof, native-record check, exact count, canonical ordinal, and unique record-ID invariant. Make single-record validation delegate to the evaluation-wide result instead of repeating 120 full scans.

```python
class PreparedHoldoutScore(ExecutionModel):
    run_id: str
    binding: EvaluatorRecordBinding
    private_score_content: bytes


class HoldoutScoringPlanItem(ExecutionModel):
    ordinal: int = Field(ge=0, lt=120)
    alias: str = Field(pattern=r"^[a-f0-9]{64}$")
    public_record_ref: ArtifactRef
    judgment: HoldoutJudgment
    prepared_score: PreparedHoldoutScore
    existing_binding: EvaluatorRecordBinding | None = None


def prepare_score(...) -> PreparedHoldoutScore:
    record = self._validated_public_record(public_record_ref, batch)
    identity = self._identity(batch, alias)
    private_score = _PrivateScore(...)
    binding = EvaluatorRecordBinding(...)
    return PreparedHoldoutScore(
        run_id=self._score_run_id(batch, record.record_id),
        binding=binding,
        private_score_content=private_score.content(),
    )
```

Change `bind_score()` to call `prepare_score()` and persist exactly the returned bytes/binding. Existing holdout tests must remain unchanged in meaning.

- [ ] **Step 5: Implement the single declared read-only `preflight()` API**

Parse the evaluator-owned file through Task 1's private reader with a 16 MiB limit. Compute blind payload and record-set hashes from native records ordered by schedule ordinal. Read `evaluation/rubric.md` from the captured clean repository and bind its SHA-256. Construct a `PreparedHoldoutScore` for every judgment without writing.

Under `evaluation_run_lease(mapping_run.id)`, inventory direct children. Reject non-`holdout_score` children. For every existing score child, require the expected deterministic run ID, exact binding JSON, exact private-score bytes, `COMPLETED` state, and one unique public record. Return 120 ordinal-sorted `HoldoutScoringPlanItem` values, attaching exact existing bindings and leaving missing ones as `None`. Recapture the repository and require the same snapshot before returning.

- [ ] **Step 6: Run Task 2 tests and verify GREEN**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_holdout_scoring.py tests/unit/test_evaluation_modes.py \
  tests/unit/test_evaluation_cutoff.py
```

Expected: PASS.

- [ ] **Step 7: Run Task 2 quality checks**

Run:

```bash
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python -m ruff check \
  src/gpu_agent/benchmark/holdout.py src/gpu_agent/benchmark/holdout_scoring.py \
  tests/unit/test_holdout_scoring.py
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python -m mypy --strict \
  src/gpu_agent/benchmark/holdout.py src/gpu_agent/benchmark/holdout_scoring.py
```

Expected: both commands exit 0.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/gpu_agent/benchmark/holdout.py \
  src/gpu_agent/benchmark/holdout_scoring.py \
  tests/unit/test_holdout_scoring.py tests/unit/test_evaluation_modes.py
git commit -m "feat: preflight evaluator holdout judgments"
```

### Task 3: Deterministic scoring session, recovery, metrics, and CLI

**Files:**
- Modify: `src/gpu_agent/benchmark/holdout_scoring.py`
- Modify: `src/gpu_agent/cli.py`
- Modify: `src/gpu_agent/benchmark/__init__.py`
- Test: `tests/unit/test_holdout_scoring.py`
- Test: `tests/integration/test_benchmark_cli.py`

**Interfaces:**
- Consumes: Task 2 `HoldoutScoringPlan` and `PreparedHoldoutScore`.
- Produces: `HoldoutScoringController.score(evaluation_run_id: str, private_binding_run_id: str, labels_path: Path, repository: Path) -> HoldoutScoringResult`.
- Produces CLI command `gpu-agent benchmark score-holdout` with the exact options in the spec.
- Persists evaluator-only `holdout-scoring/input-binding.json`, `bindings.json`, `metrics.json`, and `result.json`.

- [ ] **Step 1: Write failing session identity, lifecycle, crash, and concurrency tests**

Assert the deterministic ID is SHA-256 of the exact domain-separated evaluation/mapping string, a session is top-level with public external origin, and two package hashes cannot share a session.

```python
def test_scoring_session_rejects_a_second_package(scoring_fixture):
    first = scoring_fixture.controller.score(
        scoring_fixture.evaluation_run_id,
        scoring_fixture.mapping_run_id,
        scoring_fixture.labels_path,
        scoring_fixture.repository,
    )
    changed_path = scoring_fixture.write_changed_package()
    with pytest.raises(ValueError, match="holdout scoring package conflicts"):
        scoring_fixture.controller.score(
            scoring_fixture.evaluation_run_id,
            scoring_fixture.mapping_run_id,
            changed_path,
            scoring_fixture.repository,
        )
    assert scoring_fixture.load_result(first.scoring_run_id).package_hash == first.package_hash
```

Inject one failure after each durable boundary: session creation, `QUEUED → RUNNING`, input binding, score ordinals 0/1/119, bindings artifact, metrics artifact, and result artifact. For each, rerun the same package and assert one session, exactly 120 score children, exact artifacts, and no duplicate events.

Run two threads and then two helper subprocesses with the same package and assert identical results. Repeat both races with two different canonical packages for the same evaluation/mapping pair; assert exactly one package hash wins the input binding, the loser reports conflict, and no score child contains decisions from the losing package.

- [ ] **Step 2: Run session tests and verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_holdout_scoring.py \
  -k 'session or lifecycle or crash or recovery or concurrent or package_race'
```

Expected: FAIL because `score()` and scoring-session artifacts do not exist.

- [ ] **Step 3: Implement the session claim, state transitions, and ordered scoring**

Use an in-process lock plus an owner-only, no-follow flock file named from the deterministic session ID. Under the lock, re-run Task 2 `preflight()`, create/recover the top-level evaluator session, validate its kind/binding/public external origin, transition a new session `QUEUED → RUNNING` exactly once, and exact-persist input binding before the first missing score. A recovered `RUNNING` session with an input binding must match it exactly; a pre-binding crash may be claimed by either package because no decision is yet durable. Iterate the 120 plan items in ordinal order; reload exact existing bindings and call `bind_score()` only for missing entries.

```python
for item in plan.items:
    binding = item.existing_binding
    if binding is None:
        binding = controller.bind_score(
            plan.batch,
            item.alias,
            item.public_record_ref,
            labels=item.judgment.labels,
            score=item.judgment.score,
            should_be_inconclusive=item.judgment.should_be_inconclusive,
            private_holdout_passed=item.judgment.private_holdout_passed,
        )
    bindings.append(binding)
```

- [ ] **Step 4: Implement terminal artifacts and strict completed recovery**

Call the existing metrics API only after every binding reloads, passing its full native context:

```python
metrics = aggregate_grouped(
    bindings,
    public_store=self.public,
    evaluator_store=self.evaluator,
    run_binding=self.binding,
    schedule_verifier=self.schedule_verifier,
)
```

Persist bindings, metrics, and result through `put_if_absent_exact()`, transition the `RUNNING` session to its finalizing stage, and only then transition it to `COMPLETED`. Never attempt `QUEUED → COMPLETED`. When the session is already `COMPLETED`, do not call `put` or `transition`; reload and exact-verify input, all three terminal artifacts, every score child, and metrics hash, then return.

- [ ] **Step 5: Run session/recovery tests and verify GREEN**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_holdout_scoring.py \
  -k 'session or lifecycle or crash or recovery or concurrent or package_race or completed'
```

Expected: PASS.

- [ ] **Step 6: Write failing CLI and privacy-canary tests**

Invoke the real Typer command against the synthetic family. Assert missing/relative/repository-local/RunStore-local labels fail before store mutation; safe labels complete; optional metrics output is `0600` and exact; stdout has only ID/count/hash. Insert a unique private canary into labels and private identities, then byte-scan the repository, public store, stdout, and stderr for its absence.

- [ ] **Step 7: Implement CLI with stable errors and safe output**

The command opens the configured family, validates external paths through Task 1, captures the repository, calls the controller, optionally publishes metrics, and maps exceptions to these public codes without raw payloads:

```text
HOLDOUT_LABEL_PACKAGE_INVALID
HOLDOUT_SCORING_EVIDENCE_MISMATCH
HOLDOUT_SCORING_CONFLICT
HOLDOUT_SCORING_FAILED
```

Success output is exactly:

```text
scoring_run_id <32hex>
scored 120/120
metrics_sha256 <64hex>
```

- [ ] **Step 8: Run Task 3 tests and quality checks**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_holdout_scoring.py tests/integration/test_benchmark_cli.py
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python -m ruff check \
  src/gpu_agent/benchmark/holdout_scoring.py src/gpu_agent/cli.py \
  tests/unit/test_holdout_scoring.py tests/integration/test_benchmark_cli.py
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python -m mypy --strict \
  src/gpu_agent/benchmark/holdout_scoring.py src/gpu_agent/cli.py
```

Expected: all commands exit 0.

- [ ] **Step 9: Commit Task 3**

```bash
git add src/gpu_agent/benchmark/holdout_scoring.py src/gpu_agent/cli.py \
  src/gpu_agent/benchmark/__init__.py tests/unit/test_holdout_scoring.py \
  tests/integration/test_benchmark_cli.py
git commit -m "feat: score complete holdout evaluations"
```

### Task 4: Shared release evidence resolver and scoring-session requirement

**Files:**
- Modify: `src/gpu_agent/benchmark/release.py`
- Test: `tests/unit/test_release_gate.py`
- Test: `tests/unit/test_holdout_scoring.py`

**Interfaces:**
- Consumes: Task 3 deterministic scoring-session ID, input binding, ordered bindings, metrics, and result models.
- Produces: `ReleaseEvidenceRoots`, `ReleaseEvidenceResolution`, and `ReleaseEvidenceFreezer.resolve(...)`.
- Preserves: `ReleaseEvidenceIndex.derive(selection, ...)` and existing reason-code behavior.
- Changes: derived evidence `private_scoring` contains both private mapping and deterministic scoring-session IDs.

- [ ] **Step 1: Write failing canonical-resolution tests**

Using complete synthetic native evidence, assert the resolver derives corpus IDs in ledger order and acceptance arrays in sorted order. Assert the candidate selection round-trips through `ReleaseEvidenceIndex.derive()` with no reason codes.

```python
def test_freezer_resolves_the_same_selection_checked_by_release_gate(release_fixture):
    resolution = ReleaseEvidenceFreezer.resolve(
        release_fixture.roots,
        release_fixture.public,
        release_fixture.evaluator,
        release_fixture.family,
        release_fixture.repository,
    )
    checked = ReleaseEvidenceIndex.derive(
        resolution.selection,
        release_fixture.public,
        release_fixture.evaluator,
        release_fixture.family,
        release_fixture.repository,
    )
    assert checked == resolution.evidence
    assert checked.reason_codes == []
```

- [ ] **Step 2: Run resolver tests and verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_release_gate.py -k 'freezer or canonical_resolution'
```

Expected: import/attribute failure because roots, resolution, and freezer do not exist.

- [ ] **Step 3: Refactor the deriver into one optional-selection resolver**

Introduce immutable roots and resolution models:

```python
class ReleaseEvidenceRoots(ExecutionModel):
    development_evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    holdout_evaluation_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    private_binding_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    release_test_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")


class ReleaseEvidenceResolution(ExecutionModel):
    selection: ReleaseEvidenceSelection
    evidence: ReleaseEvidenceIndex
```

Refactor `_ReleaseEvidenceDeriver` into a resolver that derives each component from roots. When an expected selection is supplied by `ReleaseEvidenceIndex.derive()`, retain the existing stage-specific comparisons and reason codes such as `CORPUS_SELECTION_MISMATCH`, `PRIVATE_SCORING_INCOMPLETE`, and `RELEASE_ACCEPTANCE_INVALID`. Do not replace all mismatches with one generic comparison.

- [ ] **Step 4: Write failing scoring-session and Mode-E uniqueness tests**

Assert the resolver rejects:

- 120 exact score children with no session;
- a `RUNNING` or `FAILED` session;
- altered input binding, bindings order, metrics, or result hash;
- a completed session from another evaluation/mapping pair;
- duplicate Mode-E diagnosis run IDs even when the set still has nonzero entries.

Expected reason codes are `PRIVATE_SCORING_INCOMPLETE` for session/score defects and `RELEASE_ACCEPTANCE_INVALID` for repeated Mode-E lineage.

- [ ] **Step 5: Require exact completed session and non-collapsed Mode-E lineage**

Derive the scoring-session ID from the two roots, load it from evaluator storage, and exact-check all session artifacts against the already validated score children and metrics. Before constructing `live_llm`, collect the ordered Mode-E record IDs and diagnosis IDs and require equal list length and unique-ID count:

```python
mode_e_ids = [
    record.lineage.diagnosis_run_id
    for evaluation in (development, holdout)
    for item, record in zip(evaluation.schedule.items, evaluation.records, strict=True)
    if item.mode == "E"
]
if len(mode_e_ids) != len(set(mode_e_ids)):
    raise ValueError("Mode-E diagnosis lineage is not unique")
```

- [ ] **Step 6: Run release gate regressions**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_release_gate.py tests/unit/test_holdout_scoring.py
```

Expected: PASS, including all pre-existing release reason-code tests.

- [ ] **Step 7: Run strict quality checks**

Run:

```bash
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python -m ruff check \
  src/gpu_agent/benchmark/release.py tests/unit/test_release_gate.py
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python -m mypy --strict \
  src/gpu_agent/benchmark/release.py
```

Expected: both commands exit 0.

- [ ] **Step 8: Commit Task 4**

```bash
git add src/gpu_agent/benchmark/release.py tests/unit/test_release_gate.py \
  tests/unit/test_holdout_scoring.py
git commit -m "refactor: derive canonical release evidence roots"
```

### Task 5: Freeze-selection publication and CLI

**Files:**
- Modify: `src/gpu_agent/benchmark/release.py`
- Modify: `src/gpu_agent/cli.py`
- Test: `tests/unit/test_release_gate.py`
- Test: `tests/integration/test_benchmark_cli.py`

**Interfaces:**
- Consumes: Task 1 atomic writer and Task 4 `ReleaseEvidenceFreezer.resolve()`.
- Produces: `ReleaseEvidenceFreezer.freeze(..., output: Path) -> FrozenReleaseSelection`.
- Produces CLI command `gpu-agent release freeze-selection` with the exact options in the spec.

- [ ] **Step 1: Write failing explicit-root, read-only, gate, drift, and privacy tests**

Patch only the second repository snapshot capture to differ and assert no output. Build complete-looking evidence with one gate deficiency (15 public cases, seven private templates, 119 scores, or one skipped release test) and assert no output and the precise reason code. Add wrong-kind, wrong-store, nonterminal, and mismatched-binding variants for each of the four roots.

Create newer valid-looking development, holdout, mapping, and release-test decoy runs after the four selected roots. Assert the derived selection uses only the explicit roots and their cutoff; this proves the freezer never performs a latest-run scan. Snapshot every byte and directory entry under both RunStores and the ledger before success and before every failure variant, and assert the snapshots are identical afterward.

Seed unique canaries in alias, nonce, label, private case/template ID, and evaluator path fields. On success, byte-scan the canonical selection and captured stdout/stderr; assert none of those canaries or absolute evaluator paths appear. Opaque evaluator run IDs are the only private-side identifiers allowed in the selection.

```python
def test_freezer_never_publishes_when_repository_changes(release_fixture, monkeypatch):
    monkeypatch.setattr(
        release_fixture.freezer,
        "capture_repository",
        release_fixture.first_then_changed_snapshot,
    )
    with pytest.raises(ValueError, match="release repository changed"):
        release_fixture.freeze()
    assert not release_fixture.output.exists()
```

- [ ] **Step 2: Write failing CLI publication and round-trip tests**

Invoke the not-yet-implemented command with each wrong root type/store/status, relative/repository/store-local output, symlinked output, unsafe parent, identical existing output, and different existing output. On success assert stdout includes only path, digest, cutoff, and aggregate counts. Feed the written file to `_derive_release_evidence()` and `release check`; assert complete synthetic evidence passes without modifying either store.

- [ ] **Step 3: Run all freezer/publication tests and verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_release_gate.py tests/integration/test_benchmark_cli.py \
  -k 'freezer or freeze_selection or publish or repository_changes or explicit_roots or privacy'
```

Expected: FAIL because `freeze()` and `FrozenReleaseSelection` do not exist.

- [ ] **Step 4: Implement gate-first atomic freeze**

Capture the repository with the expected development commit, resolve canonical evidence, create a manifest through `ReleaseManifest.from_evidence()`, and require `ReleaseGate.check(manifest, evidence).passed`. Recapture and compare the repository, serialize selection as `model_dump_json(indent=2) + "\n"`, and call Task 1's writer.

```python
result = ReleaseGate().check(ReleaseManifest.from_evidence(evidence), evidence)
if not result.passed:
    raise ValueError("release evidence is incomplete: " + ",".join(result.reason_codes))
if capture_repository_snapshot(repository, expected_commit=actual.commit) != actual:
    raise ValueError("release repository changed")
digest = write_private_atomic_new(
    output,
    selection.model_dump_json(indent=2).encode() + b"\n",
    repository=repository,
    forbidden_roots=(public.root, evaluator.root),
)
```

- [ ] **Step 5: Implement CLI and stable failure codes**

Open the configured family, create `ReleaseEvidenceRoots` from four options, validate the external output, and call `freeze()`. Map failures without raw evaluator payloads:

```text
RELEASE_ROOTS_INVALID
RELEASE_EVIDENCE_INCOMPLETE
RELEASE_REPOSITORY_CHANGED
RELEASE_SELECTION_OUTPUT_UNSAFE
RELEASE_SELECTION_OUTPUT_CONFLICT
```

- [ ] **Step 6: Run Task 5 tests and quality checks**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_release_gate.py tests/integration/test_benchmark_cli.py
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python -m ruff check \
  src/gpu_agent/benchmark/release.py src/gpu_agent/cli.py \
  tests/unit/test_release_gate.py tests/integration/test_benchmark_cli.py
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python -m mypy --strict \
  src/gpu_agent/benchmark/release.py src/gpu_agent/cli.py
```

Expected: all commands exit 0. The true release-marked E2E remains closed without real evidence.

- [ ] **Step 7: Commit Task 5**

```bash
git add src/gpu_agent/benchmark/release.py src/gpu_agent/cli.py \
  tests/unit/test_release_gate.py tests/integration/test_benchmark_cli.py
git commit -m "feat: freeze canonical release selections"
```

### Task 6: Operator runbook, distribution, and final zero-cost acceptance

**Files:**
- Create: `docs/v2-operator-runbook.md`
- Modify: `docs/v2-release-status.md`
- Modify: `README.md`
- Modify: `tests/unit/test_distribution_resources.py`
- Modify: `pyproject.toml` only if the current package-data rules omit a required non-Python resource.

**Interfaces:**
- Consumes: final CLI help and stable error/output contracts from Tasks 3 and 5.
- Produces: an operator sequence from external signer provisioning through scoring, selection, manifest derivation, and check without embedding a private key.

- [ ] **Step 1: Write failing distribution/help smoke tests**

Extend the distribution resource list with the new runbook and add subprocess help assertions for both commands:

```python
def test_operator_commands_are_exposed():
    scoring = runner.invoke(app, ["benchmark", "score-holdout", "--help"])
    freezing = runner.invoke(app, ["release", "freeze-selection", "--help"])
    assert scoring.exit_code == 0
    assert "--labels" in scoring.output and "--private-binding-run-id" in scoring.output
    assert freezing.exit_code == 0
    assert "--release-test-run-id" in freezing.output and "--output" in freezing.output
```

- [ ] **Step 2: Run smoke tests and verify RED**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_distribution_resources.py -k 'operator or distribution'
```

Expected: FAIL because the runbook does not exist in the source distribution expectation.

- [ ] **Step 3: Write the operator runbook and stable status text**

Document, in this order:

1. clean final commit and Python environment checks;
2. external Ed25519 public-key provisioning and the existing JSON stdin/stdout signer protocol;
3. production family provisioning;
4. 16 public and eight evaluator-only registrations;
5. pricing attestation and explicit user-approved total/unit caps;
6. signed 240-unit development and 120-unit holdout evaluations;
7. creation and secure placement of the 120-record label package;
8. `score-holdout`, `collect-evidence`, `freeze-selection`, `derive-manifest`, and `check` commands;
9. wheel/sdist hashes, push, CI, tag, and release.

Every path example must be absolute and outside the checkout where required. State explicitly that
the repository ships no production signer/private key and that the release remains closed until
real evidence passes.

- [ ] **Step 4: Build and inspect wheel/sdist**

Run:

```bash
/home/you/conda_env/agentic-gpu-debugger/bin/python -m build
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  tests/unit/test_distribution_resources.py
```

Expected: build exits 0 and resource tests pass. Inspect archive listings in the test rather than manually trusting build output.

- [ ] **Step 5: Run the complete zero-cost test suite**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest -q \
  -m 'not gpu and not container and not live_llm and not release and not release_evidence'
```

Expected: exit 0 with no failed tests. Record the exact passed/deselected counts in the task report.

- [ ] **Step 6: Run final static and repository checks**

Run:

```bash
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python -m ruff check src tests
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python -m ruff format --check src tests
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python -m mypy --strict src
git diff --check
git status --short
```

Expected: Ruff/mypy/diff checks exit 0. Status contains only the intended Task 6 files before commit; generated `dist/` remains ignored.

- [ ] **Step 7: Commit Task 6**

```bash
git add docs/v2-operator-runbook.md docs/v2-release-status.md README.md \
  tests/unit/test_distribution_resources.py pyproject.toml
git commit -m "docs: add v2 evidence operator runbook"
```

## Final branch verification

- [ ] Generate a review package from the base commit through Task 6 HEAD and run one whole-branch architecture/security review.
- [ ] Resolve every blocking finding with a failing regression test before changing production code.
- [ ] Re-run the complete zero-cost suite, Ruff, strict mypy, build, and distribution smoke tests after the final fix.
- [ ] Confirm no GPU or provider run was started and no private identity, label, score, signer key, or evaluator path appears in Git-tracked bytes.
- [ ] Merge the reviewed commits into `feat/v2-release`; do not start final 16+8/360 evidence until that branch is clean and frozen.
