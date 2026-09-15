# V2 Native Release Provenance Implementation Plan

> **Status:** approved continuation after Task 5 of the release-foundation plan was rejected and reverted. This plan contains no paid model run and no claim that Portfolio Release has passed.

**Goal:** Make every release claim derivable from native, immutable execution artifacts that were bound to the repository and toolchain before execution, so a later manifest, receipt, test report, or synthetic record cannot create evidence.

**Architecture:** Add an immutable release binding to release-relevant runs at creation time; produce corpus and evaluation records only through controllers that derive fields from registered RunStore artifacts; keep holdout identity and scoring in the evaluator store; select exact run IDs in one frozen release snapshot; and perform release testing through a controller-owned two-stage command. Old runs without the new binding remain useful demonstrations but cannot satisfy this release gate.

**Scope boundary:** This plan builds and tests the zero-cost provenance path. It does not create 16+8 validated cases, does not add pricing attestation, does not invoke a provider, does not run live GPU acceptance, and does not generate `evaluation/release-manifest.json`.

---

## Invariants shared by every task

- A release binding is written when a run is created. No API may attach or replace commit, tree, toolchain, prompt, model, corpus, or pricing facts after a run becomes terminal.
- Repository identity is `commit + tracked_tree_hash + clean=true`; the controller obtains it from the checkout. A caller-supplied commit string is not proof.
- Toolchain identity is derived from the locked configuration and the runtime evidence recorded by the execution backend. Callers cannot supply an unrelated toolchain hash.
- Release loading starts from an explicit immutable snapshot of exact run IDs. Unselected historical runs neither count nor poison the result.
- Public artifacts never contain private case IDs, template/operator IDs, source paths, Oracle implementation, evaluator labels, or private record paths.
- Unit tests may use fake subprocess/container ports, but the positive fixture must traverse the same production producers and native artifact schemas. Tests may not write summary booleans or evaluation records directly into RunStore to make the gate pass.
- A missing producer, binding, artifact, or cross-store link is a typed closed-gate reason, never an inferred success.

## Task 1: Immutable repository and runtime bindings

**Files:**
- Create: `src/gpu_agent/provenance.py`
- Modify: `src/gpu_agent/contracts.py`
- Modify: `src/gpu_agent/store.py`
- Modify: `src/gpu_agent/environment.py`
- Modify: `src/gpu_agent/execution/isolated.py`
- Modify: `src/gpu_agent/service.py`
- Modify: `src/gpu_agent/cli.py`
- Create: `tests/unit/test_provenance.py`
- Modify: `tests/unit/test_store.py`
- Modify: `tests/unit/test_service.py`

**Contracts:**

```python
class RepositorySnapshot(ExecutionModel):
    commit: str
    tracked_tree_hash: str
    clean: Literal[True]

class RunBinding(ExecutionModel):
    repository: RepositorySnapshot
    purpose: Literal["corpus_validation", "evaluation", "release_acceptance"]
    toolchain_lock_hash: str | None
    prompt_version: str | None
    model_config_hash: str | None

RunStore.create_run(kind, *, binding: RunBinding | None = None, parent_run_id=None)
```

- [ ] Write tests proving dirty trees, abbreviated commits, symlinked repositories, mutable tracked files, caller/actual HEAD disagreement, and post-creation rebinding are rejected.
- [ ] Implement `capture_repository_snapshot(repo)` with fixed `git` argv, bounded output, no shell, a clean-index/worktree check, and a canonical tracked-tree hash. Tests inject a process port; release CLI uses the real port.
- [ ] Persist `RunBinding` inside `RunManifest` at `create_run`; never expose a setter. Parent/child release-relevant runs must inherit and exactly match their parent's repository binding.
- [ ] Compute the toolchain-lock hash from `containers/toolchain.lock.json` through `read_regular`. At execution time compare the binding to the backend's recorded image/base/version/policy evidence; a lock hash alone is not runtime proof.
- [ ] Thread the optional binding through `ApplicationService` and release controllers without changing ordinary development commands.
- [ ] Verify focused tests, all offline tests, Ruff, strict mypy, and commit `feat: bind release runs before execution`.

## Task 2: Native, case-bound corpus validation

**Files:**
- Create: `src/gpu_agent/benchmark/validation.py`
- Modify: `src/gpu_agent/benchmark/models.py`
- Modify: `src/gpu_agent/benchmark/builder.py`
- Modify: `src/gpu_agent/benchmark/executor.py`
- Modify: `src/gpu_agent/cli.py`
- Modify: `tests/unit/test_corpus_registration.py`
- Modify: `tests/integration/test_benchmark_cli.py`
- Modify: `tests/gpu/test_mutation_validation.py`

**Contracts:**

```python
class CaseExecutionObservation(ExecutionModel):
    case_id: str
    template_id: str
    mutation_id: str
    role: Literal["clean", "mutant"]
    split: Literal["public", "private"]
    source_hash: str
    harness_hash: str
    input_set_hash: str
    toolchain_hash: str
    oracle_id: str
    target_tool: SanitizerTool
    expected_finding: str
    build_ref: ArtifactRef
    runtime_ref: ArtifactRef
    sanitizer_refs: list[ArtifactRef]
    oracle_ref: ArtifactRef

class CaseValidationArtifact(ExecutionModel):
    clean_run_id: str
    mutant_run_id: str
    clean_observation_hash: str
    mutant_observation_hash: str
```

- [ ] Write negative tests for reused runs, role swaps, unrelated case/template, mismatched source/harness/input/toolchain hashes, missing Oracle, target-tool substitution, incomplete sanitizer execution, timeout, and claimant-supplied success booleans.
- [ ] Implement a controller that creates clean and mutant runs with Task 1 bindings, executes the existing backend/Oracle, and persists `CaseExecutionObservation` before terminalization. Derive every outcome from typed `BuildResult`, `ExecutionResult`, `SanitizerResult`, and Oracle artifacts already registered in that run.
- [ ] Change `BenchmarkBuilder.register` to accept a hash-checked `CaseValidationArtifact` or exact source run IDs and derive `CaseManifest`; do not accept externally constructed `CaseExecution` as release evidence.
- [ ] Require unique clean/mutant run pairs per registered case. Bind both roles to the exact case metadata and reject reuse across unrelated cases or public/private splits.
- [ ] Make `gpu-agent benchmark validate` use the native controller path. Old runs lacking `RunBinding` or observations return `CASE_EXECUTION_ATTESTATION_UNAVAILABLE` and cannot be upgraded in place.
- [ ] Adapt the live mutation test to the production controller, but only collect it in zero-cost verification; do not run GPU here.
- [ ] Verify and commit `feat: derive corpus registration from native runs`.

## Task 3: Native evaluation lineage and private scoring

**Files:**
- Modify: `src/gpu_agent/benchmark/evaluation.py`
- Modify: `src/gpu_agent/benchmark/executor.py`
- Modify: `src/gpu_agent/benchmark/metrics.py`
- Modify: `src/gpu_agent/service.py`
- Create: `src/gpu_agent/benchmark/holdout.py`
- Modify: `tests/unit/test_evaluation_runner.py`
- Modify: `tests/unit/test_evaluation_modes.py`
- Modify: `tests/integration/test_benchmark_cli.py`

**Contracts:**

```python
class EvaluationLineage(ExecutionModel):
    diagnosis_run_id: str
    diagnosis_hash: str
    evidence_hash: str
    provider_invocation_hashes: list[str]
    candidate_run_id: str | None
    verification_run_id: str | None
    public_verification_hash: str | None

class EvaluatorRecordBinding(ExecutionModel):
    public_record_id: str
    public_record_hash: str
    private_case_id: str
    private_template_id: str
    private_score_hash: str
```

- [ ] Write tests proving arbitrary record IDs/hashes, empty diagnoses, provider/config mismatch, missing mode-appropriate invocation, candidate/verification mismatch, private-label omission, record replay, and forged idempotency/reservation values are rejected.
- [ ] Extend `PublicEvaluationRecord` with public lineage that resolves to real public RunStore artifacts. Persist records only through `EvaluationExecutor`; make record construction internal or require a validated lineage object.
- [ ] For modes A–C, prove the expected deterministic controller path and zero provider calls. For D, prove RuleRouter and zero provider calls. For E, prove the bound provider invocation(s), prompt version, configured/response model policy, `store=false`, usage, and terminal diagnosis.
- [ ] Recompute attempt idempotency keys and reserved cost from the frozen schedule and bindings when loading. Reject extra as well as missing attempt/record artifacts.
- [ ] Add a holdout preparation controller that creates cryptographically opaque aliases from a random evaluator-owned nonce plus private identities, persists only aliases publicly, and stores the exact public-record/private-score binding in the evaluator RunStore. Regex shape is not evidence.
- [ ] Make evaluator labels/scores hash-bind to the public record without copying private identity or labels into public artifacts.
- [ ] Keep production paid execution closed until the separate pricing attestation exists.
- [ ] Verify and commit `feat: bind evaluation records to native run lineage`.

## Task 4: Explicit immutable release snapshot

**Files:**
- Rewrite: `src/gpu_agent/benchmark/release.py`
- Modify: `src/gpu_agent/benchmark/__init__.py`
- Rewrite: `tests/unit/test_release_gate.py`
- Create: `tests/integration/test_release_snapshot.py`

**Contracts:**

```python
class ReleaseEvidenceSelection(ExecutionModel):
    repository: RepositorySnapshot
    public_case_run_ids: list[str]
    private_case_run_ids: list[str]
    development_evaluation_run_id: str
    holdout_evaluation_run_id: str
    private_binding_run_id: str
    acceptance_run_ids: dict[str, list[str]]
    release_test_run_id: str

class ReleaseEvidenceIndex(ExecutionModel): ...

ReleaseEvidenceIndex.derive(selection, public_store, evaluator_store, actual_repository)
ReleaseGate.check(manifest, index)
```

- [ ] Write anti-forgery tests for nonexistent, duplicate, cross-split, unselected, stale, wrong-store, wrong-parent, wrong-hash, nonterminal, reused, extra and omitted run IDs.
- [ ] Load only exact selected runs and their referenced children. Validate all selected bytes through `RunStore.read`; ignore unrelated historical runs after safely validating store boundaries.
- [ ] Derive corpus counts/tool families, corpus hash, A–E coverage, repeats, record totals, model/prompt/toolchain bindings, private scoring completeness, and native acceptance evidence from Tasks 1–3. A manifest contains claims only.
- [ ] Require 16 public, 8 private, four public cases per sanitizer family, distinct private template/operator identities, complete development and holdout batches, and no stopped/failed/unknown-cost unit.
- [ ] Keep a positive unit/integration path that uses the actual Task 1–3 producers with fake execution ports. Direct `RunStore.put` of summary or evaluation result objects is a test failure.
- [ ] Verify and commit `feat: derive releases from an explicit evidence snapshot`.

## Task 5: Two-stage release test controller without JUnit bootstrap

**Files:**
- Create: `src/gpu_agent/release_controller.py`
- Modify: `src/gpu_agent/cli.py`
- Modify: `tests/conftest.py`
- Create: `tests/unit/test_release_controller.py`
- Create: `tests/e2e/test_release_evidence.py`
- Modify: `tests/e2e/test_release_acceptance.py`
- Modify marker declarations in the real isolation, four-tool, private-Oracle, live-LLM, mode and corpus tests.

**Protocol:**

1. `gpu-agent release collect-evidence` captures the actual clean repository snapshot, invokes a fixed pytest argv itself (`-m release_evidence --require-live --junitxml=<controller temp>`), checks exact collected node IDs and marker metadata from a controller plugin, requires exit code 0/no skipped tests, normalizes bounded output, and stores an invocation envelope before terminalizing its run.
2. `gpu-agent release check` loads the frozen selection and manifest, derives the index, runs `ReleaseGate`, and exits nonzero on any reason. This final gate is not an input to its own evidence report, eliminating the bootstrap cycle.

- [ ] Write tests rejecting caller-provided XML, arbitrary commands, missing flags, hand-written node paths, partial `-k` selections, changed marker sets, skipped tests, dirty/different trees, nonzero exit, and a negative-control test counted as positive evidence.
- [ ] The production controller creates its own temporary JUnit/marker report and invokes a fixed executable/argv without a shell. It captures command, repository snapshot, collection hash, exact node IDs, exit status, and normalized JUnit hash before terminalization.
- [ ] Use a dedicated `release_evidence` marker only on positive evidence-producing tests. Keep the offline “gate closes without manifest” test unmarked. The positive `release` acceptance test always requires and validates frozen artifacts; it never passes because the manifest is absent.
- [ ] Add an integration test with a temporary minimal pytest package that exercises the real subprocess controller. Repository release node allowlists remain fixed and separately unit-tested.
- [ ] Verify and commit `feat: run release acceptance through a trusted controller`.

## Task 6: Truthful status documentation and whole-plan verification

**Files:**
- Modify: `IMPLEMENTATION_PLAN_CN_V2.md`
- Modify: `README.md`
- Modify: `docs/acceptance.md`
- Modify: `docs/evaluation-report.md`
- Modify: `docs/execution-environment.md`
- Modify: `docs/demo.md`

- [ ] State exactly which native provenance producers exist and which live artifacts are still absent. Historical `8bd3cea` evidence remains demonstration-only and cannot be re-attested.
- [ ] Document the explicit selection snapshot, public/evaluator split, two-stage release command, clean-tree requirement, and why old artifacts fail closed.
- [ ] Keep the release blocked on native GPU validation of 16+8 cases, production opaque holdout batch preparation, pricing attestation, explicit user-approved cost ceiling, complete 360-unit paid evaluation, and same-commit release-evidence execution.
- [ ] Run collection; focused provenance/corpus/evaluation/release tests; all unit/integration/e2e tests excluding live markers; Ruff; format check; strict mypy; `pip check`; `git diff --check`; secret/private-path/large-file scans.
- [ ] Run the positive release command only in closed-gate mode and verify a typed nonzero result without starting GPU/provider work. Do not create a release manifest.
- [ ] Obtain independent final review over the full plan range and fix every Critical/Important finding before calling the zero-cost provenance foundation complete.
- [ ] Commit `docs: describe native release provenance and blockers`.

## Completion boundary

This plan is complete when the repository can no longer convert declarative summaries, old runs, synthetic evaluation records, handwritten JUnit, or unrelated historical artifacts into release evidence, and all zero-cost tests/reviews pass. Portfolio Release remains blocked until the real corpus, pricing approval, GPU/provider runs, evaluator scoring, frozen selection, and final release check are executed.
