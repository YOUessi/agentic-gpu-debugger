# V2 Release Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the zero-cost implementation gaps that prevent the existing GPU debugging prototype from running a durable five-mode evaluation and enforcing a trustworthy release gate.

**Architecture:** Keep `ApplicationService` as the only production entry to diagnosis and verification. Add explicit evaluation-mode policy at that boundary, persist evaluation batches in a public `RunStore`, expose controller-only benchmark commands through Typer, and make release verification derive facts from immutable artifacts instead of trusting manifest claims. Corpus expansion and paid 360-unit execution are intentionally a later plan because they require real GPU evidence and an explicit API cost ceiling.

**Tech Stack:** Python 3.11, Pydantic 2, Typer, pytest, Ruff, strict mypy, existing `RunStore`, existing provider and isolated GPU backend contracts.

**Spec:** `/home/you/projects/agentic-gpu-debugger/PROJECT_SPEC_CN_V2.md` (§§17–23), with `/home/you/projects/agentic-gpu-debugger/IMPLEMENTATION_PLAN_CN_V2.md` T11–T12 as the accepted implementation requirements.

## Global Constraints

- No DeepSeek or other paid provider call is allowed in this plan; tests use real local components plus bounded fakes only at the provider/network boundary.
- Model-generated CUDA never executes on the host; all candidate execution remains in `IsolatedGPUBackend`.
- Private inputs, truth, and artifacts never enter the public `RunStore`, model view, repository, CLI output, or release artifact.
- Modes A–E use the same provider, schemas, input, toolchain, RAG index, prompt version, and physical budgets; only evidence availability or acquisition policy differs.
- Each case/mode has at least three repeats, is scheduled serially on one GPU, and has randomized order with a recorded seed.
- An absent cost ceiling stops before the first provider call. A batch persists each completed unit before starting the next.
- `VERIFIED_FIXED` remains scoped to the recorded candidate, toolchain, input suite, Oracle results, and required checks.
- Release evidence must be bound to the current 40-character commit, exact configuration hashes, non-zero required test counts, real immutable run artifacts, 16 public cases, 8 private cases, and all five evaluation modes.
- Do not alter global Python, CUDA, ROS, Docker, or Conda configuration.

---

### Task 1: Deterministic Full-Suite Collection

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/unit/test_release_gate.py`
- Test: `tests/e2e/test_release_gate.py`

**Interfaces:**
- Consumes: pytest configuration in `[tool.pytest.ini_options]`.
- Produces: `python -I -m pytest --collect-only -q` collects both same-named release test modules without import mismatch.

- [ ] **Step 1: Capture the failing baseline**

Run:

```bash
/home/you/conda_env/agentic-gpu-debugger/bin/python -I -m pytest --collect-only -q
```

Expected: exit 2 with an import mismatch between `tests/unit/test_release_gate.py` and `tests/e2e/test_release_gate.py`.

- [ ] **Step 2: Configure isolated pytest imports**

Set the pytest default import mode in `pyproject.toml`:

```toml
addopts = "--strict-markers --import-mode=importlib"
```

- [ ] **Step 3: Verify collection and the free baseline**

Run:

```bash
/home/you/conda_env/agentic-gpu-debugger/bin/python -I -m pytest --collect-only -q
/home/you/conda_env/agentic-gpu-debugger/bin/python -I -m pytest tests/unit tests/integration -q
```

Expected: collection exits 0; unit/integration tests pass with only declared prerequisite skips.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "test: isolate pytest module collection"
```

---

### Task 2: Durable Evaluation Batch Contract

**Files:**
- Modify: `src/gpu_agent/benchmark/evaluation.py`
- Modify: `src/gpu_agent/benchmark/__init__.py`
- Modify: `tests/e2e/test_evaluation_run.py`
- Create: `tests/unit/test_evaluation_persistence.py`

**Interfaces:**
- Consumes: `RunStore.create_run`, `put`, `transition`, `load`, and `read`; existing `EvaluationRecord` and `EvaluationManifest`.
- Produces: `EvaluationRunner(store, case_ids, execute, max_cost_usd, max_unit_cost_usd, random_seed)` and `run(mode, split, repeats) -> EvaluationManifest`. A batch run has kind `evaluation`, persists `evaluation/schedule.json` before execution, each `evaluation/records/<ordinal>.json` before the next unit, and `evaluation/manifest.json` at terminal completion.

- [ ] **Step 1: Write failing durability and hard-cap tests**

Tests must establish these observable behaviors:

```python
def test_record_is_durable_before_next_unit(store): ...
def test_unit_reservation_stops_before_cost_cap_can_be_exceeded(store): ...
def test_unexpected_executor_failure_preserves_completed_records(store): ...
def test_resume_rejects_commit_or_schedule_mismatch(store): ...
```

The executor fake may inspect `store` but assertions must target persisted artifacts and terminal reason codes, not mock call counts alone.

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/you/conda_env/agentic-gpu-debugger/bin/python -I -m pytest tests/unit/test_evaluation_persistence.py tests/e2e/test_evaluation_run.py -q
```

Expected: failures because the runner neither owns a `RunStore` batch nor reserves per-unit cost.

- [ ] **Step 3: Implement immutable schedule and incremental records**

Add a typed schedule item and extend the manifest with `run_id`, `commit`, `prompt_version`, `toolchain_hash`, `model_config_hash`, `schedule_hash`, `expected_units`, and `executed_units`. Use zero-based `ordinal` as the artifact name; never overwrite an artifact. The runner must create and persist the shuffled schedule before invoking `execute`.

Before each unit:

```python
if spent + max_unit_cost_usd > max_cost_usd:
    stopped_reason = "COST_CAP_RESERVATION_REQUIRED"
    break
```

After a returned record, persist it before updating the in-memory list. Unexpected exceptions produce a typed terminal batch reason and leave already persisted records readable; they do not synthesize successful records.

- [ ] **Step 4: Verify GREEN and regressions**

Run the focused tests, then:

```bash
/home/you/conda_env/agentic-gpu-debugger/bin/python -I -m pytest tests/unit tests/integration tests/e2e/test_evaluation_run.py -m "not gpu and not container and not live_llm and not release" -q
```

- [ ] **Step 5: Commit**

```bash
git add src/gpu_agent/benchmark/evaluation.py src/gpu_agent/benchmark/__init__.py tests/e2e/test_evaluation_run.py tests/unit/test_evaluation_persistence.py
git commit -m "feat: persist cost-bounded evaluation batches"
```

---

### Task 3: Production Evaluation Modes and Benchmark CLI

**Files:**
- Create: `src/gpu_agent/benchmark/executor.py`
- Modify: `src/gpu_agent/service.py`
- Modify: `src/gpu_agent/agent/orchestrator.py`
- Modify: `src/gpu_agent/cli.py`
- Create: `tests/unit/test_evaluation_modes.py`
- Create: `tests/integration/test_benchmark_cli.py`

**Interfaces:**
- Consumes: `ApplicationService`, `AgentOrchestrator`, `RuleRouter`, `EvaluationRunner`, `EvaluationRecord`, `BenchmarkBuilder`, and existing provider/verification contracts.
- Produces: `EvaluationExecutor.execute(case_id, template_id, mode, repeat) -> EvaluationRecord`; controller CLI group `gpu-agent benchmark validate` and `gpu-agent benchmark evaluate`.

- [ ] **Step 1: Write failing mode-separation tests**

The tests must prove externally visible evidence policy:

```python
def test_mode_a_exposes_only_source_and_runtime(): ...
def test_mode_b_adds_frozen_retrieval_without_tools(): ...
def test_mode_c_uses_precollected_tools_without_planner(): ...
def test_mode_d_uses_rule_router_and_never_planner(): ...
def test_mode_e_uses_planner_and_never_rule_substitution(): ...
```

All modes must produce the same typed diagnosis and candidate schema, record physical provider/tool counts, and preserve failures as that mode's failures.

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/you/conda_env/agentic-gpu-debugger/bin/python -I -m pytest tests/unit/test_evaluation_modes.py tests/integration/test_benchmark_cli.py -q
```

Expected: failures because no production executor or benchmark CLI exists.

- [ ] **Step 3: Add an explicit acquisition policy boundary**

Introduce a controller-owned literal mode and route acquisition without changing verification semantics:

```python
EvaluationMode = Literal["A", "B", "C", "D", "E"]
```

- A: source plus ordinary runtime, followed by final diagnosis/patch calls.
- B: A plus frozen official-document retrieval.
- C: A plus pre-collected required sanitizer evidence.
- D: bounded `RuleRouter` acquisition plus the shared final diagnosis/patch provider.
- E: bounded model Planner acquisition; provider failure remains failure and cannot fall back to D.

Refactor only the acquisition decision boundary needed to select the mode. Keep preparation, isolated build/run, patch registration, verification, and usage recording shared.

- [ ] **Step 4: Expose safe controller commands**

Create `benchmark_app = typer.Typer(no_args_is_help=True)` and attach it as `benchmark`. `validate` accepts controller-owned serialized `CaseExecution` files and writes only to the configured corpus store after `BenchmarkBuilder.validate`. `evaluate` requires explicit `--mode`, `--split`, `--repeats >= 3`, `--max-cost-usd`, and `--max-unit-cost-usd`; before execution it prints case × mode × repeat and the hard maximum cost. Missing caps exit before provider construction.

- [ ] **Step 5: Verify GREEN**

Run focused tests and `python -I -m gpu_agent benchmark --help`; assert CLI integration tests use an injected executor and never read credentials.

- [ ] **Step 6: Commit**

```bash
git add src/gpu_agent/benchmark/executor.py src/gpu_agent/service.py src/gpu_agent/agent/orchestrator.py src/gpu_agent/cli.py tests/unit/test_evaluation_modes.py tests/integration/test_benchmark_cli.py
git commit -m "feat: execute five controlled benchmark modes"
```

---

### Task 4: Complete Metrics and Blind Evaluation Artifacts

**Files:**
- Modify: `src/gpu_agent/benchmark/metrics.py`
- Modify: `src/gpu_agent/benchmark/evaluation.py`
- Modify: `tests/unit/test_metrics.py`
- Modify: `tests/integration/test_evaluation_views.py`

**Interfaces:**
- Consumes: persisted `EvaluationRecord` objects and hidden `HiddenTruth` available only to the evaluator.
- Produces: grouped summaries by case and template, plus family/root/location, evidence precision, citation precision, retrieval hit@k, compile rate, Oracle rate, verified rate, regression detection, inconclusive precision/recall, budget exhaustion, latency, and known cost with explicit denominators.

- [ ] **Step 1: Write failing denominator and grouping tests**

Cover empty data, timeouts in the end-to-end denominator, N/A truth, unknown costs, repeated case/template grouping, tool collection cost, irrelevant citations, and missing retrieval labels.

- [ ] **Step 2: Verify RED**

Run `python -I -m pytest tests/unit/test_metrics.py tests/integration/test_evaluation_views.py -q` and confirm missing metrics/grouping fail.

- [ ] **Step 3: Implement minimal typed metric additions**

Never infer relevance from citation existence. Metrics requiring labels return `Metric(value=None, n=0, numerator=0)` when labels are unavailable. Repeats increase record denominators but never `case_count` or `template_count`.

- [ ] **Step 4: Verify GREEN and commit**

```bash
git add src/gpu_agent/benchmark/metrics.py src/gpu_agent/benchmark/evaluation.py tests/unit/test_metrics.py tests/integration/test_evaluation_views.py
git commit -m "feat: report complete controlled evaluation metrics"
```

---

### Task 5: Evidence-Backed Release Gate and Status Documentation

**Files:**
- Modify: `src/gpu_agent/benchmark/release.py`
- Modify: `tests/unit/test_release_gate.py`
- Modify: `tests/e2e/test_release_gate.py`
- Modify: `IMPLEMENTATION_PLAN_CN_V2.md`
- Modify: `README.md`
- Modify: `docs/acceptance.md`
- Modify: `docs/evaluation-report.md`
- Modify: `docs/execution-environment.md`
- Modify: `docs/demo.md`

**Interfaces:**
- Consumes: persisted benchmark-case manifests, evaluation manifests/records, public and evaluator `RunStore` roots, current commit, exact public configuration hashes.
- Produces: `ReleaseGate.check(manifest, evidence) -> ReleaseGateResult`, where `evidence` is derived by a controller-owned loader from immutable artifacts and not copied from the release manifest.

- [ ] **Step 1: Write failing anti-forgery tests**

Tests must reject:

```python
expected == executed == 0
evidence_run_ids == {required_key: ["nonexistent"] for required_key in required_keys}
manifest.commit != current_commit
changed toolchain/corpus/model hashes
duplicate cases or templates counted across public/private splits
evaluation missing any mode, repeat, case, or persisted record
release marker selecting only the lightweight manifest test
```

- [ ] **Step 2: Verify RED**

Run the unit release tests and collect `tests/gpu tests/e2e -m release`; confirm the forged manifest currently passes and release coverage is incomplete.

- [ ] **Step 3: Implement derived evidence verification**

Keep `ReleaseManifest` declarative, but validate it against a separately derived `ReleaseEvidenceIndex`. The loader must open bounded regular files through existing safe readers, load real run manifests/artifacts, verify artifact hashes, require terminal `COMPLETED` states, deduplicate case/template identities, and bind every referenced record to the current commit/config/schedule. A non-empty string is never evidence by itself.

- [ ] **Step 4: Mark the real release acceptance surface**

Release selection must cover isolation, four tools, private Oracle aggregation, live LLM evidence, five modes, corpus counts, and the final manifest without rerunning paid calls when frozen same-commit evidence is valid.

- [ ] **Step 5: Reconcile documentation with exact current state**

Update stale milestone status and environment statements. Expand the clean-checkout public demo with exact preparation, full non-secret hashes, a current-commit run-report fixture, and a concrete negative control. Continue to state that Portfolio Release is blocked until 16+8 and the paid batch actually pass; do not fabricate a release manifest.

- [ ] **Step 6: Verify and commit**

Run unit/integration/e2e offline tests, collection, Ruff, strict mypy, `pip check`, `git diff --check`, secret/private-path scan, and the release test expecting a truthful closed gate while corpus/evaluation evidence is absent.

```bash
git add src/gpu_agent/benchmark/release.py tests/unit/test_release_gate.py tests/e2e/test_release_gate.py IMPLEMENTATION_PLAN_CN_V2.md README.md docs
git commit -m "feat: bind releases to immutable evaluation evidence"
```

## Completion Boundary

This plan is complete when all zero-cost implementation paths above are reviewed and pass, while the release gate still truthfully remains closed for missing 16+8 corpus and paid five-mode evidence. The next plan expands/validates the corpus, runs a cost-estimation pilot, requests an explicit user-approved API ceiling, executes the frozen 360-unit batch, generates `evaluation/release-manifest.json`, and performs the final Portfolio Release verification.
