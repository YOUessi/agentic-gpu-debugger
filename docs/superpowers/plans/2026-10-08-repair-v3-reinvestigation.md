# Repair v3 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for the coordinator and superpowers:test-driven-development for all behavior changes. Independent model-context and budget-continuation tasks may run in parallel; one fresh whole-branch review follows integration.

**Goal:** Public self-check failures can trigger a budgeted investigation of the failed candidate and update the diagnosis used by the next patch.

**Architecture:** Keep the original patch base and parent diagnosis. A coordinator creates isolated public candidate investigation runs, continuing the same provider, call gate and acquisition ledger; a conditional model context describes source roles explicitly.

**Tech Stack:** Python 3.11/3.12, existing Pydantic contracts, RunStore and isolated CUDA backend; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-08-repair-v3-reinvestigation-design.md`

## Global Constraints

- Default `public-repair-v2`, traditional diagnose and frozen A–E evaluation retain their behavior.
- The original source remains the only patch base; candidate evidence never enters its bundle.
- The provider, call budget, acquisition ledger and deadline are shared across investigation episodes.
- Private validation stays outside the public feedback loop; uncertain calls stop the loop.
- New code, work records and results are stored on the dedicated GitHub branch.

## Review Focus

- All Sanitizers CLEAN but public numeric output wrong: V3 permits evidence-backed diagnosis.
- A failure after the last candidate or an unavailable tool: no wasted investigation or private verification.
- Candidate source lines differ from the original: current citations cannot be reused as original evidence.
- Budget exhaustion during a new episode: counts and deadline do not reset, including format retries.
- Multiple failed episodes: the current scoped diagnosis and cumulative usage remain correctly attributable.

## Task 1: Conditional public repair context and prompts

**Files:** `agent/models.py`, `agent/provider.py`, `agent/prompts.py`; `tests/unit/test_repair_public_context.py`.

**Interfaces:** `PublicRepairContext` with source hashes, previous diagnosis, public checks and current functional-failure flag;
`PublicEvidence.repair_context`; `select_prompt(kind, payload) -> (version, instructions)`.

- [x] Add RED tests for unchanged V2 serialization/prompts, contextual V3 requests and cross-scope citation rejection.
- [x] Implement optional context and conditional prompt/telemetry version.
- [x] Run `.venv/bin/python -m pytest -q tests/unit/test_repair_public_context.py` and preserve passing evidence.

## Task 2: Continue acquisition budgets in a new candidate workspace

**Files:** `agent/orchestrator.py`, `agent/policy.py`; `tests/unit/test_repair_budget_continuation.py`.

**Interfaces:** `AgentOrchestrator.continue_in_workspace(backend, handle, stdin_ref) -> AgentOrchestrator`;
`LLMCallGate.begin_repair_cycle() -> None`. Coordinator copies child budget/usage back in finally.

- [x] Add RED tests proving shared ledger/gate/deadline, fresh candidate action scope and no retry reset.
- [x] Implement continuation and V3-only numeric-failure evidence sufficiency.
- [x] Run `.venv/bin/python -m pytest -q tests/unit/test_repair_budget_continuation.py`.

## Task 3: Coordinate public failure attribution, investigation and revision

**Files:** new `src/gpu_agent/repair_coordinator.py`; `repair.py`, `service.py`, `cli.py`;
public evidence projection in `agent/orchestrator.py`; `tests/unit/test_repair_reinvestigation.py`.

**Interfaces:** `RepairCoordinator` owns current scoped diagnosis and trigger decisions, sharing the original orchestrator;
`repair_candidates(..., coordinator=None)` preserves its existing candidate return type.

- [x] Add failing end-to-end tests for new diagnosis reaching revise_patch while original base and parent evidence remain fixed.
- [x] Test limits, unavailable and inconclusive outcomes, functional-only failure, and one-time private verification.
- [x] Implement native public child prepare/build/run, exact-source context export and cumulative parent usage records.
- [x] Expose `--reinvestigate` and bounded `--max-reinvestigations` on the existing repair command.
- [ ] Run targeted repair, provider, agent and evaluation regression tests, then zero-cost CI checks. Targeted integration: 101 passed; full offline CI is running.

## Task 4: Independent review, GPU validation and handoff

**Files:** dated `docs/repair-log/2026-10-08-repair-v3-reinvestigation.md`, runbook, changelog; a focused GPU smoke command/test when needed.

- [x] Review the complete implementation against the spec; address material findings with regression tests. All four findings have regression coverage.
- [ ] Commit the branch through GitHub, then use an isolated Tang checkout to run the committed GPU validation.
- [ ] Record actual environment and results, separate scripted-provider control-flow proof from live model evidence.
- [ ] Open a reviewable PR against `design/v2-operator-workflow`; report completed validation and remaining effect-evaluation work.
