# Task 3 Authority Stabilization Plan

> This plan supersedes further patch rounds for Task 3 in
> `2026-09-16-native-release-provenance.md`. Tasks 4–6 remain gated until this
> plan passes an independent whole-plan review.

## Goal

Make native evaluation evidence prove, rather than merely claim, that the
canonical schedule was frozen before any evaluation child executed; preserve
historical recovery at the signed corpus cutoff; and derive public/private
verification and Mode-D failure semantics from the same native facts.

## Non-goals and safety

- Do not run GPU, containers, network providers, or paid APIs.
- Do not read provider credentials.
- Production schedule signing remains externally provisioned and hard-closed
  when unavailable. Test signing remains explicitly TEST_ONLY.
- Do not begin release snapshot work until all tasks below are independently
  approved.

## Task 1: Freeze-before-execute authority state machine

**Files:** `benchmark/schedule_authority.py`, `benchmark/evaluation.py`,
`service.py`, `store.py`, authority/evaluation tests.

- A signing request is valid only while the evaluation parent is `QUEUED`.
- It binds the exact manifest/event/artifact prefix and proves zero child runs.
- Commit the signed receipt before the parent transitions to `RUNNING`.
- `ApplicationService.diagnose` accepts evaluation work only for a `RUNNING`
  parent whose signed receipt resolves and whose ordinal is claimed.
- Extra/pre-sign diagnosis children, post-execution sealing, copied receipts,
  and altered prefixes fail closed.
- Recovery is explicit for crashes before receipt, after receipt, and after the
  RUNNING transition; it never permits physical work before authority.

## Task 2: Non-bypassable execution lease

**Files:** `benchmark/executor.py`, `store.py`, concurrency tests.

- Open and acquire the canonical per-unit lock inside the concrete unbound
  execution body; do not dispatch through an instance-replaceable lock method.
- Compare `fstat(fd)` with `stat(path)` device/inode after acquisition.
- Expose no `__wrapped__` or callback path around the lease.
- Runner execution and recovery use concrete unbound implementations for
  record validation, attempts, records, terminalization, and manifest loading;
  instance monkeypatches cannot replace any trust-boundary method.
- Final manifest and downstream release consumers independently reload and
  validate native records rather than trusting Runner-produced summaries.
- Concurrent and crash-recovery tests prove at most one physical diagnosis or
  provider dispatch for an ordinal.

## Task 3: Historical cutoff propagation

**Files:** `benchmark/schedule_authority.py`, `benchmark/evaluation.py`,
`benchmark/holdout.py`, `benchmark/metrics.py`, recovery tests.

- Persist the authority cutoff in schedule, alias batch, attempt, record, and
  evaluator score bindings.
- Resume, holdout resolution/scoring, and metrics reload the universe at that
  exact cutoff rather than the current corpus head.
- Adding later public/private cases cannot invalidate or enlarge an existing
  signed run.
- Cutoff substitution, future registrations, and mixed-cutoff records fail.

## Task 4: One verification derivation and private-safe projection

**Files:** `verification/engine.py`, `verification/models.py`,
`benchmark/executor.py`, verification/evaluation tests.

- Implement one pure derivation from ordered native children and evaluator
  suite spec to counts, checks, findings, reason, failure stage, limitations,
  verdict, and public projection.
- Producer and validator both call that derivation; no parallel failure table.
- Correctly represent build failure, runtime tool error, sanitizer tool error,
  later private-child failure, and oracle failure as truthful terminal records.
- Public projection uses only public child/index-0 facts. Private findings,
  counts, check outcomes, reasons, limitations, and suite shape remain only in
  evaluator storage.
- Reparse sanitizer raw logs and recompute ordinary/instrumented oracles from
  exact input/output bytes.

## Task 5: Exact Mode-D dispatch accounting

**Files:** `agent/orchestrator.py`, `benchmark/executor.py`, Mode-D tests.

- Persist distinct attempted, dispatched, completed, and failed states for
  sanitizer, retrieval, and source acquisition.
- Increment budget counters according to the same transition used by the
  physical producer, including `KnowledgeError` and pre-dispatch timeout.
- Replay starts from the frozen default budget and verifies every pre/post
  counter, remaining allowance, evidence snapshot, action argument, and final
  audit.
- Legitimate failed acquisition becomes terminal `INCONCLUSIVE`; forged
  counters, false dispatch, or evidence advancement fail closed.

## Task 6: Whole-plan verification and Task 3 acceptance

- Run focused adversarial tests for every finding above.
- Run the complete offline suite, Ruff, format check, strict mypy, diff check,
  secret-pattern scan, and GPU collect-only.
- Independent reviewer examines the full range from `8eee211` and reruns
  adversarial reproductions.
- Task 3 is complete only with no Critical or Important finding. Minor findings
  must be fixed or explicitly documented without weakening release claims.
