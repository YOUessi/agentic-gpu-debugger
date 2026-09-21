# SDD ledger — plan: docs/superpowers/plans/2026-09-21-holdout-evaluator-execution-isolation.md

## Baseline

- BASE: `26bb978b6d57e5e5aa49a967f402d7a6926d2c6a`
- Branch: `design/v2-operator-workflow`
- Workspace: linked Git worktree; not a submodule.
- Initial state: clean.
- Verification ruling: the latest broad offline baseline predates only the two documentation commits in BASE. Do not repeat already-passing broad tests at task start; run focused affected tests for Tasks 1–4 and exactly one full offline regression in Task 5.
- Live-work ruling: no DeepSeek calls, live GPU acceptance, corpus regeneration, push, or publication during Tasks 1–5.

## Preflight conflict scan

| Scope | Shared interface / risk | Ruling |
| --- | --- | --- |
| Task 1 ↔ Task 3 | `ApplicationService`, workflow visibility, executor construction | Task 1 first makes a single workflow store-derived; Task 3 then supplies the correct public/evaluator service. Task 3 must preserve Task 1's visibility API and tests. |
| Task 2 ↔ Task 3 | evaluation lineage, holdout transaction, split routing | Task 2 owns models and deterministic transaction lifecycle; Task 3 only routes execution/recovery through those APIs. |
| Task 3 ↔ Task 5 | holdout record validation | Task 3 establishes routing and recovery; Task 5 centralizes commitment resolution and scoring/release consumers without recreating routing logic. |
| Task 4 ↔ Tasks 1/3 | `service.py`, production factories | Task 4 wires exact family stores into the already-established APIs; it must not rewrite workflow or executor semantics. |
| Task 4 ↔ live evidence | family pin schema changes | Old production family is intentionally invalid after schema/path pinning. Final evidence must use a newly provisioned family; never mutate or silently migrate the old one. |
| Task 5 ↔ Task 2 | alias-mapping child inventory | `holdout_execution` stays top-level with public external origin. Alias-mapping direct children remain reserved for `holdout_score`. |
| All tasks | illustrative snippets vs repository reality | Approved design/spec and invariants are authoritative; adapt test scaffolding mechanically where current APIs differ, recording any material deviation. |

## Task status

- Task 1 — APPROVED at `f2c5ad0a7abff538824df63798d60bf9c6704763` (three commits); focused `73 passed`; Ruff/mypy/diff-check passed
- Task 2 — APPROVED at `12e6150` (four commits); focused Task 2, store, integration, validator, and Mode E adversarial checks passed; Ruff/mypy/diff-check passed
- Task 3 — APPROVED at `eb7adbb`; focused Mode E privacy/authorization and persistence checks
  passed; independent review found 0 Blocker, 0 Major, 0 Minor
- Task 4 — APPROVED at `0047567`; remediation and broader Task 4 checks passed; independent
  rereview found 0 Blocker, 0 Major, 0 Minor
- Task 5 — pending

## Review/fix ledger

- Task 1 review wave 1: CHANGES_REQUESTED; 0 Blocker, 2 Major, 1 Minor.
  - Major: replace evaluator root scan in `derive_verification()` with audit-parent-scoped children.
  - Major: evaluator-local verification must inherit and strictly validate diagnosis `external_origin`; current `None` assumption rejects the approved holdout lineage.
  - Minor: migrate `tests/gpu/test_candidate_verification.py` to exact evaluator `RunStore` constructor without running GPU.
- Task 1 review wave 2: original 2 Major + 1 Minor closed; CHANGES_REQUESTED with 1 new Major, 1 new Minor.
  - Major: evaluator-local persisted derivation must reject same-store `external_origin.visibility == "evaluator"`; only no origin or exact public origin is legal.
  - Minor: child inventory schema version must reject booleans and require strict integer `1`.
- Task 1 review wave 3: APPROVED; 0 Blocker, 0 Major, 0 Minor.
- Task 2 review wave 1: CHANGES REQUIRED; 1 Blocker, 2 Major, 1 Minor.
  - Blocker: evaluator native record validation is not fully artifact-derived, allowing forged summary/candidate/provider/verification fields into a terminal blind projection.
  - Major: recovery can misclassify orphan/unsafe deterministic state as absent and reserve can mutate an incomplete terminal parent.
  - Major: remaining development constructor/package export compatibility regression after discriminated lineage migration.
  - Minor: restore three name-only test renames; use explicit Task 2 selectors and report the deferred Task 3 runner failure honestly.
- Task 2 review wave 2: original Blocker/Majors/Minor closed; CHANGES REQUIRED with 2 residual Major findings.
  - Major: bind candidate/verification exact external origin plus candidate parent/base/patched source provenance; replace early-exit adversarial tests with independently discriminating valid Mode E mutations.
  - Major: serialize deterministic reserve/recover and use fd-scoped no-follow identity validation to close check-then-use and same-attempt concurrency races.
- Task 2 review wave 3: R1 closed; R2 core closed; CHANGES REQUIRED with 1 Major, 1 Minor.
  - Major: revalidate lock pathname↔fd identity, type, mode, nlink, and root after `flock` and before/after the critical section.
  - Minor: add an unchanged valid Mode E complete+recover control alongside the ten single-field adversarial cases.
- Task 2 review wave 4: APPROVED; 0 Blocker, 0 Major, 0 Minor.
- Task 2 review wave 4: APPROVED; 0 Blocker, 0 Major, 0 Minor.
- Task 3 review wave 1: CHANGES REQUIRED; 1 Blocker, 3 Major, 1 Minor.
  - Closed Blocker: explicit public allowlist with empty diagnosis and a real Mode E multi-canary
    byte scan with nonzero provider calls.
  - Closed Major: raw reserved-ID bypass replaced by controller-revalidated, sealed one-use start
    capability under pinned public/evaluator leases.
  - Closed Major: deterministic candidate/verification slots remove evaluator-root enumeration.
  - Closed Major: constructor now validates exact family/store/binding/verifier identities before
    any public or evaluator mutation.
  - Closed Minor: real Mode E provider `STARTED` recovery proves `1 -> 1` physical-call idempotency.
- Task 3 review wave 2: residual 1 Blocker + 1 Major closed at `b818e9a`.
  - Closed Blocker: public projection now discards provider-controlled failure reasons; a real
    Mode E uppercase limitation canary is evaluator-only under a complete public byte scan.
  - Closed Major: importable capability fields are no longer authorization. The issuing controller
    holds the exact one-use object identity, pops it before validation, and rejects forgery and
    replay before source, GPU/backend, or provider work.
  - Verification: focused `9 passed in 119.58s`; persistence `25 passed in 32.13s`; Ruff, mypy,
    and diff-check passed.
- Task 3 review wave 3: residual 1 Major closed at `eb7adbb`.
  - Closed Major: removed the importable seal/capability and caller-addressable registry. Reserved
    diagnosis now carries exact controller/batch/prepared authority and invokes a self-validating
    authorize-and-start operation before source I/O; every callable entry independently verifies
    signed public and deterministic evaluator authority.
  - Verification: forged/replay `2 passed in 2.52s`; reviewer set `4 passed in 59.19s`;
    persistence `25 passed in 32.03s`; Ruff, mypy, and diff-check passed.
- Task 3 review wave 4: APPROVED; 0 Blocker, 0 Major, 0 Minor.
- Task 4 review wave 1: CHANGES REQUIRED; 0 Blocker, 2 Major, 1 Minor.
  - Closed Major: production provisioning now validates pre-existing store safety, resolved
    non-overlap, and physical identity before any family mutation.
  - Closed Major: all controller authority and marker files now enforce owner, exact mode,
    single link, no-follow fd/path identity, and bounded stable reads.
  - Closed Minor: successful development/holdout factory tests verify the complete service,
    controller, batch, and verifier graph.
- Task 4 review wave 2: APPROVED at `0047567`; 0 Blocker, 0 Major, 0 Minor. Verification:
  remediation `27 passed, 46 deselected`; broader Task 4 `44 passed, 77 deselected`; Ruff,
  mypy, and diff-check passed.
