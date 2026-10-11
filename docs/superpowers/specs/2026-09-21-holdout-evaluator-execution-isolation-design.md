# Holdout Evaluator Execution Isolation Design

## Status

Approved direction for the V2 release blocker discovered before paid evaluation. This design
changes the execution boundary only; it does not authorize provider spend, weaken the signed
schedule, or treat the already collected `f5ce2c0` corpus evidence as valid for the repaired
commit.

## Problem

The V2 operator contract requires private holdout sources, identities, validation logs, labels,
and evaluator paths to remain outside Git and the public RunStore/output. The current
`f5ce2c0` implementation selects the evaluator corpus for a holdout schedule, but constructs an
`ApplicationService` whose store is public. `EvaluationExecutor.execute_scheduled()` resolves a
private case and passes its source path to `ApplicationService.diagnose()`, which writes the full
`kernel.cu` bytes and all downstream agent/provider evidence to that public store.

The failure is architectural rather than an artifact-name mistake:

- the evaluation parent, schedule, attempts, and public records correctly belong in the public
  store;
- the diagnosis child, source snapshot, model exchanges, candidate, and verification evidence
  for holdout belong in evaluator-controlled storage;
- `EvaluationRunner`, `EvaluationExecutor`, `ApplicationService`, the agent loop, and candidate
  verification currently assume those two roles use the same public `RunStore`.

There is also an operator-path ambiguity. `ApplicationService` addresses private verification
evidence as `GPU_AGENT_EVALUATOR_ROOT/runs`, while corpus-family configuration records one exact
evaluator store. A production layout is valid only when those paths resolve to the same pinned
store where required; the CLI and runbook must make that relationship explicit before mutation.

## Goals

1. Keep the public evaluation coordinator: signed schedule, attempts, claims, sanitized records,
   cost projections, and terminal manifest remain in the public store.
2. Keep development evaluation behavior unchanged: development diagnosis and its evidence remain
   public.
3. Execute every holdout diagnosis entirely in evaluator storage. This includes source bytes,
   build/run/Sanitizer evidence, prompts and provider invocation artifacts, generated candidate,
   verification evidence, private identities, and full native lineage.
4. Persist only a blind `PublicEvaluationRecord` for each holdout unit in the public evaluation
   parent. It must contain the scheduled alias, bounded public outcomes, usage/cost totals, and no
   private source, case/template/operator identity, evaluator path, evaluator artifact reference,
   or provider payload.
5. Preserve signed schedule validation, exact corpus cutoff, attempt idempotency, crash recovery,
   cost ceilings, external scoring, and evidence-derived release checks.
6. Fail before provider or GPU execution when the public/evaluator paths, family identities, or
   split-store bindings disagree.

## Non-goals

- Do not relax the private-source contract.
- Do not encrypt private artifacts and store the ciphertext in the public store.
- Do not add another public CLI for private identities or private batch execution.
- Do not redesign the A–E acquisition modes, model prompts, scoring rubric, or release thresholds.
- Do not reuse corpus or evaluation evidence bound to the pre-fix commit.

## Chosen Architecture

### Public coordinator and evaluator worker

`EvaluationRunner` continues to own a public `RunStore`. It creates and activates the signed
evaluation parent, persists schedule/attempt/claim artifacts, enforces cost reservations, and
stores the final public record for every ordinal.

`EvaluationExecutor` receives two explicit execution capabilities:

- a public `ApplicationService` for development units;
- an evaluator `ApplicationService` plus `HoldoutController` for holdout units.

It selects the service from the signed item's split. A caller cannot override that choice with a
path or visibility flag. The selected service store must be the exact store recorded by the
configured corpus family.

For a development item, the existing public path remains unchanged. For a holdout item, the
executor resolves the scheduled alias privately, reserves an evaluator execution transaction,
runs diagnosis and verification in the evaluator service, persists the complete native result in
that evaluator transaction, and derives an in-memory blind projection. Only that projection is
returned to the public runner.

### Evaluator execution transaction

Each holdout ordinal has one deterministic evaluator transaction controlled by
`HoldoutController`. Its identity is derived from the immutable public evaluation run ID,
schedule hash, corpus cutoff, ordinal, and attempt idempotency key under the existing evaluator
authority. It is not supplied by the model or CLI.

The evaluator transaction binds:

- public evaluation run ID and ordinal;
- exact schedule and attempt hashes;
- corpus cutoff and holdout proof;
- private case/template identity resolved from the alias map;
- evaluator diagnosis run ID and terminal artifact hashes;
- full `EvaluationRecord` and the hash of its blind public projection;
- provider usage/cost derivation and verification lineage.

The public store never receives this transaction ID, its path, private identities, or evaluator
artifact references. The public record is joined back to evaluator evidence only inside trusted
holdout scoring and release validation through the evaluator-owned ordinal binding.

For holdout, `record_id` is a deterministic blind public identifier derived from the public
evaluation run, schedule hash, and ordinal; it is not an evaluator diagnosis/candidate/
verification run ID. Holdout public lineage contains only content commitments needed to bind the
projection. Evaluator run IDs and `ArtifactRef` values remain in the evaluator transaction.
Development keeps its existing native public run lineage.

### Store-aware application workflow

`ApplicationService` and the agent/verification components become visibility-preserving instead
of public-only. All artifacts produced by one workflow use `service.store.visibility`; no method
accepts an independently supplied visibility string.

The following behavior is required:

- evidence repositories open a public or evaluator view matching their store;
- source snapshots, acquisition policies, budgets, tool results, provider invocations, diagnosis,
  candidates, and verification results stay in the selected store;
- child runs inherit the selected store and exact release binding;
- code that enumerates candidate or verification children is parent-scoped and cannot discover
  unrelated evaluator corpus, label, or Oracle runs;
- agent evidence construction may read the selected diagnosis source but never receives the
  private case identity, target finding, labels, alias nonce, or Oracle truth;
- private Oracle/verification audit artifacts remain evaluator-only and are never included in the
  blind public projection.

Hard-coded `"public"` artifact writes in this workflow are replaced with store-derived
visibility. Public-only report/export APIs remain public-only and reject evaluator references.

### Exact store configuration

The corpus family remains the authority for exact store identities. Production configuration
must satisfy all of the following before a schedule is reserved:

- `GPU_AGENT_RUN_ROOT` resolves exactly to the family public store;
- `GPU_AGENT_EVALUATOR_ROOT/runs` resolves exactly to the family evaluator store;
- public store, evaluator root/store, controller root, signer root, checkout, and external private
  source root obey the existing non-overlap rules;
- both stores reproduce the device/inode/visibility recorded by the family;
- development uses the public service and public corpus;
- holdout uses the evaluator service and evaluator corpus while retaining a public coordinator.

The runbook will provision the evaluator store as the `runs` child of an owner-only evaluator
root and will export the parent as `GPU_AGENT_EVALUATOR_ROOT`. The CLI rejects the previous
ambiguous layout before creating alias, schedule, evaluation, GPU, or provider artifacts.

## Data Flow

### Development

1. Public coordinator verifies the family, reviewed pricing, provider policy, and signed schedule.
2. Public service executes diagnosis and verification against a public registered case.
3. Native `EvaluationRecord` is validated and its public form is stored under the public parent.

### Holdout

1. Public coordinator creates the evaluation parent and signed blind-alias schedule.
2. Public attempt and claim are persisted for the next ordinal.
3. `HoldoutController` validates the proof and resolves the alias only in evaluator context.
4. It reserves or reloads the deterministic evaluator execution transaction for that ordinal.
5. Evaluator service reads the external private source and executes diagnosis, provider calls, a
   candidate, and verification entirely in the evaluator store.
6. Evaluator controller validates native artifacts, persists the full private record and lineage,
   and derives a blind public projection with a public record ID and content commitments.
7. Public runner verifies the projection against the signed item and evaluator-owned projection
   hash, then persists only the projection.
8. External adjudication and holdout scoring join the public record to the evaluator transaction
   without publishing the private side.

## Recovery and Idempotency

- A public attempt remains the authority that a unit started and reserves the maximum unit cost.
- An evaluator execution transaction is deterministic for that attempt. Recovery verifies exact
  bytes and resumes only states explicitly supported by the existing diagnosis/provider recovery
  contracts.
- A completed evaluator execution is never rerun. Its blind projection must reproduce byte for
  byte; otherwise recovery fails closed.
- A provider invocation in `STARTED` state remains ambiguous and causes the existing stopped or
  failed outcome. Recovery must never issue a second paid request under a new key.
- The public record is written only after the evaluator transaction is terminal. If evaluator
  completion exists but public persistence was interrupted, recovery revalidates and writes the
  same projection without another GPU or provider call.
- Any cross-store hash, origin, binding, ordinal, cutoff, or schedule mismatch terminates the
  evaluation without advancing to the next unit.

## Cost and Provider Boundary

The existing total and per-unit caps remain signed schedule inputs. Mode E is still the only mode
allowed to invoke the provider. Pricing and model configuration remain bound to the same clean
commit, prompt version, reviewed rate card, and provider endpoint/model.

For holdout, request/response and invocation state are evaluator artifacts. The public projection
may include numeric token counts, latency, physical-call count, and derived USD cost, because
those are necessary for budget enforcement and aggregate reporting; it may not contain prompt,
response, source, patch content, provider artifact IDs, or evaluator paths.

## Failure Handling

- Configuration disagreement: fail before family mutation, schedule signing, GPU, or provider.
- Private-source or identity detected in a public artifact: fail the affected unit and the release
  privacy gate; never redact after publication and continue.
- Evaluator execution incomplete: retain evaluator evidence, keep the public attempt reservation,
  and apply the existing ambiguous-attempt stop rule.
- Projection mismatch: fail closed; do not regenerate a different record from the same execution.
- Public persistence interruption after evaluator completion: recover from the exact evaluator
  binding without re-execution.
- Development regression: fail the targeted regression tests; its public execution path must not
  be routed through the evaluator worker.

## Verification Strategy

Implementation is test-driven. The first tests must fail on `f5ce2c0` for the observed reason.

1. **Leak reproduction:** execute one synthetic holdout unit containing unique canaries in source,
   case/template/operator identity, provider payload, evaluator path, labels, and nonce. Assert the
   source and all private canaries exist only in evaluator storage and are absent byte-for-byte
   from the entire public store, CLI output, report, and public record.
2. **Development preservation:** execute the corresponding development unit and prove its source
   and native evidence remain public with unchanged record semantics.
3. **Cross-store lineage:** tamper each origin, ordinal, schedule hash, attempt hash, cutoff,
   projection hash, private identity binding, and evaluator run state; every variant must fail.
4. **Recovery:** interrupt after evaluator completion but before public record persistence. Resume
   and prove no second GPU/provider call occurs and the same public bytes are installed.
5. **Paid-call ambiguity:** leave an evaluator provider invocation in `STARTED`; resume must stop
   without a replacement request.
6. **Configuration:** mismatched `GPU_AGENT_RUN_ROOT`, evaluator parent, family store, symlink,
   overlap, or visibility must fail before any run/store mutation.
7. **Existing privacy suite:** retain all alias/label/Oracle canary protections and add full private
   source/provider artifact scanning.
8. **Affected regression:** run focused service, agent, verification, evaluation, holdout,
   persistence, schedule-authority, CLI, release-gate, and private-visibility suites, followed by
   the complete offline suite, Ruff, formatting, mypy, and build checks.

After the repaired commit is reviewed and clean, provision a new production family with the exact
store layout, rerun the required 16 public and 8 private native GPU registrations on that commit,
and only then execute the signed 240 development plus 120 holdout evaluation units. Previously
successful units bound to `f5ce2c0` are historical evidence and cannot satisfy the repaired
release gate.

## Acceptance Criteria

- One clean reviewed commit implements the split-store execution boundary.
- Public store byte scan contains no holdout source or private canary across all five modes.
- Evaluator store contains complete native diagnosis/provider/verification lineage for every
  holdout ordinal.
- Public records remain alias-only and validate against signed schedule and evaluator projection
  hashes without evaluator paths or identities.
- Development evaluation remains behaviorally compatible.
- Exact store configuration fails closed before side effects.
- Resume never duplicates a completed GPU unit or an ambiguous/terminal paid invocation.
- Focused and full zero-cost verification pass before live evidence is regenerated.
- New-commit 16+8 corpus evidence, 240+120 evaluation evidence, scoring, release-test collection,
  frozen selection, derived manifest, and final release check all pass before V2 is called
  publishable.
