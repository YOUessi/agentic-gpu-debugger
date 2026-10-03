# Task 4 implementation report

Base: `eb7adbb` — `fix: make holdout authorization self-validating`.

## Outcome

- Corpus-family schema v3 preserves the public/evaluator path strings and additionally pins each
  store's resolved path, device, inode, and visibility. Each side's marker contains its exact pin.
- `CorpusFamily.open()` parses and validates schema/store identity before opening the ledger, and
  opens the ledger in existing-only mode. It does not create/chmod a ledger, init lock, key, or
  store while rejecting an invalid or legacy family.
- Production configuration requires exact absolute `GPU_AGENT_RUN_ROOT` and
  `GPU_AGENT_EVALUATOR_ROOT/runs` identities, owner-only real directories, and non-overlap with
  the controller and repository before services, provider availability, alias preparation,
  schedules, or runs.
- Release service construction no longer reaches `ApplicationService.configured()` for evaluation.
  It receives the already validated family stores: `(public, evaluator)` for the coordinator and
  `(evaluator, evaluator)` for the holdout worker.
- The production holdout factory now passes the complete Task 3 service/controller/batch triple.
  Invalid mode/split values are rejected before configured runner construction.
- The runbook now provisions the evaluator store as the `runs` child of the exported evaluator
  root and explicitly requires a new family instead of migrating prior evidence.

## RED evidence

The initial focused tests failed in all eight intended ways:

```text
8 failed, 48 deselected in 0.65s
```

The failures showed the missing shared validator and workflow-visibility factory, and proved that
the former schema accepted both a copied marker on a replacement inode and an old schema without
rejection.

## GREEN evidence

Focused Task 4 selector after the final changes:

```text
29 passed, 77 deselected in 1.15s
```

The production CLI adversarial subset includes wrong public path, wrong evaluator parent,
symlinked evaluator root, replaced store inode, and wrong pinned visibility. Every case exits with
the stable pre-cost error, leaves the complete directory tree byte-identical, and records zero
provider availability calls. The unit boundary additionally covers overlap and unsafe permissions.

Static verification:

```text
ruff: All checks passed!
mypy: Success: no issues found in 4 source files
git diff --check: exit 0, no output
```

## Scope and remaining work

- No full offline suite, GPU/container command, network request, DeepSeek/provider call, corpus
  regeneration, push, or publication was run.
- Task 5 still owns evaluator commitment resolution, scoring/release consumer migration, the one
  complete offline regression, and final review remediation.
