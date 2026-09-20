# V2 Holdout Scoring and Release Selection Design

## Status and intent

This design closes the last missing controller operations between a completed signed
development/holdout evaluation and the existing evidence-derived release gate. It adds two
auditable workflows:

1. bind one evaluator-owned adjudication package to every record in the 120-unit holdout run,
   persist grouped metrics, and support exact crash recovery;
2. derive one canonical `ReleaseEvidenceSelection` from four explicit native roots and publish
   it atomically outside the Git checkout.

The target user is the trusted benchmark operator. Success means the operator no longer has to
call `HoldoutController.bind_score()` 120 times or hand-author release-selection JSON, while all
existing public/evaluator, provenance, cost, signing, and release-gate boundaries remain intact.

This design does not execute GPU work or model calls, create the eight private cases, approve an
API budget, or place an Ed25519 private key in this repository.

## Chosen approach

The implementation will add controller services behind two CLI commands:

```text
gpu-agent benchmark score-holdout \
  --evaluation-run-id <32hex> \
  --private-binding-run-id <32hex> \
  --labels /absolute/controller/holdout-labels.json \
  --repository /absolute/clean/checkout \
  [--metrics-output /absolute/controller/holdout-metrics.json]

gpu-agent release freeze-selection \
  --development-evaluation-run-id <32hex> \
  --holdout-evaluation-run-id <32hex> \
  --private-binding-run-id <32hex> \
  --release-test-run-id <32hex> \
  --output /absolute/controller/release-selection.json \
  --repository /absolute/clean/checkout
```

Both commands derive public/evaluator stores exclusively from
`GPU_AGENT_CORPUS_FAMILY_ROOT`. Callers cannot substitute arbitrary store roots. Both commands
are zero-provider operations.

Two alternatives are rejected:

- A shell script that loops over `bind_score()` and writes selection JSON is shorter, but cannot
  prevent two different label packages from interleaving or prove exact coverage.
- Automatic scoring inside `EvaluationRunner` couples blinded execution to hidden truth and
  weakens the evaluator boundary. Scoring remains a separate post-evaluation operation.

## Shared controller artifact boundary

A small shared module will provide bounded private-file reads and no-replace atomic writes for
operator artifacts. It will be used by both workflows and by the existing external release-path
validation.

Input files must be absolute and outside:

- the bound Git checkout;
- the public RunStore;
- the evaluator RunStore.

The reader rejects missing files, symlinks, non-regular files, files not owned by the current OS
user, group/other permissions, multiple hard links, identity changes between path and opened file
descriptor, and content over the command-specific size limit.

Output parents must already exist, be owned by the current user, have no group/other permissions,
and be opened with `O_DIRECTORY | O_NOFOLLOW`. Publication uses a `0600` single-link temporary
file in the same directory, `fsync`, and Linux `renameat2(RENAME_NOREPLACE)` with directory file
descriptors, followed by directory `fsync`. The command fails closed when that no-replace primitive
is unavailable; it does not fall back to an overwrite-capable rename or a two-link publication
window. A crash before rename may leave an owner-only dot-prefixed temporary file but never a
published partial target. If the target already contains byte-identical canonical content, retry
succeeds idempotently. A different existing target is never overwritten or removed.

These checks protect against ordinary path substitution, concurrent operators, partial writes,
and uncertain process termination. They do not claim to isolate data from a malicious process
running as the same OS user with direct write access to controller storage.

## Holdout label package

The external evaluator supplies one canonical JSON package. The package carries no private case
or template identifiers and no alias-map nonce.

```python
class HoldoutJudgment(ExecutionModel):
    blind_id: str                 # 64 lowercase hex
    public_record_hash: str       # 64 lowercase hex
    blind_payload_hash: str       # 64 lowercase hex
    labels: EvaluationLabels
    score: Score
    should_be_inconclusive: bool
    private_holdout_passed: bool


class HoldoutLabelPackage(ExecutionModel):
    schema_version: Literal[1] = 1
    evaluation_run_id: str
    private_binding_run_id: str
    schedule_hash: str
    aliases_hash: str
    corpus_cutoff: int
    expected_record_count: Literal[120] = 120
    record_set_hash: str
    rubric_hash: str
    judgments: list[HoldoutJudgment]  # exactly 120
```

`blind_id` is the existing SHA-256 identifier returned by
`PublicEvaluationRecord.blind()`. `blind_payload_hash` is SHA-256 over canonical JSON for that
blind projection. `record_set_hash` is SHA-256 over the canonical, schedule-ordinal-ordered list
of `(ordinal, public_record_hash, blind_payload_hash)` tuples. `rubric_hash` must match the bytes
of the tracked `evaluation/rubric.md` in the clean repository bound to the evaluation run.

`Score` is an evaluator judgment, not a model self-assessment. This workflow proves that the
judgment is bound immutably to the exact blinded payload and public record; it does not claim
that schema validation proves the evaluator's judgment is substantively correct.

## Holdout scoring preflight

`HoldoutScoringController.score()` performs every non-mutating validation before creating a
scoring session or score child:

1. Capture a clean repository snapshot and require the evaluation binding's exact commit.
2. Open the configured production family and require public/evaluator visibility and family
   membership.
3. Load the public evaluation root, require `COMPLETED`, purpose `evaluation`, and verify its
   external schedule receipt.
4. Require `split=holdout`, `selection=all`, exactly modes A-E, three repeats, eight unique
   aliases, and the complete 8 × 5 × 3 Cartesian product.
5. Require exactly 120 attempts and record artifacts with canonical ordinals, unique record IDs,
   a complete terminal manifest, no stopped reason, and native evaluation-record validation.
6. Reconstruct the `HoldoutBatch` from the explicitly supplied evaluator mapping run and its
   public alias origin. Require its binding, cutoff, alias payload hash, schedule proof, nonce
   recomputation, and private corpus identities to match.
7. Require package header fields to equal the verified evaluation, mapping, schedule, cutoff,
   alias hash, record-set hash, and tracked rubric hash.
8. Require 120 unique judgments whose blind-ID set exactly equals the record blind-ID set, and
   require each public-record and blind-payload hash to match.
9. Under an fd-scoped mapping-child inventory, inspect every pre-existing `holdout_score` child.
   Its deterministic run ID, record binding, private identity, and canonical private-score bytes
   must exactly equal the corresponding judgment in this package. Unexpected child kinds,
   conflicting scores, duplicate records, or more children than records fail before any session
   or score write. A verified exact subset is eligible for same-package recovery.
10. Recapture the repository and require the same snapshot before any evaluator write.

Missing, duplicate, extra, failed, stopped, mismatched, or ambiguous input fails before the first
write. No partial package is accepted.

## Scoring session, idempotency, and metrics

Preflight is followed by one deterministic evaluator-only session:

```text
scoring_run_id =
  sha256("holdout-scoring-v1:<evaluation-run-id>:<private-binding-run-id>")[:32]
kind = holdout_scoring
external_origin = the public evaluation run
binding = the evaluation binding
```

The session is top-level in the evaluator store, not a child of the alias mapping. The existing
release gate requires every direct mapping child to be a `holdout_score` and would correctly
reject a session child of another kind.

A session-wide file lock serializes the complete scoring operation. Before the first per-record
write, the session persists `holdout-scoring/input-binding.json`, containing the two root IDs,
schedule/alias/cutoff bindings, record-set hash, rubric hash, and canonical package hash. A retry
must match that artifact exactly. A different package for the same evaluation/mapping pair fails
before it can mix decisions with existing scores.

Judgments are processed in schedule ordinal order. Each entry calls the existing
`HoldoutController.bind_score()` with the verified record reference and the alias read from that
record. The preflight-verified exact subset is reloaded rather than rewritten; remaining entries
use existing deterministic score IDs, per-record locking, `put_if_absent_exact()`, and score
reload validation. A crash leaves the session `RUNNING`; the same package revalidates all existing
children and fills the remainder. A conflicting partial set never gains a session input binding.

After 120 bindings reload successfully, the controller calls `aggregate_grouped()` with the
persistent bindings and native stores. It stores:

- `holdout-scoring/bindings.json`, ordered by schedule ordinal;
- `holdout-scoring/metrics.json`, containing the full `GroupedMetricSummary`;
- `holdout-scoring/result.json`, containing safe counts and artifact hashes.

Only then does the session become `COMPLETED`. Metrics never exist for an incomplete mixed set.
All three terminal artifacts use `put_if_absent_exact()`. A `COMPLETED` session retry is strictly
read-only: it reloads and exact-verifies the input binding, ordered bindings, metrics, result, and
all score children, then returns the existing result without attempting to add an artifact.
If requested, `--metrics-output` receives the same canonical evaluator-only metrics through the
shared atomic external writer.

Successful CLI output is limited to session ID, `scored 120/120`, and the metrics SHA-256. It
does not print paths, aliases, labels, private identities, or per-record decisions. Full grouped
metrics stay in evaluator-controlled storage because even opaque per-alias results are not part
of the public projection.

## Canonical release selection

`ReleaseEvidenceFreezer` accepts exactly four caller-selected roots:

- completed signed development evaluation;
- completed signed holdout evaluation;
- completed evaluator alias-mapping run;
- completed release-test run.

It never scans for the newest run. From those roots it derives every other selection field:

- `repository`: a fresh clean snapshot matching both evaluation bindings;
- public/private case registration IDs: the two authoritative store targets in ledger commit
  sequence through the evaluations' common cutoff;
- `four_tools`: each validated public case's mutant validation run;
- `isolation`: exactly the supplied release-test run;
- `private_oracle`: all exact, terminal `holdout_score` children bound one-to-one to holdout
  records;
- `live_llm`: all unique Mode-E diagnosis run IDs from development and holdout records;
- the two evaluation IDs and private binding ID: the supplied roots after complete validation.

The resolver also derives the deterministic `holdout_scoring` session ID from the holdout
evaluation and mapping roots. It requires that session to be `COMPLETED` and exact-verifies its
input binding, ordered bindings, metrics/result hashes, and one-to-one agreement with all 120 score
children. A directory containing 120 manually created or mixed-package score children is not
eligible for freezing. The session ID is native derived evidence rather than a fifth caller root;
the derived evidence index records both the mapping and scoring-session IDs under private scoring.

Corpus arrays preserve ledger sequence. Acceptance arrays use sorted run IDs for deterministic
JSON. The acceptance key set is exactly `four_tools`, `isolation`, `private_oracle`, and
`live_llm`. Duplicate IDs within a category or reuse across selection categories are rejected;
the shared resolver first requires the count of Mode-E records to equal the count of unique
diagnosis run IDs and only then sorts those IDs. Both freezer and existing derive behavior are
updated together, so a set can never silently collapse repeated Mode-E lineage.

The freezer and `ReleaseEvidenceIndex.derive()` must share one resolver for schedule, corpus,
mapping, score, release-test, and acceptance validation. There will not be a second weaker
directory-enumeration implementation. The existing derive API continues to return stable reason
codes when checking a supplied selection; the freezer uses the same resolver without a supplied
selection to construct the canonical candidate.

Before publication, the candidate is passed through complete evidence derivation,
`ReleaseManifest.from_evidence()`, and `ReleaseGate.check()`. Any missing 16+8 corpus, private
diversity, four-tool coverage, 240+120 evaluation units, score binding, Mode-E lineage, test
count, or binding prevents selection publication. There is no `--allow-incomplete` option.

The repository is captured again immediately before output and must equal the initial snapshot.
The canonical selection is then written using the shared no-replace atomic writer. An identical
existing selection is an idempotent success; different content at the requested path is
`RELEASE_SELECTION_OUTPUT_CONFLICT`.

The freezer prints only the output path, selection SHA-256, common cutoff, and aggregate run
counts. The selection may contain opaque evaluator run IDs but never private identities, aliases,
nonce, labels, scores, evaluator artifact paths, or store paths.

## Failure and recovery semantics

Stable CLI failures do not echo raw private exceptions or payloads.

Holdout scoring distinguishes invalid package/input, evidence mismatch, existing-content
conflict, and execution failure. A failure before session creation has no store mutation. A
crash after the input binding is durable resumes only with the identical package. An existing
`FAILED` score child or different immutable score artifact is a conflict requiring operator
review, not an automatic overwrite.

Selection freezing is read-only with respect to stores and ledger. Root, evidence, gate,
repository-drift, unsafe-output, and output-conflict failures leave no selection file. A crash
before publication leaves no target; a crash after atomic publication is recovered by exact
content comparison.

Concurrent corpus registrations after the signed cutoff do not enter the selection. Concurrent
unfinished scoring causes the freezer to fail closed; the operator reruns after scoring reaches a
terminal state.

## External signer boundary

Production code continues to contain only the external command client and Ed25519 verification.
The operator runbook will document the JSON stdin/stdout contract, executable ownership/mode
requirements, public-key provisioning, and how to set the signer command. It will not include a
private key, generate a production key, persist a private key in either RunStore, or ship a signer
that silently downgrades the external trust boundary. Tests may continue to use test-only signing
support clearly marked as non-production.

## Files and test strategy

Expected implementation surface:

- new `src/gpu_agent/benchmark/controller_artifacts.py` for secure external artifact I/O;
- new `src/gpu_agent/benchmark/holdout_scoring.py`;
- refactor `src/gpu_agent/benchmark/release.py` around a shared resolver/freezer;
- extend `src/gpu_agent/cli.py` with the two commands;
- add operator instructions under `docs/` and update V2 release status;
- add unit, integration, recovery, privacy-canary, and CLI tests.

TDD coverage must include:

- label package exact coverage, all header/hash mismatches, missing/duplicate/extra judgments,
  wrong schedule/cutoff/mapping/rubric, and malformed/unsafe external files;
- two different packages racing for one session, same-package concurrency, every artifact-boundary
  crash/retry, a pre-existing subset of 1–119 score records, and metrics only after 120/120;
- absence of labels/private identities/canaries from the public store, repository, and CLI output;
- wrong/missing/cross-store release roots, cutoff/config mismatch, corpus ordering, extra/missing
  score children, duplicated Mode-E lineage, and all release-gate insufficiency reasons;
- repository drift during resolution, repository-local/relative/symlink output rejection, `0600`
  atomic publication, byte-identical retry, and different-content no-overwrite;
- round-trip of the frozen selection through the existing derive/check commands.

The final verification set is the complete zero-cost suite, Ruff formatting/checks, strict mypy,
sdist/wheel resource smoke tests, and command help/safe-failure tests. Real 16+8 GPU registration
and the paid 360-unit evaluation begin only after these tracked changes are reviewed, committed,
and frozen.

## Completion criteria

This implementation phase is complete when both commands are present, tested, documented,
fail-closed, and can run against synthetic native stores without GPU/provider access; the release
gate must still truthfully remain closed when real production evidence is absent. The subsequent
execution phase freezes the final commit, provisions a new production family and external signer,
registers 16+8 cases, obtains an explicit user-approved API cap, runs and scores 360 units, freezes
selection/manifest, passes the release gate, builds artifacts, and only then publishes V2.
