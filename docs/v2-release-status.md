# V2 release status

This document separates implemented code, verified native evidence, and remaining release
work. A passing unit test or a configured command is not counted as a live GPU/model result.

## Latest recorded baseline: e80ce75

The 16 public + eight private registrations and signed 240+120 evaluations are complete on
e80ce75. All 120 holdout native chains and the staged-input hashes were checked; the failed
initial holdout remains preserved. The frozen 18-test GPU/isolation suite passed as supplemental
evidence. See [results](evaluation-report.md) and [closeout checks](e80ce75-closeout-CN.md).
Later development changes are not automatically covered by that frozen experiment.

## Verified core

- T01–T02: Python 3.11/CUDA 12.8 environment and trusted clean-kernel evidence.
- T03–T04: isolated Docker GPU execution, memcheck evidence, scoped patching, trusted
  public/private Oracle, and fail-closed verdicts.
- T05–T06: versioned official-document retrieval and one recorded DeepSeek diagnose → patch
  → strict verify path.
- T07: real clean/fault pairs for memcheck, racecheck, initcheck, and synccheck.
- T09: strict four-tool verification with public and private inputs.
- T10 foundation: native validation, crash-safe corpus ledger, registration gates, and public
  seed batch evidence. The e80ce75 family contains all 16 public and eight private registrations.

## Development evidence is not the final release evidence

- The 9d75699 development evaluation completed 240 units. Its failures remain part of
  that baseline; selected regression successes from later code versions are not merged
  into its scores.
- At f81052a the remaining case_0006/C regression passed native verification, while
  case_0015/E still failed. The v7 attempt at 0a56f08 diagnosed that unit but rejected both
  generated patches under the new caller-mask source contract (8 calls / 24,443 tokens).
- The subsequent v8 implementation adds controller-computed counterexamples to the existing
  single patch retry. It does not supply a reference fix or add another repair loop.
- At cdd629f, case_0015/E/repeat=1 passed both standard and full/strict native verification:
  all four Sanitizers CLEAN, VERIFIED_FIXED. Diagnosis run 06b852e8d5a7423397d49fd66a3d4927;
  strict verification run 3889794668a3aa5afeb1f919efc2fffa. The model produced the correct
  candidate on its first patch call, so this run does not demonstrate the causal benefit of
  counterexample feedback. Eight calls / 23,579 tokens; no previously successful unit rerun.
- The current retrieval corpus is 2026-09-24.2 (79 unchanged verified chunks, cuda-lex-v3).
  The five newly added queries now have chunk-level labels and achieve 5/5; overall lexical
  retrieval remains 17/29, not a claim of perfect retrieval. Annotation provenance is mixed.
- See [the sync/retrieval repair record](sync-participation-review-CN.md) for code boundaries,
  historical disagreements, tests, and native regression evidence.

## Implemented release code awaiting final live evidence

- Deterministic lexical, vector-cosine semantic, and hybrid retrieval comparison over 29
  public development labels. Lexical remains the default; labels are never ranking inputs.
- External Ed25519 schedule-authority client and production family public-key provisioning.
- Reviewed pricing attestation bound to repository commit, provider, model, prompt, and source
  content hash.
- Native A–E evaluation scheduling, cost reservations, holdout aliases, blind views, metrics,
  and evaluator-private score bindings.
- Split-store execution keeps holdout source/model/candidate/verification evidence in the exact
  evaluator family store while the public coordinator retains only blind commitments. Production
  family schema v3 pins resolved path, device, inode, and visibility; older families are rejected
  without migration and cannot supply final evidence.
- Evidence-derived release gate that validates ledger membership, 16+8 counts, sanitizer
  coverage, private diversity, 240 development + 120 holdout units, same-commit/config/cutoff
  lineage, live acceptance selections, and release-test evidence.
- Resumable evaluator-only scoring for one exact 120-record label package, canonical selection
  freezing from four explicit roots, and no-replace publication outside the checkout/RunStores.
- A production operator sequence for the external Ed25519 signer, family provisioning, 16+8
  registration, authorized record-only provider use, 240+120 execution, scoring, freezing, and release.
- Python 3.11/3.12 zero-cost CI, MIT distribution metadata, and verified sdist/wheel resources.

The detailed sequence is [the V2 production evidence operator runbook](v2-operator-runbook.md).
The repository contains only the external signer client and Ed25519 verification path; it ships
no production signer or private key.

## Remaining formal evidence requirements

- For any proposed newer evaluated version, evidence must match that commit. The completed
  e80ce75 corpus/evaluations are retained as a baseline, not a reason to rerun them implicitly.
- Freeze the clean repository commit, corpus cutoff, toolchain, prompt, model configuration,
  reviewed price source for accounting, record-only cost policy, and signed development/holdout
  schedules. There are no dollar caps or balance-based stops.
- Retain all 360 completed e80ce75 A–E unit outcomes, including failures, timeouts,
  inconclusive results, provider usage, latency, and nullable cost.
- Supply the canonical 120-record evaluator label package, complete blind scoring/private score
  bindings, and generate the real evaluation report.
- Capture a zero-skip, zero-failure release-test run on the same commit/config/cutoff.
- Generate the selection and manifest under a controller-owned directory outside the Git
  checkout, set `GPU_AGENT_RELEASE_SELECTION` and `GPU_AGENT_RELEASE_MANIFEST` to those
  absolute paths, derive the evidence index, and pass `gpu-agent release check` from a clean
  checkout. Repository-local release artifacts are rejected because they would invalidate the
  repository snapshot they claim to bind.

The final gate uses the same controller-owned external paths both as explicit CLI inputs and as
the release-test environment. First freeze the selection from the four reviewed roots, then derive
and check the manifest:

```bash
export GPU_AGENT_RELEASE_SELECTION=/srv/gpu-agent-controller/release-v2/release-selection.json
export GPU_AGENT_RELEASE_MANIFEST=/srv/gpu-agent-controller/release-v2/release-manifest.json

gpu-agent release freeze-selection \
  --development-evaluation-run-id DEVELOPMENT_EVALUATION_RUN_ID \
  --holdout-evaluation-run-id HOLDOUT_EVALUATION_RUN_ID \
  --private-binding-run-id PRIVATE_ALIAS_MAPPING_RUN_ID \
  --release-test-run-id RELEASE_TEST_RUN_ID \
  --output /srv/gpu-agent-controller/release-v2/release-selection.json \
  --repository /opt/releases/agentic-gpu-debugger

umask 077
set -o noclobber
gpu-agent release derive-manifest \
  --selection /srv/gpu-agent-controller/release-v2/release-selection.json \
  --repository /opt/releases/agentic-gpu-debugger \
  >/srv/gpu-agent-controller/release-v2/release-manifest.json
set +o noclobber

gpu-agent release check \
  --selection /srv/gpu-agent-controller/release-v2/release-selection.json \
  --manifest /srv/gpu-agent-controller/release-v2/release-manifest.json \
  --repository /opt/releases/agentic-gpu-debugger
```

Until real native evidence closes every blocker and the final check reports `passed: true`, V2 is
not a final Portfolio Release. Documentation, synthetic tests, or built distributions do not open
that gate.
