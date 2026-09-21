# V2 release status

This document separates implemented code, verified native evidence, and remaining release
work. A passing unit test or a configured command is not counted as a live GPU/model result.

## Verified core

- T01–T02: Python 3.11/CUDA 12.8 environment and trusted clean-kernel evidence.
- T03–T04: isolated Docker GPU execution, memcheck evidence, scoped patching, trusted
  public/private Oracle, and fail-closed verdicts.
- T05–T06: versioned official-document retrieval and one recorded DeepSeek diagnose → patch
  → strict verify path.
- T07: real clean/fault pairs for memcheck, racecheck, initcheck, and synccheck.
- T09: strict four-tool verification with public and private inputs.
- T10 foundation: native validation, crash-safe corpus ledger, registration gates, and public
  seed batch evidence. The current final family does not yet contain the required membership.

## Implemented release code awaiting final live evidence

- Deterministic lexical, vector-cosine semantic, and hybrid retrieval comparison over 24
  public development labels. Lexical remains the default because it scored highest on this
  frozen development set.
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
  registration, explicit budget approval, 240+120 execution, scoring, freezing, and release.
- Python 3.11/3.12 zero-cost CI, MIT distribution metadata, and verified sdist/wheel resources.

The detailed sequence is [the V2 production evidence operator runbook](v2-operator-runbook.md).
The repository contains only the external signer client and Ed25519 verification path; it ships
no production signer or private key.

## Hard blockers before Portfolio Release

- Freeze and real-GPU validate/register 16 public cases, at least four per Sanitizer family.
- Create, validate, and register eight evaluator-only holdout cases using distinct private
  template/operator identities; no private source or identity may enter Git/public output.
- Record at least one final-commit multi-step Agent investigation satisfying T08.
- Freeze the clean repository commit, corpus cutoff, toolchain, prompt, model configuration,
  reviewed price source, total cost cap, unit cap, and signed development/holdout schedules.
- Execute all 360 A–E units serially, retaining failures, timeouts, inconclusive results,
  provider usage, latency, and cost.
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
