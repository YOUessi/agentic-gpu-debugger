# V2 production evidence operator runbook

This runbook is for the trusted release operator. It describes the production sequence; it is
not evidence that the sequence has been run. The V2 release stays closed until the real 16+8
corpus, 240+120 signed evaluations, evaluator scoring, fixed release suite, and final release
check all pass on one clean commit and one corpus cutoff.

The repository ships no production schedule signer and no production private key. Production
signing belongs to an independently controlled executable. Never place its private key in this
checkout, either RunStore, a build artifact, CI variables available to untrusted jobs, or release
output.

## 1. Freeze the commit and environment

Use a fresh checkout at an absolute path. Stop if it is dirty, if `HEAD` is not the reviewed
commit, or if the locked environment is not healthy.

```bash
cd /opt/releases/agentic-gpu-debugger
test -z "$(git status --porcelain)"
FINAL_COMMIT="$(git rev-parse HEAD)"
test "$(git rev-parse --show-toplevel)" = /opt/releases/agentic-gpu-debugger
/home/release/conda_env/agentic-gpu-debugger/bin/python --version
/home/release/conda_env/agentic-gpu-debugger/bin/python -I -m pip check
```

Record `FINAL_COMMIT`, the reviewed `containers/toolchain.lock.json` hash, prompt version, model
configuration, and the intended provider in the controller change record before doing any live
work. Do not amend or rebuild the commit after evidence collection starts.

## 2. Provision the external Ed25519 signer

Provision an Ed25519 key pair outside the checkout and both RunStores using the organization's
key-management procedure. Put only the reviewed public key at
`/srv/gpu-agent-controller/keys/schedule-authority.pub`. Keep the private key behind the external
signer; this project has no production key-generation or signing command.

The executable configured below must be an absolute, non-symlink path to a regular file owned by
the operator, executable by the owner, and not writable by group or other users:

```bash
export GPU_AGENT_SCHEDULE_AUTHORITY_COMMAND=/srv/gpu-agent-signer/bin/schedule-authority
test -x /srv/gpu-agent-signer/bin/schedule-authority
```

For each invocation, the application sends exactly one
`EvaluationScheduleSigningRequest` JSON object on stdin. It has `schema_version: 4` and binds the
transaction and evaluation IDs, target store, schedule and immutable `RunBinding`, selection,
modes, split, repeats, random seed, record-only cost policy, case/template universe, corpus namespace
and cutoff, authority key hash, and pristine queued-run hashes. The signer must return exactly one
`EvaluationScheduleReceipt` JSON object on stdout with:

```json
{
  "state": "COMMITTED",
  "request": { "schema_version": 4 },
  "algorithm": "Ed25519",
  "payload_hash": "<64 lowercase hex>",
  "signature_hex": "<128 lowercase hex>"
}
```

The returned `request` must be byte-semantically identical to the input model. The payload is the
domain separator `gpu-agent-evaluation-schedule-v3\0` followed by canonical JSON for that request;
`payload_hash` is its SHA-256 and `signature_hex` is the Ed25519 signature. The command must exit
zero, emit no stderr, and finish within the bounded client timeout. The client invokes it without
a shell and with only `PATH=/usr/bin:/bin`, `LANG=C`, and `LC_ALL=C`; it does not pass provider
credentials.

## 3. Provision the production corpus family

The controller root, public store, evaluator store, and checkout must be distinct absolute paths.
Keep the controller and stores owner-only. Provisioning records only the public signing key.

```bash
install -d -m 0700 /srv/gpu-agent-data/v2-public
install -d -m 0700 /srv/gpu-agent-private/v2-evaluator
install -d -m 0700 /srv/gpu-agent-private/v2-evaluator/runs

gpu-agent benchmark provision-family \
  --controller-root /srv/gpu-agent-controller/family-v2 \
  --public-store /srv/gpu-agent-data/v2-public \
  --evaluator-store /srv/gpu-agent-private/v2-evaluator/runs \
  --repository /opt/releases/agentic-gpu-debugger \
  --schedule-public-key /srv/gpu-agent-controller/keys/schedule-authority.pub

export GPU_AGENT_CORPUS_FAMILY_ROOT=/srv/gpu-agent-controller/family-v2
export GPU_AGENT_RUN_ROOT=/srv/gpu-agent-data/v2-public
export GPU_AGENT_EVALUATOR_ROOT=/srv/gpu-agent-private/v2-evaluator
```

Reopening that family must reproduce its namespace and store identities. Never substitute ad hoc
store roots on later scoring or release commands. `GPU_AGENT_EVALUATOR_ROOT` names the owner-only
parent; its `runs` child is the exact evaluator store pinned by the family. The previous layout
that pinned the parent itself is invalid. Provision a new family; do not edit or migrate an old
family configuration or reuse evidence registered under it.

## 4. Validate and register exactly 16+8 cases

First produce native clean and mutant validation run IDs through the reviewed GPU controller. For
each of the 16 public cases, register its exact pair in the public store:

```bash
gpu-agent benchmark validate PUBLIC_CLEAN_RUN_ID PUBLIC_MUTANT_RUN_ID \
  --corpus-root /srv/gpu-agent-data/v2-public \
  --visibility public
```

Repeat exactly 16 times and verify at least four public cases cover each Sanitizer family. Then,
from controller-only private inputs under `/srv/gpu-agent-private/cases-v2`, produce and register
eight evaluator-only pairs:

```bash
gpu-agent benchmark validate PRIVATE_CLEAN_RUN_ID PRIVATE_MUTANT_RUN_ID \
  --corpus-root /srv/gpu-agent-private/v2-evaluator/runs \
  --visibility evaluator
```

Repeat exactly eight times. The eight private cases must have distinct private case, template, and
operator identities. Do not copy their sources, identities, validation logs, or an alias map into
the checkout or public store. Before proceeding, reconcile the family ledger to exactly 16 public
and eight evaluator registrations at the intended cutoff.

## 5. Attest pricing and obtain explicit budget authorization

Review the provider's current HTTPS price source and retain its exact bytes in controller storage.
Load provider endpoint/model settings in the trusted shell, then create an owner-only attestation
bound to `FINAL_COMMIT` and the source-content hash:

```bash
export OPENAI_BASE_URL=https://provider.example/v1
export OPENAI_MODEL=reviewed-production-model

gpu-agent benchmark attest-pricing \
  --repository /opt/releases/agentic-gpu-debugger \
  --commit "$FINAL_COMMIT" \
  --input-usd-per-million REVIEWED_INPUT_RATE \
  --output-usd-per-million REVIEWED_OUTPUT_RATE \
  --source-uri https://provider.example/pricing \
  --reviewed-at 2026-09-21T00:00:00 \
  --source-content-hash REVIEWED_SOURCE_SHA256 \
  --output /srv/gpu-agent-controller/attestations/v2-pricing.json

export GPU_AGENT_PRICING_ATTESTATION=/srv/gpu-agent-controller/attestations/v2-pricing.json
```

Real provider calls require user authorization. Do not infer authorization from an attestation,
an earlier run, or available account credit. Record actual token usage and calculable costs only;
there are no per-unit or total dollar limits, balance checks, or cost-triggered stops. Unknown
cost stays null. Load the provider API key only into this trusted controller shell and never
print or persist it. The existing 40-request unit boundary prevents unbounded agent loops.

## 6. Run the signed 240+120 evaluations

Use all five modes and three repeats. The development command must report
`16 case × 5 mode × 3 repeats = 240 units` before execution:

```bash
gpu-agent benchmark evaluate \
  --mode all --split development --repeats 3 \
  --corpus-root /srv/gpu-agent-data/v2-public \
  --case-root /opt/releases/agentic-gpu-debugger/benchmarks/public \
  --repository /opt/releases/agentic-gpu-debugger \
  --commit "$FINAL_COMMIT" \
  --toolchain-hash REVIEWED_TOOLCHAIN_HASH \
  --model-config-hash ATTESTED_MODEL_CONFIG_HASH
```

Record the completed development `run_id`. Then run the private holdout against only the
evaluator-controlled source root. It must report `8 case × 5 mode × 3 repeats = 120 units`:

```bash
gpu-agent benchmark evaluate \
  --mode all --split holdout --repeats 3 \
  --corpus-root /srv/gpu-agent-private/v2-evaluator/runs \
  --case-root /srv/gpu-agent-private/cases-v2 \
  --repository /opt/releases/agentic-gpu-debugger \
  --commit "$FINAL_COMMIT" \
  --toolchain-hash REVIEWED_TOOLCHAIN_HASH \
  --model-config-hash ATTESTED_MODEL_CONFIG_HASH
```

Do not continue on a stopped, failed, unsigned, partial, wrong-commit, wrong-cutoff, or
wrong-configuration run. Preserve all 360 unit outcomes, including failures and inconclusive
records.

## 7. Prepare the 120-record evaluator label package

The external evaluator must adjudicate the blind holdout projection and create one canonical
`HoldoutLabelPackage` with exactly 120 unique judgments. Its header must bind the completed
holdout evaluation, evaluator alias-mapping run, signed schedule, alias hash, corpus cutoff,
record-set hash, and tracked rubric hash. Each judgment binds one blind ID, public-record hash,
blind-payload hash, labels, score, inconclusive decision, and private-holdout decision. It must not
contain private case/template identities or the alias-map nonce.

Place it outside the checkout and both RunStores, in an existing owner-only directory, as a
single-link regular file owned by the operator:

```bash
install -d -m 700 /srv/gpu-agent-controller/adjudication
install -m 600 /srv/external-evaluator/v2-holdout-labels.json \
  /srv/gpu-agent-controller/adjudication/v2-holdout-labels.json
```

Reject symlinks, extra hard links, group/other permissions, partial transfers, and post-review
changes. The evaluator should transfer the file through an independently authenticated channel.

## 8. Score, collect, freeze, derive, and check

Bind the complete label package. The optional metrics copy remains evaluator-controlled and must
also be outside the checkout and both stores:

```bash
gpu-agent benchmark score-holdout \
  --evaluation-run-id HOLDOUT_EVALUATION_RUN_ID \
  --private-binding-run-id PRIVATE_ALIAS_MAPPING_RUN_ID \
  --labels /srv/gpu-agent-controller/adjudication/v2-holdout-labels.json \
  --metrics-output /srv/gpu-agent-controller/adjudication/v2-holdout-metrics.json \
  --repository /opt/releases/agentic-gpu-debugger
```

Require `scored 120/120` and retain the scoring session ID and metrics SHA-256. Then collect the
fixed release suite on the same completed development evaluation:

```bash
gpu-agent release collect-evidence \
  --repository /opt/releases/agentic-gpu-debugger \
  --development-evaluation-run-id DEVELOPMENT_EVALUATION_RUN_ID
```

The release-test run must finish with zero skips and zero failures. Create an owner-only output
directory, then freeze the canonical selection from exactly four roots; this command never scans
for a newer run and never publishes incomplete evidence:

```bash
install -d -m 700 /srv/gpu-agent-controller/release-v2

gpu-agent release freeze-selection \
  --development-evaluation-run-id DEVELOPMENT_EVALUATION_RUN_ID \
  --holdout-evaluation-run-id HOLDOUT_EVALUATION_RUN_ID \
  --private-binding-run-id PRIVATE_ALIAS_MAPPING_RUN_ID \
  --release-test-run-id RELEASE_TEST_RUN_ID \
  --output /srv/gpu-agent-controller/release-v2/release-selection.json \
  --repository /opt/releases/agentic-gpu-debugger
```

Derive the manifest without overwriting an existing file, and check it against native evidence:

```bash
export GPU_AGENT_RELEASE_SELECTION=/srv/gpu-agent-controller/release-v2/release-selection.json
export GPU_AGENT_RELEASE_MANIFEST=/srv/gpu-agent-controller/release-v2/release-manifest.json

umask 077
set -o noclobber
gpu-agent release derive-manifest \
  --selection /srv/gpu-agent-controller/release-v2/release-selection.json \
  --repository /opt/releases/agentic-gpu-debugger \
  > /srv/gpu-agent-controller/release-v2/release-manifest.json
set +o noclobber

gpu-agent release check \
  --selection /srv/gpu-agent-controller/release-v2/release-selection.json \
  --manifest /srv/gpu-agent-controller/release-v2/release-manifest.json \
  --repository /opt/releases/agentic-gpu-debugger
```

Proceed only when the final JSON says `"passed": true` and has no reason codes. A generated file,
a completed command, or a green offline test is not a substitute for this real evidence check.

## 9. Build, hash, push, run CI, tag, and release

From the unchanged clean checkout, run the complete offline regression and static checks required
by the release process, then build both artifacts with the already locked build backend and no
isolated dependency download:

```bash
cd /opt/releases/agentic-gpu-debugger
/home/release/conda_env/agentic-gpu-debugger/bin/python -m build --no-isolation
sha256sum /opt/releases/agentic-gpu-debugger/dist/*.whl \
  /opt/releases/agentic-gpu-debugger/dist/*.tar.gz \
  > /srv/gpu-agent-controller/release-v2/distribution-sha256.txt
```

Inspect the wheel and sdist contents, verify the hashes from an independent release host, and
install-smoke the wheel outside the checkout. Then perform these state-changing steps in order,
using the repository's protected release procedure:

1. Push the reviewed commit/branch.
2. Wait for required CI on that exact commit; CI is offline evidence only.
3. Create the approved signed version tag on `FINAL_COMMIT` and push the tag.
4. Wait for tag/release CI and recheck artifact hashes.
5. Publish the immutable release and attach the verified wheel, sdist, hashes, and evidence
   summary without evaluator-private data.

Do not push, tag, or publish if the checkout moved, the real release check closed again, an
artifact hash differs, or any private key, label, score, identity, alias, nonce, or evaluator path
appears in Git-tracked bytes or public release output.
