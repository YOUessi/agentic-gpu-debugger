# 2026-11-24 — targeted live recovery results

Frozen implementation: `1e825aa43d51eccf05c8b5953d5705d0554e1c65`. Precommitted plan: [targeted recovery pilot](2026-11-24-targeted-recovery-pilot-plan.md). Actual plan JSON SHA256 `280f88e597275749306e400f891d9e9fb527b4272b66c8f5d152d648c94c1343`. All four units completed; every result, public tool ledger, and provider cap was rechecked. No cherry-picking, no new Planner experiment, no extended AST, and no private verifier.

## Configuration integration

The recovery script accepts build/numeric failures from hash-pinned prior action ledgers or single-step reports, and an autonomous from-mutant mode. It records immutable experiment config, operator prefix vs live model turns, the actual handoff capabilities/feedback, actual provider output caps, and partial outcomes on provider errors. The runner receives the same explicit provider settings as the real wrapper, rather than defaulting its journal to 4096. Diagnostic feedback is integrated as its own opt-in flag; the previous AST editor and verifier remain unchanged. 124 focused tests, 123 strict-mypy files, Ruff lint/format passed on the frozen code. HardSet V2 native-contract CI passed; full CI was cancelled for the implementation SHA after the plan documentation update.

Qwen3-8B-AWQ, local vLLM server seed 43, 8192 output tokens, 16 total tool steps, 18 physical-call ceiling, 600 s wall, 180 s request timeout, 6 compile-only admissions, 2 full checks. Operator prefixes consume tool steps but are excluded from live decision metrics. One trial per unit; these are descriptive results, not population estimates. C leaves compile gate and revert off to isolate the enlarged output cap.

## Results (live-created candidates only)

| Unit | Prefix steps | Live steps | Calls | Tokens | Accepted live edit | Build passed | Numeric dispatch | Full public pass | Finish | Terminal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A-build-h001 | 3 | 13 | 13 | 76794 | 1 | 0 | 0 | 0 | no | STEP_BUDGET_EXHAUSTED |
| A-build-h003 | 3 | 7 | 7 | 35809 | 1 | 1 | 1 | 1 | yes | WORKSPACE_DIFF_STAGED_UNVERIFIED |
| B-numeric-h001 | 5 | 1 | 2 | 15209 | 0 | 0 | 0 | 0 | no | LLM_INCOMPLETE_MAX_OUTPUT_TOKENS |
| C-from-mutant-h002 | 0 | 16 | 16 | 73725 | 1 | 0 | 0 | 0 | no | STEP_BUDGET_EXHAUSTED |

A: 2/2 model-selected and accepted rollbacks; 2/2 produced a new accepted AST edit; 1/2 compiled, dispatched, passed public checks, and finished. B: 1/1 reproduced numeric failure with active diagnostics, 0 live revised candidates; truncated at 8192 (not step-budget exhaustion). C: full 16-step loop, first action completed, first AST edit at turn 3, full checker admitted at turn 4 but build failed before numeric dispatch. No new strict-fix claim. Total measured cost: 38 physical model calls, 201537 tokens, zero format retries. Do not combine the three heterogeneous objectives into a single repair-rate estimate.

### A-h001

Operator turns 1–3 reproduce exact old compile-failed candidate. Live turn 4 rollback restores original SHA. Turn 5 before probe at 129, turn 6 read file, turns 7–9 identical 129 probes rejected; turn 10 Inspect; turn 11 NOOP; turn 12 new accepted candidate; turn 13 real compile fails (line 16, j is not a template); turn 14 Inspect; turns 15–16 NOOP. The model again uses the wrong AST locations (n07 loop initialization, n08 declaration), inserting the Softmax corrections into the wrong scope. No full public checker admission. Revert was successful, repair failed.

### A-h003

Operator turns 1–3 reproduce const assignment build failure. Live turns 4–10: revert OK, Inspect OK, AST edit OK, compile OK, diff OK, full checker OK, finish OK. New candidate SHA `992a461c7f14930797781d15be985706ce5eb4a57215f82a98c73c4194861c2e`. Actual edit n06: `total += a[start + j] + b[start + j];`; n03: removes const from valid and adds the guard `if (valid == 0) valid = 1;`. Compile child `34e1908577444e71a4bd384d6d4f90e8`, checker child `4324fb4c87e443ee86ec685beff6ce6f`; the exact candidate is bound to compile and checker evidence. No rejected actions, no incomplete, 7 live model calls. This is the first live conditional compile-failure recovery in this project that reaches public pass and Finish. Not independent verification.

### B-h001

Operator turns 1–5 reproduce the exact numerator-only candidate, compile it, then execute the full checker. Failure: FINITE_NUMERIC_MISMATCH, size 32, group 0, finite/nonzero outputs, not tail, four sanitizer tools CLEAN with numeric_output_matches=false. Handoff contains these results and no revert capability. Live turn 6 after probe; next physical model decision terminated with LLM_INCOMPLETE_MAX_OUTPUT_TOKENS. Output 8192, reasoning 8191, input 2945, partial output chars 0, service explicitly max_output_tokens. No new AST edit, no second candidate build or numeric dispatch. Initial prefix compile/dispatch are not credited to the model. This is a generation-budget failure, not evidence that public diagnostics are ineffective.

### C-h002

First response completed: input 1827, output 718, reasoning 627. Turn 2 Inspect, turn 3 accepted AST edit of n08 using `const float scale = rsqrtf(sum_sq / valid + 1.0e-5f);`; replaced x's declaration rather than original scale node. Turn 4 full checker fails compilation at line 18 (x undefined). Turn 5 after probe, turns 6–8 repeat same probe and are rejected; turn 9 Inspect; turns 10–16 repeated NOOP and repeated-rejected edits, no second accepted candidate. 16 completed live model calls, no incomplete. The output problem is resolved for this sampled trajectory, but target selection and failure recovery remain deficient.

## Evidence

| Unit | Public run ID | Raw controlled-recovery.json SHA256 |
|---|---|---|
| A-build-h001 | 2aa5d88af2534066afc2ce021f6927bf | 85a224cc2a0c518e1505236ed10f37c05dfb0fbc5309c7664273841ae3df17d4 |
| A-build-h003 | d0b379c4fa064a37b10318459a1d21d4 | 8f06b8892087f6ce8ad58d4d652ed049b6a408747d1fb46e46c2be728de4015e |
| B-numeric-h001 | 06889510d9b442f9bc0b000a6d595fb9 | e335c96c3bf0493ef776a4e5c571e21df74e2b6739970675c15cbe9e0d44b42e |
| C-from-mutant-h002 | 8f7b61fb19c443d4993c329f70428763 | 91567f28e634c9796b0b1561118a67a7e48368b1951f1e37ec7fae24ec2ff0f4 |

See [machine-readable stage results](2026-11-24-targeted-recovery-pilot-results.json). Raw artifacts on Tang under `/home/you/gpu-native-harness-validation-20261124/p3-targeted-live-recovery/`. Actual journal, child-check provenance and provider telemetry were re-read after all four units ended. GPU 204 MiB used, 0% utilization, shared lock free.

## Next gate

Do not start Planner-only or broad factorial experiments yet. The build-recovery chain now has one genuine end-to-end public-success sample. Prioritize a controlled replay of B's exact failed second-decision input to test a reasoning-budget intervention (hold cap/model/payload fixed; do not silently auto-retry). Next isolate state-scoped Probe parameter filtering and AST target binding/selection: A-h001 and C-h002 each made three redundant probes, and both repeatedly submitted NOOP edits. Report wrong-target compile errors independently of accepted AST edits. Do not attribute the single h003 success solely to rollback, or claim increased general repair reliability without repetitions and matched controls.
