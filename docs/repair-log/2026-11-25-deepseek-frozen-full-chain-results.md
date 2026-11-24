# P7 — DeepSeek V4-Pro frozen full-chain confirmation (2026-11-25)

Four units executed in the precommitted order on Tang using the existing recovery CLI and independent verify-repo CLI. All four completed, and all four exact public-passing candidates independently returned VERIFIED_FIXED / ALL_REQUIRED_CHECKS_PASSED. No Harness source changes. Frozen commit `be3dafd6988972dfdb51cf2faee6984103565973`; precommitted [plan and exact-input results](2026-11-24-deepseek-exact-input-comparison.md). The previous compound launch was blocked by tool safety screening; this run used the existing supported CLI directly, with existing project Provider configuration (no keys in command arguments or reports), the shared GPU lock and separate verifier invocations. No screening bypass, new tool, larger budget or model substitution. Final source checkout was clean at the same SHA; GPU 208 MiB, 0% utilization, lock free.

## Configuration

Model requested and returned deepseek-v4-pro, official Responses API, SDK 3.13.0, existing adapter reasoning.effort=none, provider default sampling (no explicit temperature, top_p, seed); all 38 physical invocations reported reasoning_tokens=0. Precision, exact weights revision and serving implementation are not disclosed. This is a model-system comparison with local Qwen3-8B-AWQ default Thinking, **not a pure model-capacity ablation**. 16 total tool steps including operator prefix, 18 physical calls, 8192 output cap, 180s per-request timeout, 600s wall budget, six compile-only admissions, two full checker admissions. Binding and state-scoped actions on. Planner and extended AST off. A compile/revert on, B compile/diagnostics on and revert off, C no compile/revert/diagnostics, identical to P5. Original failures reproduced from the same hash-pinned prior evidence. C from original mutant.

## Results

| Unit | Tool steps (live / total) | Physical calls | Valid physical outputs | Tokens | Agent wall (s) | Strict wall (s) | AST / build / numeric / public | Strict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| A-build-h001 | 11 / 14 | 13 | 11/13 | 61375 | 64.96 | 11.64 | 1 / 1 / 1 / 1 | VERIFIED_FIXED |
| A-build-h003 | 8 / 11 | 8 | 8/8 | 32397 | 42.04 | 11.65 | 1 / 1 / 1 / 1 | VERIFIED_FIXED |
| B-numeric-h001 | 9 / 14 | 10 | 9/10 | 48053 | 61.78 | 12.63 | 1 / 1 / 1 / 1 | VERIFIED_FIXED |
| C-from-mutant-h002 | 6 / 6 | 7 | 6/7 | 23841 | 37.86 | 11.56 | 1 / 1 / 1 / 1 | VERIFIED_FIXED |

38 physical calls, 34 valid structured outputs (89.47% of physical calls), 30/34 first-attempt logical decisions valid (88.24%), all four format retries recovered. No incomplete output. All 34 live tool actions accepted or ran successfully at the controller boundary; probes can return a failing diagnostic without tool rejection. No binding rejects, no wrong-target writes, zero duplicate Probe requests. There were two exact same-source repeat inspections of the operator prefix following rollback, which are necessary to reestablish AST evidence, and zero live-to-live identical requests. Offline audit reconstructed each final source byte-for-byte from original source and observed AST evidence, checked each old-code binding and child dispatch/pass and independently verified strict projection identity. Operator review: five of five accepted live edit items correct target and scope; all four live edit actions yield strict-passing final candidates. Unparsed invalid responses are not included in the target denominator.

Input tokens 161491, output 4175, total 165666, cached input 25344, actual reasoning 0. Agent wall sum 206.64s, strict verifier wall sum 47.48s, excluding scheduling gaps. Actual billed monetary cost unavailable (not inferred from cross-tokenizer counts).

## Mechanism evidence

### A-build-h001

Live edit turn 11:
```diff
- denominator += expf(x);
+ denominator += expf(x - maximum);
```
```diff
- out[i] = expf(own) / denominator;
+ out[i] = expf(own - maximum) / denominator;
```
Schema errors: physical call 5: reason string_too_long (max_length=300); physical call 7: action.repo_replace_syntax.edits.1.replacement_code string_too_short (min_length=1). Recovered by existing format retry policy.
Final SHA `88dd297b8a0bc53dc3b93b7d1a4678551346b4855d7139b4edaddecef0070ba4`, exact strict verdict VERIFIED_FIXED.

### A-build-h003

Live edit turn 8:
```diff
- total += a[start + j];
+ total += a[start + j] + b[start + j];
```
No format failure.
Final SHA `dd2f7ba2889c6870416f3aabca6bcdf18b16a00fea0cbe1fd1124ea3b17519e4`, exact strict verdict VERIFIED_FIXED.

### B-numeric-h001

Live edit turn 9:
```diff
- denominator += expf(x);
+ denominator += expf(x - maximum);
```
Schema errors: physical call 8: action.repo_run_public_checks.phase extra_forbidden. Recovered by existing format retry policy.
Final SHA `88dd297b8a0bc53dc3b93b7d1a4678551346b4855d7139b4edaddecef0070ba4`, exact strict verdict VERIFIED_FIXED.

### C-from-mutant-h002

Live edit turn 4:
```diff
- const float scale = rsqrtf(sum_sq / 32.0f + 1.0e-5f);
+ const float scale = rsqrtf(sum_sq / static_cast<float>(valid) + 1.0e-5f);
```
Schema errors: physical call 5: action.repo_run_public_checks.reason extra_forbidden. Recovered by existing format retry policy.
Final SHA `2a6b46543d50286749e7eb310fa1ed630db93c6d32dd25693c38ca2dc607b567`, exact strict verdict VERIFIED_FIXED.

B's original numerator-only candidate genuinely reproduced numeric mismatch with all_zero=false. The live model added the missing denominator stabilization, with no need for rollback. It recovered despite the earlier single-step rationale's erroneous all-zero statement. Diagnostic causal benefit remains unknown without a same-model diagnostics-off control. A and B converge on the same h001 source SHA; four successful trajectories correspond to three distinct registered task candidates, not four independent task types.

## Comparison and limits

P5 Qwen: 1/4 strict-passing units (h003), 41/42 complete typed physical responses, one incomplete, zero format retries, five binding rejects, two manually adjudicated semantic miswrites, zero duplicate probes. P7 DeepSeek: 4/4, no incomplete, four schema-invalid responses all corrected, zero binding rejects and zero semantic miswrites. This supports an important model-system limitation under the frozen controller: both formerly failing semantic edit tasks now reach Strict Fix, numeric recovery reaches Strict Fix, and h003 positive control holds. It does not show that DeepSeek's raw structured output rate is higher, or establish a stable population success rate or pure capacity effect. Reasoning and service stack differ, and the task set is small and repeatedly studied. No claim of cost advantage without actual bills. Next useful experiment is fixed-seed/frozen-budget repeated evaluation and eventually independent registered tasks, not additional Harness features. No additional run has been performed in this round.

## Provenance

| Unit | Agent public run | Controlled-recovery report SHA256 | Strict projection public run | Strict projection SHA256 |
|---|---|---|---|---|
| A-build-h001 | `2cbab1142b6a49b7bd6800a9c95f0157` | `d68cd4a94830ac781f6285816dd19d55c51dc5478580c6d9a23c519c31de1c4c` | `d753091d50b64e968dacb0417ff0705a` | `987936e8e588a274bb1df25f520fc927feb0614fe86033c34b26a9f1b6956e3e` |
| A-build-h003 | `a58da62920c846799b5da9dc8aa27e66` | `e86a5882e6d2fe9acdd060b6e1abb8b1cebe5831957eddd2d27e2269225d878f` | `cba7cd9a0a564d5cac9cec2c5b9f5ec6` | `524aa289c98cc969c21539d3a9c0d00047574be1141ef3e41a8b1d839a4806b6` |
| B-numeric-h001 | `e4a9f7f19dcf479197e774510fcb5781` | `c72788a0f3d873a0df7aa22db7d5a7be1304c99a27af80467f5126742889fa22` | `40f861dda13f4d76aa60c2a147461d54` | `987936e8e588a274bb1df25f520fc927feb0614fe86033c34b26a9f1b6956e3e` |
| C-from-mutant-h002 | `6b1e6884a8c54b34bcadc355603023df` | `a27275d37132973ca803f1183a0dbe56cc90fb2ed1c236f9c0d52ff0a5815f54` | `d146ecc162e549daa3379512336f535a` | `9acacd95463f3fde40ca5f98a51fc83d8497f2be1e5a9a882a47c9d877b01483` |

Raw reports and segregated evidence: `/home/you/gpu-native-harness-validation-20261124/p7-deepseek-frozen-full-chain/`. No private hidden inputs or expected outputs in these records. [Machine-readable results](2026-11-25-deepseek-frozen-full-chain-results.json).
