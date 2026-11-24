# P5 — frozen bound/scoped live rerun, 2026-11-24

All four precommitted trials completed on Tang, in original order, without changing source, model, tool availability policy, prompt, reasoning settings, or budgets mid-run. Source SHA `be3dafd6988972dfdb51cf2faee6984103565973`, plan SHA256 `4592159fbf5d5eb69420184cda767a790ce675b0d901b94f5e29ba6feef98fba`. Qwen3-8B-AWQ, server seed 43 (not a guarantee of deterministic per-request sampling), default thinking, 8192 output tokens, 16 total tool turns including operator prefix, 18 physical model calls, 6 independent compile admissions, 2 full checker admissions. Both AST target binding and state-scoped actions enabled; Planner and Extended AST off. A uses compile/revert; B compile/diagnostics, no revert; C no compile/revert/diagnostics, as previously frozen. No new functionality or model comparison. Read-only post-run audit reconstructs every accepted edit from original registered source and observed AST pages; validates all rejected action diff hashes and final worktree bytes, exact request caps and provider roles, checker lifecycle and Strict Fix candidate identity. Audit script SHA256 `c4baf867a6932c1d1338bd925718fa3c723336c502eb9dabbdcd81efcf38cc27`.

## Results (live-created candidates, operator prefix excluded)

| Trial | Live turns / calls | Tokens | AST accepted | Build pass | Dispatch | Public pass | Strict Fix | Binding rejection | Semantic wrong-target edit | Duplicate Probe |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| A-build-h001 | 13 / 13 | 71749 | 1 | 0 | 0 | 0 | not run | 3 | 1 | 0 |
| A-build-h003 | 8 / 8 | 44192 | 1 | 1 | 1 | 1 | VERIFIED_FIXED | 0 | 0 | 0 |
| B-numeric-h001 | 4 / 5 | 44280 | 0 | 0 | 0 | 0 | not run | 0 | 0 | 0 |
| C-from-mutant-h002 | 16 / 16 | 75271 | 1 | 0 | 0 | 0 | not run | 2 | 1 | 0 |

42 calls, 235492 tokens, no format retries. Prior P3: 38 calls, 201537 tokens. Do not combine heterogeneous trial objectives into one population-level repair rate, claim token savings, or infer separate causal benefits of the two new flags.

## Failure taxonomy and source review

- **A-build-h001** — STEP_BUDGET_EXHAUSTED. Turn 15: n08 in the maximum-reduction loop, not the denominator loop, receives denominator accumulation. Original denominator statement and numerator remain unchanged. NVCC turn 16: denominator undefined. Rejections: {"REDUNDANT_READ":1,"CUDA_AST_TARGET_BINDING_MISMATCH":3,"CUDA_SYNTAX_NOOP":2}.
- **A-build-h003** — WORKSPACE_DIFF_STAGED_UNVERIFIED. Turn 8, n06: total += a[start + j]; -> total += a[start + j] + b[start + j];. One minimal model edit, build turn 9, checker turn 10, finish turn 11. Strict same-SHA pass. Rejections: {}.
- **B-numeric-h001** — LLM_INCOMPLETE_MAX_OUTPUT_TOKENS. Four distinct after-probes: 32,33,65,129; all NUMERIC_MISMATCH. Fifth live invocation uses 3828 input / 8192 output / 8191 reasoning tokens, 0 output_text chars; service confirms max_output_tokens. No live revision; numeric diagnostic benefit unknown. Rejections: {}.
- **C-from-mutant-h002** — STEP_BUDGET_EXHAUSTED. Turn 13 accepted n08 x declaration with exact old-code witness, wrapped declaration in if (j == threadIdx.x) and added accumulation. Original following accumulation uses x outside its new scope; scale divisor remains unchanged. Full checker turn 14 BUILD_FAILED, line 18 x undefined; after-probe turn 16 repeats build failure. Rejections: {"REDUNDANT_READ":1,"CUDA_AST_TARGET_BINDING_MISMATCH":2,"CUDA_SYNTAX_REPEATED_REJECTED_EDIT":3,"CUDA_SYNTAX_NOOP":1}.

**Binding vs semantic target correctness.** Five real mismatched-witness edits were rejected with no source side effects; no accepted writes had a mismatched SHA/alias/old-code witness. Manual review of the three accepted live edit actions finds two semantic target/scope mistakes (h001 and h002). Report `binding_mismatch_write_count=0`, `wrong_target_write_count=2` under this explicit operator-review rubric, not zero semantic miswrites. C's failure is scope/quality rather than proof of an inconsistent alias/witness pair. No duplicate same-SHA/phase/size Probe submissions or repeats of completed Probe requests were observed in all four sessions (previous P3: six redundant Probe rejections). B chose four distinct yet unproductive-for-repair probes; legal action diversity alone does not ensure progress.

**Budget classification.** A-h001 and C-h002 terminate at the 16-step ceiling, but have repeated rejections and a non-buildable candidate, not an otherwise successful recovery interrupted only by insufficient budget. B has explicit output-cap failure, not step exhaustion. No newly produced numeric-failing candidate was reached in B; the reproduced operator prefix failure is not a live model numeric recovery result.

## Independent strict result

A-h003 model selected revert at turn 4 and then made a single minimal correction at turn 8. The exact new candidate SHA `dd2f7ba2889c6870416f3aabca6bcdf18b16a00fea0cbe1fd1124ea3b17519e4` passed compile (`b236c9e122ca45f992ea825f33d9b574`), full checker (`91239c4109564a51b1494b166a456a56`) and independent verifier: VERIFIED_FIXED / ALL_REQUIRED_CHECKS_PASSED. Strict public projection SHA256 `fa1c98011d0a326d9fc4151a37b30fb0e3fad774ce3fd97c35ae4c6273dd02b0`. This is a second successful conditional recovery and independent verification of h003, with a distinct candidate from P3. It establishes repeated occurrence on this same task, not stable generalization.

## Evidence

| Unit | Public run | Raw report SHA256 | Independent post-run audit SHA256 |
|---|---|---|---|
| A-build-h001 | `ec719d88bd8340339dcd17cd71fe09a0` | `2a7531fba3aa2137973a7631587cf2fcdedd16606510931b4d7cfa5f67a4d708` | `b249348c8c978d3906a8a772044512cd5c7cc4a4a18d7de243be55e62453d84c` |
| A-build-h003 | `33290e0a53bf4fe7ba394f92fd9d6fd9` | `07fc6b478edaee9e39766879c0dba71e3b3efc4f67f996decede2d2f0b116f36` | `728426ec7485f03b9fe8b515158e38da1ba7ca3dbf7eb7b88e2a29aa53c151f6` |
| B-numeric-h001 | `a11de714c7ef4b569b03350806cf6c58` | `5c42ea91f951ad1b69d7ac400c02290ac403c14885fdd5cc0e1be6d991893ad8` | `f1b77f4de556fd1549cc755e6d798ab27c74c12adcec75b573b997aae8401251` |
| C-from-mutant-h002 | `bb3dc9daac484e03b34bc4d4e855360e` | `827808d32ea2a08a60cdff04769bbb2b004a4066b54056c13d9f023fb91f7ee0` | `a04849c7b11bc53327dbaf2494d2ce1fcf31f98cfa930803d6f0527f6d61cf96` |
Raw artifacts: `/home/you/gpu-native-harness-validation-20261124/p5-bound-scoped-live-rerun/`. Strict hidden evidence remains isolated; only public projection recorded. Source reconstruction passed all four. GPU released, 204 MiB used, 0% utilization, shared lock free at last inspection.

## Decision

No further Harness features or larger step cap justified by these observations. h003 recovery repeats, but A-h001 and C-h002 still fail on semantic positioning/scoping and NOOP/repeated action behavior, and B remains generation-limited. Next preferred study: a stronger model on the identical frozen Harness and task conditions (or exact failed input as a controlled capability diagnostic), retaining all attempts and identical verifier. Do not claim a stronger model is installed: inspected local /home/you/models contains only Qwen3-8B and Qwen3-8B-AWQ; unquantized 8B is not a distinct stronger-capacity model. No stronger-model run has been performed in this round.
