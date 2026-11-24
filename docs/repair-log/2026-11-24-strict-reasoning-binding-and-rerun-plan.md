# P4: reasoning intervention and AST target binding — 2026-11-24

Independent h003 Strict Fix: [exact candidate verification](2026-11-24-h003-independent-strict-fix.md), VERIFIED_FIXED / ALL_REQUIRED_CHECKS_PASSED.

## B exact-input reasoning replay

Original failed invocation payload SHA256 `29c583ac6e9d96a104f8cee0a69c01bf99ae466b74e91f8ffd9f567e55962dc1` was reconstructed from frozen `1e825aa43d51eccf05c8b5953d5705d0554e1c65` workspace, verified checkpoint and public artifacts. The hash matches the original Invocation, unlike previous approximate replays. Original response was incomplete at 8192 output / 8191 reasoning tokens. New trials used the same payload, prompt, Qwen3-8B-AWQ, Responses API and strict JSON schema, cap 8192, no tool execution or retries. SDK 3.13.0, vLLM 0.19.1; server seed 43. Runner blob `b6afdc6381db955f76f05636dbb35cb40764611a` from `0ad68ef9f5a286692aa3e4d95fb38d8fa8d6719b`. The first default arm completed before an operator cancellation of a previously slow fetch workflow; remaining arms restarted the service at the same server seed. Server seed is not request-level deterministic sampling.

Identical wire schema SHA256 `056934f79ea48c9fdf14d106883c3ac73a98052c4c8e439d0ba90dab5b1b956d`, instructions SHA256 `e5c1f420129ebf611ce334a069fa3514eeb8baa12c901c044bbf218a1dc19643`, wire input SHA256 `a0e7c482020bc4ae5d2e540326f52460d9949f082f58fec881a1ca6da3968959`. The local model's original chat template does not consume reasoning_effort; the ResponsesRequest translates effort to template kwargs. Native thinking_token_budget exists in Chat Completions, not this ResponsesRequest; it was not assumed supported or tested using a changed endpoint. The disabled-thinking service used an exact template copy with only a leading Jinja set enable_thinking=false. Task instructions/schema unchanged; rendered input is four tokens longer.

| Arm | Input | Output | Reasoning | Output text tokens (local retokenization) | Wire status / result | Action quality |
|---|---:|---:|---:|---:|---|---|
| default | 2945 | 170 | 169 | 0 (no output text) | completed / LLM_INVALID_OUTPUT | no action |
| effort low | 2945 | 5825 | 5735 | 87 | completed / valid | duplicate after probe 32 |
| effort none | 2945 | 7660 | 7567 | 90 | completed / valid | duplicate after probe 32 |
| server thinking disabled | 2949 | 63 | 0 | 0 | completed / LLM_INVALID_OUTPUT | no action |

Effort values really reached SDK request. Neither low nor none demonstrates a hard reasoning budget (none still spends 7567). The template-based intervention produced zero reported reasoning tokens but no parsed output text. All four were non-incomplete in this replay; two valid actions were redundant in the exact recorded state. Do not claim successful task progress from legal schema alone. No GPU invoked. Retokenized output-text count is separate from provider total-minus-reasoning tokens. No change to production reasoning policy justified.

Artifacts (under Tang /home/you/gpu-native-harness-validation-20261124/p4-strict-and-binding):
- reasoning-plan.json SHA256 `a5d15c67cc6c6821a8e1f316a0b2b2974b0e7f2ce05d1dc4ffae64c8ae13c8d5`
- reasoning-default.json `a7bb2e47ecfbe9e7b35e8ff61aa6f275c9f7b4ee17b0af27d026e50b82503570`
- reasoning-effort-low.json `c930f195842913e6265d812127d77bab2207a9b775c7fdc953b2126f1cba5b18`
- reasoning-effort-none.json `f4e768cd2f83c96bcf6a077f27b3b0a39ea06fe3990dfefbc36fcdcffb855408`
- reasoning-thinking-disabled.json `a8422343a429c8b7020e58c63048ea8507ef8fe21371f4fd6c7ead010f8eb84a`

## Bound editor and scoped action profile

Frozen source `be3dafd6988972dfdb51cf2faee6984103565973`. `--ast-target-binding` independently enables required expected_old_code in dynamic wire schema; host verifies exact pinned old code against alias and source SHA before any write. Missing/mismatched witness is rejected atomically with requested alias, actual and expected code, and no_source_written=true. Legacy field omission and old grammar retained. `--state-scoped-actions` supplies untested public_probe_remaining_sizes (empty means no Probe capability) and current-revision rejected_action_state derived from verified tool ledger, not compacted history. No rejected node blacklist. Resume verifies probe signatures against child attestation and action ledger; profiles bound into checkpoint and journal.

Controlled replay retained actual previously generated wrong aliases and replacement strings, and explicitly added operator old-code witnesses. This is a mismatch test, not a live model's ability to generate the correct witness.
- h001, A-build-h001 turn 12, wrong n07 (loop initializer) with own declaration witness; entire two-edit batch rejected. Source/diff unchanged, 0 compiles/checks/model calls. Run `db797266847542fe8f8f850768d96568`, report SHA256 `1eb466066bbe1f27a1ca08791bd61e439b7c82b3da533107fb2dcfe5eb417fef`.
- h002, C-from-mutant-h002 turn 3, wrong n08 (x declaration) with scale declaration witness; rejected before source mutation. Run `2931150a0c2f4fc4b9caa850a2ebe905`, report SHA256 `1fa6c28b1f1c996d58115d2b9fbb6710d79bd8badf5babb744b81669bb860d42`.
- Binding only detects inconsistent alias/witness pairs; it cannot prove intent or disambiguate identical old-code strings in different semantic scopes.
- Final regression 130 passed in 67.00s; strict mypy 123 files; Ruff check/format pass. Log SHA256 `1cc8cffaf6ac63dbdde57b5a871f56e8649536965ed57485c6eca5b1b3d64dc5`, typecheck `62f7c4c63dd61757257e3e6457752556858069091c5f9afd703f3e2aaee660f7`. Legacy strict-schema fixture hash identical across old and new `0fbc1dff50d66049d0873b0dde9c7090792166e6afef16b73b5483ff4cab520f`.

## Precommitted live rerun

Same four units and prior evidence as targeted-recovery-pilot-plan. Same default thinking, local model, server seed 43, 8192 output tokens, 16 total tool steps, 18 physical model calls, 600 s wall, 180 s request timeout, 6 compile-only and 2 full checker caps. Both new constraints enabled; Planner and Extended AST disabled. A h001 and h003 compile/revert; B h001 compile + diagnostics, no revert; C h002 original mutant, compile/revert/diagnostics disabled as before. Historical prefix copies actual old-code witnesses to reproduce prior faulty bytes, not to teach the new correction. Operator prefix excluded from live repair metrics, included in step budget. All four outcomes retained. This evaluates a combined constraint profile, not a causal separation between the two changes. No stronger-model or extra-sampling arm included unless persistent live wrong-target errors justify a separate exact-input study.
