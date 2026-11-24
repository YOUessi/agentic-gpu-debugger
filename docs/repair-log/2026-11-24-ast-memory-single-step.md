# 2026-11-24 · AST working-memory repair and one-shot editing gate

## Frozen source and scope

Experiment source: `1d41cfb22bd50a37d4c524a121ba6c292ef23c40` (clean detached checkout on Tang RTX 4090).
Parent baseline `bf7f872a` / previous real experiment source `473bbc21`:
Qwen3-8B-AWQ achieved 0/1 effective edits and 0/1 public Checker reaches in each
of ReAct+AST and Planner+AST, with 12 tool steps per arm. No causal planner
conclusion was supported.

The new controller pins public AST pages outside history compaction, uses
source-order short aliases, binds the current source SHA, cursor, symbol and
observed aliases in structured-output grammar, invalidates old evidence after
a successful edit, re-enumerates checkpointed evidence, and audits full typed
actions with rejection codes. The old hash-addressed AST checkpoints are
intentionally incompatible with this interface. Truncated code nodes are
explicitly marked and not eligible for model edits.

## CPU and CI validation

87 focused CPU tests passed (38.76 s) on the formatted implementation;
regression includes compaction after 5 later reads, cursor and alias state,
checkpoint rehydration, NOOP / repeated rejected edits / absent alias / stale
SHA, and one-step edit with GPU only after a changed candidate. Ruff lint and
format both passed for all 17 changed Python files. GitHub HardSet V2
native-contract workflow passed; full CI was still in progress when checked.
The fake-checker CPU tests are NOT real model or GPU results.

## Physical Qwen3-8B-AWQ experiment, h001

The controller prefilled four public pages (25 nodes), then exposed only one
edit action. Planner and Reflect disabled. No reference patch, Oracle,
hidden test source, expected outputs or private evaluator exposed. One physical
model call was permitted. The operator ran the registered public GPU checker
only after a legal changed candidate was accepted.

| Metric | Result |
|---|---:|
| Physical model calls | 1 |
| Legal non-NOOP edits | 1/1 |
| Public Checker reached | 1/1 |
| Public CUDA checks passed | 1/1 |
| Independent hidden verification | Not run |
| Input / output / total tokens | 4197 / 3512 / 7709 |
| Reasoning tokens reported by provider | 3283 |
| Prefilled AST pages / nodes | 4 / 25 |

The model selected n12 and n14, shifted each input sum by maximum before
exponentiation, rather than changing the exponentiation statements themselves.
The two edits were accepted atomically. This is an independently generated
mathematically equivalent stable-softmax candidate, not a scripted answer.

Original source SHA256: `47e69f940ca6c8b9ab287b5ee5c9aed7973f4d1eeb7ac41b7e855a01609c9bce`.
Candidate SHA256: `4d7d5fd3d3384cb610bda8128f3b580ed5202e269336b4d325e94b6f2e5d190c`.
Public Checker: `PUBLIC_CHECKS_PASSED`, run `7fb5e1d7a6bf4219985683921e77fe8f`.

Evidence hashes from Tang:
- single-step-result.json: `ab57e2ef6e221081c754cdc4cee110049fea57bcf7745902f4046ccc4a35c4b3`
- single-step-snapshot.json: `ab96145f6281444f47fed932130a1de4619f9ec5476d8ba0833f62e7b00c2bad`
- CPU regression log: `9d405b7d32a249c86f77b9011019ea768ba4b4db1857c4347e606d40f645a27f`

Raw artifacts: `/home/you/gpu-native-harness-validation-20261124/ast-memory-regression/`.
A first vLLM startup used a partial Hugging Face snapshot and failed before
any physical model decision (missing tokenizer). Retrying from the existing
complete local Qwen3-8B-AWQ model produced the recorded trial; the startup
failure is not a failed model decision. GPU lock released and server shut down.

## Interpretation and next gate

The one-shot experiment proves that this 8B model can generate a legal,
publicly correct h001 repair when provided current AST evidence, so the old
zero-edit outcome cannot be attributed solely to lack of model capability.
It does not isolate the contribution of each bundled controller change, nor
prove the full autonomous loop or cross-case generalization. A fresh no-Planner
ReAct+AST h001 loop on this same frozen implementation must verify zero
duplicate-page inspections, zero invalid aliases, an effective edit and public
Checker reach before repeating Planner-only comparisons. No strict-fix or
planner gain is claimed.
