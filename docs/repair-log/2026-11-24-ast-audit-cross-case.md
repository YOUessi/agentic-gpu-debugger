# 2026-11-24 · AST audit closure and two-seed cross-case diagnostic

## Frozen implementation and engineering gate

Frozen implementation: `dc7f5aabd0bcef0cb7aad28cadfcf040f687c130`. Changes from `bd0d20c` are restricted to the AST executor, bounded public audit/replay, recovery validation and tests. Planner, Reflect, public Probe and independent Verifier implementations were not changed. The edit interface retains two-digit aliases and does not require a new intent field.

- New registered AST tool events persist exact model-visible public observations, full typed actions, rejection details, observation SHA256 and rejected-edit fingerprints.
- Replay validates the public artifact, action identity, contiguous turns, observation hash and rejected-edit fingerprint. Legacy hash-only observations cannot be reconstructed.
- Resume cross-checks the history, last action, active inspected pages, parser output and rejection ledger against the recorded public events, rejecting tampered checkpoints.
- Stable rejection codes carry actionable public corrections, including the current/expected SHA, alias, original code and public requirement for NOOP.
- The new executor prompt has its own version `gpu-repository-ast-executor-2026-11-24-v1`. Other modes retain their existing prompt.
- 93 focused CPU tests passed in 38.86 s; Ruff lint and formatting passed; mypy --strict checked 120 source files successfully. GitHub HardSet V2 native-contract validation passed. Full CI was still in progress at the last check.

## Experiment

Qwen3-8B-AWQ, Tang RTX 4090 Laptop GPU, local vLLM. Two separate server initializations with seed 41 and 43, each runs h001, h002, h003 in that order. The seed is the server initialization seed, not an independently set per-request seed. Each trial is a fresh isolated worktree with the same source revision, operator-prefilled AST state, one physical model decision and at most one real public GPU Checker, only after a valid non-NOOP edit. No reference code, hidden evaluator, Planner, Reflect or test doubles enter the model trial. No failed trial was discarded or replaced.

| Server seed | Case | Valid edit | Checker | Outcome | Total tokens |
|---:|---|---|---|---|---:|
| 41 | h001 | no | not reached | CUDA_SYNTAX_NOOP | 5789 |
| 41 | h002 | yes | PASS | PUBLIC_CHECKS_PASSED | 4267 |
| 41 | h003 | no | not reached | CUDA_SYNTAX_REPLACEMENT_PARSE_ERROR | 4297 |
| 43 | h001 | yes | FAIL | NUMERIC_MISMATCH | 5706 |
| 43 | h002 | yes | PASS | PUBLIC_CHECKS_PASSED | 4069 |
| 43 | h003 | yes | PASS | PUBLIC_CHECKS_PASSED | 4773 |

Summary: legal edit and public Checker reach 4/6; public pass 3/6 (3/4 reached); one numeric mismatch, one mixed NOOP rejection, one syntax rejection. No provider errors or missing node aliases. All six full public audit replays passed. This is not a statistical estimate of population accuracy. Operator-prefilled pages cannot measure autonomous Inspect efficiency.

## Failure analysis

- h001 seed 41: two changed declarations correctly subtract the maximum (n14, n12), alongside two unchanged entries (n15, n13). The atomic request is rejected, no partial edit is applied and no GPU test is run. This is a mixed-NOOP action assembly problem, not evidence of complete inability to formulate a repair. The unsubmitted subset was not independently tested in this trial.
- h001 seed 43: changes only the numerator, leaving the denominator unshifted. A legal candidate reaches the actual public Checker and fails NUMERIC_MISMATCH at size 32. This is a distinct incomplete semantic repair.
- h002: both seeds independently replace the fixed divisor 32 with valid, yielding the same candidate and a real public pass.
- h003 seed 41: tries to replace an early return with a comment, rejected by syntax parser. Seed 43 correctly adds the b contribution to the group sum and passes.

## Limits and next experiment

Do not claim a Planner gain, a strict fix, or a causal prompt regression based on a different-seed comparison to the earlier h001 success. The two previously identified infrastructure failure mechanisms are not observed in these one-step trials, but end-to-end robustness still needs independent full ReAct-only trajectories on this new revision. A targeted mixed-NOOP diagnostic should report every unchanged entry and suggest omitting it while maintaining atomic semantics; do not silently accept a subset or force arbitrary text changes. Then compare a stronger model on the same single-step contract when a suitable provider is available. No stronger-model or hidden verifier run was performed in this batch.

## Evidence

See [machine-readable summary](2026-11-24-ast-audit-cross-case.json) for full selected action replacements, outcome, public run IDs and SHA256 hashes. Raw output on Tang: `/home/you/gpu-native-harness-validation-20261124/ast-audit-closure/`. Both vLLM runs finished; GPU lock released and memory returned to 204 MiB.
