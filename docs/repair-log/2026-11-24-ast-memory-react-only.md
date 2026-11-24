# AST working-memory regression: real no-Planner h001 loop (2026-11-24)

Frozen implementation: `1d41cfb22bd50a37d4c524a121ba6c292ef23c40`. Tang RTX 4090, local Qwen3-8B-AWQ, vLLM. ReAct + AST, evidence_first, no Planner, no Reflect, no batch-edit fallback. Limits: 12 tool steps, 16 model calls, 2 public checks. Same frozen implementation as the [one-step diagnostic](2026-11-24-ast-memory-single-step.md).

## Result

The actual model selected all six actions and both replacement statements itself. No answer, private reference implementation, hidden Checker, or scripted edit was given to the model.

| Turn | Action | Result |
|---|---|---|
| 1 | repo_read_file, line 27 | OK |
| 2 | repo_inspect_syntax, offset 0, limit 8 | OK |
| 3 | repo_inspect_syntax, offset 8, limit 8 | OK |
| 4 | repo_replace_syntax: n13 and n15 | OK, non-NOOP |
| 5 | repo_run_public_checks | PASS (actual GPU Checker) |
| 6 | finish | OK |

Model-selected replacements:

```cpp
denominator += expf(x - maximum);
out[i] = expf(own - maximum) / denominator;
```

| Acceptance metric | Observed |
|---|---:|
| Repeated same-version/same-page Inspect | 0 |
| Node-not-found rejections | 0 |
| Rejected tool actions (all types) | 0 |
| Legal non-NOOP edit actions | 1 (2 nodes) |
| Public Checker reaches / trials | 1/1 |
| Public Checker passes / trials | 1/1 |
| First accepted edit turn | 4 |
| First public pass turn | 5 |
| Physical model calls / tool turns | 6 / 6 |
| Model tokens (total) | 31,543 |

Final status: `DIFF_STAGED_UNVERIFIED`. Public functional success does **not** establish independent hidden verification or Strict Fix. CLI process exited 0, vLLM shut down, GPU memory returned to 204 MiB and shared GPU lock released.

## Interpretation and next gate

The previously recorded no-Planner and Planner runs at the old harness each produced zero effective edits and zero public Checker reaches. One-shot h001 now generates a valid alternate softmax repair and passes the real public Checker; the complete no-Planner loop also independently produces a stable-softmax repair and passes. This supports a Harness-side failure mechanism on h001 and demonstrates the bundled changes can unblock this path. It does not establish which individual change caused the improvement, cross-case generalization, or a Planner effect. Do not report a success-rate improvement from this one-case comparison as a population estimate. Replicate across seeds and h002/h003; only then resume a matched Planner-only comparison if the baseline gate remains satisfied.

## Audit

The full action records and artifact hashes are in [the machine-readable summary](2026-11-24-ast-memory-react-only.json). Local raw run directory: `/home/you/gpu-native-harness-validation-20261124/ast-memory-regression/qwen-react-only-h001`. Public run `f7d2064b2d454db6b09f742ba2d325a0`; public Checker run `13865474748e4f029ef0a7777798ab1d`.

The separate full CI run at bca11dd failed the strict type check for a runtime `list[edit]` annotation; commit a061b11 replaces it with `GenericAlias(list, edit)`, preserving the runtime schema. HardSet native-contract validation passed at bca11dd. The real experiment was not rerun on the annotation-only follow-up commit.
