# 2026-11-24 — P0 budget fairness, checker lifecycle, cross-case AST closure

## Frozen implementation

`52be179d46ab8be335a93abcf1d1ee986970bbc6` (Draft PR #16). Equal-executor comparison reserves logical decision and format retry quotas independently for executor, planner, and reflection. For 12 tool steps and 2 executor retries the ceilings are 14 physical calls for ReAct, 18 for hierarchical (including one decision and one retry each for plan and reflect). Actual calls and token costs remain counted. Old diagnostic/patch workflows retain their prior behavior. This fixes the protocol; no fresh Planner comparison was run.

Public checker request, admission, numeric dispatch, completion, and pass are now separate. Numeric dispatch is a backend invocation marker, not proof of successful GPU execution. Compilation failure counts as a completed failed checker, not a numeric dispatch. Legacy records cannot establish missing dispatch evidence. Failed-session audit recovers AST accepted edits and checker evidence without depending on a final result.json. The AST prompt was already split from exact-text prompts.

## Engineering validation

- Tang targeted regression: **104 passed**, 59.94s; strict mypy 121 source files; Ruff lint/format passed.
- Regression SHA256 `038aa189b57744d723d825f34ca22d74ddb1424fad4efc01571578a1636dc5b1`; typecheck SHA256 `0cd98cfd5b35e0e85b86d75459966f02562f9c10fe59fb3d86271d1b9fa33e5c`.
- Real macOS isolated checkout: the two original verifier-preflight regressions passed (2 passed, 33 deselected, 34.52s). Only the controller-created temp path is canonicalized, not arbitrary model-supplied paths. These are preflight tests, not hidden GPU verification.
- Fixtures cover all three tasks, first public failure followed by second edit/check, multipage h003 resume, stale revision rejection, partial audit, and role-specific retry budgets. Scripted fixtures are not real-model repair evidence.

## Live from-mutant trials (no Planner)

Local Qwen3-8B-AWQ, vLLM server seed 43, 12 tool steps / 14 physical calls / 2 public checks, existing statement editor and public feedback. Frozen SHA above. Order h002, h003, h001. No hidden evaluation. All outcomes retained; no retry/cherry-picking of failed sessions.

| Case | Physical calls | Tokens | First accepted edit | Accepted edits | Checks admitted / completed / numeric dispatch / pass | Outcome |
|---|---:|---:|---:|---:|---|---|
| h002 | 1 | 5925 | none | 0 | 0/0/0/0 | LLM_INCOMPLETE before any tool |
| h003 | 12 | 49719 | 4 | 2 | 2/2/0/0 | BUILD_FAILED twice; candidate added assignment to const valid; later fixed group sum but retained compile error |
| h001 | 12 | 52237 | 5 | 3 | 1/1/0/0 | BUILD_FAILED after replacing wrong AST slots; subsequent edits exhausted budget before rechecking |

Numeric dispatch refers only to full Checker, not probes (h001 2 probes; h003 1). The live sample is 0/3 public passes and is not a statistical estimate of repair probability. All 25 physical calls were executor, no format retries. Earlier h001 success is not contradicted by a single different trajectory. Accepted edit does not imply semantic correctness or compilability.

| Case | Run ID | metrics.json SHA256 | actions.json SHA256 |
|---|---|---|---|
| h002 | 0211e04e09c74f9f8c3c4dcd204c8ac4 | 19b79f27bbab0331a5ca1ba87eb627c07023e3f338549108b5901e2062fda792 | 37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570 |
| h003 | 7cdf25a34f764a3b9dbc4d3e97d07e0a | 871a94707810ca01738d8938ca1efe2fd8c02ec42b7fa374af3aae4d69a6d078 | d5d377d154be04a9c02103615a7011a0a5cb2f63b5e603beedb3e281a3f75647 |
| h001 | f9b65ea401844ec9b7444cdf39e483de | e2e699070761064c3ee40f3a526a537c0bd60621c66bf3ded1494b991c892eb0 | 0dc4ba91f2c298376f238b57d62d3a63630623f59f738488d4cc0c0c985dfcc0 |

Raw public logs live on Tang in `/home/you/gpu-native-harness-validation-20261124/p0-fair-budget-closure/react-seed43-{case}/`. h002 partial audit recovers the one physical call and token usage despite no completed tool.

## Separate P1 branch

`experiment/ast-editor-diagnostics-v2-20261124` implements opt-in extended AST and bounded diagnostics; validated implementation `b9f3eb789f52c3c4530d614b9fd7128ba5afc202`. See its `docs/repair-log/2026-11-24-extended-ast-public-diagnostics.md`. Do not combine this with the P0 frozen trials. Neither a new Planner-only comparison nor a strict-fix statistic is justified yet.
