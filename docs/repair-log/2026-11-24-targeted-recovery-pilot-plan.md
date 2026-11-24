# 2026-11-24 — precommitted targeted recovery pilot

Implementation freeze: `1e825aa43d51eccf05c8b5953d5705d0554e1c65`. The experiment runner records requested settings and checks them against the journal and actual physical provider invocations. This is an engineering diagnostic, not a powered causal or strict-fix benchmark.

All units: Qwen3-8B-AWQ (local vLLM, server initialization seed 43); 8192 output-token cap; 16 total tool steps (operator prefix included); 18 physical model-call ceiling (two executor format retries); max 6 independent compiles, max 2 full public check admissions, existing public probe quota; 180 s per-request deadline, 600 s session wall budget; syntax editing enabled, batch exact-text edits disabled, no Planner or extended AST. No private verifier. One run per unit, no failed-run replacement.

| Order | Unit | Starting point | Compile gate | Revert | Public diagnostics |
|---:|---|---|---|---|---|
| 1 | A-build-h001 | exact earlier model-produced compile-failed candidate | on | on | off |
| 2 | A-build-h003 | exact earlier model-produced compile-failed candidate | on | on | off |
| 3 | B-numeric-h001 | exact earlier numerator-only numerically failed candidate | on | off | on |
| 4 | C-from-mutant-h002 | unmodified registered mutant | off | off | off |

C retains the original ReAct tools to isolate the larger output cap from the compile-gate change. A/B reproduce the original failure with real compilation/checker before giving control to the live model. The earlier action is not credited to the live model. A measures recovery via either a direct corrective edit or a rollback followed by a new edit; rollback itself is not success. B requires a distinct live candidate SHA, attested build, full-check numeric dispatch, and public acceptance. C measures the complete autonomous trajectory, not conditional recovery. Report admitted, compiled, dispatched, passed and finish separately. An incomplete response, wall-budget/timeout, or step-budget exhaustion is a distinct failure type.

| Prior artifact | SHA256 | Candidate SHA256 |
|---|---|---|
| h001 full-loop actions.json | 0dc4ba91f2c298376f238b57d62d3a63630623f59f738488d4cc0c0c985dfcc0 | a3a89df8bb0aa97a3631097129b36969bc8ea9c6b6c2dd43615689b8278a6488 |
| h003 full-loop actions.json | d5d377d154be04a9c02103615a7011a0a5cb2f63b5e603beedb3e281a3f75647 | acfe74458931f8fc7fff369c9f5a26425b6357214d8a87aa9df44a52319503b8 |
| h001 numerator-only single-step-result.json | f62ba346378f6a42f1ceb4699a12bc484fb7abdf80ec75ed72eec32f6361bc4d | 992264e8d9c65ba504357399a1d8229fe03a6396f215f0e226da5618ea9c9f73 |

Frozen implementation validation on Tang: 124 focused tests passed in 62.97s; strict mypy, 123 source files; Ruff lint and formatting passed. CPU regression log SHA256 `e468439cefbacbe310f861d4b76699f6a2bc3064f6a685f89ce91142fb2e1346`; strict typecheck log SHA256 `62f7c4c63dd61757257e3e6457752556858069091c5f9afd703f3e2aaee660f7`.

Operator plan and raw artifacts: `/home/you/gpu-native-harness-validation-20261124/p3-targeted-live-recovery/`. The plan was validated before the first model call, including all three historical artifact hashes. Results will be appended as a separate immutable report; this file contains the precommitted plan only.
