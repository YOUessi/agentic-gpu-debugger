# Formal benchmark protocol — 2026-11-25

**Candidate release:** `e6f31d0f7d45491e0781d80a825e3d2cf4886801` (P7 result-bearing release); **executed frozen runtime:** `be3dafd6988972dfdb51cf2faee6984103565973`. Preserve the original runtime checkout and its source-hash/base-commit payload semantics; release-only documentation and the evaluation controller do not change the Harness. The manifest pins the runtime source/benchmark/CLI Git objects, CUDA image, Python dependencies, and local Qwen model files. See [machine-readable manifest](2026-11-25-formal-benchmark-manifest.json).

## Counting and status

The previous DeepSeek result was four complete trajectories on three registered development tasks, with one from-mutant trajectory (h002), three conditional recovery trajectories (A-h001, B-h001 and A-h003), and three unique strict-passing source SHA values. A/B h001 converge on the same source SHA. None of those four observations enter the new repeatability denominator. h001–h003 are **Development Set**, not evidence of independent task generalization.

## Phase 1 — repeatability

Two model systems, six cells per model, five repetitions per cell: 60 new trajectories. Separate from-mutant and conditional-recovery analysis. Fresh output/worktree per trial; never discard failed units or substitute another sample. Precommitted alternating model blocks and rotated cell order, Qwen server seeds 43–47 (not request-level deterministic), DeepSeek server defaults. Both use the exact frozen controller and model output cap 8192.

| Cell | Start state | Compile gate | Revert | Diagnostics |
|---|---|---|---|---|
| F-h001 | Original mutant | on | on | on |
| F-h002 | Original mutant | on | on | on |
| F-h003 | Original mutant | on | on | on |
| R-A-h001 | Hash-pinned prior compile failure | on | on | off |
| R-B-h001 | Hash-pinned numerator-only numeric failure | on | off | on |
| R-A-h003 | Hash-pinned prior compile failure | on | on | off |

The three from-mutant cells use one complete, uniform profile (including compile gate/revert/diagnostics); this is a **new predeclared protocol**, not the former C-h002 minimal-profile experiment. Conditional cells retain original A/B flags and prior candidate identities. Binding, state-scoped actions and AST on; no Planner or extended syntax. 16 total tool steps, 18 physical calls (two format retries), 180s request timeout, 600s admission budget and external 600s wall watchdog; 6 compile-only, 2 full checker, 4 functional probe admissions; one revert maximum. Prefix steps consume steps and admissions, but never count as live repair. Watchdog termination is scoped to the owned CLI; halt batch if a timeout leaves external effects uncertain. GPU is serialized under the existing shared lock. No model intervention after trial start.

Strict checks occur outside model context on every attested new public-passing candidate, even duplicate SHAs across distinct trials. Normal terminal Finish uses existing verify-repo; if a closed session passed the public checker but did not finish, an operator may use existing verify-hardset with a source-reconstructed, exact SHA-bound diff, labelled separately without claiming a completed Agent loop. If the source/provenance cannot be verified or the session/tool has uncertain side effects, mark strict verification blocked; never create a false Finish or rerun to replace the trial. Retain strict failure/inconclusive/blocked distinction. No private inputs, expected answers or private verifier feedback reach the model.

For each trial retain configuration, immutable reports, journal and action audit, stage-specific candidate SHA evidence, first accepted edit and first live accepted edit producing a strict-verified exact SHA, revert, first candidate result, reason, calls and retries, usage and physical output completeness, wall times and actual public GPU tool elapsed milliseconds (build / numeric / sanitizer), source SHA, strict projection and full outcome. These GPU timings include isolation/container overhead, not kernel-only latency. Actual billed monetary cost is null unless provided by billing; no guessed money. Public numeric/strict verdicts and compile-only metrics remain separate. Summarize primary from-mutant task-wise 5-run Strict Fix counts, and conditional recovery per cell; report Wilson 95% intervals as descriptive small-sample intervals, unique SHA counts, failure types and costs including all failed attempts. Never pool recovery trajectories as independent tasks.

## Phase 2 — sealed independent holdout (not yet available)

Predeclare seven defect strata, at least three independently authored single-kernel tasks per stratum (target minimum 21). Record defect, difficulty and edit type in sealed task manifest. A separate author/evaluator, not this Harness development loop, constructs and validates tasks and hidden inputs, commits their hashes and freezes public evidence and scoring before releasing task sources only at evaluation. No case-specific repair hints, task-derived prompt tuning or post-hoc exclusions. The present runtime explicitly restricts registered tasks to h001–h003; new holdout registration and evaluator extension is a real engineering prerequisite, not already implemented. Review changes for evaluation plumbing only; freeze a new version and re-run every comparator if runtime behavior changes. Keep multi-file/macro/signature scope separate, no generic CUDA claims. Holdout failures remain failures of the frozen version; improvements require a new sealed holdout or a new, clearly labeled evaluation.

## Phase 3 — same-DeepSeek strong baselines (not yet run)

Four arms: Direct/One-shot (one typed edit decision with static source/evidence), best existing fixed Workflow selected on Development Set and rerun using DeepSeek, ordinary Coding Agent with equivalent public read/edit/build/test access and isolation, current Repository Harness. Freeze each arm's exact prompt, tool contract and budget conversion before holdout release; all use same official DeepSeek provider configuration, output cap, wall/physical request ceilings, public GPU admission/workload limits and identical strict verifier. Compare task-level strict success and actual cost per success, including failed trials. Do not equate one step across architectures. Report unavailable billing honestly. Component ablations (Planner/Diagnostics/Revert) only after main comparison.

## Execution boundary

The benchmark runner is operator-only orchestration of existing, frozen CLIs. It never changes model-visible tools, prompts or source. Credentials stay in the existing protected configuration on Tang; private verifier inputs stay separate. Preflight is non-inference and checks the immutable manifest, source objects, prior hashes and environment. All new attempt records are append-only. No holdout or baseline results are claimed yet.
