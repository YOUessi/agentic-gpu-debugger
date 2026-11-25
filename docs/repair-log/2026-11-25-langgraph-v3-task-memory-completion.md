# LangGraph v3 task/memory runtime — implementation and acceptance

Date: 2026-11-25. Tested and comparison-frozen code SHA: `bf9e224d14e7a9382e2f40e450c0e35cda41e670`. The P8 controller and runtime remain `8724f84` and `be3dafd`, respectively.

## Implemented
- Actual repository runner and CLI: graph-native parent and Executor subgraph; SQLite checkpointer and native interrupt/resume; graph version `repository-langgraph-v3-task-memory`.
- Dependency-ready task frontier using LangGraph Send and reducer; pure parallel scoring with a single selected writer/executor; task-specific allowed-tool intersection and host-side scope guard, distinct evidence per milestone, candidate-sensitive invalidation, bounded failure/stall replanning.
- Bounded context projection preserving host budget advice, active task, current failure, plan status and recent evidence. The canonical history, source, and pinned AST map remain separate; the summary is capped at 2800 characters, not an asserted tokenizer-level input limit.
- Real cross-session LangGraph Store integration backed by a scoped persistent SQLite database. Explicit off/read/learn modes, per-repository scope, read-only enforcement, immutable repair episodes, bounded BM25 retrieval and optional local CPU embedding backend. Prior episodes are frozen into the graph for the current session and re-ranked against failure state. No raw prior source, patch, private verifier material or invented strict success enters memory.
- Absolute deadlines, effect journal head and host checkpoint digest bound to graph state; strict serializer configuration; fail-closed admission on uncertain effects; native resumption at a reconciled frontier.
- Four-arm same-model comparison runner with immutable reservations, no-resample interruption/timeout handling, separate strict-verifier process, resource ownership/idle checks, primary all-started denominator and secondary determinate denominator.

## Verification actually performed on Tang
- 10 original native tests remained passing after task scheduling and Store integration.
- 15/15 after new parallel frontier, scoped-denial, evidence, Store and context tests.
- 21/21 after actual process-death fault injection. Tested termination inside a model call, after model result, after tool result, between host/native checkpoints, plus successful resumption at a quiet boundary. Semantically identical but rewritten host checkpoint rejected.
- 25/25 new/native and comparison-specific tests before final formatting/provenance hardening.
- Final regression: **87 passed in 51.51s**; Ruff checks passed; Git diff check passed. Command:
  `python -m pytest -q tests/unit/test_gpu_harness_repo_agent.py tests/unit/test_gpu_harness_repo_public.py tests/unit/test_gpu_harness_ablation.py tests/unit/test_native_langgraph_runtime.py tests/unit/test_langgraph_runtime_comparison.py`
- Real Git worktrees, Tree-sitter, LangGraph, SQLite, process crashes. Models and public CUDA checker are test substitutes. Embedding similarity path tested with injected deterministic vectors; a real local sentence-transformer model has not been installed or benchmarked.

## Frozen comparison plan
- Tang: `~/projects/agentic-gpu-graph-native-dev-20261125/comparison-60-bf9e224d/manifest.json`.
- Canonical JSON digest `ac04b05405a7cc079f62e06cf11f611d0b136f675c5429222b652db6e9743d9f`; exact file SHA256 `510412403ffe0ca59dbec2a4aa2c93c7b8165f4651495881e946a66f8c856f37`.
- Official DeepSeek V4 Pro model configuration, four arms (legacy-react, native-react, legacy-planner, native-dag), three existing development cases, five repetitions each: 60 planned.
- Same ceilings for all arms: 16 tool turns, 24 physical model calls, 8192 output tokens/request, 180s request timeout, 600s Agent and 600s strict watchdogs, 6 compile-only, 2 public checks, 4 probes. The new 24-call ceiling accommodates planner/replanner reservations; this is not a modification of P8's 18-call protocol. Role allocations intentionally differ between architectures and are not presented as paired RNG reproduction.
- Memory off in formal comparison, independent verification only after public pass. These are repeated development tasks, not a new holdout.
- Plan and status command ran successfully with zero model calls, zero GPU checks, 0 reserved and 60 pending.

## Remaining empirical boundary
At 11:46 Beijing time the old P8 controller was still running (PID 781064), with its last progress snapshot showing 16 successes, 21 completed failures, 1 unknown timeout, 22 pending. No new paid inference or GPU evaluation initiated by this implementation round, so no measured real-model repair-rate superiority is claimed. New benchmark execution must be serialized with the existing shared GPU lock. The previously approved P8 local launcher is unaffected. Native memory opt-in and task scheduling are implemented; a future held-out test corpus and real-model evaluation remain distinct scientific validation, not missing runtime code.
