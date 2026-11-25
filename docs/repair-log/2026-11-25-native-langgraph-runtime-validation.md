# Native LangGraph runtime implementation and validation

2026-11-25. Tested implementation commit: `fbe1948fea49477c6a2e229ed60dac91d7eac21b`.

## Implemented
- Native parent graph and nested one-turn executor, typed task DAG, dependency/evidence tracking, graph-based failure classification and bounded replanning.
- Native SQLite checkpointer (LangGraph 1.2.20, checkpoint 4.3.1, checkpoint-sqlite 3.1.3, langchain-core 1.8.0), sync durability, thread identity, real interrupt and Command resume.
- Separate write-ahead model/tool effect journal with hash-chain verification and native checkpoint head binding. Uncertain or replayed effects fail closed.
- Actual repository runner integration; development CLI defaults to LangGraph and planning. Explicit legacy mode and no-hierarchical ablation; existing Python API default unchanged for older callers.
- Native nested stream and read-only checkpoint inspection; AST permission, public checks and independent strict verifier boundaries preserved.

## Actual CPU test evidence
- First native suite: 6/6; first combined regression: 68/68.
- Expanded native suite: 10/10 (6.63 s).
- Final combined command: `python -m pytest -q tests/unit/test_gpu_harness_repo_agent.py tests/unit/test_gpu_harness_repo_public.py tests/unit/test_gpu_harness_ablation.py tests/unit/test_native_langgraph_runtime.py`.
- **72 passed in 46.95s.** Ruff and formatter checks passed on all 8 touched Python files.
- Tests use a real Git ToolHost, actual Tree-sitter AST identity/permission checks, actual SQLite persistence and LangGraph subgraph execution. The model and CUDA checker are deterministic test substitutes. The provider adapter test verifies planner/reflect role admission and prompt marker with an injected port, not a live provider request.
- CLI help confirmed `--runtime` defaults to langgraph and exposes native review/stream options.
- The direct SSH clone lacked authorization; the test checkout was obtained through the repository's existing authorized fetch and a separate local clone. No change to frozen working trees or their dependencies.

## Isolation and limitations
P8 controller remained at `8724f84`, frozen runtime at `be3dafd`. Development test dependencies were added to a separate CPU virtual environment; no new paid inference or CUDA evaluation was initiated. No claim of higher real-model Strict Fix rate, throughput, or cost advantage. The existing host checkpoint is kept as the physical-workspace rehydration proof and is compared against native graph state. Unknown external effects remain non-replayable; arbitrary crash boundaries are not all resumable. The graph deadline gates admission, while any future real benchmark still requires a hard process watchdog. No multi-agent fan-out or cross-session semantic memory is claimed. The new architecture needs a fresh frozen, same-model comparison after P8, and real-model plan/replan response quality has not yet been validated.
