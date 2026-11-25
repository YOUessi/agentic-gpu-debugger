# Graph-native repository Coding Agent (development branch)

## Motivation and migration boundary

The previous `run_tool_agent` creates a four-node planner/model/tool/reflect graph
and invokes `graph.compile().invoke()` without a checkpointer. It is a model/tool
loop, not a persistent graph runtime. The old repository checkpoint can reconstruct
the worktree and tool observations, but does not recover the native graph frontier.
The former planner has a fixed four-milestone schema and reflection fires once at
half budget rather than based on a failure.

This development branch introduces the actual `gpu-agent harness repo-run --runtime
langgraph` path. The CLI defaults to LangGraph and planning; `--runtime legacy` is
an explicit control, `--no-hierarchical` an ablation. The old Python API default is
kept for existing callers and benchmarks until they explicitly opt in. P8's frozen
checkout, task order, and trial budgets remain untouched. No hidden verifier is
added as a model tool.

## Runtime architecture

    START -> refresh -> budget_gate
                            |-- planner -> refresh
                            |-- replan -> refresh
                            |-- executor (compiled nested StateGraph)
                            |      choose -> conditional tool dispatch
                            |      inspect/edit/compile/probe/public_check/revert/finish
                            |-- terminal
    executor -> reconcile -> checkpoint -> refresh / review / terminal
    review -> native interrupt + explicit Command(resume) -> refresh / terminal

The state contains a versioned task DAG, task dependencies and evidence-derived
statuses, candidate identity, failure evidence, immutable plan revisions, typed
decisions/observations, append-only graph events, model-call accounting, absolute
deadline, and review state. Task dependencies affect ready/stale task selection
in the next model context; the model still owns the next action and must satisfy
the independent host permissions. Build/probe/public-check evidence is attached
to the exact candidate. A new edit invalidates stale candidate evidence.
A successful public check is not a Strict Fix.

The global planner uses the official provider's audited typed SDK path and a
new variable-length DAG schema. Failure-driven replanning revises the full DAG
under reserved role-specific model-call budgets, instead of periodically invoking
a fixed-stage reflection. Planner outputs cannot mark their own tasks done or
grant capabilities. Task progress is derived by the controller.

## Persistence and effect boundary

- Native SQLite checkpointer, one thread ID bound to the public run ID, sync
  durability, per-invocation executor subgraph with inherited checkpointer.
- Native interrupt at a completed tool boundary, optional candidate review after
  public pass, explicit operator Command(resume) or rejection.
- Independent hash-chained write-ahead effect journal records model planner,
  executor, replan and tool admissions/completions. An uncertain or already
  admitted effect must not be replayed. Graph, effect journal, workspace and
  legacy compatibility projection must agree before native resume.
- Existing repository checkpoint remains the host rehydration proof, not the
  source of graph scheduling state. There is no automatic replay of interrupted
  paid/GPU operations, no rollback of external reality via time travel.
- Absolute deadline persists across resume. Graph checks admissions; the
  provider retains its own remaining-time and physical-call constraints. A
  future benchmark still needs its independent hard process watchdog for
  long-running tools.
- Operator inspection via `gpu-agent harness inspect-graph --output ...` reads
  native checkpoint state/history, not a newly constructed model or host.

## Acceptance

Real model-independent worktree tests must exercise the actual CLI/API runner,
nested graph, native SQLite checkpoints, failure-driven replan, human pause and
resume, exact-once completed actions, stale evidence, and fail-closed recovery
when an effect journal/checkpoint disagree. Real CUDA/model improvement requires
a separate frozen benchmark with equal resource accounting. CPU tests do not
establish a Strict Fix rate or an advantage over the legacy harness.
