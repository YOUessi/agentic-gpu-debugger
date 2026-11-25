# Native DAG — three directed reliability fixes

## Scope and provenance

This is a development-only reliability repair. The original P8 study and the completed 60-trial four-arm architectural comparison were not edited or rerun. Frozen source of the completed comparison: daded0d6f2709e9f6e6c1f9825c40ffbf315538e. The new work is on a separate branch.

The original comparison was three development CUDA defects x 5 repetitions x four arms. A reproducible, read-only analysis is scripts/analyze_langgraph_runtime_paired.py, with its actual output in docs/benchmark/2026-11-25-langgraph-paired-analysis.json.

| Pair (first vs second) | Both pass | First only | Second only | Both fail |
|---|---:|---:|---:|---:|
| Native ReAct vs Legacy ReAct | 11 | 3 | 1 | 0 |
| Native DAG vs Legacy Planner | 7 | 3 | 3 | 2 |
| Native DAG vs Native ReAct | 9 | 1 | 5 | 0 |

Native ReAct vs Legacy ReAct has only four discordant trials: exploratory exact two-sided McNemar p=0.625. The difference is not statistically compelling in this small, repeatedly tuned development set. Native DAG used 660,510 reported Tokens vs 894,796 for Legacy Planner (26.2% less), but differences in context, prompts, tool actions and role allocation confound causal attribution.

## Original five Native DAG failures (preserved)

- r02-h002-native-dag and r05-h001-native-dag: GRAPH_TASK_FRONTIER_BLOCKED, with an open Edit task. The prior scheduler pessimistically required remaining tool steps to cover every unfinished milestone at one action per milestone.
- r04-h002-native-dag: GRAPH_PLAN_INVALID_SCOPE. The first typed plan was rejected by the controller; scope was previously checked after the provider call, outside its bounded retry path.
- r03-h001-native-dag and r03-h002-native-dag: the REFLECT/REPLANNER role, not the Executor, emitted two invalid structured outputs. Provider audit shows schema failures including an overly long revision_reason. The original runtime exited without a final Agent result.

These are system-level outcomes, not evidence that planning necessarily harms repair ability.

## Fix 1: DAG liveness

graph_schedule now allows an authorized dependency-ready task whenever at least one tool step remains, without imposing the invalid remaining_steps >= unfinished_milestones gate. The hard tool-step ceiling is unchanged. The existing blocked-replan and blocked-terminal nodes record specific reasons for no admissible action, dependency blockage, missing Finish and exhausted steps, with task statuses attached. No new Agent nodes were added. Recovery accepts a fully recorded effect-free pending graph result at a safe boundary after checking ledger and host state identity; uncertain external model/GPU effects remain non-replayable.

## Fix 2: Structured Plan Scope

The model-facing structured schema now constrains registered CUDA task target paths to the operator-pinned kernel path or empty target. For generic repositories it permits only inspect, edit and finish task gates. The independent semantic validation still requires registered public checks and the authorized target scope. Invalid plans now trigger the provider's existing role-scoped two-attempt format-validation path. No additional model call allowance is introduced; rejected plans never enter the execution DAG.

## Fix 3: complete failure terminalization

A failed Planner, Reflect or Executor model output is audited as model_failed, never labeled model_selected. The LangGraph execution reaches an explicit INCONCLUSIVE terminal result rather than allowing an uncaught provider exception to leave no result.json. A private terminal-diagnostics.json records failure stage and role, retry counts, exact last complete host Checkpoint hash and turn, current source hashes, final diff hash, provider invocation status/counts and both public/effect audit hashes. No raw generated text, API keys, private evaluator artifacts or hidden inputs are copied into these records. A model failure does not create a fictitious completed tool or Checkpoint. UNCERTAIN delivery remains non-resumable.

## Default and research tracks

The operator-facing CLI now defaults to Native ReAct (LangGraph runtime without hierarchical Planner). The native DAG remains opt-in with --hierarchical. The Python API retains its legacy default for compatibility with existing callers. The completed four-arm comparison explicitly selected each runtime and hierarchy option, so its previously recorded allocation is unchanged.

## Verification

- 13/13 newly added targeted DAG, scope, exact-once, and finalization tests passed on Tang. These include real ToolHost and actual provider audit objects with injected non-paid SDK port responses.
- The broader targeted regression after changes passed 218/218 in 91.08 seconds. It covers previous repository CUDA workflows, AST tool permissions, native graph durability and crash recovery, provider contract, and Linux worker lifetime tests.
- Code lint/format and git diff checks passed.
- Formal GPU and paid-model improvement remains UNMEASURED for this repaired version. The original five failures are development regression inputs and are not re-sampled.

## Scientific guardrails

The 60 original outcomes still count as originally recorded. No selective retry of Native DAG failures is a valid performance comparison. A separate frozen code revision and independent unseen multi-step task set are needed to estimate whether Planner/DAG adds value over Native ReAct after engineering defects are corrected.
