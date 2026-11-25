# Native LangGraph: real execution and GPU comparison launch

## Execution record

The runtime used for the new acceptance and GPU comparison is frozen at
`daded0d6f2709e9f6e6c1f9825c40ffbf315538e`. This entry records actual provider and
embedding execution. The 60-trial GPU comparison has its own controller and waits
for the existing P8 controller to release the shared GPU lock.

### Real provider, pause/resume, and learned memory

On 2026-11-25 at 12:10:25–12:10:50 Asia/Shanghai, the actual DeepSeek provider
completed the native hierarchical acceptance command against a fresh two-file
Git fixture. No model factory or mocked provider was used.

| Measurement | Observed result |
|---|---|
| Model / endpoint | `deepseek-v4-pro` / `api.deepseek.com` |
| Acceptance | PASS |
| Wall time | 24.931 seconds |
| Physical model calls | 8, all COMPLETED |
| Reported model tokens | 22,876 |
| Tool turns | 7 |
| Pause | After turn 1; 2 model calls and 22 checkpoints |
| Resume | Accepted finish; 115 checkpoints |
| Exact source checks | Both required file contents correct; no extra edited files |
| Audit | Original effect prefix and provider records preserved; no duplicate effects |
| Memory | One learned episode bound to the run and final effect-journal head |
| CUDA / hidden evaluation | None in this CPU acceptance |

The session correctly reports `DIFF_STAGED_UNVERIFIED`. Exact fixture contents
are independently checked by the operator script; this is not a claim that CUDA
execution or hidden verification passed. The complete sanitized result is in
[`2026-11-25-langgraph-live-resume-acceptance.json`](2026-11-25-langgraph-live-resume-acceptance.json).

The reproducible command is `scripts/validate_langgraph_live_provider.py`, with
an explicit paid-call opt-in, a fresh output directory, and the provider
configuration supplied through the existing environment. It preserves failed
outcomes and does not retry a run.

### Real execution found a scheduler defect

The first live run, at `bf9e224d`, completed both exact edits in 40.541 seconds,
using 12 physical calls, 35,517 reported tokens, and 10 tool turns. Its original
trajectory is preserved in
[`2026-11-25-langgraph-real-provider-initial.json`](2026-11-25-langgraph-real-provider-initial.json).

A helper read displaced an earlier inspection witness because task evaluation
selected the latest matching observation. This moved a completed prerequisite
past its dependent edit and reopened the wrong task scope. The fix preserves the
earliest admissible witness after the existing dependency, candidate, failure,
revert, and distinct-observation filters. The exact observed sequence now has a
regression test. The two live runs are development checks with different goal
wording and pause/memory configuration; their counts are not a paired performance
comparison.

The launch review also found that a detached SDK worker could survive an external
controller kill. Linux workers now install a parent-death signal before importing
the SDK, then compare their parent PID to the controller's captured PID. Real
process tests cover SIGTERM, SIGKILL, and parent death before guard installation.

### Verification on the actual execution host

The frozen runtime passed **202 tests, with 3 initially skipped**, in 76.73
seconds. Those three were the new parent-death tests: the host Python lacks
`os.pidfd_open`. The test observer was then changed to use Linux subreaper and
bounded `waitpid` polling, leaving the runtime unchanged. All three real process
tests passed on the same host in **1.38 seconds**, importing the frozen runtime.
The test file was executed outside the frozen checkout, so its Git state and
manifest stayed unchanged.

The 205 selected cases cover repository tools, public-test integration,
ablations, native persistence/recovery, task scheduling, comparison admission,
provider contract, provider telemetry, request timeout handling, and actual
process termination. This is a targeted verification set, not a claim that every
repository test was run.

### Real pretrained embeddings

The actual `LocalSentenceEmbedder` and `SQLiteRepairStore` ran with CPU-only
PyTorch and the content-locked `paraphrase-MiniLM-L3-v2` model. The memory module's
SHA-256 is unchanged between the tested source and the frozen runtime above.

- Functional checks: **18/18** passed, including actual 384-dimensional vectors,
  offline inference, persisted embeddings, namespace isolation, read-only
  enforcement, repeatability, and lesson sanitization.
- Fixed semantic examples: **4/5** top-1 hits, versus **0/5** for BM25-only on the
  same queries. A compile-description miss is retained without fixture tuning.
- No GPU or paid model API was used for this embedding validation.

The fixed queries, original failed perfect-recall expectation, separate
functional/retrieval metrics, two-process repeatability, package versions,
official model hash, download lock, and reproducible scripts are preserved in
[`2026-11-25-real-embedding-validation/`](2026-11-25-real-embedding-validation/).
These five controlled examples demonstrate the retrieval path; they are not a
held-out estimate of retrieval quality.

## GPU comparison already launched

The controller was started on Tang as PID `933632` from its own detached process
session. Its first observed log line was `WAITING_FOR_SHARED_GPU_LOCK`. It holds
no trial reservation and makes no model or GPU calls while waiting. On acquiring
the lock, it revalidates the frozen code, dependencies, task inputs, budgets, and
allocation before admitting the first trial.

| Frozen design | Value |
|---|---|
| Arms | legacy-react, native-react, legacy-planner, native-dag |
| Cases | h001, h002, h003 |
| Repetitions | 5 per arm/case; 60 total |
| Model | deepseek-v4-pro |
| Per-trial ceilings | 16 tool steps, 24 physical calls, 8,192 output tokens per call |
| Time ceilings | 180 seconds per request; 600 seconds each for agent and strict phases |
| Public tools | At most 6 compile-only calls, 2 full public checks, 4 probes |
| Memory / review | Off in every comparison arm |
| Canonical manifest SHA-256 | `97a3399a4f6f85d1c4699f7e1a7954fecc24a0b7e213a60e4252831bde9cd897` |
| Exact manifest file SHA-256 | `d9093386b0bfe0fcdfd4c0f5d7bb9881e4e9647a1fac29f8bd100d9fd82522df` |

The frozen manifest is
[`2026-11-25-langgraph-comparison-manifest-daded0d6.json`](../benchmark/2026-11-25-langgraph-comparison-manifest-daded0d6.json).
The earlier `bf9e224d` plan admitted zero trials and remains untouched. Existing
P8 checkouts and trials are not modified or resampled. The new comparison keeps
timeouts and unknown outcomes in its primary denominator and refuses to replay
reserved attempts. Private evaluator artifacts remain outside agent tools,
memory, and the public result export.

No GPU comparison success rate is asserted by this launch record. Trial results
must come from the new controller's actual public and independent strict
verification evidence.

## Automatic public result export

`scripts/watch_langgraph_comparison.py` publishes the frozen manifest, current
summary, controller identity/state, and bounded public trial records to the
dedicated branch `results/langgraph-comparison-daded0d6`. It publishes initially,
when execution starts, after each five new results, and at a terminal state. It
uses the comparison runner's existing denominator calculations and preserves
timeouts and unresolved reservations.

The exporter is observational: it never starts, retries, resumes, or changes a
trial. It reads no private evaluator output or raw agent/model logs. Git writes
use a separate temporary index and a non-forced push, preserving the existing
controller's HEAD, index, and worktree. A temporary local bare-repository check
verified export filtering, timeout/unknown accounting, parent commit identity,
unchanged worktree/index/HEAD, and an unchanged restart producing no new commit.
