# P8 watchdog isolation and accounting — 2026-11-25

## Scope

Controller code validated at `8724f84875bf49c4c23e518d384ca23e3d52a45b`; original runtime remains `be3dafd6988972dfdb51cf2faee6984103565973`. Original manifest SHA256: `47d38ec2b9c363dd4acb29074f46aad6dd9913dac2bc35bc10bcd0d3ba050610`. The separate timeout amendment has SHA256 `75f244c4608f8b11908c2c533401114ca81b54fe21540d2eba0396963d2514f9`. No change to original manifest, Harness, prompts, model configuration, budgets, task order, or verifier.

## Frozen timeout

- Trial: r03-qwen-R-B-h001; Qwen3-8B-AWQ; replicate 3.
- Execution: 600.008 seconds, exit -15, watchdog.
- Outcome: OUTCOME_UNKNOWN / TIMEOUT; candidate correctness UNKNOWN; strict_fix_within_budget=false; retry_same_replicate=false.
- Verified journal: 46 events; last complete tool boundary turn 11 repo_replace_syntax REJECTED (event 45); last event turn 12 model_started (event 46). Checkpoint's journal head matches event 45; checkpoint counter is 44. No checkpoint resumed.
- Provider invocation artifacts: seven distinct physical calls, six completed, one uncertain. Completed request elapsed time totals 479.273s. Last started at 07:42:01.893 Beijing time; termination near 07:43:30, about 88.2 seconds later. Remaining elapsed time about 32.5s. Server log HTTP responses are not treated as direct request-ID matches. No exact split of queueing, generation, transport and parsing is available.
- Observed usage: 20,962 input, 35,225 output, 34,303 reasoning (subset of output), 6,352 cached, 56,187 total tokens. Last request usage unavailable, so total is a lower bound.
- Public artifacts record 6 builds (13.622s), 5 run invocations (2.035s), 4 sanitizers (2.381s), including setup and prefix evidence; these are recorded subprocess times, not kernel-only timing. Counts are not claimed as live-only actions. 7 public runs and 187 artifact refs passed inventory verification.
- Timeout record SHA256: `1a1c02ccca8a889af0e7520ff837b877d721107e8392a0e5f75fb84630b49b54`.
- Journal SHA256: `b05f0d18a4f1c6335bf44effa2635fe551306a8843f75dfa1400b53f034ef62d`.
- Checkpoint SHA256: `489d57ce5b454f37caceebc37569e6f86e0519c040235d0fb7dbf2ba934bf1ac`.

## Durable isolation

Existing 213 ledger events retained. Appended event 214, WATCHDOG_ISOLATED, with timeout record digest, no_resample=true and cleanup proof. New ledger head `003799c4a36d89db9c1293148f0023a1c7110a991fd0eacc7728a8d4e19aca04`. Under shared GPU lock, observed 706 MiB memory in use, no compute processes, no running containers, no trial-owned processes, port 8127 closed. Released lock after audit. No new model or GPU evaluation actions on this attempt.

## Current interim metrics

| Model | Started | Determinate | Strict successes | Timeouts | Primary | Secondary | Observed tokens | Agent wall s | Strict wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek | 18 | 18 | 12 | 0 | 66.67% | 66.67% | 715137 | 827.526 | 141.950 |
| Qwen | 15 | 14 | 3 | 1 | 20.00% | 21.43% | >=688141 | 4573.157 | 34.382 |
| Total | 33 | 32 | 15 | 1 | 45.45% | 46.88% | >=1403278 | 5400.683 | 176.332 |

Primary Wilson 95% descriptive intervals: DeepSeek 43.75–83.72%, Qwen 7.05–45.19%, pooled 29.84–62.01%. Qwen hypothetical timeout sensitivity 20.00–26.67%; the upper endpoint is not a measured success. Actual billed cost unavailable. Observed total agent+strict wall time per success: 80.790s DeepSeek and 1535.846s Qwen; observed tokens per success 59594.75 and >=229380.33. These are trial execution times excluding scheduler/service startup, not monetary or kernel benchmarks. Interim task distributions are unequal and contain repeated development tasks.

## Validation and status

23/23 scheduler unit tests passed; Ruff passed. Frozen paths unchanged. Dry run independently verified 32 existing determinate records and 15 successes, found 27 pending units, and performed zero model calls and zero new GPU checks. The next original pending trial is r03-qwen-R-A-h003.

The execution launch was blocked by the tool's safety check. No alternate launch attempted. The offline isolation and reporting were completed independently; controller is not running, 27 trials remain pending, and no additional formal samples were started. A future legitimate resumption must preserve the existing ledger and use the original pending trial order. Restarting Qwen with the original block seed cannot reconstruct its previous RNG stream or cache; document the restart rather than claiming exact continuation.
