# 2026-11-23 · Frozen paired REAL-model GPU Agent P0 experiment

## Research question

Does an **evidence-first context** or **atomic multi-snippet edit action** improve a real LLM's ability to debug registered CUDA kernels, holding source, model endpoint/name, case and resource budget fixed? The historical script-driven 3 vs 6 turns did **not** answer that question. Neither fake CUDA nor scripted model outcomes may be counted as LLM successes.

## Frozen four-arm design

Use the same frozen SHA of this project, the same versioned HardSet V2 manifest, public numerical oracle and the **same** LLM API configuration for four arms. The two manipulated factors are:

| Arm | Context | Multi-snippet edits |
|---|---|---|
| legacy_single | legacy: list/read file before editing | no |
| legacy_batch | legacy | yes |
| evidence_single | full registered mutant source in initial snapshot | no |
| evidence_batch | full registered mutant source in initial snapshot | yes |

Cases are h001 Softmax32, h002 RMSNorm tail, h003 two-stage center. Default preflight is a one-case h001 pilot; for the complete *development* comparison specify all three cases. Each case starts in a distinct isolated SHA-pinned Worktree. No model response, public GPU result or Strict Verifier result is visible to another trial. The order of four arms is **precommitted** and rotated over case index/repetition, not optimized in response to earlier results. Provider sampling seeds are **not claimed fixed** because this model endpoint does not guarantee seed support; `--repetitions` counts independent trials, not identical RNG seeds.

An optional `workflow_D` arm uses the original fixed-probe, diagnosis-and-patch Workflow; it differs in tools and prompts. Report it as a **descriptive reference**, not as an isomorphic ablation arm. Do not compare its `tool_turns` (fixed graph) to freely selected repository tool calls.

## Faithful budgets and independent results

Each free-choice arm receives the **same max model requests, max LangGraph steps, max public GPU checks**, public functional requirement and independent CPU oracle. The existing model provider physically audits usage, rejects uncertain calls and stores usage metadata. No GPU or billable model call happens during preflight.

A completed arm can obtain a public result from the pinned Docker CUDA backend (fixed tests + four Sanitizers). **Only after the Agent's session is terminal and publicly accepted**, a separate operator-only Strict Verifier runs secret hidden inputs in a new evaluator RunStore; these are never model tools. Even a `VERIFIED_FIXED` from HardSet experimental strict V2 is **not** signed production holdout certification.

The trial result distinguishes `EXPERIMENTAL_STRICT_FIXED`, `STRICT_NOT_FIXED`, `PUBLIC_NOT_ACCEPTED`, `MODEL_OR_RUNTIME_ERROR`, and `INCONCLUSIVE`. Raw SDK exception messages are withheld to avoid credential/prompt leakage. Model token usage is `null` if missing—not zero or imputed. **All precommitted trial indices must appear** in the final report. Any trial failure is recorded and never automatically retried or dropped from the denominator. The controller aborts the full sweep if the frozen source SHA, registry SHA, model name or endpoint host changes.

The protocol does not yet claim equal total dollar budget across model types or across the fixed workflow, because pricing, internal retry counts and GPU checks differ. Token/call/step/GPU counts and wall time are reported separately; cost USD remains unknown unless verified pricing is supplied.

## Commands (real API requires explicit operator setup)

```bash
REPO=/absolute/path/to/agentic-gpu-debugger
OUTPUT=/outside/the/repo/new-agent-paired-experiment

# Safe read-only preflight (no network/LLM, no Docker, no output created):
gpu-agent harness sweep-repo \
  --repository "$REPO" --output "$OUTPUT" \
  --cases h001,h002,h003 --repetitions 1 \
  --max-steps 12 --max-model-calls 14 \
  --max-public-checks 2 \
  --evaluator-secret-file /operator-private/secret.bin

# Only run AFTER OPENAI_BASE_URL, OPENAI_MODEL and OPENAI_API_KEY
# have been injected securely into the operator process environment.
# A non-OpenAI compatible endpoint must explicitly support store=false:
# export GPU_AGENT_STORE_FALSE_SUPPORTED=1
gpu-agent harness sweep-repo \
  --repository "$REPO" --output "$OUTPUT" \
  --cases h001,h002,h003 --repetitions 1 \
  --max-steps 12 --max-model-calls 14 \
  --max-public-checks 2 \
  --evaluator-secret-file /operator-private/secret.bin \
  --execute --allow-paid-calls --allow-real-gpu

# Optional fixed Workflow D for descriptive comparison only:
# append --include-workflow-d to preflight and real run.
```

`--execute` without both explicit approvals returns a refusal. Missing API configuration or a missing/non-private (not 0600) evaluator secret results in `preflight.json` and `frozen-plan.json` with typed blockers **before trial/GPU/API action**. This blocked result is not counted as an experimental failure, because no trial actually began. Once a real trial begins, it is never auto-retried, and its `unit-NNN.json` is written exactly once. A source drift or other fatal integrity violation aborts the sequence rather than silently continuing with changed inputs.

Expected outputs for a completed run:

```text
output/
  preflight.json        # no API key or hidden input
  frozen-plan.json      # SHA, model name/host, order, budgets
  unit-001-.../         # immutable agent and external evaluator evidence
  unit-001.json         # outcome, measured tokens/steps, failure class
  ...
  paired-report.json    # all planned units with no survivor bias
```

## What is still genuinely missing

The Tang session used for the preceding script-driven GPU check had **none** of `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY` or the store=false setting. Without an authorized live provider configuration, this experimental runner must report `MODEL_NOT_CONFIGURED` and **must not** invent a DeepSeek result. Prior 2026-11-21 D/E live DeepSeek runs and 2026-11-23 free-choice failure are **historical, non-paired** evidence; their 3/3 and 0/1 cannot be used as new 2×2 scores.

Only after a real provider is connected, run h001 pilot, then h002/h003, keep all failed attempts, conduct separate verification, and compute paired differences and confidence intervals. More HardSet cases and out-of-distribution defects are required before strong claims about model generalization, Agentic RL reward quality or improved success rates.
