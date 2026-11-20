# 2026-11-21 · LangGraph CUDA Repair Agent end-to-end GPU smoke (scripted-model control)

## What this really establishes

**This was a complete native GPU execution of the newly implemented Agent control loop. The decisions and patch strings were supplied by a test scripted-model double, not by an actual DeepSeek/OpenAI LLM. Do not call it an LLM-repair win or compare its model efficiency with a genuine Agent or D rule baseline.**

Runtime source: \`ca01ec6deb02ecade2a23e21ded52552d9409f83\`. Tang: NVIDIA RTX 4090 Laptop GPU, existing project CUDA 12.8 / Docker \`IsolatedGPUBackend\`. No paid model requests; physical LLM calls = 0, dollar cost unknown/null.

Original source: pinned \`h001\` Softmax32 mutant, incorrect on large logits; functional requirement is stable group-wise Softmax. Scripted E planner chose the legal public Functional Probe size 65, then stopped. The fake model diagnosis cited the real probe evidence. Two model-output diff strings were preconstructed: first a scope-valid comment-only patch preserving the bug; then a scope-valid replacement matching the known clean kernel. No oracle, hidden outputs or private evaluator data were used by the decision graph.

### Controller/Graph actual trace

Root public RunStore ID: \`f01907655ab14f99a04f60199473c93a\`.

| Agent step | Real execution / outcome |
|---|---|
| \`investigate\` | Mode E scripted planner chose size 65, actual isolated CUDA probe ran and observed \`PROGRAM_FAILED\` |
| \`diagnose\` | Fake model returned source-line and real-public-probe-ID cited root-cause hypothesis |
| \`patch\` attempt 1 | Scope-valid model-double diff; candidate SHA256 \`9e99406389b68f9c41827e57ef83d05d104fe8942d07c6e07b6820023bc34335\` |
| \`public_selfcheck\` attempt 1 | Real isolated CUDA build/run: \`passed=false, PROGRAM_FAILED\`, first failing public shape \`n=33\` |
| \`patch\` attempt 2 | Fake model saw public failure and returned new full diff against original source; candidate SHA256 \`88dd297b8a0bc53dc3b93b7d1a4678551346b4855d7139b4edaddecef0070ba4\` |
| \`public_selfcheck\` attempt 2 | Real GPU build, public numeric checks and four sanitizer inspections: \`PUBLIC_CHECKS_PASSED\` |
| Final operator-only evaluator | Separate private HMAC-derived hidden inputs and four sanitizer checks: \`VERIFIED_FIXED / ALL_REQUIRED_CHECKS_PASSED\` |

The parent RunStore has exactly **8 audited Agent phase events**, two public candidate child runs, and **no private/evaluator/hidden artifacts in the parent manifest**. Final evaluator has separate public and evaluator stores and was invoked only after the Agent loop exited.

Final generated report:

- \`verified=true\`, \`verdict=VERIFIED_FIXED\`, \`candidates=2\`, \`probes=1\`, \`model_physical_calls=0\`;
- \`agent-report.json\` SHA256: \`faf1b628839308968e4b1a44f8238542a5d5150b1380e7cafa1f48d0752fa0f6\`;
- raw run storage: \`/home/you/gpu-agent-autonomous-validation-20261121/scripted-full-agent-h001-ca01ec6/\`, including \`investigation/public/\`, \`public-candidate-01/\`, \`public-candidate-02/\`, and \`strict/evaluator/runs/\`.
- Original source checkout: \`/home/you/gpu-agent-autonomous-validation-20261121/source-ca01ec6/\`, clean and SHA-pinned.

### Test evidence and hard limitations

- Tang's venv on source \`ca01ec6\`: **131 passed** across the HardSet Agent/LLM Probe unit groups (excluding unrelated full suite).
- A later PR commit \`869ab366d3889a310775d326a2176e2afce67a9d\` added actual provider-wire MockPort regression for the new \`diagnose_hardset_v2\` and \`propose_patch_hardset_v2\` functions. Targeted GitHub workflow [#44834582874](https://github.com/YOUessi/agentic-gpu-debugger/actions/runs/44834582874) passed **135 offline tests**. Full Python3.11/3.12 CI requires separate confirmation.
- Attempting the genuine DeepSeek E paid run in the previous branch hit an execution environment safety block. No actual HardSet model-produced diagnosis, diff or verification result exists, and this report makes no such assertion.
- **The next research milestone is actual authorized E LLM-generated repairs**, including failures, original candidate code hash, full public selfcheck, final independent strict score, model calls/input/output/cached tokens and GPU execution cost. Only then compare with an equally provisioned D baseline, multiple independent cases and repeated seeds.

**This smoke proves the complete Agent engine works against real CUDA backends and reacts to public failures. It does NOT establish that the LLM can solve the new task or beat the rule system.**
