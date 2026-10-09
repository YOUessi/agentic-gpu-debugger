# V2 production evidence operator runbook

This runbook is for the trusted release operator. It describes the production sequence; it is
not evidence that the sequence has been run. The V2 release stays closed until the real 16+8
corpus, 240+120 signed evaluations, evaluator scoring, fixed release suite, and final release
check all pass on one clean commit and one corpus cutoff.

The repository ships no production schedule signer and no production private key. Production
signing belongs to an independently controlled executable. Never place its private key in this
checkout, either RunStore, a build artifact, CI variables available to untrusted jobs, or release
output.

## 1. Freeze the commit and environment

Use a fresh checkout at an absolute path. Stop if it is dirty, if `HEAD` is not the reviewed
commit, or if the locked environment is not healthy.

```bash
cd /opt/releases/agentic-gpu-debugger
test -z "$(git status --porcelain)"
FINAL_COMMIT="$(git rev-parse HEAD)"
test "$(git rev-parse --show-toplevel)" = /opt/releases/agentic-gpu-debugger
/home/release/conda_env/agentic-gpu-debugger/bin/python --version
/home/release/conda_env/agentic-gpu-debugger/bin/python -I -m pip check
```

Record `FINAL_COMMIT`, the reviewed `containers/toolchain.lock.json` hash, prompt version, model
configuration, and the intended provider in the controller change record before doing any live
work. Do not amend or rebuild the commit after evidence collection starts.

## 2. Provision the external Ed25519 signer

Provision an Ed25519 key pair outside the checkout and both RunStores using the organization's
key-management procedure. Put only the reviewed public key at
`/srv/gpu-agent-controller/keys/schedule-authority.pub`. Keep the private key behind the external
signer; this project has no production key-generation or signing command.

The executable configured below must be an absolute, non-symlink path to a regular file owned by
the operator, executable by the owner, and not writable by group or other users:

```bash
export GPU_AGENT_SCHEDULE_AUTHORITY_COMMAND=/srv/gpu-agent-signer/bin/schedule-authority
test -x /srv/gpu-agent-signer/bin/schedule-authority
```

For each invocation, the application sends exactly one
`EvaluationScheduleSigningRequest` JSON object on stdin. It has `schema_version: 4` and binds the
transaction and evaluation IDs, target store, schedule and immutable `RunBinding`, selection,
modes, split, repeats, random seed, record-only cost policy, case/template universe, corpus namespace
and cutoff, authority key hash, and pristine queued-run hashes. The signer must return exactly one
`EvaluationScheduleReceipt` JSON object on stdout with:

```json
{
  "state": "COMMITTED",
  "request": { "schema_version": 4 },
  "algorithm": "Ed25519",
  "payload_hash": "<64 lowercase hex>",
  "signature_hex": "<128 lowercase hex>"
}
```

The returned `request` must be byte-semantically identical to the input model. The payload is the
domain separator `gpu-agent-evaluation-schedule-v3\0` followed by canonical JSON for that request;
`payload_hash` is its SHA-256 and `signature_hex` is the Ed25519 signature. The command must exit
zero, emit no stderr, and finish within the bounded client timeout. The client invokes it without
a shell and with only `PATH=/usr/bin:/bin`, `LANG=C`, and `LC_ALL=C`; it does not pass provider
credentials.

## 3. Provision the production corpus family

The controller root, public store, evaluator store, and checkout must be distinct absolute paths.
Keep the controller and stores owner-only. Provisioning records only the public signing key.

```bash
install -d -m 0700 /srv/gpu-agent-data/v2-public
install -d -m 0700 /srv/gpu-agent-private/v2-evaluator
install -d -m 0700 /srv/gpu-agent-private/v2-evaluator/runs

gpu-agent benchmark provision-family \
  --controller-root /srv/gpu-agent-controller/family-v2 \
  --public-store /srv/gpu-agent-data/v2-public \
  --evaluator-store /srv/gpu-agent-private/v2-evaluator/runs \
  --repository /opt/releases/agentic-gpu-debugger \
  --schedule-public-key /srv/gpu-agent-controller/keys/schedule-authority.pub

export GPU_AGENT_CORPUS_FAMILY_ROOT=/srv/gpu-agent-controller/family-v2
export GPU_AGENT_RUN_ROOT=/srv/gpu-agent-data/v2-public
export GPU_AGENT_EVALUATOR_ROOT=/srv/gpu-agent-private/v2-evaluator
```

Reopening that family must reproduce its namespace and store identities. Never substitute ad hoc
store roots on later scoring or release commands. `GPU_AGENT_EVALUATOR_ROOT` names the owner-only
parent; its `runs` child is the exact evaluator store pinned by the family. The previous layout
that pinned the parent itself is invalid. Provision a new family; do not edit or migrate an old
family configuration or reuse evidence registered under it.

## 4. Validate and register exactly 16+8 cases

First produce native clean and mutant validation run IDs through the reviewed GPU controller. For
each of the 16 public cases, register its exact pair in the public store:

```bash
gpu-agent benchmark validate PUBLIC_CLEAN_RUN_ID PUBLIC_MUTANT_RUN_ID \
  --corpus-root /srv/gpu-agent-data/v2-public \
  --visibility public
```

Repeat exactly 16 times and verify at least four public cases cover each Sanitizer family. Then,
from controller-only private inputs under `/srv/gpu-agent-private/cases-v2`, produce and register
eight evaluator-only pairs:

```bash
gpu-agent benchmark validate PRIVATE_CLEAN_RUN_ID PRIVATE_MUTANT_RUN_ID \
  --corpus-root /srv/gpu-agent-private/v2-evaluator/runs \
  --visibility evaluator
```

Repeat exactly eight times. The eight private cases must have distinct private case, template, and
operator identities. Do not copy their sources, identities, validation logs, or an alias map into
the checkout or public store. Before proceeding, reconcile the family ledger to exactly 16 public
and eight evaluator registrations at the intended cutoff.

## 5. Attest pricing and obtain explicit budget authorization

Review the provider's current HTTPS price source and retain its exact bytes in controller storage.
Load provider endpoint/model settings in the trusted shell, then create an owner-only attestation
bound to `FINAL_COMMIT` and the source-content hash:

```bash
export OPENAI_BASE_URL=https://provider.example/v1
export OPENAI_MODEL=reviewed-production-model

gpu-agent benchmark attest-pricing \
  --repository /opt/releases/agentic-gpu-debugger \
  --commit "$FINAL_COMMIT" \
  --input-usd-per-million REVIEWED_INPUT_RATE \
  --output-usd-per-million REVIEWED_OUTPUT_RATE \
  --source-uri https://provider.example/pricing \
  --reviewed-at 2026-09-21T00:00:00 \
  --source-content-hash REVIEWED_SOURCE_SHA256 \
  --output /srv/gpu-agent-controller/attestations/v2-pricing.json

export GPU_AGENT_PRICING_ATTESTATION=/srv/gpu-agent-controller/attestations/v2-pricing.json
```

Real provider calls require user authorization. Do not infer authorization from an attestation,
an earlier run, or available account credit. Record actual token usage and calculable costs only;
there are no per-unit or total dollar limits, balance checks, or cost-triggered stops. Unknown
cost stays null. Load the provider API key only into this trusted controller shell and never
print or persist it. The existing 40-request unit boundary prevents unbounded agent loops.

## 6. Run the signed 240+120 evaluations

Use all five modes and three repeats. The development command must report
`16 case × 5 mode × 3 repeats = 240 units` before execution:

```bash
gpu-agent benchmark evaluate \
  --mode all --split development --repeats 3 \
  --corpus-root /srv/gpu-agent-data/v2-public \
  --case-root /opt/releases/agentic-gpu-debugger/benchmarks/public \
  --repository /opt/releases/agentic-gpu-debugger \
  --commit "$FINAL_COMMIT" \
  --toolchain-hash REVIEWED_TOOLCHAIN_HASH \
  --model-config-hash ATTESTED_MODEL_CONFIG_HASH
```

Record the completed development `run_id`. Then run the private holdout against only the
evaluator-controlled source root. It must report `8 case × 5 mode × 3 repeats = 120 units`:

The evaluation source root is a **runtime snapshot**, not necessarily the original private
validation tree. Each `<case-id>/public_input/` directory must contain the exact registered
`kernel.cu`, `vector_io.cpp`, `vector_api.h`, `json.hpp`, and `input.json` bytes. Private batch
validation can resolve a shared harness and an input elsewhere through `source-manifests.json`;
the evaluation service instead reads `input.json` adjacent to the kernel. Passing that original
tree directly can therefore validate successfully and then fail before the first model call.

Before scheduling, stage all eight snapshots in a new evaluator-owned directory (directories
0700, files 0600), resolving files only through the validated private manifests. Check every
source hash and the registered input hash before and after copying. Keep the original private
tree unchanged, reject duplicate basenames, and never copy truth, reference implementations,
alias maps, or labels into the runtime snapshot. Use that snapshot root as `--case-root` below.
Do not fabricate default inputs, change registry hashes, or overwrite a started evaluation to
recover from a layout error; retain the failed run and explicitly record any replacement run.

```bash
gpu-agent benchmark evaluate \
  --mode all --split holdout --repeats 3 \
  --corpus-root /srv/gpu-agent-private/v2-evaluator/runs \
  --case-root /srv/gpu-agent-private/evaluation-snapshots-v2 \
  --repository /opt/releases/agentic-gpu-debugger \
  --commit "$FINAL_COMMIT" \
  --toolchain-hash REVIEWED_TOOLCHAIN_HASH \
  --model-config-hash ATTESTED_MODEL_CONFIG_HASH
```

Do not continue on a stopped, failed, unsigned, partial, wrong-commit, wrong-cutoff, or
wrong-configuration run. Preserve all 360 unit outcomes, including failures and inconclusive
records.

## 7. Prepare the 120-record evaluator label package

The external evaluator must adjudicate the blind holdout projection and create one canonical
`HoldoutLabelPackage` with exactly 120 unique judgments. Its header must bind the completed
holdout evaluation, evaluator alias-mapping run, signed schedule, alias hash, corpus cutoff,
record-set hash, and tracked rubric hash. Each judgment binds one blind ID, public-record hash,
blind-payload hash, labels, score, inconclusive decision, and private-holdout decision. It must not
contain private case/template identities or the alias-map nonce.

Place it outside the checkout and both RunStores, in an existing owner-only directory, as a
single-link regular file owned by the operator:

```bash
install -d -m 700 /srv/gpu-agent-controller/adjudication
install -m 600 /srv/external-evaluator/v2-holdout-labels.json \
  /srv/gpu-agent-controller/adjudication/v2-holdout-labels.json
```

Reject symlinks, extra hard links, group/other permissions, partial transfers, and post-review
changes. The evaluator should transfer the file through an independently authenticated channel.

## 8. Score, collect, freeze, derive, and check

Bind the complete label package. The optional metrics copy remains evaluator-controlled and must
also be outside the checkout and both stores:

```bash
gpu-agent benchmark score-holdout \
  --evaluation-run-id HOLDOUT_EVALUATION_RUN_ID \
  --private-binding-run-id PRIVATE_ALIAS_MAPPING_RUN_ID \
  --labels /srv/gpu-agent-controller/adjudication/v2-holdout-labels.json \
  --metrics-output /srv/gpu-agent-controller/adjudication/v2-holdout-metrics.json \
  --repository /opt/releases/agentic-gpu-debugger
```

Require `scored 120/120` and retain the scoring session ID and metrics SHA-256. Then collect the
fixed release suite on the same completed development evaluation:

```bash
gpu-agent release collect-evidence \
  --repository /opt/releases/agentic-gpu-debugger \
  --development-evaluation-run-id DEVELOPMENT_EVALUATION_RUN_ID
```

The release-test run must finish with zero skips and zero failures. Create an owner-only output
directory, then freeze the canonical selection from exactly four roots; this command never scans
for a newer run and never publishes incomplete evidence:

```bash
install -d -m 700 /srv/gpu-agent-controller/release-v2

gpu-agent release freeze-selection \
  --development-evaluation-run-id DEVELOPMENT_EVALUATION_RUN_ID \
  --holdout-evaluation-run-id HOLDOUT_EVALUATION_RUN_ID \
  --private-binding-run-id PRIVATE_ALIAS_MAPPING_RUN_ID \
  --release-test-run-id RELEASE_TEST_RUN_ID \
  --output /srv/gpu-agent-controller/release-v2/release-selection.json \
  --repository /opt/releases/agentic-gpu-debugger
```

Derive the manifest without overwriting an existing file, and check it against native evidence:

```bash
export GPU_AGENT_RELEASE_SELECTION=/srv/gpu-agent-controller/release-v2/release-selection.json
export GPU_AGENT_RELEASE_MANIFEST=/srv/gpu-agent-controller/release-v2/release-manifest.json

umask 077
set -o noclobber
gpu-agent release derive-manifest \
  --selection /srv/gpu-agent-controller/release-v2/release-selection.json \
  --repository /opt/releases/agentic-gpu-debugger \
  > /srv/gpu-agent-controller/release-v2/release-manifest.json
set +o noclobber

gpu-agent release check \
  --selection /srv/gpu-agent-controller/release-v2/release-selection.json \
  --manifest /srv/gpu-agent-controller/release-v2/release-manifest.json \
  --repository /opt/releases/agentic-gpu-debugger
```

Proceed only when the final JSON says `"passed": true` and has no reason codes. A generated file,
a completed command, or a green offline test is not a substitute for this real evidence check.

## 9. Build, hash, push, run CI, tag, and release

From the unchanged clean checkout, run the complete offline regression and static checks required
by the release process, then build both artifacts with the already locked build backend and no
isolated dependency download:

```bash
cd /opt/releases/agentic-gpu-debugger
/home/release/conda_env/agentic-gpu-debugger/bin/python -m build --no-isolation
sha256sum /opt/releases/agentic-gpu-debugger/dist/*.whl \
  /opt/releases/agentic-gpu-debugger/dist/*.tar.gz \
  > /srv/gpu-agent-controller/release-v2/distribution-sha256.txt
```

Inspect the wheel and sdist contents, verify the hashes from an independent release host, and
install-smoke the wheel outside the checkout. Then perform these state-changing steps in order,
using the repository's protected release procedure:

1. Push the reviewed commit/branch.
2. Wait for required CI on that exact commit; CI is offline evidence only.
3. Create the approved signed version tag on `FINAL_COMMIT` and push the tag.
4. Wait for tag/release CI and recheck artifact hashes.
5. Publish the immutable release and attach the verified wheel, sdist, hashes, and evidence
   summary without evaluator-private data.

Do not push, tag, or publish if the checkout moved, the real release check closed again, an
artifact hash differs, or any private key, label, score, identity, alias, nonce, or evaluator path
appears in Git-tracked bytes or public release output.

## Repair v3

本节说明显式启用的公开开发工作流 `public-repair-v3`。默认 `gpu-agent repair` 仍使用
`public-repair-v2`；传统 `diagnose`、冻结 A–E 评测及上文 V2 release 证据流程保持原行为。
Repair v3 支持 D/E 调查模式：当前 CLI 使用 E，没有 `--mode` 参数；应用层
`ApplicationService.repair(..., mode="D")` 可选择 D。A–C、绑定评测的运行和 evaluator
store 不能使用这一多轮修复入口。入口与限制见 [CLI](../src/gpu_agent/cli.py) 和
[应用服务](../src/gpu_agent/service.py)。

### 启用与前置条件

沿用已配置的隔离 GPU 后端、provider、知识库和公开功能规格。输入目录需要受支持、
绑定原始 `kernel.cu` SHA256 的 `task.json`，有效的公开输入，以及当前支持的
`vector_api.h` 接口。自带公开案例提供相应文件。前置检查失败时会报告
`PUBLIC_TASK_UNAVAILABLE`、`PUBLIC_TASK_INVALID`、`PUBLIC_INPUT_INVALID` 或
`PUBLIC_INTERFACE_UNSUPPORTED`，不会通过重新调查来绕过规格检查。

```bash
gpu-agent repair benchmarks/public/case_0009/public_input --allow-paid-calls \
  --reinvestigate --max-reinvestigations 1 --max-candidates 3 --max-llm-calls 40
gpu-agent report RUN_ID
```

将第二条命令的 `RUN_ID` 替换为 repair 输出的实际 ID。`--allow-paid-calls` 是发送模型
请求的显式授权；费用只记录，不设美元硬上限。参数以 `gpu-agent repair --help` 为准：

| 参数 | 默认值与范围 | 含义 |
| --- | --- | --- |
| `--reinvestigate` | 默认关闭 | 显式选择 v3；省略时保持 v2。 |
| `--max-reinvestigations` | 默认 1，范围 0–3 | 限制整次 repair 的重新调查次数；单独设置不启用 v3，设为 0 时不启动重新调查。 |
| `--max-candidates` | 默认 3，范围 1–20 | 包含首次候选在内的候选上限；最后一个候选失败后不再调查。 |
| `--max-llm-calls` | 默认 40，范围 1–40 | 限制整次 repair 的物理模型请求，包含初始调查、重新调查、补丁和格式重试。 |

达到重新调查上限不会单独结束 repair：控制器记录 `REINVESTIGATION_LIMIT` 决策后，
可继续使用最近取得的有效诊断修订，直到候选或总预算用尽。提高候选数或重新调查次数
不会增加总调用、工具采集或时间配额。

### 何时重新调查

每个候选先执行公开自检。只有明确的 `FAILED` 才进入修订决策；工具不可用、超时、
取消、截断等 `UNAVAILABLE` 状态直接停止。编译错误和首次普通运行失败通常直接修订；
公开 Sanitizer finding、公开功能输出错误，以及相邻失败候选出现相同的非编译检查
结果签名，可触发重新调查。分类依据公开检查状态，不能当作已经查明根因。
决策由 [RepairCoordinator](../src/gpu_agent/repair_coordinator.py) 保存。

重新调查会创建以原始诊断 run 为父级的独立 `repair_reinvestigation` public 子运行，
重新准备、编译和运行失败候选，再继续受控工具调查。该候选内仍要求先取得 memcheck
结果，再运行其他 Sanitizer；同一候选内仍拒绝没有信息增益的重复动作。

若控制器对该子运行的实际公开输出重新执行功能检查并确认失败，会把结果绑定到当前
输出工件，允许 v3 在没有 Sanitizer finding 时完成有依据的功能故障诊断。上轮自检失败
本身不能满足这个条件；v2 的证据要求保持原样。投影与引用检查见
[AgentOrchestrator](../src/gpu_agent/agent/orchestrator.py)。

### 诊断与补丁的源码作用域

父 run 的 `diagnosis.json` 保留原始源码语义。每次候选调查只将当前候选的源码、实际
工具证据及本次检索文档放入新的 `PublicEvidence`。`PublicRepairContext` 中的上一轮
诊断、`previous_diagnosis_source_sha256`、候选 hash 和失败自检反馈是带来源的上下文；
它们不是当前事实，也不会扩大新诊断的合法 citation 集合。

修订反馈同时提供 `diagnosis_source` 和 `diagnosis_source_sha256`。诊断行号与位置必须
按这份 hash 匹配的源码解释；`previous_candidate_source` 只对应最近失败的公开检查。
例如 C1 已重新调查、随后 C2 失败，但重新调查额度已耗尽时，生成 C3 所用的 diagnosis
仍可能属于 C1。不能把其行号当作 C2 的行号。无论 diagnosis 属于哪个候选，返回的
完整 replacement diff 都必须应用到原始 `public_source`，上下文行和删除行也必须
来自原始源码。具体模型契约见 [V3 提示词](../src/gpu_agent/agent/prompts.py)。

首次调查与首次补丁沿用旧提示契约；带 V3 investigation context 或 V3 修订反馈的模型
请求使用 `public-repair-v3-2026-10-08-v1`。应按各物理调用的 `prompt_version` 核对版本，
不能把整次运行中的所有调用都视为使用 V3 提示词。

### 共享预算与自检计数

初始调查与所有候选重新调查复用同一 provider、调用 gate、采集 ledger 和截止时间。
启动新调查只重新预留最终诊断与补丁的尾部调用，不清空已经使用的次数或格式重试。

| 资源 | 整次 repair 的限制与统计口径 |
| --- | --- |
| 物理模型请求 | 最多 40 次，可由 `--max-llm-calls` 调低；重试也计数。 |
| 修复阶段时间 | 原始 600 秒截止时间由调查、补丁和公开自检共享；各工具超时按剩余时间裁剪，不为候选重新计时。 |
| Agent 动作 | 初始调查加全部重新调查共享最多 38 步。 |
| 调查 Sanitizer | 初始调查加全部重新调查共享最多 4 次实际采集调用。 |
| 官方文档检索 | 初始调查加全部重新调查共享最多 3 次实际 RAG 调用。 |
| 候选公开自检 Sanitizer | 与调查采集分开计数；每个候选最多执行四种工具，前置检查或工具不可用可提前结束。受候选上限与同一修复截止时间约束。 |

因此“调查最多 4 次 Sanitizer”不等于“整个 repair 最多 4 次”。最终独立 strict verifier
沿用原有执行和限额机制，其检查也不能记成 Agent 调查采集。共享计数实现见
[预算 gate](../src/gpu_agent/agent/policy.py) 与 [公开自检循环](../src/gpu_agent/repair.py)。

### 产物位置与停止原因

public RunStore 默认在 `.gpu-agent/runs`，可由 `GPU_AGENT_RUN_ROOT` 指定。每个 run 的
`<RUN_ROOT>/<RUN_ID>/manifest.json` 列出工件名称、hash 和 `relative_path`；下表是工件的
逻辑名称，不是可以直接拼接的文件路径。实际内容保存在 manifest 指向的
`<RUN_ID>/artifacts/<ARTIFACT_ID>`，应通过 RunStore 读取并校验 hash。

| 所属运行 | 逻辑工件名称 | 核查内容 |
| --- | --- | --- |
| 父 run | `repair/policy.json` | 实际采用的 v2/v3、候选上限和重新调查上限。 |
| 父 run | `repair/<N>/candidate.json`、`repair/<N>/result.json` | 第 N 个候选 hash 与其自检结果；自检前发现重复时可能只有 candidate 工件。 |
| 父 run | `repair/<N>/decision.json` | 失败后的修订/重新调查决定及 reason；最后一个候选不再生成后续决策。 |
| 父 run | `repair/<N>/feedback.json` | 下一次修订实际使用的公开反馈、`diagnosis_source` 与来源 hash。 |
| 父 run | `repair/<N>/reinvestigation.json` | 重新调查 run ID、候选 source hash、新诊断和 `budget_after`。 |
| 父 run | `repair/summary.json` | 最终 `stop_reason`、选中候选、各轮结果、调查子 run 列表、累计采集用量和自检 Sanitizer 次数。 |
| 父 run | `agent/budget.json`、`agent/final-budget.json`、`agent/budget-audit.json` | 累计预算、repair 结束时的调用/剩余时间和预约审计；最终值查看 final-budget。 |
| 父 run | `agent/acquisition-usage.json`、`agent/usage-summary.json` | 实际调查采集次数、物理模型调用次数及是否使用 synthetic provider。 |
| 重新调查子 run | `repair/context.json`、`repair/lineage.json` | 历史上下文、父 run、自检 run、source hash、提示版本和 `budget_before`。 |
| 重新调查子 run | `diagnosis.json`、`evidence/bundle.json`、`actions/<STEP>/step.json`、`actions/<STEP>/decision.json` | 当前候选的诊断、证据和受控动作轨迹。 |
| 自检子 run | `self-check.json`、`self-check-usage.json` | 公开自检结果及该次实际 Sanitizer 调用数。 |
| provider 所属父 run | `provider/<INVOCATION_ID>/<STATE>.json` | 每次物理模型调用的状态、prompt version、时间和用量。 |

`repair/summary.json` 只在进入候选循环后产生；前置规格检查或首次诊断失败时，先查看
CLI 错误、父 run 的诊断 limitations 与实际可用产物。常见停止原因如下：

| `stop_reason` | 含义与下一步 |
| --- | --- |
| `PUBLIC_CHECKS_PASSED` | 公开自检通过，随后调用一次原有独立 strict verifier；仍须读取最终 verdict。 |
| `CANDIDATE_LIMIT` | 候选数用尽；最后一个失败候选不再调查。 |
| `REPEATED_CANDIDATE` | 新候选源码 hash 已出现，停止空转；与触发调查的重复公开失败信号不同。 |
| `PUBLIC_CHECK_UNAVAILABLE` | 自检证据不可用；查看该轮 checks，尤其 `interruption`，不能按通过继续。 |
| `REINVESTIGATION_INCONCLUSIVE` | 子运行未建立有效诊断；查看该轮 `reinvestigation_limitations` 和子运行 diagnosis。 |
| `REVISION_REJECTED` | 新补丁未通过原始源码匹配或补丁范围等校验。 |
| `AGENT_BUDGET_EXHAUSTED`、`LLM_TIMEOUT`、`LLM_UNCERTAIN_INVOCATION` 等固定错误码 | 修订或调查遇到预算/模型边界；发生在调查内部时也可能作为 inconclusive 的 limitation 保存。 |
| `PUBLIC_REPAIR_SOURCE_MISMATCH` 等来源错误码 | 候选、自检或上下文来源不匹配；停止而不混用证据。 |

`REINVESTIGATION_LIMIT` 通常是 `decision.reason`，表示本轮改为直接修订，不能误读为
repair 已停止。模型请求为 `UNCERTAIN` 时不盲目重放，也不新建 provider 清空状态。

### 公开边界与结果解释

重新调查与修订只使用公开功能规格、公开输入、候选源码和实际公共工具证据；不能读取
private holdout、ground truth、隐藏参考实现或独立 verifier 的结果来指导下一候选。
公开自检通过后才执行一次现有严格独立验证，隐藏验证失败不会返回模型继续修复。
`PUBLIC_CHECKS_PASSED` 不是最终成功；CLI 仅在最终 verdict 为 `VERIFIED_FIXED` 时以 0
退出。容器隔离、受限补丁、来源校验和 V2 release 门禁均保持原机制。

本节说明已实现的行为契约，不宣称 GPU/LLM 能力收益或修复率提升已验证。脚本化 GPU
冒烟、真实模型实验、实际失败轮次与环境限制以
[2026-10-08 逐轮记录](repair-log/2026-10-08-repair-v3-reinvestigation.md) 为准；脚本化
provider 的闭环验证不能替代同一冻结案例、模型、输入和预算下的 V2/V3 效果比较。
