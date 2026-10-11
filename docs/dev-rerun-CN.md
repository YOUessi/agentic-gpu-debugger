# 开发集 240 单元重跑步骤（V2.1 契约）

目的：在修复后的干净 commit 上重跑 16 case × 5 mode × 3 repeat。结果只作开发诊断，
不是发布证据（发布还需 holdout、评分与 release check，见 `v2-operator-runbook.md`）。

以下 `$REPO` 指一个**全新 clone** 的绝对路径，`$PY` 指项目 Python 3.11/3.12 解释器。

## 0. 冻结代码

```bash
# 在工作区提交全部改动后：
git -C /home/you/.codex/worktrees/a352/agentic-gpu-debugger status --porcelain   # 必须为空
SHA=$(git -C /home/you/.codex/worktrees/a352/agentic-gpu-debugger rev-parse HEAD)
REPO=/home/you/releases/agentic-gpu-debugger-v21-${SHA:0:7}
git clone /home/you/.codex/worktrees/a352/agentic-gpu-debugger "$REPO"
git -C "$REPO" checkout "$SHA"
cd "$REPO"
$PY -m pip install --no-deps --no-build-isolation -e .
$PY -c 'import gpu_agent; print(gpu_agent.__file__)'   # 必须位于 $REPO/src/gpu_agent
```

评测 binding 现在包含 `runtime_code_hash`：被 import 的 `gpu_agent` 不在 `$REPO/src`
（例如 editable 安装仍指向旧目录）或批次中途改了代码，都会在开始前或当个单元停止。

## 1. 离线检查（不花钱、不需要 GPU）

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -I -m pytest -m 'not gpu and not container and not live_llm and not release' -q
ruff check src tests && mypy src/gpu_agent
```

恢复只接受已经持久化且能重放验证的 completion；仅有 attempt/claim、或诊断后崩溃
但结果尚未持久化时，返回 `AMBIGUOUS_STARTED_ATTEMPT`，不会重新调用模型或 GPU。
离线检查必须取得实际结果；不能把测试收集成功、缺依赖或旧失败当成通过。

## 2. 新建 corpus family 并重新验证 16 个公开 case

公开 kernel 去掉了泄漏提示（注释、提示性函数名）且 case_0001 不再硬编码 n=257，
14 个 mutant hash 发生变化（case_0002、0003 的 kernel 未改）；本轮使用新的 family，
保留旧 family 和实验记录用于追溯（runbook 第 3 节）。

```bash
install -d -m 0700 /home/you/gpu-agent-v21/public
install -d -m 0700 /home/you/gpu-agent-v21/evaluator/runs
gpu-agent benchmark provision-family \
  --controller-root /home/you/gpu-agent-v21/controller \
  --public-store /home/you/gpu-agent-v21/public \
  --evaluator-store /home/you/gpu-agent-v21/evaluator/runs \
  --repository "$REPO" \
  --schedule-public-key <你之前使用的 schedule signer 公钥>
export GPU_AGENT_CORPUS_FAMILY_ROOT=/home/you/gpu-agent-v21/controller
export GPU_AGENT_RUN_ROOT=/home/you/gpu-agent-v21/public
export GPU_AGENT_EVALUATOR_ROOT=/home/you/gpu-agent-v21/evaluator

bash gpu_batch.sh preflight
bash gpu_batch.sh all                         # 真实 GPU 验证 16 个 case，只验证不注册
$PY -m gpu_agent benchmark run-seeds --repository "$REPO" \
  --data-root /home/you/gpu-agent-v21 --register   # 全部 VALIDATED 后再注册
```

任何 case 出现 `TARGET_FINDING_MISSING`：说明改名/去注释后目标工具不再稳定检出，
先停下把日志发给我，不要改 hash 绕过。

## 3. API 冒烟（只记录费用，限制物理请求次数）

```bash
export OPENAI_BASE_URL=... OPENAI_MODEL=... OPENAI_API_KEY=...   # DeepSeek
# 3a. planner 结构化输出：3 次请求，打印每次的无值诊断
GPU_AGENT_PLANNER_SMOKE_CALLS=3 $PY -I -m pytest \
  tests/integration/test_live_planner_smoke.py -m live_llm -s --require-live
# 3b. 一个 race case 走完整诊断+补丁；包括格式重试在内最多 40 次请求
gpu-agent diagnose benchmarks/public/case_0002/public_input --allow-paid-calls \
  --max-llm-calls 40
gpu-agent verify <上一步 run_id> --generated-candidate
```

## 4. 定价证明与评测

```bash
gpu-agent benchmark attest-pricing --repository "$REPO" --commit "$SHA" \
  --input-usd-per-million <审核单价> --output-usd-per-million <审核单价> \
  --source-uri <价格页 https URL> --reviewed-at <日期> --source-content-hash <页面 sha256> \
  --output /home/you/gpu-agent-v21/controller-attest/pricing.json
export GPU_AGENT_PRICING_ATTESTATION=/home/you/gpu-agent-v21/controller-attest/pricing.json
export GPU_AGENT_SCHEDULE_AUTHORITY_COMMAND=<你之前使用的 signer 可执行文件>
```

不要用美元上限模拟“只跑五个单元”：调度顺序随机，前五个不保证覆盖 A–E，
且按实际费用累计时，总上限为单元上限的五倍也不保证恰好停止在第五个。
先完成上面的结构化输出和 race 闭环冒烟，再运行完整开发集。
费用策略固定为只记录：记录实际 usage 和可计算费用，
未知费用保留 null，不因美元金额中断；每单元仍有 40 次物理模型调用边界。

```bash
gpu-agent benchmark evaluate --mode all --split development --repeats 3 \
  --corpus-root /home/you/gpu-agent-v21/public --case-root "$REPO/benchmarks/public" \
  --repository "$REPO" --commit "$SHA" \
  --toolchain-hash <toolchain lock hash> --model-config-hash <attest 输出的 model config hash>
$PY -m gpu_agent.benchmark.dev_report /home/you/gpu-agent-v21/public <run_id>
```

这条命令是完整的 240 单元运行。不要在同一批次中途修改代码或重新安装其他 checkout。
运行后导出汇总：

```bash
$PY -m gpu_agent.benchmark.dev_report /home/you/gpu-agent-v21/public <run_id> --json > dev-report.json
```

报告按模式给出：状态分布、失败原因、诊断数、冻结族标签命中、补丁数、验证 verdict、
物理 LLM 调用数与已知费用。批次停止（`stopped_reason` 非空）时把 run_id 和报告发给我。
