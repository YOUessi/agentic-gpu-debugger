# 2026-10-08 — Repair v3 失败候选重新调查

## 任务与来源

- 用户确认在现有架构基础上优化失败后的诊断更新，优先实现 Repair v3。
- GitHub 基线：`design/v2-operator-workflow@ab46153c0190aa73ed11305b4181a46dccf2be5f`。
- 工作分支：`feat/repair-v3-reinvestigation`。
- 设计与计划：`docs/superpowers/specs/2026-10-08-repair-v3-reinvestigation-design.md`、
  `docs/superpowers/plans/2026-10-08-repair-v3-reinvestigation.md`。

## 已确认的问题与决定

1. 原 `repair_candidates` 每次都使用首次 diagnosis，没有重新获取证据的路径。
2. 失败候选可能消除最初 OOB、暴露其他错误；重新调查原始源码不能覆盖这种情况。
3. 新建 BudgetLedger 会重置真实计数；采用共享 ledger/gate、候选独立 evidence scope。
4. 子运行诊断的行号/citations 不能覆盖父运行原始诊断；单独保存，作为修订上下文使用。
5. 固定 self-check 和 Agent 调查的 Sanitizer 成本分开记录，不混用统计口径。

## 环境与验收范围

开发在独立云端 checkout `/workspace/scratch/cca353f9d23f/agentic-gpu-debugger` 进行，
解释器为该目录的 `.venv/bin/python`（Python 3.12.14）。Tang 上已有未提交工作的项目目录不作开发用途；
真实 GPU 验证将从 GitHub 提交建立另一个独立检出。

验收包括：默认 V2 兼容、失败候选独立调查、有效新诊断进入原始基准补丁、纯功能错误的 D/E 路径、
跨运行引用隔离、共享预算与截止时间、不可用停止、异常清理、三轮修订的诊断源码对应，以及原生 GPU 控制流。
CPU fake-backend 测试、脚本 provider 的 GPU 测试、真实模型效果评价分别记账；本次不运行付费模型评价。

## 第一轮：补齐失败后的调查与诊断更新

### 问题、原因与方案

原 V2 每次 self-check 失败后调用 `revise_patch`，但始终沿用第一次 diagnosis。
补丁可能修掉原始缺陷、产生另一种错误，因此重复解释原始证据不足以支持下一轮修改。
简单重新实例化 Agent 又会重置工具 ledger；把候选证据并回原始 run 则会混淆源码行号和引用。

采用独立 public 子运行承载候选的 prepare/build/run 和调查，继续使用同一 provider、gate、实际 ledger
与截止时间。父级保留最初 diagnosis；新诊断连同源码 SHA256 单独保存。补丁始终从原始快照生成。
编译失败直接修订；工具 finding、当前公开功能失败或重复运行失败按规则决定是否重新调查。

### 实际改动

- 新增 `RepairCoordinator`，负责分类、候选调查、作用域和预算汇总。
- `AgentOrchestrator.continue_in_workspace` 复用实际 ledger，重建候选内动作去重状态。
- `PublicRepairContext` 和条件提示词将历史假设与当前证据分开；没有上下文时保留旧 payload 和 prompt。
- CLI 用 `--reinvestigate` 显式选择 V3；候选上限默认 3，重新调查默认最多 1 次。
- 固定 self-check Sanitizer 次数单独保存，不能冒充调查的 4 次工具上限。

### 测试与结果

新增控制流测试首先得到 8 项失败，确认旧策略不接受 V3 且 CLI 没有新参数。实现接入后，
新控制流与旧修复/公开功能检查的较窄组合得到 **37 passed**，退出码 0（4.00 秒）。
模型上下文与预算续接由独立任务开发，并在最终组合测试中重新验证。

这些测试只替换模型或 GPU 子进程，真实 RunStore、证据引用、源码 hash、补丁校验和控制器仍运行；
这证明控制流与边界，不能作为真实 GPU 或模型能力证据。

## 第二轮：纯功能错误与测试输入的真实性

### 问题与原因

新增 D/E 用例发现：候选数值输出错误而 memcheck CLEAN 时，原规则路由继续寻找 Sanitizer finding，
不能利用已由控制器检查的公开功能错误结束调查。修复时仅在 V3 当前候选确有公开功能失败的条件下，
允许 D 选择 finish；E 使用同一证据充分性规则。V2 的证据要求不变。

D 的第一次回归仍失败；检查保存的 action/diagnosis 发现，它尚未进入候选阶段：D 查询原生 finding
`Invalid __global__ write`，测试的单片段语料只包含 `writes`，旧测试 tokenizer 未匹配，结果是
`NO_INFORMATION_GAIN`。该问题属于合成 fixture 的覆盖缺口。测试改用生产支持的 `cuda-lex-v6`，
让 D 的原生查询与 E 的脚本查询都能在同一片段上建立初始诊断，没有修改生产检索或放宽证据要求。

另外，在测试 deadline 计数时发现原 fixture 的输出缺少 numeric wire-format 字段，输入最初也缺少 `n`，
因此测试提前停在输入/输出检查而未触达 Sanitizer。补齐合法 `n/a/b` 输入与 `dtype/shape/values` 输出，
并断言具体停止码，使该测试真实到达预期边界。修正后才观察到下轮描述的计数失败。

### 结果

D、E 均可基于当前候选的真实格式公开输出和 CLEAN Sanitizer 形成带合法引用的诊断。
这里的子进程输出仍是合成数据；Tang 测试将使用真正的 CUDA 数值错误验证同一路径。

## 第三轮：独立评审与退出、溯源保护

### 发现与修复

独立评审没有发现 Critical 问题，提出两项 Important 和两项 Minor：

1. C1 调查后生成 C2，若 C2 再失败且达到调查上限，诊断仍属于 C1，而最新失败源码已是 C2。
   旧 V3 指令误要求用 C2 解释 C1 的行号。现明确使用 `diagnosis_source` 及其 hash；
   `previous_candidate_source` 只解释最近失败检查。补充 C1/C2 行内容不同的两种 provider 提示词测试，
   以及完整三轮修订的控制器测试。
2. 原生后端 `shutil.rmtree` 清理可能抛出 `PermissionError` 等 `OSError`，仅捕获
   `BackendInfrastructureError` 会跳过子 diagnosis、累计预算和父 summary。候选调查与固定 self-check
   都明确处理这两类清理错误，保存不可用状态或 inconclusive 原因，保留最后已检查候选。
3. V3 修订候选曾复制首次候选的旧 prompt version。现在修订候选记录实际 V3 版本；首次和 V2 不变。
4. 固定 Sanitizer 次数曾在 `gate.timeout(60)` 之前递增，deadline 耗尽时会记录未执行的一次。
   现在先取得有效 timeout、构造请求，再计数和调用后端。

### 回归证据

清理原生错误、第三轮 prompt 溯源和两种 self-check 清理错误的新增用例先得到 **4 failed, 11 passed**。
deadline 用例修正输入/输出后单独得到预期失败：实际后端调用 0 次，记录却为 1 次。
两项 provider 提示词测试先 RED，再与其所属文件合计 **32 passed**。

修复后的集成命令（在上述开发目录执行）：

```bash
.venv/bin/python -m pytest \
  tests/unit/test_repair_reinvestigation.py \
  tests/unit/test_iterative_repair.py \
  tests/unit/test_public_repair_correctness.py \
  tests/unit/test_repair_budget_continuation.py \
  tests/unit/test_repair_public_context.py -q
```

结果：**101 passed in 6.25s**，退出码 0。原始输出位于相邻任务目录
`/workspace/scratch/cca353f9d23f/repair-v3-integration-final.log`。
同版本 `ruff check src tests` 通过，`ruff format --check src tests` 显示 175 个文件无需格式化；
`mypy --strict src/gpu_agent` 显示 77 个源文件无问题；`pip check` 无依赖冲突。

## 第四轮：完整 CPU 回归与 Tang GPU 验证（进行中）

初始全套 baseline 在约 43% 进度后主动中断，退出码 130，不作为通过证据。
其环境包含 SOCKS 代理，而环境未安装 `socksio`，已有 SDK 构造测试会受此影响。
只对离线测试子进程移除代理变量后，该既有单测已通过；没有修改 SDK 实现或全局环境。

当前最终离线回归命令：

```bash
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u http_proxy -u https_proxy .venv/bin/python -m pytest -x -q \
  --durations=15 -m 'not gpu and not container and not live_llm and not release'
```

命令正在运行，完整结果尚待记录，不将进度中的测试点数当作通过总数。

新增 `tests/gpu/test_repair_v3_gpu.py`，直接使用原生后端与公开案例：原始 OOB → 首候选改为 `a-b`
→ 公开数值错误 → 候选重新调查得到 CLEAN memcheck 和数值错误事实 → original-base 修订为 `a+b`
→ 公开功能检查及四种 Sanitizer 全通过。provider 是脚本，未构造私有 evaluator，也不声称 VERIFIED_FIXED。
本地已收集到 1 个测试，但执行为 skip（容器环境不可用）；这不是 GPU 通过。

Tang 预检已实际返回：RTX 4090 Laptop GPU、驱动 580.178.04、Python 3.11.16、Docker 29.1.3。
实现已提交为 `dd92a364c4ddad4f1afd41c50e33c614cca11a18`；Tang 独立检出位于
`/home/you/projects/agentic-gpu-debugger-repair-v3-20261008`，已核对相同 HEAD。

### Tang 第一轮失败：测试收尾的工作目录已销毁

精确命令（在上述 Tang checkout 执行）：

```bash
/home/you/conda_env/agentic-gpu-debugger/bin/python -I -m pytest \
  tests/gpu/test_repair_v3_gpu.py::test_public_repair_v3_reinvestigates_numeric_failure_on_real_gpu \
  --require-live \
  --gpu-run-root /home/you/projects/agentic-gpu-debugger-repair-v3-20261008/runs/repair-v3-20261008/public \
  --release-evidence-report /home/you/projects/agentic-gpu-debugger-repair-v3-20261008/runs/repair-v3-20261008/pytest-report.json \
  -q -s
```

结果为 **1 failed in 28.84s**，退出码 1。父 run `4e6905f091084c9eaa0847cfb133e18d`，
候选调查 run `7bff9d8d54a348988c86b42077f072f6`。
实际 CUDA 流程、数值失败、新诊断、最终公开检查、四种 Sanitizer、引用隔离和预算断言已执行通过，
但最后的容器清理检查抛出 `RuntimeError: cannot inspect active containers`，所以整项仍记失败。
原始 stdout 和 pytest 报告保留在该 `runs/repair-v3-20261008` 目录。

原因已由堆栈和 `_docker` 实现确认：`active_containers()` 使用该 backend 的 `workspace_root` 作为
Docker 查询进程的 cwd；self-check 和子调查返回后，其 `TemporaryDirectory` 已删除，查询无法启动。
这属于新增测试的生命周期使用错误，不能将其解释为候选仍有计算错误，也不能直接忽略清理断言。

修复仅改变 GPU 测试：仍逐一查询每个 backend 的精确 owner 标签，但统一使用尚存在的原始控制目录启动
真实 `docker ps --all`，要求命令成功且无残留容器。不修改 production backend，不重建候选工作目录，
不模拟 Docker 返回。下一轮只重跑这一个失败的 GPU 测试，保留上述失败记录。

## 尚未评价的范围

没有执行真实 LLM 的 V2/V3 同条件收益对照、私有 holdout 或 release acceptance。
本轮的成功标准是新增工作流正确并有原生 GPU 证据；总体修复率和成本收益仍需独立实验。
