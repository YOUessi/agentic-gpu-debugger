# 2026-10-09：Repair v3 首次真实模型试用

## 背景、目标和验收条件

用户在 Repair v3 实现、离线回归和脚本化 provider 的真实 GPU 验证完成后，要求
“使用一下”。本轮使用 Tang 上既有的真实模型配置，对预先选定的公开案例
`case_0009` 执行一次 E 模式 Repair v3，观察真实调查、补丁、公开自检和最终独立验证。
实现与前一轮验证见 [2026-10-08 记录](2026-10-08-repair-v3-reinvestigation.md)，
操作契约见 [Repair v3 runbook](../v2-operator-runbook.md#repair-v3)。

本轮验收要求是：确认实际执行交付源码，保留真实模型调用和原生 GPU 证据，完整记录
候选与检查结果，并如实区分“启用 V3”“发生修订”“发生候选重新调查”和“最终修复
通过”。第一候选直接通过是正常结果；不为展示重新调查而制造错误、增加针对案例的
提示或重复抽样。本轮不是 V2/V3 修复率对照，也不改变冻结 A–E 评测。

GitHub 继续作为源码和记录的事实源；Tang 专用 checkout 只用于本次真实模型与 GPU
运行。本文件按上海日期命名，精确时间同时保留 UTC。

### 执行版本与环境

| 项目 | 已确认值 |
| --- | --- |
| 仓库 | `YOUessi/agentic-gpu-debugger` |
| 分支 | `feat/repair-v3-reinvestigation` |
| 执行提交 | `93ba8452b3f9607aab8eaa594ec2403dd35450a6` |
| Tang checkout | `/home/you/projects/agentic-gpu-debugger-repair-v3-20261008` |
| 启动前状态 | 已 fast-forward 至上述提交；工作树干净。 |
| Python | `/home/you/conda_env/agentic-gpu-debugger/bin/python`，3.11.16 |
| OpenAI SDK | 3.13.0 |
| 原生执行后端 | `IsolatedGPUBackend` |
| 工件记录的 NVCC | `12.8.93` |
| 工件记录的 Compute Sanitizer | `2025.1.0.0 (build 35583870)` |
| 编译目标 | `sm_89` |
| 已配置真实模型 | `deepseek-v4-pro` |
| 已配置 endpoint | `https://api.deepseek.com` |
| 兼容能力声明 | `GPU_AGENT_STORE_FALSE_SUPPORTED=1` |
| 公开调查模式 | E；当前 `repair` CLI 默认模式。 |

执行时通过 Conda Python 的 `-I` 和 `runpy` 引导固定导入上述 checkout 的 `src`，
provider worker 同样固定源码路径；未修改共享环境的 editable 安装。此项用于避免
另一份旧 checkout 的导入路径影响本次结果。

| 版本绑定 | SHA256 / 配置 |
| --- | --- |
| `runtime_code_hash` | `b7c08c2ecc750eacad028cd08989d5e3f6a429a6e630d2fa717c3e32c78265c6` |
| 工具链锁 hash | `3880152ba598d514a27ca364a4e812abb8f2ec9e21a7ac3be784c2d2def6ac91` |
| 本轮执行镜像 ID | `sha256:ec7f38d73b44d6e363f23cf5f2c22d7bdb93865b2a05d4110f068a7591b5f515` |
| 知识库 corpus hash | `15c53142f8949ba1d852a3227fd11460060adec359fda9ee0cf42c52011b8600` |
| 知识索引文件 SHA256 | `6ddc3cf2443ff983f08d0dcae44ca5f68ba28033beeb391a162422f9936ed27a` |
| 索引 tokenizer | `cuda-lex-v6` |
| 知识库版本约束 | `cuda=12.8.1;compute-sanitizer=2025.1.0.0` |
| 索引预检 | 92 个 chunks 全部与上述版本约束兼容。 |

既有知识索引已复制为任务专用只读文件
`/home/you/gpu-agent-repair-v3-live-20261009-01/knowledge-index.json`。本次使用已有的
官方文档本地索引；启动时仅确认索引加载与兼容性预检，结束后才从实际运行工件确认
本轮检索 1 次，详见下文动作与预算记录。模型凭据仅由控制器加载，本文件不记录
凭据值或完整环境变量。

## 证据目录和当前总状态

任务根目录：`/home/you/gpu-agent-repair-v3-live-20261009-01`。

| 位置或标识 | 用途与当前状态 |
| --- | --- |
| `public/` | 本次公开 RunStore；对应 `GPU_AGENT_RUN_ROOT`。 |
| `verification/` | 本次独立验证目录；对应 `GPU_AGENT_EVALUATOR_ROOT`，服务在其下使用 `runs/`。 |
| `knowledge-index.json` | 本轮冻结、只读的公开知识索引。 |
| `started.json` | 启动前保存实际应用 argv，不含凭据。 |
| `experiment.log` | 真实 CLI 的 stdout 与 stderr 合并输出。 |
| `completion.json` | 外层监控等待应用结束后写入起止 UTC、退出码与总耗时；本轮 CLI 退出码为 1。 |
| 启动时父监控 PID | `540233`。 |
| 启动时真实 CLI PID | `540240`。 |
| 运行起点 UTC | `2026-10-08T16:58:58.585377+00:00`。 |
| 运行起点上海时间 | `2026-10-09 00:58:58.585377 +08:00`。 |
| 运行终点 UTC | `2026-10-08T17:00:18.771510+00:00`。 |
| 运行终点上海时间 | `2026-10-09 01:00:18.771510 +08:00`。 |
| 完整应用耗时 | `80.186` 秒。 |
| 父诊断 run | `177a6daf715a4b53bfc47eeca11cf831`。 |
| 公开自检 run | `e0f0f915b89040f6924dc032ced9a715`。 |
| 候选重新调查 run | `a22d2eea2497495cb6ea567a96223cd4`。 |
| 最终注册 candidate run | `cdc7f53c5b40744a01978a98a19993cb`，仍是失败的第一候选。 |

本文件首次起草时只确认预检和启动，没有预填成功。本次结束后追加确认：**真实
Repair v3 已触发候选重新调查，但本次修复未成功**。第一候选仍有 racecheck finding；
重新调查完成 memcheck 后耗尽共享 Sanitizer 采集额度，无法再取得新 racecheck
证据，最终停止于 `REINVESTIGATION_INCONCLUSIVE`。只生成 1 个候选，未生成第二个
补丁，也未执行独立验证。上述 PID 保留为启动记录，应用已经结束。

已确认真实模型共 10 次物理调用，全部 `COMPLETED`；本次无请求 `UNCERTAIN`、无格式
重试。费用未知，不记为 0。此处结果已经与导出的
[summary.json](artifacts/2026-10-09-repair-v3-live-case0009/summary.json) 和
[provider-usage.json](artifacts/2026-10-09-repair-v3-live-case0009/provider-usage.json)
交叉核对。完整公开附件和取证范围列在本文的“取证与记录核查”一节。

## 第一轮：预选公开案例的真实运行与失败定位

### 1. 问题与影响

此前 Repair v3 已有离线覆盖与脚本化 provider 驱动的原生 GPU 闭环，但那些结果不能
证明真实模型在实际使用时会正确选择工具、给出有效补丁，或在公开失败后完成新的
候选诊断。本轮补足真实使用证据，并保留未触发 V3 调查的可能结果。

预选输入为 `benchmarks/public/case_0009/public_input`，包含源码绑定的 `task.json`、
`kernel.cu` 与 `input.json`。公开算法是 `vector-add-cpu-v1`；`n=32`，数组 `a` 全为
`1.0`、`b` 全为 `2.0`，功能要求为 32 个 `3.0`，使用现有 float32 检查与
`atol=1e-5`、`rtol=1e-5`。

| 公开输入绑定 | SHA256 |
| --- | --- |
| 原始 `kernel.cu`，与 `task.json.source_sha256` 一致 | `394dc3f945580df9ea74437ef75b2873848ca1d20522c031c7ef02b6f9cd8c11` |
| `input.json` | `6f78f9ba58b9b82becab90fa9e82bc6c046946889f510f519d6824a1ed552149` |
| `task.json` 文件 | `e39a262e8e048629df79f8523cd8bd0f68bd2a43521d726a3873f5f8c766e3ed` |

### 2. 原因与证据

静态读取公开源码可见，主 vector-add 计算之后还会启动 `folded_warp_writes`：32 个
线程通过 `threadIdx.x & 15U` 写入 16 个 shared-memory 槽位。这是本轮预期调查的
共享内存写冲突，预期主要由 racecheck 提供证据；这里的静态判断不替代本次原生
Sanitizer 的实际结果。

[2026-09-30 真实调查记录](2026-09-30-agent-live-validation.md) 中，该公开案例的
E 模式候选曾发生语义失败；[随后的一次 repair 记录](2026-09-30-iterative-repair.md)
又在首候选通过。历史说明它适合真实试用，同时也说明不能事先保证本次会触发重新
调查。历史成功与失败均不计入本轮结果。

### 3. 应该如何验证

沿用已有 provider、公开知识库和隔离 GPU 后端，在固定提交上执行正常 CLI，不注入
脚本化模型回答。提前限定最多 3 个候选、1 次候选重新调查、40 次物理模型调用，
单请求超时 120 秒，整次修复共享 600 秒截止时间。提高候选或调查上限不会额外增加
共享调用、工具采集和时间配额。

每个候选先做公开自检。若达到公开功能或 Sanitizer 失败的调查条件，则核查实际
`REINVESTIGATE` 决策、独立 `repair_reinvestigation` 子运行及其诊断来源；若第一候选
直接通过，则记录 V3 已启用但没有重新调查。公开自检通过后再读取原有严格独立
验证的结果，不能把 `PUBLIC_CHECKS_PASSED` 单独当作最终 `VERIFIED_FIXED`。

### 4. 实际准备与改动

Tang 专用 checkout 已同步至交付提交并核对干净状态。控制器采用固定 `src` 路径的
引导方式启动主进程及 provider worker，冻结知识索引至新的只读副本，并把公开运行
和独立验证目录配置到本次任务根目录。

外层监控为 Desktop Commander 启动的 Python here-doc，没有另外创建监控脚本文件。
构造子进程环境时先移除旧 `OPENAI_`、`GPU_AGENT_` 变量及 `PYTHONPATH`、`PYTHONHOME`，
再用 `shlex` 解析既有 `provider.env` 的字面赋值，仅载入 `OPENAI_BASE_URL`、
`OPENAI_MODEL`、`OPENAI_API_KEY` 和 `GPU_AGENT_STORE_FALSE_SUPPORTED`，随后设置本次
目录、知识版本和 120 秒请求超时。没有 `source` 执行配置文件；凭据仅保留在控制器
及子进程环境内存中。

本轮启动前没有修改生产代码、提示词、公开案例、正确性标准、工具链锁或共享
editable 安装。既有私有 corpus 不作为模型上下文，独立验证结果也不反馈模型。
当前文档修改用于记录本次准备、真实运行及失败分析。

### 5. 实际运行、结果与原因

工作目录为 `/home/you/projects/agentic-gpu-debugger-repair-v3-20261008`，解释器为上表
所列 Conda Python。`started.json` 保存实际应用 argv，其命令表示为：

```bash
/home/you/conda_env/agentic-gpu-debugger/bin/python -I -c "import pathlib,runpy,sys; source=pathlib.Path(sys.argv.pop(1)); sys.path.insert(0,str(source)); import gpu_agent; assert pathlib.Path(gpu_agent.__file__).resolve()==source/'gpu_agent/__init__.py'; runpy.run_module('gpu_agent',run_name='__main__')" \
  /home/you/projects/agentic-gpu-debugger-repair-v3-20261008/src \
  repair /home/you/projects/agentic-gpu-debugger-repair-v3-20261008/benchmarks/public/case_0009/public_input \
  --allow-paid-calls --reinvestigate --max-reinvestigations 1 --max-candidates 3 --max-llm-calls 40
```

实际由外层监控通过 `subprocess.Popen(argv, cwd=checkout, env=child_env, ...)` 启动，
将 stdout 写入 `experiment.log`、stderr 合并至 stdout；这不是把上面的展示字符串
重新交给 shell 执行。外层 `wait` 返回后写 `completion.json`，保留完整进程耗时，
避免只计诊断阶段而漏掉后续严格验证。

应用实际从 `2026-10-08T16:58:58.585377+00:00` 运行至
`2026-10-08T17:00:18.771510+00:00`，耗时 `80.186` 秒，CLI 退出码 **1**。外层监控
脚本退出 0 仅表示它正常收集了子进程状态，不能当作修复成功。

#### 原始调查正确定位了写冲突，第一补丁没有修复它

真实 E 模式的原始动作依次为：`run_memcheck` → `run_synccheck` → `run_racecheck`
→ `retrieve_official_docs` → `finish_diagnosis`。原始 memcheck 与 synccheck 均为
`CLEAN`，racecheck 为 `FINDING`；1 次检索返回 5 个文档 chunks。原始诊断正确指出
多个线程通过 `t & 15` 写入同一 shared-memory 槽位，根因在公开源码第 11 行的冲突
写操作。实际动作、调用前预算与 gate 决策见
[actions.json](artifacts/2026-10-09-repair-v3-live-case0009/actions.json)。

实际模型补丁只将读取位置作了以下改动：

```diff
-    if (threadIdx.x == 0) block_summary = slots[0];
+    if (threadIdx.x == 0) block_summary = slots[threadIdx.x];
```

进入该分支时 `threadIdx.x` 已经等于 0，所以两种读取在该分支内等价。补丁没有
修改第 11 行多个线程写同一槽位的操作，因此并未消除诊断所指出的冲突。这是本次
生成补丁的语义失败；实际请求全部完成，没有依据把它归因于 API 不可用。完整实际
候选补丁见 [candidate-01.diff](artifacts/2026-10-09-repair-v3-live-case0009/candidate-01.diff)。

| 第一候选绑定 | SHA256 |
| --- | --- |
| 候选 composite hash | `72dbf4f806c1d6bf9ce9b58209c04b86d35ef759c3df776bd15f2519f3e1e18b` |
| 候选 `kernel.cu` hash | `f5065948dd7278d9b3f234220bedd8f205f86d23b769dc0cce2f35c9bac4cb01` |

#### 公开自检检出了未修复的竞态

| 第一候选的公开自检 | 实际结果 |
| --- | --- |
| build | `CLEAN` |
| runtime | `SUCCESS` |
| 数值输出 | 32 个值全部为 `3.0` |
| functional | `PASSED` |
| memcheck | `CLEAN` |
| racecheck | `FINDING`，第 11 行 write-write 冲突 |
| initcheck | `CLEAN` |
| synccheck | `CLEAN` |

公开数值输出正确与共享内存竞态可以同时存在：数值要求针对 vector-add 的输出，
冲突来自额外的 `folded_warp_writes`。因此本次不能只看 32 个 `3.0` 就判补丁正确。
控制器把本轮自检判为 `FAILED`，保存决策
`REINVESTIGATE / PUBLIC_SANITIZER_FAILURE`，并实际创建了来源绑定的
`repair_reinvestigation` 子运行。

原始调查和该候选自检的 racecheck 原生日志都报告同一第 11 行 write-write 冲突，
两份日志的 SHA256 均为
`5adbb40a8b058a7ca9a2852019d8991adde003e1e3576d3d39e2899b7193ecc2`。
这支持“候选仍保留原写冲突”的判断；原始日志引用及其 run 归属见 summary 中的
`native_evidence`，不能因日志内容相同就把两个不同运行的工件混为一个。

#### 重新调查在共享 Sanitizer 额度处停止

候选子调查首先执行 `run_memcheck`，结果 `CLEAN`。此前原始调查已执行 3 次
Sanitizer，所以这次 memcheck 后，累计调查采集达到 `4/4`。下一次 planner 要求
`run_racecheck`，被 `AGENT_BUDGET_EXHAUSTED` 拒绝；再下一次 planner 仍提出
`run_racecheck`，再次被拒绝后停止。两次被拒绝的 racecheck 计划都没有执行为新的
原生工具采集，不能计入实际 Sanitizer 次数。

| 资源 | 实际用量与上限 | 本轮影响 |
| --- | --- | --- |
| 调查 Sanitizer | `4/4`，原始调查 3 次 + 候选调查 memcheck 1 次 | 已耗尽，阻止候选 racecheck。 |
| 固定公开自检 Sanitizer | 4 次，独立计数 | 不计入上述调查采集额度；其 racecheck finding 已触发重新调查。 |
| 官方文档检索 | `1/3` | 未耗尽。 |
| 物理模型调用 | `10/40` | 未耗尽。 |
| Agent 步数 | `8/38` | 未耗尽。 |
| 修复阶段截止时间剩余 | `521.5699779320275` 秒 | 时间未耗尽。 |

因此，子诊断中的 `AGENT_BUDGET_EXHAUSTED` 特指调查 Sanitizer 配额，而不是 40 次
LLM 请求用完、600 秒到期或 API 超时。原始调查与候选调查共享预算的约束得到执行，
但这次动作分配没有给重新调查保留完成 memcheck 及目标 racecheck 的采集空间。

公开自检中已有同候选的 racecheck finding；当前流程将它用于失败分类和调查上下文，
子调查仍需要重新建立当前候选的有效诊断，未受控复用该自检证据来替代新的工具
采集。这一实际路径暴露了后续改进点：需要考虑原始调查与重新调查之间的可行预算
分配，或验证同候选公开自检证据能否在严格来源校验下被安全复用。本次没有实施
这种改动，也没有通过重置计数、扩大额度或重复抽样覆盖本次失败。

#### 真实模型用量与提示版本

共 10 次真实物理请求，全部 `COMPLETED`、`attempt=0`；没有格式重试，没有
`UNCERTAIN`。其中 planner 8 次、diagnose 1 次、patch 1 次。

| 提示版本 | 实际调用 |
| --- | --- |
| `m3-2026-10-01-v12` | 前 7 次：planner 5、diagnose 1、patch 1。 |
| `public-repair-v3-2026-10-08-v1` | 后 3 次：候选重新调查的 planner。 |

不能把整次运行中的所有请求都称为 V3 提示，也不能把原始诊断和第一次补丁误记为
候选重新调查的调用。此次 tokens 合计 `31,744`，其中 input `29,742`、output
`2,002`、cached `12,672`。cached 是输入用量中的缓存统计，不另外加到 total。
API usage 中 `reasoning_tokens=0`。逐调用求和与 summary 的汇总值一致。费用未知，
记录为 `null`，不记为 0。

#### 最终运行状态

`repair/summary.json` 记录 `stop_reason=REINVESTIGATION_INCONCLUSIVE`、
`reinvestigations=1`。新诊断为 `INCONCLUSIVE`，limitation 为
`AGENT_BUDGET_EXHAUSTED`。最终只保留并注册了第一候选，没有生成第二补丁，
独立验证为 `NOT_RUN`。该注册候选是失败尝试的留存，不表示候选已通过。

本轮取得了“真实错误候选 → 公开自检失败 → 实际候选重新调查 → 有限预算下明确
停止”的使用证据，尚未取得“重新诊断成功 → 下一候选修复通过”的真实证据。

公开 RunStore 的工件名是逻辑名称；实际文件路径以 manifest 的 `relative_path` 为准，
通过 `RunStore.read(ref)` 读取并校验 hash。`gpu-agent report` 可辅助阅读公开诊断、
最终候选、验证和模型用量，但当前不展示完整 V3 逐轮视图，须另读 repair 工件。

### 6. 本轮结论与下一步

本次预选的真实试用已执行完成，修复结果失败，失败链条已定位：原始诊断正确，
第一补丁只做了无效的读取改动；公开自检发现冲突后，V3 实际启动子调查，但共享
Sanitizer 采集额度不足以完成目标工具的重新取证。这是模型补丁语义与后续调查
预算分配两个具体问题，不能合并描述为“模型出错”或“API 不稳定”。

本轮没有追加抽样、改代码或扩大预算。公开 allowlist 附件的导出与交叉核对已完成，
详见下节。仍未解决的工程问题是“调查预算的可行分配 /
同候选公开自检证据的受控复用”，本次没有实施该项改动。下一次相关开发应保留
有限总预算、来源绑定和公共证据边界，并针对本例暴露的可行性问题验证，不能用
重置计数或无限抽样代替修复。

## 取证与记录核查

运行结束后，使用 `EvidenceRepository.view` 读取并验证原始调查、候选重新调查和
公开自检三份 bundle 及其嵌套工件引用的 hash；只按公开 allowlist 导出本次必要
结果，没有导出私有 holdout。仓库内包含以下 8 个文件：

| 附件 | 内容与范围 |
| --- | --- |
| [provenance.json](artifacts/2026-10-09-repair-v3-live-case0009/provenance.json) | 被测提交、源码导入、输入与知识库 hash、实际 argv、开始/结束、运行后一致性和清理状态。 |
| [summary.json](artifacts/2026-10-09-repair-v3-live-case0009/summary.json) | 本轮策略、候选公开自检、决策、诊断摘要、预算、真实工具结果和原生日志引用。 |
| [provider-usage.json](artifacts/2026-10-09-repair-v3-live-case0009/provider-usage.json) | 10 次物理调用的种类、终态、版本、时间、用量和错误码；未复制外部请求/响应 ID。 |
| [actions.json](artifacts/2026-10-09-repair-v3-live-case0009/actions.json) | 8 个实际 planner 动作、调用前预算和允许/拒绝决策。 |
| [candidate-01.diff](artifacts/2026-10-09-repair-v3-live-case0009/candidate-01.diff) | 唯一实际生成的公开案例补丁，保留失败版本。 |
| [evidence-index.json](artifacts/2026-10-09-repair-v3-live-case0009/evidence-index.json) | 53 条选定公共工件元数据，保留逻辑名称、实际相对路径、hash 和字节数；不是原始工件全文包。 |
| [report.txt](artifacts/2026-10-09-repair-v3-live-case0009/report.txt) | 明确标注的公共结果摘要，由核对后的工件汇总；不是原生 CLI stdout 或完整产品 report。 |
| [artifact-inventory.json](artifacts/2026-10-09-repair-v3-live-case0009/artifact-inventory.json) | 其余 7 个附件的 SHA256 与字节数清单；导入仓库后逐项重算一致。 |

完整产品报告保留为 Tang 任务目录的 `report-full.txt`，SHA256 为
`713691c5a0fa64495c497d1774f8778afba4fa7de91e73a7e33f4d36e3c3a488`。仓库中的
`report.txt` 采用公共摘要，未复制完整外部请求/响应 ID 或检索文档全文；完整原始
应用输出保留在同一任务目录的 `experiment.log`。上述区别也写入附件本身，避免
把后处理摘要冒充原始命令输出。

取证辅助脚本曾发生两次只读字段使用错误：第一次误用 `ExecutionResult.status`，
查实际模型定义后改用 `runtime_status`；下一次误用 `ToolResult.payload`，改用
`typed_payload` 后完成导出。两次均是记录脚本失败，不是新的模型/GPU 实验，也
没有因此重跑任何 API 或 GPU 操作。

运行后核对执行提交、`runtime_code_hash` 和知识索引 hash 均与启动前一致。
`docker ps --all --quiet --filter label=io.gpu-agent.owner` 成功，退出码 0，返回
0 个带项目 owner 标签的容器。此项是本次结束后的实际只读清理核查，不据空日志
推定清理成功。

本轮只新增运行记录与公共附件。文档核查确认链接存在、JSON 可解析、附件字节数
与 hash 一致，并重新汇总 provider 用量；未为记录整理重跑已经结束的真实实验。

## 最终状态与交接

应用运行已经结束，CLI 退出 1。本轮真实模型调用和候选重新调查均已发生，修复
未成功；独立验证未运行。实际失败已完整保留，没有替换为额外抽样的成功结果。

本次试用记录与公开证据完整，源码无修改。目录
`artifacts/2026-10-09-repair-v3-live-case0009/` 包含 8 个文件，已完成实际内容、数值、
hash、版本和清理证据核对。本文及附件的记录版本以所在 Git 提交为准；被测源码固定为
`93ba8452b3f9607aab8eaa594ec2403dd35450a6`，记录整理未改动被测实现。

单个案例只能说明这次真实使用的结果，不能据此宣称 V3 比 V2 提高修复率或降低成本。
