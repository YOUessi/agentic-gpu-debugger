# 2026-11-23｜百度 / 腾讯 / 阿里 / 字节 / 小红书 Agent/AI 应用/AI 算法 JD 对标与研发路线

> 事实边界：核查的是公开 JD，不保证实时仍可投递；官方页面优先。腾讯部分样本为社招、字节阿里部分完整职责来自第三方 JD 聚合，不与实时官方确认混为一谈。任何“通过”必须绑定对应 Git SHA/实验 ID；未运行的实验标记为待验证。

## 一、具体招聘证据（官方优先）

| 企业与具体职位 | 可核查要求 | 官方/聚合来源 |
|---|---|---|
| 百度 Agentic RL 实习 J107371，2026-11-18 | 多工具长程 PPO/GRPO、稀疏奖励与信用分配、PRM/ORM、Rollout、DeepSpeed/FSDP、SFT→RM→RL、Agent 评测 | https://talent.baidu.com/jobs/detail/INTERN/179ca572-c5bb-48fc-a7d8-8335235aa745 |
| 百度 AI Agent 开发实习 J106681，2026-10-19 | Harness/Skills/MCP、Agent 状态机、Context/Memory、多 Agent、Trace/Bad Case、Token & latency、评测自动化 | https://talent.baidu.com/jobs/detail/INTERN/162291c8-43e8-4cc0-9c96-3fb0a7ebb720 |
| 百度 代码智能体架构实习 J101404 | 多 Harness 轨迹合成清洗、SFT/OPD/RL、沙盒训练、vLLM/SGLang、CUDA/推理性能 | https://talent.baidu.com/jobs/detail/INTERN/397771ff-709c-42f5-9fd7-2e4112e0768d |
| 百度 AI Agent 研究&开发实习 J107330，2026-11-13 | Harness RSI、AutoResearch、真实实验反馈、对照与消融、Agent Loop、Context、Memory、工具调用 | https://talent.baidu.com/jobs/detail/INTERN/8d2918ff-2a7e-4721-b94a-947d73d18bd9 |
| 百度 Agent 算法实习 J103894 | 难任务评测、推理时搜索 scaling、verifier-guided search、拒绝采样、技能/记忆、自我改进 | https://talent.baidu.com/jobs/detail/INTERN/70a8d8a6-5aff-47ad-88c8-24aa47ab1812 |
| 百度 文本大模型算法实习 J106676，2026-10-19 | SFT/LoRA/DPO/RLHF、数据筛选配比、Transformers/DeepSpeed、Bad Case 闭环 | https://talent.baidu.com/jobs/detail/INTERN/869332a3-f1ce-4a02-b0e8-d407a9666ae1 |
| 小红书 Agent 产品研发实习 | 企业 Harness、Scheduler、Sandbox/Approval、Trace/Replay/Observability、真实业务和测试 | https://job.xiaohongshu.com/campus/intern/position/20736 |
| 小红书 AI Agent 算法实习 | Agentic post-training、记忆/个性化、Context 优化、搜索、SFT/DPO/PPO/GRPO | https://job.xiaohongshu.com/campus/intern/position/20284 |
| 小红书 Agentic AI 研究实习 | Planner/推理/Memory/Tool、Simulator、Benchmark、Data Pipeline、Agent UI DSL | https://job.xiaohongshu.com/campus/position/20145 |
| 腾讯 混元 Agent Harness Engineer | tracing/observability、Eval A/B/regression、Debug、数据管道、严谨代码；**社招**能力标杆 | https://careers.tencent.com/jobdesc.html?postId=2052685072754196480 （全文可交叉参考 https://www.jdwatch.work/jobs/jcrdqkxs9） |
| 腾讯 大模型强化学习框架 | RL 训练基础设施、DeepSpeed/FSDP/Megatron、Verl/ROLL/AReal、显存/通信优化；**社招** | https://careers.tencent.com/jobdesc.html?postId=2072330929078190080 |
| 阿里 AI Agent 算法工程师实习 | Agent 全生命周期、SFT/RL、Planning、RAG、评测、Qwen/DeepSeek，**第三方全文** | 官方 https://campus-talent.alibaba.com/campus/position/199903900038 ; 收录 https://jobs.ultraai.site/jobs/alibaba_campus/internship:199903900038 |
| 字节 AI 搜索/推荐系统 Data AML | Agent Skills、数据质量、特征生成、A/B 分析、分布式/高可用，**校招而非该职位实习** | 官方 https://jobs.bytedance.com/campus/position/7672773197708413237/detail ; 收录 https://jobs.ultraai.site/jobs/bytedance_campus/7672773197708413237 |
| 字节 AI Agent 研发开发者服务 | 需求→编码→测试→上线、RAG/MCP、上下文工程、服务可靠性；**社招、第三方收录** | https://jobs.ultraai.site/jobs/bytedance/7485248634141804807 |

注意招生资格/毕业时间窗口、实习时长/每周到岗天数及官网存活状态须单独核验。

## 二、合并 JD 后的 3 类岗位，不混淆难度

1. **Agent Harness / AI 应用工程**：LangGraph 状态机、Tool Contract、MCP、Skills、审批沙箱、上下文工程、Memory 更新、RAG 索引版本、Replay、Observability、真实业务端到端。验收：任务成功率、错误恢复、延迟、Tokens、成本、可维护性。
2. **Agent 算法研究**：Evidence-aware Planning、inference-time search/verifier 反馈、失败归因、测试时预算分配、自我改进、轨迹学习。验收：同一个模型同样预算，独立任务的成功率与置信区间比规则基线实际提高。
3. **大模型后训练/AI 算法/Infra**：PyTorch/Transformer、SFT/LoRA/DPO/GRPO、Rollout/Reward/PRM、数据清洗、vLLM/SGLang、CUDA profiling/分布式。验收：独立验证的训练增益、GPU 性能与成本，而非只有框架名和 Loss 曲线。

## 三、当前 CUDA Agent 的代码与证据对照

| 能力 | 可核查现有内容 | 缺口与门槛 |
|---|---|---|
| 自研 Agent Harness | LangGraph Model→Tool Loop、Typed Actions、控制器权限、Git Worktree、SessionJournal/Resume | 多模型 provider 适配/重播语义、长任务复盘与错误恢复指标不完善 |
| 公共 GPU 工具 | HardSet h001/h002/h003 注册入口；h001 的真实 Docker NVCC+Oracle+四 sanitizer | 真实模型自主动作 A/B；更多 case 与分布外任务；验证器有效性边界 |
| 独立校验 | h001 脚本化候选已通过 experimental strict V2 9/9 hidden + 4/4 sanitizer | 不能称真实 LLM 成功，不能代替 signed holdout |
| Context/记忆 | 有确定性压缩、历史工具索引、检查点与重复读取拒绝 | 目前未证明对真实 LLM 的决策收益；持久语义/过程记忆与更新评测不足 |
| 轨迹与成本 | RunStore 公共事件、模型物理请求日志、Token/失败报告、首次 patch/check 统计 | 公开多模型公平对照表、统计置信区间、延迟与美元成本、错误分布 |
| Agentic RL | 当前 CUDA 仓库没有可证实的 SFT→GRPO 训练闭环 | 必须先建立 trajectory + reward 正确性，再训练并独立检验 |

**不能在简历把脚本化模型说成 DeepSeek 自主修复；不能把 h001 一例的通过率宣传为泛化结果。**

## 四、按优先级的大块研发，非按周排期

### P0｜模型自主决策：直接针对先前 14 工具 / 晚 patch 的失败

任务：给注册公共 CUDA Case 提供**证据优先上下文**，让模型首轮看到完整当前 mutant，而不是反复列目录；引入**同一文件 2-4 段不重叠的原子更新**，减少逐段 edit 的动作数量；保留 legacy 与 evidence_first 两种模式做消融。补拒绝无效重复操作、余额提醒、候选哈希绑定与恢复检查。

验收：同一 case、同一模型、同一温度/seed 配置、同一任务与模型请求上限，记录成功率、动作轨迹与费用。脚本化对照只能证明工具能力，**不算模型收益**。先以 h001/h002/h003 做开发/回归，再构造未见过的 CUDA 故障族作为独立评测。不能用 clean source 训练或引导模型。

### P1｜基准与成本可信度

- **Baselines**：固定规则 D、原自由决策 Agent、增强版自研 Harness；所有组运行同一模型配置与工具/GPU/Token 上限。
- **Metrics**：public pass、independent strict/holdout pass、first valid edit/check、重复动作、无效调用、模型请求与 Tokens、真实 GPU 编译次数、延迟、公开计算成本。
- **Protocol**：至少跨任务和随机种子；记录所有失败及基础设施故障，不对失败重抽样；开发集调参与独立测试隔离；统计置信区间与每题配对结果。
- **Gate**：提升任务成功率或相同风险下显著减小成本，且差异能在独立测试重复出现。若失败，先改决策上下文/工具反馈，不继续堆方法。

### P2｜Skills、MCP、Memory/RAG、Trace Replay

用确实需要的 CUDA 诊断技能构建有版本、签名、触发条件与权限的 Skills：例如 numerical instability、race/sync、out-of-bounds。MCP 放在工具互操作层，不取代自研执行授权；检索文档按发布版本管理，文档更新时自动标记索引过期并重建。前端增加单次 Trace 的动作回放、源码 Diff、GPU 反馈与预算时间线、失败聚类。**只有对用户任务有用的工具才引入。**

### P3｜从可验证轨迹到 Agentic Post-training

1. 从 RunStore 派生统一的 `state/action/observation/reward/terminal` 轨迹，不包含 hidden 输入或 verifier secret；过滤重复、非法动作和不确定故障。
2. 先训练 **SFT/LoRA** 工具选择基线，保留训练前/后独立结果；再考虑 DPO/拒绝采样蒸馏。
3. 如果过程与结果 Reward 能被独立验证且已有可重复改进，再进行 TRAIN-only GRPO/Agentic RL；使用独立测试集、GPU 成本约束、置信区间分析。
4. 扩展到 Deep Research Agent 的检索/证据链任务，验证不是只对 Softmax 这类已知 CUDA Mutant 有效。

### P4｜交付可面试的应用，而不是再开一堆空仓库

在现有 CUDA Agent FastAPI/React Console 中提供有权限门控的运行配置、SSE 流式事件、可恢复会话、审批 UI、公共/隐藏评测隔离、诊断 Trace Replay 与 A/B 比较面板。选至少一个真实研发使用流程做端到端演示（例如 CUDA PR 修复、公开回归测试、生成审计报告），并记录失败率、运行成本与恢复场景。

## 五、非 CUDA Agent 的补充路线

- **AI 应用开发岗位**：再把 Deep Research Agent 用作检索/知识库/搜索场景，展示检索更新、引用正确性、结构化工具调用和线上服务；不必让 CUDA Agent 冒充业务搜索系统。
- **通用 AI 算法岗位**：需要可复现 PyTorch/Transformer、SFT/LoRA/DPO/GRPO 的真实训练；多模态岗位可使用 VLM/视频项目，补视频理解、评价与适量微调实验。
- **训练/推理 Infra 岗位**：要有 vLLM/SGLang、KV cache、batching、CUDA profiling 或通信性能对比数据；当前的 GPU 调试器只是其中一个证明。

## 六、项目交付的证明标准

每个功能必须有：冻结代码 SHA、输入/模型/预算版本、源数据与工具调用来源、关键失败轨迹、对照组、自动化测试、真实验收结果、成本与局限。**若无真实模型评测，不在简历写“成功率提升 X%”或“利用 Agentic RL 提升模型能力”。**


## 七、2026-11-23 实现结果与下一道真实模型门禁

针对 P0 已在同一 PR 的实验性分支落地：

- `context_mode=legacy|evidence_first`：注册 h001/h002/h003 案例时，新模式模型首轮直接获取当前公开 mutant 的源码；不引入 clean/oracle。
- `repo_replace_many`：2–4 个相同文件内相互独立、不重叠的 exact snippets 必须全部校验后原子提交；不允许修改测试/控制器/私有路径。
- `allow_batch_edits=True|False`（CLI `--disable-batch-edits`）：作为第二个独立消融变量；连同 context mode 固定在 Journal、Checkpoint 和最终公开指标中，Resume 不得改变。
- **2×2 同一模拟决策器对照**：Legacy+逐段=5 次动作；Legacy+Batch=4；Evidence-first+逐段=4；Evidence-first+Batch=3。四组都在 Fake GPU Checker 上获得同一个候选哈希；**不是真实 DeepSeek 的成功率提升**。
- **实际 Tang RTX 4090 Docker GPU**：固定源码 `06a8923542f78ee47a4c367ffd0a40c083ee66d2`，新版脚本化决策 3 工具动作、1 次 public GPU 检查、四个 Compute Sanitizer 全 CLEAN，0 物理模型调用。完整审计见 [报告](../repair-log/2026-11-23-evidence-first-atomic-gpu-validation.md)。

接下来真实 DeepSeek/其他公开模型实验采用四组 2×2 配对加规则 D 基线，比较独立 Strict Fix 与费用/质量曲线；所有组冻结相同 case/硬件/model/请求数与 GPU 预算，记录每一次失败的 RunStore，不挑选成功轨迹再比较。受审计模型环境缺失时保持 `MODEL_NOT_CONFIGURED`，不能用脚本化模拟数据替代。

### 什么还不应立即做

**不**因为 JD 写了 MCP/Skills/Multi-Agent/GRPO 就一次性加入所有框架。Skills/MCP 先按真实可调用的 CUDA 调试工具设计；Memory/RAG 必须能证明更新/召回正确性；Agentic RL 必须有真实轨迹、独立 Reward 与训练前后优势。当前最优先的仍然是对比 **规则 D vs 真实自由决策 Agent**，解决“Agent 尚未优于规则系统”的核心事实。
