# 2026-11-23 · Paired CUDA Agent 2×2 对照实验：真实执行预检与阻塞报告

## 本轮实际执行

操作者配置了 **12 个预先冻结的计划单元**：`h001/h002/h003` 各对应 `legacy_single`、`legacy_batch`、`evidence_single`、`evidence_batch`；统一每个单元 `max_steps=12`、`max_model_calls=14`、`max_public_checks=2`，一次重复。未启用参考 Workflow D，以免把不同工具策略直接混入四因素对照。

Github 执行源码提交 `95a9b1f7d5153faa4e0a864303cc3974fb4fc83b`；Tang 固定 tar 归档 SHA256：`e5cf54ea7d35875746d748bd7374d55a4cf987d3d8c2858f8f1e4f9de9451917`；tar 重建的本地 Git SHA：`7d1cf380c8b4b84dfe32741555ed78451d59417a`。它们不是同一个 Git hash，归档哈希用于建立来源关联。

先运行完全**无模型/GPU 副作用**的 `gpu-agent harness sweep-repo`，之后显式传入 `--execute --allow-paid-calls --allow-real-gpu` 和独立 0600 评测密钥文件，验证真正试图启动时能否执行。Tang 机器 RTX 4090 和 Docker 可用，但当前 Shell 环境缺少 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。

## 结果

| 事件 | 实际结果 |
|---|---|
| 冻结对照计划 | 12 个单元，h001/h002/h003 × 4 |
| 只读预检 | `ready=false`，阻塞：`MODEL_NOT_CONFIGURED` |
| 正式执行入口退出码 | **3**（预检未通过） |
| 新建输出 | 仅 `preflight.json`、`frozen-plan.json` |
| 真正创建的 Unit 运行目录 | **0** |
| 真实模型物理请求 / Token | **0 / 0**，不是“模型失败” |
| 本轮公开 CUDA 检查 | **0** |
| 本轮独立隐藏验证 | **0** |
| 四组真实 DeepSeek 修复率 | **未测量**；不能伪报 `0/12` |

- `preflight.json` SHA256：`c1279193ffed3c398d6b8caf20277513d9dae50daf13cca31dd55b01bb8645e9`。
- `frozen-plan.json` SHA256：`4b8c8910dc16d82bb4f94900788989c2ed782d9d8ad95d70d14`。
- 运行审计保存在 Tang：`/home/you/gpu-native-harness-validation-20261123/paired-p0-95a9b1f/`。
- 源码 SHA 与预检摘要见 [机器可读记录](2026-11-23-paired-preflight-model-unavailable.json)。**没有密钥值或私有评测数据上传 GitHub。**

## 恢复真实研究所需的最小条件

由操作者向受审计的运行进程安全提供 `OPENAI_BASE_URL`（HTTPS）、`OPENAI_MODEL`（固定模型版本）、`OPENAI_API_KEY`。若调用 DeepSeek 等非 OpenAI 官方兼容提供商，需确认可强制 `store=false`，并设置 `GPU_AGENT_STORE_FALSE_SUPPORTED=1`；不满足则 Fail Closed。

**不要覆盖**当前阻塞实验目录。正确做法是在同一 GitHub 冻结提交、同样的 12 单元预算和独立新输出目录中运行真实模型。每一次真实试次将保存模型物理请求/Token、工具调用、候选 SHA、公开 GPU 结果与独立 Strict V2 verdict。未通过或不确定的执行也必须记录；不得重复采样后只挑成功记录。

以本轮为止，已完成的是 **可复现对照实验执行器、失败归档规则与真实阻塞门禁验收**，尚不是模型自主决策收益验证。
