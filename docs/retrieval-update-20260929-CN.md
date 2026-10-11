# 检索修复与实际环境接通（2026-09-29）

按用户要求回补的六轮详细经过（包含失败、原因、方案、实际修改、命令与验证边界）
见 [逐轮修复记录](repair-log/2026-09-29-retrieval.md)。后续记录遵循
[记录规范](repair-log/README.md)，本文件保留为概览及历史中间状态。

## 当前结果

已补齐索引内容、排序和实际使用路径，不再只有一个独立的 metadata CLI。
新的 `2026-09-29.1 / cuda-lex-v6` 索引含 92 个片段，哈希为
`15c53142f8949ba1d852a3227fd11460060adec359fda9ee0cf42c52011b8600`。
原有 29 条 development 查询和标签均未修改：lexical hit@5 为 **29/29**，
semantic 23/29，hybrid 27/29；默认仍为 lexical。
五条同步、竞争、初始化查询的既有 chunk 级标签全部命中。
其他查询主要按 source 级标签计分：命中来源不等于已经证明模型回答正确。
这是用于开发修复的集合，不是独立盲测或普遍检索正确率保证。

## 改动和依据

- `evidence` 来源原先只校验、不生成片段。新增显式 opt-in，将实际文档版本
  JS 和发布说明 `updates-in-2025-1` 正文入库。旧配置默认行为不变；版本核验、
  来源限制、滚动手册段落白名单不放宽。元数据片段明确标注文档版本不等于工具版本。
- NVIDIA 样例保留紧邻 kernel 的原始块注释，不生成摘要，不写入案例答案。
- v5 的标识符拆分基础上，v6 保留原词并加入保守的英文复数/第三人称词形。
- 检索实测发现同一长文档的多个片段挤占前五。v6 lexical 在 k>=3 时优先每个
  匹配来源最多两段，再以原排名补足剩余位置；不会加入没有词项匹配的文档。
  这是公开的 source-diversity 策略变更，不使用 query ID、相关标签或特殊来源名单。
  k<3 和旧分词版本的选择逻辑保持不变。
- 当前 Conda 环境之前导入 `/home/you/releases/agentic-gpu-debugger-v2/src/gpu_agent`；
  已经通过不升级依赖的 editable 安装切换至本工作区。
  `/home/you/.config/agentic-gpu-debugger/provider.env` 仅更新索引路径，密钥未变。
  新开启的 shell 需要重新 source 该文件，已运行进程不会被自动替换。

## 复核和产物

本机产物根目录：
`/home/you/.codex/worktrees/a352/agentic-gpu-debugger/.cache/gpu-agent/knowledge/`。

- `20260929-v5/`：本次从六个批准官方来源完整抓取的 sources、receipts、index、比较报告。
- `20260929-v6/`：同一批 92 个片段的 v6 索引、重建凭据、比较报告、configured-check.json。
  configured-check 验证真实环境的代码路径、索引哈希、工具版本和 29/29 查询命中。
- 使用 `scripts/build_knowledge.py --repository <repo> --output <新目录>` 可按当前
  manifest 重新构建；输出目录必须不存在。官方资料发生变化时仍按原规则拒绝。

首次相关测试 105 passed、2 failed：v6 加载器版本白名单遗漏，以及候选清单版本
未同步。两项均修正；仅针对受影响模块加 Agent 回归复测，31 passed。
两批覆盖去重后共 128 项，所有观察到的失败已复测通过；没有声称一次全量运行通过。
新增 Agent 路径测试经过真实 service/orchestrator、存储和引用检查；provider 与
容器子进程为模拟，不能算真实 GPU/模型实验。
Ruff 检查/格式检查通过，`mypy --strict src/gpu_agent` 的 72 个文件通过。

旧 corpus 和冻结的实验记录均未覆盖。本次没有模型/API/GPU 实验，没有重算历史
诊断率，也不能把新检索得分拼接进旧版本的评测结论。

---

## 以下为前一轮中间状态（已由上述修复取代）

## 已实现

- `gpu-agent knowledge metadata`：只读抓取凭据，返回来源、文档版本、抓取时间、
  内容哈希和凭据引用。必须提供预期凭据文件 SHA256；未知来源、重复凭据、
  URL/版本/固定内容哈希不一致会拒绝。没有凭据的来源明确返回 NO_FETCH_RECEIPT。
- `cuda-lex-v5`：保留精确代码符号，同时对驼峰、下划线、限定名拆出词语。
  例如 particleScatter 可匹配 particle scatter，完整 API 名匹配仍保留。
  文档标题匹配继承 v4。算法不读取开发问题 ID 或标签。
- reindex 脚本增加 `--tokenizer-version`，能从已有可信片段离线构建新索引。
  未改旧版本分词行为、未改 29 条标签、未加入模型生成的文档摘要。

## 实测

相同 79 个片段、相同 29 条开发查询：v4 lexical 18/29，v5 lexical 20/29。
semantic 15/29、hybrid 17/29。8 条非正文来源问题仍计为正文检索未命中，
另有 1 条正文排序漏命中。没有把元数据存在当成发布说明正文命中。

新索引 hash：031cabe22ed3a036fb1dbd44345e019cbe6477ae2e50633ce77c4f95cc93ebe7。
本机索引、原始比较和元数据输出在 `/tmp/gpu-agent-retrieval-20260929-rQafQ1/`。
这是开发集改进结果，不是新的独立盲测结果。

97 项相关测试通过，包括四组未出现在原 29 条查询里的自编标识符/自然语言测试、
不存在内容的查询、旧版本行为、索引保存加载、篡改凭据拒绝、真实 CLI 入口。
这些合成测试证明对应功能，不冒充真实文档相关性盲评。
Ruff 通过；src/gpu_agent strict mypy（72 个文件）通过。

## 使用

```bash
gpu-agent knowledge metadata \
  --manifest knowledge/sources.json \
  --receipts /home/you/gpu-agent-knowledge-20260924/receipts.json \
  --expected-receipts-sha256 b678681697c74971f7dcfc2d4def94b93d3c09b220dc11f18b311a1bf35b2da6 \
  --source-id sanitizer-version-probe
```

不带 --source-id 可列出所有来源。该入口不访问网络，不调用模型，不修改实验目录。
新正文索引可通过 GPU_AGENT_KNOWLEDGE_INDEX 显式选用；本轮没有改现有 provider 配置。

## 边界与未完成项

元数据入口是独立 CLI，尚未作为 Agent 的新动作接入；没有修改已有模式契约。
文档版本不是本机工具版本，来源清单中的兼容范围也不是重新探测的工具版本。
凭据仅证明当地记录过抓取；不等于本次重新验证网页或拥有发布说明正文。
“2025.1 具体更新了什么”仍需存储并核验对应正文，不能从凭据自动编答案。
当前未重新切分或扩充正文：已经定位到的代码词项问题先通过通用分词处理；
剩余自然语言语义漏检尚未解决，未引入新的 embedding 服务。
本轮没有 DeepSeek/GPU 调用，也没有运行整个离线套件或覆盖旧实验结果。
