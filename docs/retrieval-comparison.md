# V2 本地检索比较

V2 同时提供三种完全离线、无模型费用的候选检索方法：

- `lexical`：现有 BM25，保留为兼容默认值；
- `semantic`：将 CUDA 同义概念、token 与 token bigram 映射到固定 2048 维稀疏向量，
  使用余弦相似度排序；
- `hybrid`：对 BM25 和向量排序执行确定性的 reciprocal-rank fusion（RRF）。

`semantic` 不是把 BM25 改名成 embedding。实现会先进行显式、可审计的 CUDA 概念
规范化，再使用 SHA-256 feature hashing 构造 L2 归一化向量，并计算余弦相似度。
它也不是神经网络 embedding：不会下载模型、访问网络或调用付费 API，对当前小型官方
文档语料而言，这个边界比引入不可固定的远程 embedding 更适合复现。

## 开发标注与方法选择

### 2026-09-28 核对补充（不替换历史结果）

当前 development 清单是 29 条、`mixed_development` 标注，并非下面历史段落所述的
24 条全人工标注。固定索引 `094bbdf0a4fd28c4eead13199a50d5b9d4b61d64eddc8c23df2ac28a6217d443`
的 lexical hit@5 仍为 17/29。新增只读覆盖诊断区分：

- 17 条命中；
- 4 条相关内容存在，但没有排进前五；
- 8 条期待的来源不在索引：`sanitizer-version-probe` 和
  `sanitizer-release-evidence` 在来源清单中标为 `chunk_strategy=evidence`，用于版本核验，
  并不作为检索正文入库。

因此不能把全部 12 条未命中都说成排序失败。没有删除查询、修改标签、把 17/29
改成更高的通过率，也没有因此更改已冻结索引或 Agent 配置。四条排序漏命中仍保留。

离线复现（不联网、不调用模型、不修改索引）：

```bash
python -m gpu_agent.knowledge.coverage /absolute/path/to/index.json knowledge/retrieval-eval.json
```

下面保留的是旧的 24 条实验记录。

### 独立 v4 索引试验

`cuda-lex-v4` 在 v3 原有词项、停用词及 API 名匹配规则之外，增加文档标题词项。
没有查询 ID、预期来源 ID 或案例答案参与排序。v1–v3 的行为和存储格式保持兼容。
使用相同 79 个正文片段，新索引 hash 为
`77c5f03ae70b403b531efe9c7a3d8fc762f99afec5ad4ece4e968fd898ecae1c`，
开发查询比较为 lexical 18/29、semantic 15/29、hybrid 17/29。
这是开发集上的结果，不能宣称独立泛化提升；剩余漏命中没有删除或重标。

本机新索引和报告在 `/home/you/gpu-agent-knowledge-20260928-v4/`，旧 v3 索引未覆盖，
现有 provider 配置和已冻结实验也没有切换。需要使用新配置时可显式设置
`GPU_AGENT_KNOWLEDGE_INDEX=/home/you/gpu-agent-knowledge-20260928-v4/index.json`。

[`knowledge/retrieval-eval.json`](../knowledge/retrieval-eval.json) 包含 24 条人工编写、人工
标注的 development 查询。标签只引用 `knowledge/sources.json` 中的公开 `source_id`，
不读取 evaluator store、private holdout、Oracle 或评测答案。加载器拒绝除
`split=development` 之外的输入。

比较报告逐查询保留以下字段：

- 查询 ID、相关 source ID、实际检出的 source ID；
- hit@k 是否命中；
- 本次检索 wall-clock latency；
- 每种方法的总命中数、hit@k、总延迟和平均延迟；
- 由 development hit@k 选出的默认方法。

报告同时固定 corpus hash、development 标注 hash、lexical tokenizer 版本、semantic encoder
版本和 hybrid fusion 版本，使排序配置可以随证据一起复核。

默认方法只按 development 命中数选择；同分时使用固定优先级
`hybrid > semantic > lexical`。延迟只进入报告，不参与选择，避免机器负载改变默认值。

在 2026-09-20 对本机已固定的 corpus
`a3a593f2f0f7a77e478ae1aad917123788aad430f397d21610065f565834726c` 执行 24 条、`k=5`
的比较结果如下：

| 方法 | hit@5 | 本次平均延迟 |
| --- | ---: | ---: |
| lexical | 13/24（0.542） | 1.174 ms |
| semantic | 10/24（0.417） | 1.224 ms |
| hybrid | 12/24（0.500） | 1.307 ms |

因此该 corpus 的 V2 默认方法保持 `lexical`。延迟数字只描述这次本机运行，不应外推为
其他硬件的性能结论；排名确定性由测试单独验证。

## 复现与导出

下面的代码读取已有本地索引、运行三种方法，并将完整逐查询报告写到操作者选择的位置：

```python
from pathlib import Path

from gpu_agent.knowledge.retrieve import KnowledgeIndex
from gpu_agent.knowledge.semantic import (
    compare_retrieval_methods,
    load_evaluation_suite,
    write_comparison_report,
)

index = KnowledgeIndex.load(Path("/absolute/path/to/index.json"))
suite = load_evaluation_suite(Path("knowledge/retrieval-eval.json"))
report = compare_retrieval_methods(index, suite)
write_comparison_report(report, Path("/absolute/path/to/retrieval-comparison.json"))
```

同一 corpus、query、toolchain version 和 method 的候选顺序是确定的；延迟字段自然会随
运行环境变化。`KnowledgeIndex.retrieve(..., method="lexical" | "semantic" | "hybrid")`
可用于单次检索，原有未传 `method` 的调用保持 BM25 行为。
