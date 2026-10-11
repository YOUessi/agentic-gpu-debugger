# 知识库扩充流程（同步 / 竞争 / 未初始化读取）

扩充前：`knowledge/sources.json`（corpus 2026-09-15.1）只收录编程指南 device-memory 一节和
三段 memcheck 段落，没有 warp 同步、`__syncthreads`、racecheck、synccheck、initcheck 的内容。
有 finding 时控制器要求文档证据，这类 case 只能检索到无关文档。

候选已于 2026-09-24 经工程审阅提升到 corpus 2026-09-24.1，
范围和版本依据见 `REVIEW-2026-09-24.md`。下面保留可复核流程：

1. 在可访问 docs.nvidia.com 的机器上（沿用 manifest 的抓取策略与 HTML 选择逻辑）：

   ```bash
   python -m gpu_agent.knowledge.review knowledge/sources.json \
       docs/knowledge-expansion/candidates.json > /tmp/knowledge-review.json
   ```

   输出每个候选锚点下的全部段落、所在标题、内容 sha256、长度、是否可 pin，以及页面上
   不存在的锚点（`missing_anchors`，候选锚点 id 未经抓取确认）。

2. 人工审阅：
   - 只保留与 CUDA 12.8 / Compute Sanitizer 2025.1.0.0 相符的段落；Sanitizer 手册是
     滚动版本（13.4），只能 pin 有适用版本证据的 `<p>` 段落；`pinnable: true` 只是结构标记，
     不能替代兼容性审阅。发布说明不能单独证明新版内容向后兼容；
   - 超过 1400 字符的块会让 ingest 失败，需改选更细的锚点。

3. 提升为正式 corpus（一次独立、可审查的提交）：
   - 在对应 source 的 `include_anchors` 加入确认的锚点，Sanitizer 手册同时把审阅过的哈希
     写入 `approved_chunk_sha256`；
   - `corpus_version` 改为 `candidates.json` 的 `target_corpus_version`；
   - 把 `retrieval_queries` 并入 `knowledge/retrieval-eval.json`；
   - 重新 ingest，跑检索相关性评测（新查询的相关文档须进入 top-k）。

4. corpus 版本变化后，B/D/E 的证据与此前 240 单元基线不可直接比较；正式评测需在固定的
   代码 + corpus 版本上整体重跑并单独存放，不与旧版本结果拼接。
