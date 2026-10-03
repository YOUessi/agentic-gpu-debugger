# 2026-09-29：检索缺失、排序与运行环境接通的逐轮记录

## 记录来源、任务范围与验收口径

本记录按用户的新要求，回补近期检索修复过程。依据是当前工作区差异、保存的
检索报告、配置检查报告及本次对话中实际执行的工具输出。它不是当时逐秒生成的
原始日志；未保存的 pytest/stdout 完整日志、部分早期命令时间和退出码不补造。

用户要解决的是项目实际使用中的检索不足，而不是只改提示词或写一个未接通的
工具。本轮针对已定位的内容缺失、代码/自然语言匹配、结果排序及运行配置脱节。
验收证据包括：原有开发查询不改答案后的结果、入库安全约束、旧版本兼容性、
Agent 的检索/存储/引用路径，以及实际环境加载的新代码和索引。

原有开发集合为 29 条，前五条结果中有相关来源或指定片段即计命中；多数查询为
来源级标注，新增五条同步/竞争/初始化查询有片段级标注。这个口径不是模型回答
正确率，也不是 CUDA 补丁修复率。开发中已查看这些查询，不能称其为独立盲测。

工作区是 `/home/you/.codex/worktrees/a352/agentic-gpu-debugger`，修复时 HEAD 为
`87fa492`，检索修改尚未提交，因此下文新索引不能标成 `87fa492` 干净源码的产物。
解释器为 `/home/you/conda_env/agentic-gpu-debugger/bin/python`。
目标工具版本保持 CUDA 12.8.1、Compute Sanitizer 2025.1.0.0。

## 第一轮：确认缺失内容与排序失败不是同一个问题

### 问题和原因

旧 v3 索引包含 79 个片段，29 条查询只命中 17 条。开始时不能仅凭总分断言是
排序算法或模型问题，因此核对了预期来源与实际索引的交集。

其中八条查询期待的 `sanitizer-version-probe` 或 `sanitizer-release-evidence`
完全没有片段。原因是这两个来源的策略为 `evidence`，入库函数完成版本/锚点
校验后直接返回空列表。下载凭据存在不代表正文可以检索；排序无法找出不存在的内容。

另有四条是内容已经存在但没有排进前五：运行时错误描述，以及三条样例描述。
它们不能和上述八条混称“知识库没有收录”。

### 应该如何修复与实际处理

先保留原测试和索引，区分覆盖率缺口与排名缺口，再分别修复。早期 v4 先补入
文档标题词项，以解决标题信息没有参与匹配的问题。这没有补全八条缺失来源。

### 测试结果与下一轮

同样的 79 个片段与 29 条查询，v4 lexical 为 18/29，semantic 为 15/29，
hybrid 为 17/29。v4 索引哈希为
`77c5f03ae70b403b531efe9c7a3d8fc762f99afec5ad4ece4e968fd898ecae1c`。

结论是只修复了一个排名漏命中，不能宣称检索已经完成。需要继续处理词项写法差异、
真正入库缺失资料，并检查 Agent 是否能使用新结果。

## 第二轮：元数据工具与标识符拆分仍然只是部分修复

### 问题和原因

代码函数名和普通描述之间存在写法差异，例如驼峰、下划线和限定名没有拆出词语。
此外，已有抓取凭据虽然可核对来源版本，但没有方便且严格的只读查询入口。

### 应该如何修复

保留完整 API 名，同时加入通用标识符词项；元数据工具只陈述凭据中确实记录的
内容，不把版本元数据伪装成发布说明正文，也不虚构没有抓取到的段落。

### 实际如何修复

在 `src/gpu_agent/knowledge/retrieve.py` 增加 v5 分词：保留完整符号，同时拆分
驼峰、下划线和限定名。新增 `knowledge/metadata.py` 与 CLI 入口，要求调用者
提供预期 receipts 文件 SHA256，并验证来源、URL、版本、哈希、重复项和时间字段。
不存在的凭据明确返回 `NO_FETCH_RECEIPT`。这只是辅助诊断入口，不是 Agent 新动作。

### 测试结果与下一轮

未扩充正文时 v5 lexical 为 20/29，semantic 15/29，hybrid 17/29。
当时 97 项相关测试通过，Ruff 和严格类型检查通过。索引哈希为
`031cabe22ed3a036fb1dbd44345e019cbe6477ae2e50633ce77c4f95cc93ebe7`，
产物位于 `/tmp/gpu-agent-retrieval-20260929-rQafQ1/`；临时目录不保证永久保留。

还剩八条来源缺失和一条普通语言描述漏命中。独立 metadata CLI 也不能替代 Agent
的实际文档检索。因此这一轮仍是部分修复，后续必须补正文和实际使用路径。

## 第三轮：补入真实官方正文并修正切块

### 问题和原因

确认入库实现中 `chunk_strategy == "evidence"` 时直接返回 `[]`。
另一个缺口是样例切块只截取 `vectorAdd` 函数，丢掉紧邻函数的官方注释。没有
这些自然语言说明时，普通描述与代码之间可用于匹配的信息减少。

### 应该如何修复

对明确批准的 evidence 来源启用正文检索，保留原有版本证据链；样例只加入原文件
紧邻函数的说明，不生成面向测试的摘要。重新使用受限官方抓取器构建索引，旧索引不覆盖。

### 实际如何修复

`ingest.py` 新增默认关闭的 `retrievable_evidence` 和 `include_attached_comment`。
前者只允许 evidence 策略使用，后者只允许样例策略使用。版本元数据按原文生成
片段，发布说明仅提取已批准的 `updates-in-2025-1` 章节，样例保留直接相邻块注释。

`knowledge/sources.json` 显式启用这些选项，并增加限制说明：13.4 是文档构建版本，
不是安装的工具版本；发布说明片段只代表 2025.1 章节。URL 白名单、样例提交和
文件哈希、滚动手册批准段落哈希、版本检查均未放宽。

首次在受限网络中抓取官方 ReleaseNotes 时发生 `curl(6) Could not resolve host`。
这是抓取阶段的环境访问失败，不是检索算法失败。经权限批准后同一官方 URL 下载成功，
退出码 0。随后取得版本 JS 和已固定提交的 CUDA 样例，再通过项目正式 ingest 路径
抓取六个批准来源；没有把手写的段落作为官方材料入库。

新增 `scripts/build_knowledge.py`，要求输出目录不存在，保存索引、sources 快照、
逐来源 receipts 和比较报告。实际命令为：

```bash
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python scripts/build_knowledge.py \
  --repository /home/you/.codex/worktrees/a352/agentic-gpu-debugger \
  --output /home/you/.codex/worktrees/a352/agentic-gpu-debugger/.cache/gpu-agent/knowledge/20260929-v5
```

### 测试结果与下一轮

命令退出码 0，新索引含 92 个片段，哈希为
`48ac08d444487b6c482c9824e2c793eabe960eb5420f9fdb7d83788935440314`。
lexical 25/29，未命中 dev-012、dev-013、dev-016、dev-023；semantic 23/29，
hybrid 26/29。原始比较在 `20260929-v5/retrieval-report.json`。

资料已经存在，但默认检索仍未达到目标。不能停在“构建成功”，需要继续查实际排名。

## 第四轮：根据实际排名修复词形和同源结果挤占

### 问题和原因

检查上述四条查询的较长候选列表后发现：dev-012 的版本元数据排在第 13 位；
dev-013、dev-016 的发布说明首个片段排在第 8 位。靠前位置有很多来自同一份手册
的片段。相关来源已存在，因此这次是排名/上下文分配问题，不再是入库缺失。

dev-023 的描述使用 `adds`，而样例标识符拆分得到 `add`，旧规则不提供这种词形
连接。不能凭这一例证明所有语义问题都源于词形，但它支持增加通用词形匹配的尝试。

### 应该如何修复

保留原词，增加简单的英文复数/第三人称伴随词项；避免前五条被同一来源的多个
片段占满。规则不能读取查询 ID、预期来源或案例答案；旧分词版本行为不应改变。

### 实际如何修复

新增 `cuda-lex-v6`，继承 v5 拆分，并对符合条件的小写英文结尾 s 增加去 s 词项，
排除 ss/us/is 等结尾。它不是完整词形还原器，原词始终保留。

新增 `_select_lexical`：k>=3 时按原相关性排名优先每个来源最多两段，其余放入
候补，再以原排名补齐；k<3 和旧版本沿用原选择。只处理已有匹配，不插入非匹配文档。
这也可能使某篇文档的第三个高分片段让位给另一个来源，属于必须记录的排序取舍。

对同一批已抓取的 92 个片段离线重建，没有再次抓取或修改查询标签：

```bash
PYTHONPATH=src /home/you/conda_env/agentic-gpu-debugger/bin/python scripts/reindex_knowledge.py \
  .cache/gpu-agent/knowledge/20260929-v5/index.json \
  --expected-corpus-hash 48ac08d444487b6c482c9824e2c793eabe960eb5420f9fdb7d83788935440314 \
  --repository . --output .cache/gpu-agent/knowledge/20260929-v6
```

### 测试结果与下一轮

退出码 0，lexical 29/29，semantic 23/29，hybrid 27/29。五条既有片段级查询全部命中。
新索引哈希为 `15c53142f8949ba1d852a3227fd11460060adec359fda9ee0cf42c52011b8600`。
dev-012 对应来源排第 5，dev-013/016 对应来源排第 4/5，dev-023 样例排第 3。
这说明前五命中已改善，不意味着相关片段全部排第一。

排序通过后还需要检查保存/加载、原有安全边界和 Agent 调用路径，不能直接据此宣布完成。

## 第五轮：测试暴露版本接线遗漏，修正后定向回归

### 问题和原因

首次相关 pytest 得到 105 passed、2 failed，退出码 1。具体失败是：

- `test_source_diversity_and_backfill_do_not_invent_matches`：内存中 v6 索引可用，
  保存后加载却报 `Unsupported index format`，外层为 `KnowledgeCorruptError`。
  原因是构造器支持 v6，但加载器还有另一份版本白名单，更新时漏掉了这一处。
- `test_candidates_reference_manifest_sources_and_leave_the_corpus_unchanged`：
  候选清单仍声明 2026-09-24.2，而当前 manifest 已是 2026-09-29.1。
  原因是发布资料的版本声明没有同步，不是原有查询答案错误。

此外首次 Ruff 报新增测试一行超过 100 字符，随后格式化修复。这属于代码格式问题，
不是功能通过，也没有被隐藏成“全检查一次通过”。

### 应该和实际如何修复

加载器补入 v6 支持，保留未知版本拒绝和索引哈希核对；同步
`docs/knowledge-expansion/candidates.json` 的目标版本，不改查询答案。
新增测试使用独立自编文本验证原词保留、词形、来源多样性、回填、无匹配时不造结果、
版本不兼容时拒绝、正文白名单、注释边界及索引保存加载。

### 实际测试命令和结果

首次运行目录为工作区，命令为：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -I -m pytest \
  tests/unit/test_retrieval.py tests/unit/test_retrieval_completion.py \
  tests/unit/test_knowledge_metadata.py tests/unit/test_retrieval_coverage.py \
  tests/unit/test_retrieval_comparison.py tests/unit/test_knowledge_review.py -q
```

修正后只复测相关模块，并补 Agent 路径回归：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -I -m pytest \
  tests/unit/test_retrieval_completion.py tests/unit/test_knowledge_review.py \
  tests/unit/test_agent_loop.py -q
```

第二条命令退出码 0，31 passed。两批去重覆盖 128 项，不是 105+31 个独立测试。
后来 `--collect-only` 核对出 128 个测试，它只核对数量，不被算作再次执行或通过证据。
两处已观察到的失败均在第二批复测范围内并通过，没有再次运行整套项目测试。
pytest 完整控制台日志当时未另外保存到文件，以上结果来自该轮工具输出。

新增 `test_agent_retrieves_persists_and_cites_new_release_content` 经过真实 service、
orchestrator 和存储：调用检索、保存 `docs/<chunk_id>.json`、诊断中引用同一 chunk。
provider 和容器子进程是模拟，知识正文是合成 fixture；不能当作真实模型/GPU 实验。

最终 Ruff check、Ruff format --check、git diff --check 均退出 0；
`/home/you/conda_env/agentic-gpu-debugger/bin/mypy --strict src/gpu_agent` 退出 0，
报告 72 个源文件无类型问题。

## 第六轮：修复实际运行环境仍使用旧代码和旧索引

### 问题、原因与应该如何修复

读取实际 Python 导入路径发现，它仍来自
`/home/you/releases/agentic-gpu-debugger-v2/src/gpu_agent/__init__.py`。
provider 配置的索引仍是 `/home/you/projects/agentic-gpu-debugger/.gpu-agent/knowledge/index.json`。
因此即使工作区内测试通过，普通运行也可能继续用旧版，导致用户仍看到问题。

应同时切换 editable 安装和索引路径，再实际加载服务核对，不能只给用户一个新路径。
切换只影响后续运行，不覆盖旧发布源码和旧实验数据，密钥不应改动或打印。

### 实际修改和验证

经权限批准执行：

```bash
/home/you/conda_env/agentic-gpu-debugger/bin/python -m pip install --no-deps --no-build-isolation \
  -e /home/you/.codex/worktrees/a352/agentic-gpu-debugger
```

命令退出 0，没有升级依赖。仅替换
`/home/you/.config/agentic-gpu-debugger/provider.env` 的 `GPU_AGENT_KNOWLEDGE_INDEX` 行，
指向工作区 `.cache/gpu-agent/knowledge/20260929-v6/index.json`，其他配置和密钥不变。

加载该配置后，在临时 run store 中构造 `ApplicationService.configured()`，断言导入
路径位于工作区、新索引哈希正确，再使用配置中的工具版本运行原有 29 条查询。
退出码 0，实际命中 29/29，API 调用为 0。结果保存在
`.cache/gpu-agent/knowledge/20260929-v6/configured-check.json`。
验证脚本通过一次性命令执行，其完整脚本未单独作为源码文件保存；不把 JSON 报告称为脚本。

注意：已启动的 Python 进程不会自动替换导入模块，旧 shell 环境变量也不会自动更新。
后续使用旧终端时需重新加载 provider 配置，再启动程序。

## 最终状态、剩余边界与证据位置

本次默认检索的已知开发查询缺口、入库缺失和运行配置脱节已修复。Agent 的证据
存储/引用路径通过模拟集成检查，真实环境加载路径和真实官方索引查询也通过。
实际结果记录在工作区 `.cache/gpu-agent/knowledge/20260929-v5/` 和
`20260929-v6/`：前者保存本次抓取凭据及 sources，后者保存重建关系、比较结果和配置检查。

仍须明确保留的边界如下：

- semantic 仍有 6 条未命中，hybrid 仍有 2 条未命中；它们是备选比较方法，
  当前 Agent 默认使用 lexical，没有宣称三种方法都达到 29/29。
- 没有新增独立盲测，因此不能保证新查询泛化；简单词形与来源多样性也有取舍。
- 没有运行真实 DeepSeek 或 GPU 修复实验，因此没有声称修复率提高或模型错误消失。
- 本轮没有运行整个项目测试套件；检查范围是检索及受影响的 Agent 路径。
- 老实验和老 corpus 没有修改，不允许把本次结果拼进旧模型评测结论。
- 代码与记录当前尚未提交或推送。已有 `.claude-deps.tgz`、`.claude-regress.txt`
  是保留的用户文件，不属于本次修复产物。

后续若实际输入再次出现检索漏命中，应在本文追加新一轮：先记录输入、使用的
corpus/代码版本、实际候选与预期依据，再区分缺内容、匹配不足、排序问题或旧配置。
不要删除既有成功或失败记录，也不要只写“继续优化”。
