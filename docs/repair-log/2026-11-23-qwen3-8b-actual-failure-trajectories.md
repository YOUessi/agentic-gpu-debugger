# 2026-11-23 · Qwen3-8B-AWQ Coding Agent：两轮真实失败轨迹（保留负结果）

> **实际模型**：Tang RTX 4090 上的 `/home/you/models/Qwen3-8B-AWQ`，vLLM 0.19.1，localhost Responses API。非 DeepSeek，非 scripted/fake agent。每组 `max_steps=6`、`max_model_calls=8`、`max_public_checks=2`；各组实际使用 6 次物理模型请求。任务仅为 HardSet V2 h001（Softmax32）的 mutant；不是跨案例泛化实验。

## 冻结证据索引

| 实验 | 冻结执行源码 | Tang 实验目录 | paired-report.json SHA256 |
|---|---|---|---|
| Round 1：完整开放的仓库工具列表 | GitHub `59c159acecf64049c1a58ccb55278b17e10a63a6`，独立 tar SHA256 `67d7506122436aa469e53e87ac6f245f9523a632b4c4982b9ea9d94137dd919c` | `/home/you/gpu-native-harness-validation-20261123/qwen3-real-p0-59c159/h001-four-arm-pilot` | `3f431fa5c86b44d06961c05fa07f9c027ad65605c5ea4526a0e0f5fbe508c9d0` |
| Round 2：state-gated model grammar (pre-final formatting commit) | GitHub `b4e6c957130cca8de729c5d390d79d1cfccb3b90`，Tang Git pinned Worktree（提交 SHA 完全匹配） | `/home/you/gpu-native-harness-validation-20261123/qwen3-state-tool-worktree-b4e6c95/real-qwen-h001-four-arm` | `b7d64f8d3b4153bcf11944260d78fcd6e71310ad7795ab72c875dde3630eedbd` |

两轮的正式终局 **全部** `INCONCLUSIVE / STEP_BUDGET_EXHAUSTED`；没有成功源码编辑，没有真实 public CUDA checks，也不具备独立隐藏验证成功率。RunStore / checkpoint / Journal 存在于上述 Tang 私有目录。总计 **48 次真实模型物理请求、105,738 已记录 Tokens、8 次试次 0 成功、0 GPU 检查**。绝不能将其与历史 D/E 的 DeepSeek 3/3 或此前脚本化公开 CUDA 成绩混算。

## Round 1：模型有完整仓库工具，但自发循环

| Turn | legacy_single | legacy_batch | evidence_single | evidence_batch |
|---|---|---|---|---|
| 1 | repo_list_files OK | repo_list_files OK | repo_read_file OK | repo_run_public_checks REJECTED (no candidate) |
| 2 | repo_list_files OK | repo_list_files OK | repo_list_files OK | repo_list_files OK |
| 3 | repo_list_files REJECTED (duplicate) | repo_list_files REJECTED (duplicate) | repo_list_files REJECTED (duplicate) | repo_list_files OK |
| 4 | repo_list_files REJECTED | repo_list_files REJECTED | repo_run_public_checks REJECTED (no candidate) | repo_list_files REJECTED |
| 5 | repo_list_files REJECTED | repo_list_files REJECTED | repo_run_public_checks REJECTED (no candidate) | repo_list_files REJECTED |
| 6 | repo_list_files REJECTED | repo_list_files REJECTED | repo_run_public_checks REJECTED (no candidate) | repo_list_files OK |

**结论**：无候选源修改。重复列目录和过早 GPU 检查都消耗了 LangGraph 轮次。此轮实际模型 Tokens：legacy_single 8,898、legacy_batch 8,907、evidence_single 14,925、evidence_batch 14,256。公开 GPU 物理检查 0 次。

## Round 2：动态工具权限确实消除列目录，但暴露源码定位/编辑脆弱性

| Turn | legacy_single | legacy_batch | evidence_single | evidence_batch |
|---|---|---|---|---|
| 1 | repo_read_file OK | repo_read_file OK | repo_read_file OK | repo_read_file OK |
| 2 | repo_read_file REJECTED (duplicate) | repo_read_file REJECTED (duplicate) | repo_read_file REJECTED (duplicate) | repo_read_file REJECTED (duplicate) |
| 3 | repo_read_file OK | repo_read_file REJECTED (duplicate) | repo_read_file OK | repo_read_file OK |
| 4 | repo_read_file REJECTED (duplicate) | repo_read_file OK | repo_replace_snippet REJECTED | repo_replace_many REJECTED |
| 5 | repo_read_file REJECTED (duplicate) | repo_read_file OK | repo_read_file OK | repo_replace_snippet REJECTED |
| 6 | repo_replace_snippet REJECTED | repo_read_file REJECTED (duplicate) | repo_replace_snippet REJECTED | repo_replace_snippet REJECTED |

Round 2 模型物理调用：每组 6 次；Tokens：legacy_single **13,902**、legacy_batch **13,719**、evidence_single **15,969**、evidence_batch **15,162**，合计 **58,752**。全部 `PUBLIC_NOT_ACCEPTED`。模型不存在成功写入的候选 Diff，因而 public GPU checks 与隐藏 strict verifier 均未实际运行。模型耗时统计请查看各单元 `unit-NNN.json`。

### 有证据支持的具体补丁失败案例

原始冻结 h001 CUDA mutant kernel：

```cpp
// (原文件，第 15–26 行；显示行号并非文件内容)
float maximum = -FLT_MAX;
for (unsigned int j = 0; j < valid; ++j) {
    const float x = a[start + j] + b[start + j];
    maximum = fmaxf(maximum, x);
}
float denominator = 0.0f;
for (unsigned int j = 0; j < valid; ++j) {
    const float x = a[start + j] + b[start + j];
    denominator += expf(x);            // BUG: 未减最大值
}
const float own = a[i] + b[i];
out[i] = expf(own) / denominator;    // BUG: 同理未减最大值
```

- **legacy_single 第 6 步**：`repo_replace_snippet.old_text` 为 `"    17         const float x = a[start + j] + b[start + j];"`，包含阅读器渲染的 **17 行号**，在原始文件出现 0 次，控制器返回 `edit source must match exactly once`。并且这次 proposed `new_text` 只是添加注释，没有实际修复 Softmax。
- **evidence_single 第 6 步**：尝试对 `float maximum = -FLT_MAX;    for (...) {        const float x ...` 一整段文字做精确替换。模型把真实源码的多行拼成一行，原文件匹配次数 0，控制器拒绝。
- **evidence_batch 第 4 步**：`repo_replace_many` 的多片段旧文本不满足全部唯一匹配，整体被拒绝，未留下半个 patch。第 6 步 `repo_replace_snippet` 继续使用拼接为单行的 `float denominator = 0.0f;    for (...) { ... denominator += expf(x); }`，同样匹配 0 次。

这三个现象彼此不同：多余行号、错误的源码换行表示、以及模型在有限预算内无法生成真实正确的补丁。它们说明目前需要改进**模型可操作的源码编辑协议**和**定位反馈**，而不是继续添加大量 Agent 框架名。

## 审计边界与后续验证门槛

公开日志可可靠恢复每轮 **工具名称、执行状态、工具观察与拒绝原因**，checkpoint 保留当前/最后一轮结构化 `decision.action`，Provider Invocation 保存物理调用、Token 与输出哈希；但项目目前**未保存每次模型回答的完整原始正文、完整工具参数或模型内部推理链**。本报告严禁把缺失内容解释成已保存/可完全回放的逐 Token 思维链。

当前项目已设计下一版 state-scoped Schema + 权限控制器，但本报告只是对前两轮已经实际运行的负结果归档，**不宣称未来版本成功**。之后应新增独立的源码行范围编辑工具（绑定文件哈希、范围边界、受控上下文校验），并在相同真实 Qwen3 API/任务下检查是否能成功产生 Diff、执行公共 GPU 数值检查与四类 Sanitizer，再通过隔离 Strict Verifier 给出验证结果。开发集 h001 的修复也**不能**证明跨任务分布泛化。
