# 同步参与与检索修复审查（2026-09-24）

## 当前边界

基于 f81052a 修正外部补丁。原始“所有同步参与不一致都拒绝”方案没有启用。
当前源码约束仅拒绝：在可分析子集中，调用线程的自身 lane 位不在 mask 中。
所有 A–E 模式一致，不读取 case ID、故障标签、参考修复或 holdout，不返回修复代码。
其余到达/退出不确定性保持 UNANALYZABLE；GPU 验证规则不变。
重试只用原有补丁阶段额度，不增加修补循环。v7 为原因码；v8 增加从上一份候选
计算的有界数值反例（线程、mask、自身位缺失），不是给出正确目标值。

## 已修正的语义与计算

- &&、||、?: 调用的条件执行不再被当作无条件执行。
- 未建模的类型转换、窄整数、64 位提升和非不可变 uint32 绑定不猜值。
- ballot 结果限于调用者自己的 mask，不合并不相交组。
- 避免无关运算提前计算巨大位移；支持八进制字面量，不截断超宽字面量。
- 不能将源码位置上未调用的线程直接认定为动态执行中永不退出。
- 未知调用、宏控制、动态 launch、重名函数等不能作为可靠源代码约束依据。
- 审计按父诊断与候选 hash 关联，拒绝逃出公开 store 的路径。
- 31 种不同 mask 宽度及多 warp 的正反例验证，不只验证某个开发案例。

## 历史 9 项分歧的处理

原始审计共 88 条候选对照，29 条 synccheck CLEAN 中有 9 条被旧算法标记。
旧结果保留在 /tmp/gpu-sync-audit-f81052a.json。

8 项属于“具名线程未在该源码点调用”的判断，当前模型不掌握动态退出/到达，
改为未知不是宣布源码正确，也不是改变 GPU 通过标准。
新报告 /tmp/gpu-sync-audit-v3.json 保留每个调用点的原因码。

剩余候选 926f143ca1cfce72ded9688ef64edb36 为调用线程自身缺位。
用相同 CUDA 12.8 / sm_89 容器隔离编译公开候选：
PTX 中仍有 bar.warp.sync 65535，而最终 SASS 的对应 kernel 没有 warp 同步指令。
因此源码约束告警和运行时 CLEAN 可以同时存在；不能据此声称源程序符合规范，
也不能将源码分析当作 Sanitizer 的运行时发现。编译复现不是重跑历史评测。
编译记录：/tmp/gpu-sync-codegen-_i3c2jyn/ 下 kernel.ptx、kernel.sass 和 stderr。
参考：[CUDA 12.8.1 PTX 指南](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync)。

## 检索修复与可信计分

新索引 cuda-lex-v3 / corpus 2026-09-24.2：
过滤查询中的常用虚词；明确点名 API 时优先其精确标识符匹配。
无 API 命中时保留普通词法回退。不按案例或相关性标签做排序。
旧 v1/v2 加载及排序逻辑保留，79 个已验证原文 chunk 未更改。

5 项新增检索查询使用具体 chunk ID 判断，不能再因命中同一本手册的无关段落而得分。
这些标签只进入评测，不进入检索算法或模型输入；标注来源声明为 mixed_development。
实际结果：词法 17/29，语义 15/29，混合 17/29；新增内容查询均为 5/5。
不能把 5/5 宣称为全部检索通过或人工独立认证。

重建命令在 scripts/reindex_knowledge.py；只重建明确 pin 的已有可信缓存，不下载新文本。
新索引与原始迁移收据：
/home/you/gpu-agent-knowledge-20260924-v3/
corpus_hash: 094bbdf0a4fd28c4eead13199a50d5b9d4b61d64eddc8c23df2ac28a6217d443

## 验证与未完成项

相关测试 205 passed，src/gpu_agent 全部 68 个文件 strict mypy 通过，Ruff 通过。
尚不能声称 V2 发布完成。下一步固定代码与索引，只复测剩余失败单元；
保留旧结果，无论成功失败都报告一次复测的实际结果。
正式全量评测、private holdout、release gate 仍需独立验收。

## v7 一次真实复测（保留失败）

代码 0a56f08，run fc1ebfd53fe34e54b4e0620d36eda73b：诊断成功，两个补丁输出都被
sync_caller_not_in_mask 拒绝。8 次 DeepSeek 调用，24,443 tokens，无候选，未进入修复后
GPU 验证。记录：/home/you/gpu-agent-sync-regression-0a56f08/results.jsonl。
这不是已修好，也不删除或改写为成功。

v8 的改变不是再加一次重试：检查器将其独立算出的最小反例加入现有重试输入并留档。
离线集成测试核对反例 hash 来自模型上一份候选，mask 的确不含该线程自身位；
不提供修复阈值、正确 mask、case 标签或 evaluator 数据。后续复测须绑定新代码版本。
