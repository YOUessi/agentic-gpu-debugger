# 演示：从GPU证据到验证结果

## 无需模型费用即可查看的已有结果

先看仓库内[公开摘要](evidence/portfolio-summary.json)的case_0021：

1. 二维五点stencil的上边界访问存在越界。
2. E实际选择memcheck、检索文档，然后给出诊断；不是固定执行全部工具。
3. candidates包含实际模型diff；公开自检数值与四Sanitizer均通过。
4. 独立验证VERIFIED_FIXED，run `44ddb114785b4afc93a5da5b8661f9ad`，5次调用。

再看case_0022：E先memcheck再racecheck；模型移动同步屏障，第二候选仍有竞争，
第三候选重复后停止。它解释为什么退出成功或偶尔算对不能代替并发安全验证。

摘要含原result/artifact哈希和公开补丁，但不是完整原生证据包；完整日志留在操作者
实验存储。两例来自快照 `0d7db80df1bad79be0c4812ec673ec99c43e51a4`，详见
[逐轮记录](repair-log/2026-10-03-real-workloads.md)。

## 在已配置环境中现场演示

先按[README](../README.md)准备Python/Conda、锁定CUDA工具链、隔离GPU后端及文档索引。
环境元数据检查不等于GPU可执行。模型实验会产生真实费用，模型输出不确定，不能保证
每次重跑都与已有结果相同。密钥不得提交仓库或直接写在演示命令中。

无需模型的GPU验收（输出目录必须不存在）：

```bash
python -I -m gpu_agent benchmark validate-diversity \
  --repository "$PWD" --output /absolute/new/demo-native --case case_0021
```

模型配置沿用README所列OPENAI_BASE_URL、OPENAI_MODEL、OPENAI_API_KEY及兼容端点声明；
另设GPU_AGENT_KNOWLEDGE_INDEX和GPU_AGENT_KNOWLEDGE_VERSION。密钥放在仓库外受限文件中，
在自己的shell加载，不打印进日志。索引构建说明见[检索配置](retrieval-update-20260929-CN.md)。

```bash
export GPU_AGENT_RUN_ROOT=/absolute/new/demo-public
export GPU_AGENT_EVALUATOR_ROOT=/absolute/new/demo-verification
python -I -m gpu_agent repair benchmarks/public/case_0021/public_input \
  --allow-paid-calls --max-candidates 3 --max-llm-calls 40
```

repair已包含公开自检与最终独立验证，无需再手工重复verify。源码目录必须有与原始
kernel绑定的task.json及合法公开输入；预检查失败不发模型请求。使用本次打印的run_id：

```bash
python -I -m gpu_agent report YOUR_ACTUAL_RUN_ID
```

讲解公开检查、最终结论或具体失败原因、每轮补丁和调用数，不只展示程序退出码。
不为录到绿结果反复重跑。E选工具、D规则选工具；两者均可用模型诊断、补丁与修订。

## 历史单候选演示

早期run `ebce9bb96b9e45ad93b0fee942489ef7`来自commit `8bd3cea`，使用diagnose→verify，
当时5次调用并VERIFIED_FIXED（public 1/private 13）。它不是当前多轮流程的证据，
旧记录保留，不用新结果覆盖。
