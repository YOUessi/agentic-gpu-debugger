# 2026-11-23 · 仓库级 CUDA Agent：首个实验性严格独立 GPU 验证

## 固定版本与运行对象

验证的是之前通过公开检查的 **脚本化 Agent 候选**，不是新 DeepSeek 模型自主决策，也没有重新运行模型。这是第一份在仓库级自研 Harness 中把 **公开修改→公开 GPU 检查→会话结束→独立隐藏验证** 连通的真实 GPU 实验。

- 被验证的原始候选冻结版本：GitHub `786bc35725edbfd4cd1a3dcaa94589cef9d72c21`，公开候选 SHA256=`88dd297b8a0bc53dc3b93b7d1a4678551346b4855d7139b4edaddecef0070ba4`。
- 独立验证控制器源码：GitHub `f715b6f6c6327f4a0888a4c4d4a84c4519d07379`，Tar SHA256=`9103239cc94219afec10a195942cb6f9d3d5ef083ca1bf0b3fbde70fe37128d9`。
- 独立验证模块源码 SHA256=`92ef351ab100df6806d683062c426a0d77cdc3c27180a785d6aa105f752171fc`，本地相关 CPU 验收 **26 passed**。
- 执行环境：Tang NVIDIA RTX 4090 Laptop GPU、Python 3.11 锁定依赖、隔离 Docker CUDA 执行器。
- 原始 Agent 公共 RunStore ID=`dd4b207acc4b4f27a2d9c16fa6a6ac29`，原始状态 `DIFF_STAGED_UNVERIFIED`，模型实际物理调用 **0**。

## 独立 Strict Verifier 结果

| 项目 | 真实结果 |
|---|---|
| 验证类型 | HardSet **experimental** strict V2 |
| case | h001 Softmax32 |
| 实验性投影判定 | **VERIFIED_FIXED** |
| 原因 | `ALL_REQUIRED_CHECKS_PASSED` |
| 未向模型公开的隐藏数值检查 | **9/9 通过** |
| 独立隐藏 Sanitizer | **4/4 通过** |
| 私有验证工件 | **91** 项，均为 `evaluator` visibility |
| 公开验证投影 | **2** 项：候选 provenance 和 bounded verdict |
| 验证投影 RunStore ID | `f991c5128da74a5e8e3a13bd9b7ca9f7` |
| 公开验证投影 SHA256 | `987936e8e588a274bb1df25f520fc927feb0614fe86033c34b26a9f1b6956e3e` |

Private evaluator 输出保存在 Tang 的 `/home/you/gpu-native-harness-validation-20261123/repo-strict-verifier-f715b6f/strict-h001-operator-only`，**不在原 Agent 的 RunStore/工作区内**。验证生成的随机密钥由操作者留在独立私有文件中（0600），未提交 GitHub、未进入任何 Agent prompt 或模型工具。

## 操作者专用验收 CLI

```bash
# 本接口不是模型工具。须由操作者独立提供新输出目录和 0600 私有密钥。
gpu-agent harness verify-repo \
  --repository /absolute/frozen/source \
  --session /absolute/completed/repository-agent-run \
  --output /absolute/new/private-evaluation \
  --evaluator-secret-file /operator-private/experimental-secret.bin

# 独立公开轨迹统计（不访问私有 RunStore）：
gpu-agent harness summarize-repo --output /absolute/completed/repository-agent-run
```

控制器在创建任何隐藏验证输出前要求：SessionJournal 为 FINISHED、原始提交/目标 case 一致、代码修改范围恰好是注册 mutant/kernel.cu、当前源码与上一次通过的公开数值检查 hash 完全一致、公开 RunStore 子结果为不可变通过记录、PatchScope 对原始 frozen mutant 的精确 Diff 验收通过。任何部分不符即拒绝。隐藏验证**只在模型循环之后**运行，Strict Verifier 的内部对象及输入不在工具注册表内。

## 公共轨迹指标与不可混淆的结论

新 `summarize_repository_trajectory()` 对此前的真实 GPU 公开测试产生以下可复验指标：6 次工具动作，首次修改发生于第 2 次，第一次 GPU 公开检查发生于第 3 次，首次公开检查通过发生于第 5 次，首次 finish 于第 6 次；2 次公开 GPU 检查；0 次物理模型调用；0 次重复读取拒绝。**这一脚本化轨迹不代表模型效率的观测值。**

当前已经证实 **底层 Agent Harness 的受控 GPU 工程闭环与独立评测隔离可运行**；还没有证明真实 LLM 自主调试的效果，也不具备跨案例、跨分布统计意义。实验性 V2 严格投影不能被宣传为正式 signed holdout certification。对比规则 Workflow 的真实模型成功率、Token 与成本仍需在新的冻结模型实验上开展。

机器可读证据：[实验性私有验证索引](2026-11-23-repo-strict-h001-experimental.json)。
