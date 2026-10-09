# 公开案例的算法多样性扩展

## 2026-10-03追加：二维邻域与分段前缀和

以下旧四例说明保留原验收范围；当前另新增case_0021和case_0022，总计6个扩展案例。
case0021采用二维8×8线程块处理逻辑宽32的行主序网格，输出中心、上下左右邻居及b的和。
末行可不完整，越界邻居为零，行边界不可跨行；故障是计算路径漏掉上边界检查。
case0022按128元素分段计算inclusive prefix sum(a+b)，共享内存倍增步骤依赖屏障；
故障是漏掉步骤读取前的块同步，错误会影响实际prefix输出，不是附加探针。

新增两例沿用扁平数组传输ABI，公开task.json提供完整算法需求，不提供故障标签或clean代码。
注册的数值参考、clean验证参考和公开自检使用同一算法标识，最终验证仍不向模型反馈。
原生验收已覆盖clean八种边界输入、四工具、mutant和删除核心计算反例。
case0022首次因预登记读写类别顺序错误失败，原始日志核实后修正新案例元数据，5次racecheck
均检出精确类别；已通过的clean/ablation未重跑，首次失败保留。
真实Agent闭环验收见[逐轮记录](repair-log/2026-10-03-real-workloads.md)。
仍不覆盖可变显式矩阵shape接口、三维、多stream、异步生命周期、多GPU或真实第三方工程。

## 解决什么问题

旧 16 个公开案例共享 vector-add 参考，部分故障位于不影响数值输出的附加 kernel。
这能测试工具路由，却不足以说明 Agent 能理解不同算法，更可能留下“删掉附加
kernel 仍输出正确”的捷径。本次不是重复添加几组 n，而是新增四种计算语义，
故障均落在生成真实输出的计算或初始化路径上。

## 新案例与独立数值参考

- case_0017：循环移位相加，输出 `a[(i+1)%n]+b[i]`；mutant 丢失回绕处理，
  最后一个元素读取越界，由 memcheck 检测。与逐元素加法相比增加邻接索引关系。
- case_0018：三点 stencil 加 b，左右边界补零；shared tile 和 halo 被相邻线程
  读取。mutant 去掉装载完成后的块屏障，由 racecheck 检测真实输出的共享内存竞争。
- case_0019：加权直方图，最多 32 个 bin，负数取绝对值后按 bin 数取模；权重由
  b 取模得到 1–5 的整数。第一 kernel 在 shared memory 原子累计，第二 kernel
  合并块级直方图。mutant 不初始化输出累计器，由 initcheck 检出。不是额外探针。
- case_0020：每 32 元素求 `a+b` 的和，并向该组输出广播，尾组只求有效元素。
  通过 shared memory 和块屏障实现分组归约；mutant 载入时把 `a[i]` 写成
  `a[i+1]`，最后一个有效线程越过输入分配，由 memcheck 检测。目录与 oracle 名中的
  warp_reduce 指分组尺寸，并不声称是优化的 warp intrinsic 归约。早期 mask 和
  条件屏障及 shared 越界方案未通过原生验收，本批不采用、不把失败改称成功。

全部延用两数组输入、一数组输出以及 `run_vector_add` ABI；函数旧名只是 harness
兼容接口，不是算法标签。新 CPU Oracle 在 verification/oracle.py，使用 float32
语义；模型不能提交自己的 checker，不能从模型解释推导正确答案。诊断输入仍从
kernel 同目录的 input.json 读取。验证引擎和回放按受信源码 hash 选对应 Oracle。

## 如何避免简单删代码就过关

clean 与 mutant 均来自各自算法，而不是 mutant 追加到加法 kernel 后面。验收额外
从 clean 删除核心 launch，再真实运行；它必须无法通过输出契约/数值参考。对
直方图，这意味着保留的零数组不能被接受；对其余算法，未生成有限合法输出同样
失败。这个反例只证明“删除全部核心计算”的捷径被挡住，不是证明所有恶意补丁
都不可能绕过，也不代替隐藏输入和 Sanitizer 验证。

## 执行入口和保存规则

```bash
conda activate /home/you/conda_env/agentic-gpu-debugger
python -I -m gpu_agent benchmark validate-diversity \
  --repository "$PWD" --output /一个尚不存在的绝对路径
# 只检查指定案例；按需多次使用 --case
python -I -m gpu_agent benchmark validate-diversity \
  --repository "$PWD" --output /另一个新目录 --case case_0018
```

默认依次 clean、mutant、ablation。`--role mutant` 可仅补测失败角色，报告明确
标为非完整验收范围，不能把部分结果说成全量通过。已有输出目录直接拒绝覆盖。
每个角色保留 run ID、源码与输入 hash、编译日志、运行日志、Sanitizer 原始记录
及数值结果。report.json 是独立开发验收报告，不是注册 ledger 或正式评测清单。

普通执行的 mutant 超时只算症状，仍须 sanitizer 完成并命中登记类别；clean
和删除计算的反例都不能靠超时过关。racecheck mutant 要重复 5 次，其他本批
mutant 各跑一次；本批没有通过验收的新 synccheck mutant。
clean 用八种边界尺寸跑普通数值检查，在代表性输入上跑四种 Sanitizer；不是
宣称每一个尺寸都做了四工具检查。

## 版本与结论边界

原来的 corpus-registry.json、seed-batch.json、评测分母和 240/120 实验没有修改。
新定义在 diverse-registry.json，mutation 元数据原文在 diverse-mutations.json。
这四项是独立的公开开发扩展，没有冒充已注册的新 holdout，也没有获得新的模型
修复率。没有重新抽样 DeepSeek，没有按 case 加提示，没有读私有 holdout。

本次扩展证明的目标是：项目可以定义与验证不同计算语义、核心路径故障不能简单
通过删除旁路探针回避。仍不证明广泛 CUDA 调试泛化能力。二维/三维索引、矩阵
乘法、多 stream、异步生命周期、多 GPU、真实第三方项目和性能故障仍不在本批
覆盖范围；后续要增加相应算法和受信 Oracle，而不是把本次四项包装为通用能力。

实际逐轮失败、原因、改动、命令与验收结果见
[详细修复记录](repair-log/2026-09-29-case-diversity.md)。
