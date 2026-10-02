# CUDA 调试 Agent：投递介绍与讲解提纲

## 一句话定位

基于真实 GPU 工具证据的 CUDA 调试 Agent：隔离执行、Sanitizer 调查、文档检索、模型补丁、
公开自检与迭代修订，最后由独立验证判断是否修复，而不是相信模型自评。

## 简历可用描述

- 实现 Python 编排与 CUDA C++ 工作负载，集成 memcheck、racecheck、initcheck、synccheck，
  将原始工具证据、模型调用、候选补丁及验证结果持久化并绑定内容哈希。
- 构建受控工具选择与公开 GPU 自检修订流程；隔离候选代码执行，限制补丁范围，
  对重复候选、工具异常和未通过数值/内存检查的补丁停止并保留失败。
- 扩展循环移位、stencil、直方图、归约及分段scan等算法案例，配套公开功能规格和CPU参考；
  实际验证二维stencil修复成功，并保留scan残留竞争失败，避免把模型置信度当作正确性。

英文简介：Built an evidence-grounded CUDA debugging agent that combines isolated GPU execution,
Compute Sanitizer, documentation retrieval, patch generation, public self-checks and iterative
revision with independent final verification. Preserved both verified fixes and rejected repairs
with reproducible provenance rather than relying on model self-assessment.

这些句子描述已实现的技术工作，不包含未验证的生产部署、团队规模或性能提升。
如放进个人简历，请按本人实际参与和能讲清的范围使用，不补造经历。

## 建议讲解顺序

先说明问题：CUDA程序可能正常退出但仍存在竞争，数值正确也不能代替内存/同步检查。
然后展示真实成功样例case0021：E选择memcheck并检索文档，模型补上二维上边界保护，
公开功能/四工具检查以及最终独立验证均通过。仓库内有
[脱敏证据摘要和实际补丁](evidence/portfolio-summary.json)，不要求评阅者访问本机目录。

再讲工程约束：模型不能执行任意shell、修改oracle或获取隐藏验证输入；候选在隔离后端
运行；公开自检失败可以修订，最终独立验证结果不回流。任务预检查在调用模型/GPU之前，
连接失败使用安全原因码，usage未知时保持未知，不自动重放不确定请求。

最后展示失败样例case0022：模型移动屏障但未消除读写竞争，第二候选偶尔算对仍被racecheck
和instrumented功能检查拦住，第三候选重复后停止。失败不是拿来隐去的，它说明检查没有
配合模型“放水”。另有case0020/D的真实第二轮修订成功记录，见
[多轮修复记录](repair-log/2026-10-01-public-repair-correctness.md)。不要把它说成E轨迹。

## 自主调查如何解释

D用固定规则选工具，E由模型在允许的动作空间中根据现有证据选工具、检索或结束。
二者都使用模型诊断和生成补丁，也可使用同一自检修订机制。当前E有真实不同调查路径：
二维stencil用memcheck，scan在memcheck之后选择racecheck。控制器仍有先memcheck等约束。
这是自主选择的运行证据，不是E优于D的性能结论。投递介绍不需要声称后者。

## 主动交代的边界

这是Linux/NVIDIA GPU上的工程作品，不是任意CUDA代码自动修复服务。当前7类可信功能
checker仍用固定扁平数组接口，新案例是受控算法故障而非第三方生产事故。模型可能修错，
并不是所有案例都通过；不同版本的实验不拼接成整体成功率。全量release gate及旧人工
评分协议不作为本次作品投递的新增要求。具体依赖和入口见[演示说明](demo.md)。
