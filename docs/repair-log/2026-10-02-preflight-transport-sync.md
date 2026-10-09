# 2026-10-02：连接分类、功能规格预检查与版本状态同步

## 范围、原因与验收标准

用户要求完成三项：安全记录连接错误原因、提前检查功能规格、同步文档和版本状态。
采用failure-aware-execution，先做针对性离线验收，不重跑已通过的GPU/API实验。
现有主工作区HEAD为87fa492、存在82条状态变更，保留既有内容；不自动混入用户的压缩包、
依赖包和其他配置，不推送。旧实验快照19b53f9及旧结果不可改写。

## 第一轮：实现

连接异常原来仅保留SDK总类，丢失底层原因。新增有界异常链检查，只输出固定原因码，
可区分DNS、TLS、代理、拒绝连接、重置、网络不可达及超时阶段等；未知仍明确UNKNOWN。
不输出异常文本、URL、任意类型名或环境变量。新增可选字段，兼容旧记录；UNCERTAIN、
usage未知和无自动重试的行为不变，不把原因码等同于确定服务器是否收到请求。

repair原来允许缺task的任务先调用模型。现将task存在性、版本、算法及源码hash绑定、
公开输入shape/数值/参考计算检查放到创建diagnosis运行之前；复用公开自检同一纯函数，
不另写一套宽松检查。旧diagnose单候选入口保持原行为。

验收需要证明：真实适配器模拟异常可持久化分类且无敏感文本、旧记录可读、无额外调用；
坏规格/输入在创建run、provider或backend之前拒绝；正常repair及旧diagnose不回归。
本轮不提升prompt或篡改旧实验，只补实现和实际版本状态，后续记录测试结果。

## 第二轮：实际修改与离线验证（2026-10-03核对）

实际修改agent/transport.py新增固定枚举原因码、有界异常链与循环防护；provider.py将
分类放入ResponseMetadata并持久化到Invocation.transport_error。旧字段缺失默认null。
仅检查异常类型及已知errno，不匹配异常文本猜测原因，不改变自动重试、UNCERTAIN及usage。
公开任务模块提取public_expected_output，让预检查和运行后功能检查共用实现；service在
创建run之前检查任务绑定、公开数值输入、可计算的有限参考输出及现有vector_api接口。
PublicRepairInputError提供不包含原始数据的固定错误码；CLI直接显示缺规格等原因，
而非只笼统提示控制器错误。新preflight不影响legacy diagnose。

测试fixture显式补有效task，避免把缺规格任务继续当成正常repair；另写负向测试严格禁止
create_run/backend进入，证明不是在调用完模型后才失败。测试实际覆盖缺文件、非法版本、
非法算法、hash不匹配、符号链接、长度错误、非有限数、float32溢出、不支持接口及CLI错误码。
连接模拟覆盖DNS/TLS/代理/拒绝连接/重置/不可达/分阶段超时/协议错误/未知原因及循环链；
真实适配器的离线测试验证分类持久化、旧记录可读、一次调用、敏感canary不进入记录。

所有命令在主工作区执行，Python为/home/you/conda_env/agentic-gpu-debugger/bin/python。
第一批执行（环境前缀env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1）：

```
python -m pytest tests/unit/test_transport_diagnostics.py tests/unit/test_repair_preflight.py tests/unit/test_iterative_repair.py tests/unit/test_public_repair_correctness.py tests/integration/test_provider_contract.py -q
```

实际81 passed（3.71秒），无测试失败。Ruff首次发现preflight参数列表超长，格式化处理。
随后补充CLI友好原因码和接口检查，执行以下扩大后的关联回归（同一环境前缀和解释器）：

```
python -m pytest tests/unit/test_repair_preflight.py tests/unit/test_transport_diagnostics.py tests/unit/test_iterative_repair.py tests/unit/test_public_repair_correctness.py tests/unit/test_provider_output_telemetry.py tests/integration/test_provider_contract.py tests/unit/test_agent_loop.py -q
```

结果148 passed（5.77秒），退出0；该数包含前批，不相加。Ruff发现CLI提示字符串超长，
拆分相邻字面量后再次检查通过。最终检查命令：

```
/home/you/conda_env/agentic-gpu-debugger/bin/ruff check src/gpu_agent/agent/transport.py src/gpu_agent/agent/provider.py src/gpu_agent/public_task.py src/gpu_agent/service.py src/gpu_agent/cli.py tests/unit/test_repair_preflight.py tests/unit/test_transport_diagnostics.py tests/unit/test_iterative_repair.py tests/integration/test_provider_contract.py
/home/you/conda_env/agentic-gpu-debugger/bin/mypy --strict src/gpu_agent
git diff --check
```

均退出0，mypy为76个源码文件。实际导入路径核对为当前工作区src/gpu_agent/__init__.py；
runtime_code_fingerprint核对为8d5fc70ebf558e3d50ffa7c8b3d71c4d5f94b36ae13849b53daa690a51c7ba5f。
仅模拟网络异常，没有主动制造网络故障，没有API/GPU调用；旧不确定请求没有被重新发送。

## 文档与版本同步结果

新增docs/current-status-CN.md作为当前实现/实验版本对照；README链接该页；自主调查文档
撤销“尚未执行”的过时状态，保留原先48项协议并附实际结果；mode-contract及limitations
明确当前前置检查与遥测行为。v11完整48项、v12失败子集11项、当前新runtime三个状态分开。
包版本仍0.2.0、repair仍v2、prompt仍v12，不因非语义改动伪造一批新模型评测。

本轮三项实现及相关离线验证已完成；没有跑全量套件或新GPU/API实验，历史连接具体根因
仍无法重建。主工作区保留既有未提交改动，未commit/push，不把文档状态同步说成GitHub
已经更新；仓库里已有压缩包、依赖包及其他用户文件均未改动。
