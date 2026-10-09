# 历史回补 01：环境、可信执行、进程清理与隔离

回补日期：2026-09-29。本篇依据 `docs/t01-validation.md`、`docs/t02-validation.md`
和已核对的 Git 差异。文中的历史测试数量来自当时文档，不是今天重新执行的结果。
范围从专用环境建立到隔离和 provider 进程边界，不涉及后续模型修复率。

## 第一轮：系统工具链不能直接满足项目基线

### 问题与为什么出现

系统 `/usr` 提供 NVCC 11.5.119、Sanitizer 2021.3.1；项目基线要求 CUDA 12.8
对应工具，目标 GPU 为 SM 8.9。直接把系统 CUDA 当作项目环境会造成版本和目标架构
不匹配。shell 还可能注入 ROS Python 3.10 路径，使 Python 包解析不再属于专用环境。

完整 `cuda-toolkit=12.8.1` 安装曾因大包网络错误失败，随后 runtime 包出现 TLS EOF。
这些是当时观察到的下载失败；不能仅凭降低并发后成功就断定并发是唯一根因。

### 应该怎么修复

建立独立环境，不修改其他已有项目、系统驱动或全局 PATH；固定真正需要的 NVCC、
runtime 开发包和 Sanitizer API 包，用实际安装清单锁定版本。执行检查时隔离外部
Python 路径，并区分“版本信息符合要求”与“真实 CUDA 执行通过”。

### 实际怎么修复

专用 prefix 为 `/home/you/conda_env/agentic-gpu-debugger`，Python 3.11.16。
固定 cuda-nvcc 12.8.93、cuda-cudart-dev 12.8.90、cuda-sanitizer-api 12.8.93，
实际 Compute Sanitizer 为 2025.1.0.0。只在当次下载进程设置
`CONDA_FETCH_THREADS=1`；未关闭 TLS 校验，未修改全局 Conda 配置。

环境同时解析了 GCC/G++ 14.3.0，而诊断选定的 host compiler 是 `/usr/bin/g++`
11.4.0，故后端必须显式把选定编译器传给 NVCC，不能靠激活环境后的默认搜索结果。
保存实际 `conda list --explicit --sha256` 和隔离解释器的依赖清单。

### 测试结果与下一轮

当时命令包括 `python -I -m pytest tests/unit -q`、Ruff、mypy、pip check 和
`python -I -m gpu_agent env --json`。记录为 27 passed、Ruff/mypy/pip check 通过。
专用环境诊断退出 0，但明确为 `metadata_only`，`execution_verified=false`。
系统 `/usr` 诊断退出 1，返回版本/架构不支持的原因码。这是正确拒绝，不是未修复测试。
原始结构化结果在 `docs/t01-environment.json`；后续必须真实编译运行，不能把 T01
当作全部 GPU 项目已经验收。

## 第二轮：运行结束不代表进程和证据已经安全收尾

### 问题与为什么出现

T02 审查复现三类问题。第一，父进程退出后，子进程仍持有 stdout/stderr 管道，
如果只等待管道 EOF 就可能一直等。第二，持久化只同步 manifest 和 run 目录，
遗漏新建 run 在父目录中的目录项。第三，如果先写成功再做清理和证据重读，后续
清理失败也可能留下看似成功的验收记录。

### 应该和实际如何修复

进程执行不走 shell，并发排空输出；通过独立进程组管理和清理仍存活的子进程。
持久化补齐父目录 fsync。把清理、证据重读和原始源码核对放在成功验收提交之前；
失败仍写明确终态，不产生正常 acceptance。增加 typed CleanupResult、stdin/binary
哈希绑定及同一原子 manifest 内的状态审计。

CPU 参考计算在这里的用途是检查 GPU 输出数值，不是把 CUDA 测试换成 CPU 测试。
真实输入仍经过 GPU 分配、拷贝、kernel launch 和同步。选用可精确表示的二进制小数，
避免把浮点舍入差别误判为逻辑缺陷。

### 测试结果

当时执行 `python -I -m pytest tests/unit tests/integration -q`，记录 129 passed，
0 skipped；包括 45 个真实 C++ harness 边界用例。GPU 命令为
`python -I -m pytest tests/gpu/test_clean_kernel.py --require-live --gpu-run-root runs/m0 -q -s`，
记录 3 passed、0 skipped：一个正常验收，两个清理/证据读取失败注入，不是三个算法成功。
正常 run 为 `6135bcdc51c74b18baabc4a96153761f`，长度 1、257、1025 的结果一致。
摘要见 `docs/t02-acceptance.json`。原始 run 路径为当时仓库的 `runs/m0/<run_id>/`，
本次没有逐项重读该历史目录，也没有把路径存在性当作证据校验。

这一轮证明可信 clean workload 和失败收尾，不证明模型候选安全或补丁正确。

## 第三轮：限制性 umask 导致容器无法读取快照

### 问题、原因和方案

宿主以 `umask 077` 创建文件时，即便 `os.open` 传入期望 mode，最终权限仍会被
umask 收窄。容器 UID 与宿主文件所有者不同，因此可能无法读取已准备的隔离快照。
正确方向是明确设置本后端新建文件的预定权限，而非提高容器权限或放松整个目录权限。

### 实际修改和测试证据

`b6b6b1d` 在 `IsolatedGPUBackend` 的独占、NOFOLLOW 文件创建与写入后，对同一 fd
执行 `os.fchmod(stream.fileno(), mode)`。本次已核对这个差异。它没有把容器改成
privileged，也没有改动用户任意文件。该提交对应测试实际运行数量和原始 stdout
没有在本轮找到，不能写成“当时全套通过”；代码变更与后续验收结果是不同证据层级。

## 第四轮：模型调用期限与工作进程归属

### 已知问题与证据限制

`4835322` 新增 provider_process/provider_worker 并修改 provider、隔离导出和报告；
`2480a6b` 再修订 worker startup ownership，配套修改 deadline 集成测试。
提交及文件变化已核对，说明实现曾对调用截止时间和启动归属边界做过修复。
本轮没有找到足以还原当时所有调度交错的原始失败日志，因此不能虚构具体 PID、
被终止的进程、超时时间或测试通过数。

### 应该如何处理与实际状态

设计要求是调用必须有有限期限、控制器只管理自己启动且确认归属的 worker，不能
靠无限等待或杀掉无关进程处理超时。历史代码已加入独立工作进程和对应测试；
准确的失败复现应从上述两个提交的 diff 与 `test_provider_deadline.py` 逐项恢复。
本篇把它保留为“实现和测试源码可查，历史执行日志未补齐”，不假装这一轮已完整复原。

## 本篇交接

未重装环境、未运行 GPU、未调用 API。后续 60 秒请求超时、模型格式与缓存问题分别
见其他历史篇，不把不同年代的“超时”合并成同一个根因。环境版本数字是历史记录，
不是今天重新探测硬件后得到的新结论。
