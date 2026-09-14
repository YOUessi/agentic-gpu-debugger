# T01 验证记录

记录日期：2026-09-15（Asia/Shanghai）。本记录只覆盖环境诊断，不构成 M0/GPU workload 验收。

## 实际安装

- 专用 prefix：`/home/you/conda_env/agentic-gpu-debugger`；Python 3.11.16。
- NVCC 12.8.93；Compute Sanitizer 2025.1.0.0。
- 目标 RTX 4090 Laptop / SM 8.9；Driver 580.178.04。
- 本次诊断的 host compiler：`/usr/bin/g++` 11.4.0。
- Conda 还解析了环境内 GCC/G++ 14.3.0；后续 backend 必须将 Settings 中的 host compiler 显式传给 NVCC，不能把默认编译器当成已探测的编译器。
- 未修改其他 Conda 环境、系统 CUDA、显卡驱动、全局 PATH 或 ROS 配置。

初次完整 `cuda-toolkit=12.8.1` 安装因大包网络错误未完成；改为固定同 release 的 `cuda-nvcc=12.8.93`、`cuda-cudart-dev=12.8.90`、`cuda-sanitizer-api=12.8.93`。后续三个 runtime 包曾遇到 TLS EOF；核对官方地址仍可访问后，使用本次进程内 `CONDA_FETCH_THREADS=1` 重试成功。没有关闭 TLS 校验或更改全局 Conda 配置；不能据此断言并发是唯一根因。

`environment.lock.txt` 来自实际 `conda list --explicit --sha256`，同时包含 NVIDIA 官方 release channel 和 Anaconda defaults 的依赖；不是声称所有子包均来自 NVIDIA channel。`requirements.lock` 来自隔离 Python 的 `pip freeze --exclude-editable`，没有 editable 本机路径或凭据。

## 验证命令与结果

下面的 Python 命令使用上述 prefix 的解释器；`-I` 用于排除 shell 注入的 ROS Python 3.10 路径。

```bash
python -I -m pytest tests/unit -q
python -I -m ruff check src tests
python -I -m mypy src/gpu_agent
python -I -m pip check
python -I -m gpu_agent env --json
python -I -m gpu_agent env --cuda-root /usr --json
```

- 单元/受控进程测试：27 passed；包含真实 subprocess 超时、模块入口，以及独立 pytest 子进程验证 runtime/marker/collection 三类 skip 门禁。
- Ruff：通过。mypy：5 个源文件无错误。pip check：无依赖冲突。
- 项目环境：退出码 0，`ready=true`，`readiness_scope=metadata_only`，`execution_verified=false`；原始查询结果见 [JSON 记录](t01-environment.json)。
- 系统 `/usr`：退出码 1，识别 NVCC 11.5.119、Sanitizer 2021.3.1，产生 `TOOLKIT_OUTSIDE_BASELINE`、`TARGET_ARCH_UNSUPPORTED`、`SANITIZER_OUTSIDE_BASELINE`。

提交前只读审查未发现 Critical/Important 问题；审查者独立重跑 27 项测试并核对真实环境报告。审查未修改代码、索引或分支。

## 边界与下一项

未编译或执行 CUDA kernel，未运行 Sanitizer 检查，未调用模型，未启动容器。metadata ready 不证明 headers/linker/运行时正常；因此 T01 完成不等于 M0 完成。

下一项 T02：受信任 clean kernel 的真实编译、CPU 参考结果比较和证据保存。T03 之后才允许模型候选/用户代码进入隔离编译运行。
