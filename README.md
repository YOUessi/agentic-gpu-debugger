# CUDA 分段前缀和 Coding Workspace

这是一个从真实 CUDA 调试开发任务中抽取并脱敏得到的独立 Coding Workspace。

目标程序位于：

```
benchmarks/public/case_0022/public_input/kernel.cu
```

Workspace 保留了真实 CUDA 源码、公开输入和原任务使用的 C++ harness，并提供一个不包含既有
修复答案的公开检查脚本。

## 功能规格

输入为两个长度均为 `n` 的 float32 数组 `a` 和 `b`。

- 每连续 128 个元素构成一个独立 segment；
- 每个 segment 内，对 `a[i] + b[i]` 执行 inclusive prefix sum；
- 新 segment 必须重新开始累计；
- 最后一个 segment 可以不足 128 个元素，只处理有效元素；
- 保持 `run_vector_add` 接口及现有 JSON 输入/输出协议兼容。

## 任务

当前 CUDA 实现无法稳定满足上述功能规格和工程验证要求。请阅读 Workspace 中的 CUDA
源码、公开任务资料和检查工具，自主定位问题并完成修复。

要求：

1. 正确实现每 128 个元素独立的 inclusive prefix sum(a+b)；
2. 正确处理 segment 边界和不足 128 个元素的尾段；
3. 保持现有 `run_vector_add` API 与 JSON 输入输出协议；
4. 修复必须适用于合法范围内的不同 n，而不是只针对公开的 n=257；
5. 不得修改任务语义、harness 或 checker 来迁就错误实现；
6. 有 CUDA/GPU/Sanitizer 时，应尽量使用现有检查能力验证修改。

## 文件说明

- `benchmarks/public/case_0022/public_input/kernel.cu`：需要调查和修复的 CUDA 实现。
- `benchmarks/public/case_0022/public_input/input.json`：代表性公开输入。
- `benchmarks/public/case_0022/public_input/task.json`：公开任务元数据。
- `benchmarks/harness/`：编译和输入输出 harness。
- `scripts/check_case0022.py`：公开功能与运行检查。
- `Makefile`：便捷的编译/检查入口。

## 运行

```bash
python3 scripts/check_case0022.py
```

或：

```bash
make check
```

如果环境存在 `nvcc`，脚本会编译当前 `kernel.cu`。
如果同时存在可用 NVIDIA GPU，则会运行多个边界规模的数值检查。
如果还存在 `compute-sanitizer`，会运行标准 Sanitizer 工具检查。

没有 CUDA/GPU 的环境仍可以读取任务、检查代码并运行 host-side 公共规格自检；
脚本会明确标记哪些 CUDA 检查被跳过，不能把 SKIP 当作通过。

## 脱敏边界

本 Workspace 不包含已知正确 CUDA 实现、故障标签、历史修复日志、之前模型生成的候选补丁，
也不包含 evaluator-only 的最终验证资料。

最终应交付实际代码修改、问题根因说明、执行过的验证以及仍存在的环境限制。
