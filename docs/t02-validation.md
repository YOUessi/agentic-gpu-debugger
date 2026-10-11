# T02 / M0 验证记录

日期：2026-09-15（Asia/Shanghai）。T02 完成，M0 的受信任 clean kernel 验收通过。

## 交付与边界

- `RunStore`：唯一 run ID、注册且 hash 校验的不可变 blobs、原子 manifest、同一提交内的状态审计；拒绝路径穿越、符号链接、特殊文件和不匹配的 artifact 引用。私有 visibility 必须使用独立 store。
- `ProcessExecutor`：不使用 Shell，并发排空 stdout/stderr、共享 2 MiB 保留上限、流式 stdin、有限超时和取消、独立进程组清理；保留原始字节，坏 UTF-8 不丢失。
- `LocalBackend`：固定构建参数，清理继承环境，核对完整可信源码集合；执行绑定 binary/input hash，编译/运行各阶段保存日志和 typed payload；清理只接受自身登记的 workspace。
- CUDA C++：block 256，guard `i<n`，每次 CUDA API、launch、同步及 free 都检查错误。可信 host harness 负责严格 JSON 协议和有限 float32 输入/输出；parser 固定 nlohmann JSON 3.12.0 并校验上游 SHA256。

存储根目录和本地工具链是可信控制端所有；不防御同一 OS 用户/root 主动篡改。不存在“文件 hash 可以代替沙箱”的假设。用户程序、模型候选和自动 mutation 仍必须等待 T03 的隔离后端。

## 实际运行

正常验收 run：`6135bcdc51c74b18baabc4a96153761f`。完整 artifacts 位于本机 `runs/m0/<run_id>/`，不提交 Git；可公开摘要见 [t02-acceptance.json](t02-acceptance.json)。

工具链：Python 3.11.16、NVCC 12.8.93、host GCC 11.4.0、RTX 4090 Laptop / SM 8.9、Driver 580.178.04。显式指定 host compiler，未使用环境激活时可能设置的另一个编译器。

长度 1、257、1025 的输入均真实经过 CUDA 分配、拷贝、kernel launch 和同步；输出 dtype/shape/全部数值与 CPU 参考结果一致。测试采用可精确表示的二进制小数，避免 CPU double/GPU float 舍入差异掩盖逻辑错误。原始源码和输入未改变；临时 workspace 已删除，保存的每个 artifact 可从新的 Store 实例读取并校验。

## 命令与结果

激活专用环境后，在仓库根目录运行：

```bash
python -I -m pytest tests/unit tests/integration -q
python -I -m pytest tests/gpu/test_clean_kernel.py --require-live --gpu-run-root runs/m0 -q -s
python -I -m ruff check src tests
python -I -m mypy src/gpu_agent
python -I -m pip check
```

- CPU/集成：129 passed，0 skipped（含 45 个真实 C++ harness 边界用例）。
- GPU：3 passed，0 skipped。1 个正常验收 + 2 个失败注入用例；不是声称三个独立算法都成功。
- Ruff、mypy（12 个源文件）、pip check 通过。
- 构建 wheel 成功并检查包内包含可信 registry；CUDA 样例/harness 位于仓库，验收需要 checkout，不声称安装 wheel 即获得整个 benchmark corpus。

两个负向 GPU 用例分别注入清理失败和清理后的证据读取失败，均确认 run 为 `FAILED`、无活动阶段、不产生 `acceptance.json`。它们的失败 run 不算正常计算成功记录。

## 审查与实施解释

按用户要求并行分工：一个子 Agent 实现 C++/CUDA harness，一个实现 LocalBackend；主 Agent 维护 Store/Process 与 GPU 集成，另一个只读审查。

审查发现并经失败测试复现后修正了三类问题：

1. 父进程退出但子进程继承管道时，必须及时清理子进程组而非等管道关闭。
2. 新 run 的父目录项也需要 fsync；仅同步 manifest 和 run 目录不足以完整持久化目录树。
3. 清理、证据重读和原源码核对必须先于成功验收提交；清理异常仍须记录失败终态。

最终只读复核无未解决的 Critical/Important 问题。审查不代替真实 GPU 测试。

相对计划的具体化：增加 `--gpu-run-root` 保留验收 artifacts；typed CleanupResult；运行 request 记录明确绑定 stdin/binary；状态审计事件放入同一原子 manifest，未建立第二个 evidence store。源码授权 registry 由主 Agent 阅读 harness/kernel、核对 parser 上游 hash 后登记，不根据请求自报 hash 自动授权。

## 尚未完成

不包含 Sanitizer finding、OOB 修复、隐藏 Oracle、Docker 隔离或模型调用。`gpu-agent env` 仍只报告 metadata readiness；本次 execution_verified 仅属于这里的 clean-kernel 验收范围，绝不是 `VERIFIED_FIXED`。

下一项 T03：隔离编译/运行、memcheck 与真实 OOB。
