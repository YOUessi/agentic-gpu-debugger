# 公开 GPU Seed 批次运行说明

## 0. 这组命令是什么

这组入口将仓库中已有的四个公开 seed 接入原生 clean/mutant 验证控制器：
`benchmark run-seeds` 负责串行执行，`batch-report` 读取持久化摘要，
`export-batch` 导出 summary 明确列出的公开运行。它不读取 private holdout，
不调用模型，也不替代 Task 4 的 strict candidate verification。

## 1. 准备你的原开发环境

激活项目的 Python **3.11 或 3.12** 环境，然后进入仓库根目录：

```bash
cd /实际仓库路径/agentic-gpu-debugger
python --version
bash gpu_batch.sh setup
```

`setup` 做两件显式操作：

1. 对没有 `.git` 的独立源码目录建立一个本地源码快照提交；不会添加 remote 或 push。已有 Git 仓库保持原历史。
2. 以 `pip install --no-deps --no-build-isolation -e .` 安装当前 checkout。

该命令不安装驱动、CUDA、Docker 或新依赖，不降级 Python 限制，不修改容器锁，不创建 corpus family。如果缺依赖，先按原项目锁定环境解决，不能删除约束来绕过检查。现有 Git 目录有未提交修改时脚本会停止，不替你 reset 或 commit 未知修改。

也可以指定现有环境 Python 的完整路径：

```bash
GPU_BATCH_PYTHON=/你原环境/bin/python bash gpu_batch.sh setup
```

所有后续命令必须继续使用相同解释器。切换 checkout 后需重新执行 editable 安装，避免命令指向旧源码。

## 2. 先做离线回归与静态预检查

```bash
python -I -m pytest -m 'not gpu and not container and not live_llm and not release' -q
ruff check src tests
mypy --strict src/gpu_agent
ruff format --check src/gpu_agent/benchmark/batch.py \
  src/gpu_agent/benchmark/batch_models.py \
  src/gpu_agent/benchmark/batch_report.py \
  src/gpu_agent/benchmark/batch_security.py \
  src/gpu_agent/benchmark/validation.py src/gpu_agent/cli.py \
  tests/unit/test_seed_batch.py tests/unit/test_seed_batch_safety.py \
  tests/unit/test_gpu_batch_script.py
```

这些检查必须在准备真实 GPU 实验的同一 checkout 中通过。离线通过不等于 GPU 通过；失败时保留输出，不要通过删除测试或降低约束绕过。

```bash
bash gpu_batch.sh preflight
```

这一步检查四个案例的 source/harness/input hash、registry、容器工具链锁、干净 Git 状态。它**不运行 GPU**，也不能证明 Docker GPU 可用。没有 family 配置也可做静态预检查。

## 3. 使用已有的可信 corpus family

真正执行前必须配置**原来已经建立的可信 family/controller**，不能把一个空的新目录当成 family：

```bash
export GPU_AGENT_CORPUS_FAMILY_ROOT='/你已有的可信controller目录'
```

脚本读取现有 `family.json` 的 `public_store`，默认将其父目录作为 data root。例如：

```text
既有 family.json: public_store=/home/you/experiments/corpus
本批 data root:    /home/you/experiments
```

既支持 `.../runs`，也支持 `.../corpus`，不强制改名或搬迁原 store。也可显式指定，但必须与 family 一致：

```bash
export GPU_BATCH_DATA_ROOT='/home/you/experiments'
```

下列对象必须已经存在并符合既有信任边界：family root、family.json、ledger/identity.key、已绑定的 public store。data root、family root、ledger、public store 为当前用户所有的 `0700` 目录；已存在的 `.seed-batch.lock` 必须为当前用户所有、单硬链接的 `0600` 常规文件。

**没有 family 时会返回 `FAMILY_CONFIG_REQUIRED`，不会自动 `CorpusFamily.provision()`。** 权限不合格会拒绝，不会静默 chmod 你的目录。先核实路径和归属；不要对共享目录递归 chmod，也不要为绕过错误新建另一套 authority。

新批次使用既有 family/ledger，已有条目不会被覆盖。普通 `run`/`all` 只验证，不注册，因此可以先复测而不申请 corpus membership。

## 4. 先跑一个真实 memcheck 案例

```bash
bash gpu_batch.sh preflight
bash gpu_batch.sh run case_0001
```

该命令依次执行 clean 和 mutant，在隔离 GPU 后端编译、普通运行、Oracle 对比、目标 Sanitizer 检查，再交给现有原生验证控制器和 Builder 核验。不会调用付费模型，不会降级成宿主机任意 CUDA 执行。

会打印 `batch_run_id`。失败也要记住这个 ID，后续可读取已保存的进度和真实日志。目录/身份发生持续变化时系统会停止后续持久化。与项目其余 RunStore 一样，这不是针对拥有同一 OS 账号的恶意进程构建的沙箱；同账号敌手需要 VM 或专用主机边界。

## 5. 查看和导出

将下面 ID 替换为命令实际打印的值：

```bash
BATCH_ID='实际batch_run_id'
bash gpu_batch.sh report "$BATCH_ID"
bash gpu_batch.sh export "$BATCH_ID" "$HOME/gpu-batch-current-results.zip"
```

导出内容只来自 summary 中明确记录的 clean/mutant run ID，加上本批父节点报告。不是遍历全部 direct children；逐项核验父节点、case、role、binding 和 public 可见性，并在读取时校验 artifact hash。

导出不包含 controller key、ledger、evaluator store、private holdout 或私人验证审计对象。它是**公开开发批次的诊断快照**，不是可恢复的 corpus 备份，不是完整五模式评测凭据。分享前仍应查看公开源码/日志是否含个人路径等信息。

程序会生成 `inventory.json`，记录其他每个成员的大小和 SHA-256（清单自身不做递归自哈希）。已存在的输出 ZIP 不会被覆盖。

## 6. 公开 seed 全部运行

首个 case 的环境和执行路径确认后：

```bash
bash gpu_batch.sh all
```

| Case | 工具 | n | clean 与 mutant 各自的固定重复次数 |
|---|---|---:|---:|
| case_0001 | memcheck | 257 | 1 |
| case_0002 | racecheck | 256 | 5 |
| case_0003 | initcheck | 32 | 1 |
| case_0004 | synccheck | 32 | 1 |
| case_0005 | memcheck | 64 | 1 |
| case_0006 | memcheck | 33 | 1 |
| case_0007 | memcheck | 128 | 1 |
| case_0008 | racecheck | 64 | 5 |
| case_0009 | racecheck | 32 | 5 |
| case_0010 | racecheck | 96 | 5 |
| case_0011 | initcheck | 17 | 1 |
| case_0012 | initcheck | 64 | 1 |
| case_0013 | initcheck | 33 | 1 |
| case_0014 | synccheck | 32 | 1 |
| case_0015 | synccheck | 32 | 1 |
| case_0016 | synccheck | 32 | 1 |

`case_0001`–`case_0004` 是已有真实 GPU family 证据的原始 seed；
`case_0005`–`case_0016` 目前只是人工审核和静态预检查通过的候选，不能写成已通过。
每个 case 只运行它的目标 Sanitizer，不把 seed 验证冒充为 candidate 的全套 Strict
Verification。当前 registry/provenance hash 已随候选集变化，注册前必须重新生成与当前
提交绑定的真实运行证据。Task 4 的 `gpu-agent verify` 原路径继续保留。

某案例执行失败会留痕，后续案例继续；最终有失败时退出码为 1，用户取消为 130，配置错误通常为 2。`COMPLETED` 只是批次结束，逐案例的 `VALIDATED`/`FAILED` 才反映验证结果。

## 7. 显式注册（不要作为首次测试命令）

确认已有 family 配置和 GPU 验证结果后，确实需要请求注册才使用底层 CLI：

```bash
python -m gpu_agent benchmark run-seeds \
  --repository "$PWD" \
  --data-root '/既有public_store的父目录' \
  --register
```

`VALIDATED` 不等于 `REGISTERED`；注册仍要通过旧代码的 runtime attestation、原生证据重验和 ledger 唯一性规则。重复注册可能被拒绝，这是原有规则，不要删除 ledger、改 hash 或创建新 family 来“让它通过”。

## 8. 常见停止点

| 结果 | 含义与处理 |
|---|---|
| `REPOSITORY_NOT_READY` | ZIP 没有 Git HEAD，或源码仍有未提交变更；按 setup/正常 Git 提交处理 |
| `CHECKOUT_MISMATCH` | 当前安装指向别的源码目录；在新目录用原环境做 editable install |
| `FAMILY_CONFIG_REQUIRED` | 缺少已有可信 family 或 key；设置原 controller 路径 |
| `FAMILY_CONFIG_CONFLICT` | data root 不是已配置 public store 的父目录 |
| `UNSAFE_DIRECTORY` / `UNSAFE_LOCK` | 所有权、权限或对象类型不合格；先核实，不自动修复 |
| `DIRECTORY_CHANGED` / `LOCK_CHANGED` | inode/路径/权限在持有期间变化；停止，保留旧路径证据 |
| `BACKEND_UNAVAILABLE` | Docker/GPU/锁定镜像不满足；案例 `NOT_RUN` 不是程序故障 |
| `TARGET_FINDING_MISSING` | 真实工具没有稳定检出预期缺陷；保留原日志，不改 expected hash |
| `NATIVE_EVIDENCE_REJECTED` | 原有 Builder 不接受证据，不能凭 batch summary 强行注册 |

本说明不承诺 GPU 已通过。请将第一次真实运行导出的结果包用于下一轮分析。
