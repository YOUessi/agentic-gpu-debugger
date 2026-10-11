# 2026-09-29：案例代表性与核心计算路径扩展

## 任务与边界

用户要求解决案例代表性有限，并逐轮详细记录。本轮不修改已冻结实验，不把新案例
混入旧 240/120 结果，不访问私有 holdout，不启动付费模型评测。已有未提交的检索
与文档修改保留。本轮新增公开算法案例及其受信参考和测试，真实 GPU 测试只覆盖
新增/受影响路径。代码起点 HEAD 87fa492 加已有未提交修改。

## 第一轮：发现算法与验证参考都被 vector-add 限制

### 问题和证据

核对 benchmarks/seed-batch.json：16 个原案例均引用 case_0000 的同一 clean 源码。
verification/truth.py 的 oracle 限于 vector-add-cpu-v1；engine、derivation 和
benchmark/validation 都直接调用 reference_add。部分 fault kernel 不参与最终输出，
删除其 launch 可能保留加法输出。现有结果只能支持这组样例，不应外推广泛 CUDA 能力。

### 原因与应该修复的部分

根因不是案例数量少这一点本身，而是计算语义、访问结构、Oracle 与故障作用位置
过于同质。需要不同算法的 clean/mutant、各自 CPU 数值参考、非均匀输入，以及
核心计算删除后的反例验证；仅修改命名或增加边界尺寸不算新算法覆盖。

### 本轮实施计划和验收标准

新增四个公开算法：循环移位读取（边界访问）、共享内存 stencil（同步依赖）、
直方图（初始化与原子累计）、warp 分组归约（参与 mask）。继续使用已有两个输入
数组/一个输出数组的受限 ABI，入口旧名称仅为协议兼容，不代表输出仍是向量加法。
对应 Oracle 必须从受信案例定义选择，不能由模型指定；普通执行与验证回放要一致。

每个新案例要求：独立参考数值可手算验证；clean 在多组边界和非均匀输入下通过
数值与四工具检查；mutant 在核心路径被目标工具检出；删除核心计算不能通过数值
校验。新增测试不依赖模型生成正确补丁。尚未执行上述测试，本节只记录计划。

### 当前进度

已完成源码约束核查，正在实现多 Oracle 与新案例。尚无新增案例真实 GPU 通过结论。
后续在本文追加实际修改、命令、日志与失败，不以计划代替结果。

## 第二轮：实现与首次真实验收（包含一次失败）

### 实际修改

新增独立的 diverse-registry.json、四套 diverse_clean 参考与 public/case_0017–0020。
旧 corpus-registry、旧 16 案例与旧实验不变。oracle.py 新增四种 CPU 参考，truth
按源文件哈希选择 Oracle；engine、derivation、benchmark validation 同时接通，
避免在线验证与离线回放各用一套标准。pyproject 打包新资源。

新增 `benchmark validate-diversity`，只做原生验收，不注册、不创建 authority、
不调用模型。它核对输入/源码/受信 harness 哈希，通过现有隔离后端编译执行。
clean 用 1、31、32、33、127、128、129、257 八种尺寸和非均匀输入测数值，
代表输入再跑四工具；mutant 必须检出登记的目标类别；另从 clean 删除核心 launch
制作负例，要求不能通过输出契约与数值校验。基础设施失败不能冒充负例成功。

### 测试命令和结果

目录均为 `/home/you/.codex/worktrees/a352/agentic-gpu-debugger`，解释器为
`/home/you/conda_env/agentic-gpu-debugger/bin/python`（以下记为 `$PY`）。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -I -m pytest tests/unit/test_diverse_corpus.py tests/unit/test_verification_engine.py tests/unit/test_oracle.py tests/unit/test_public_corpus_definition.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -I -m pytest tests/unit/test_verification_engine.py -k new_algorithm -q
"$PY" -m gpu_agent benchmark validate-diversity --repository "$PWD" --output /home/you/gpu-agent-diversity-20260929-round1 --case case_0017
"$PY" -m gpu_agent benchmark validate-diversity --repository "$PWD" --output /home/you/gpu-agent-diversity-20260929-round2 --case case_0018 --case case_0019 --case case_0020
```

第一组 97 passed（35.18s，退出 0）；随后新增的真实 engine/replay 路径模拟测试
4 passed、55 deselected（10.58s，退出 0），不能把 deselected 算通过。
首次 strict mypy 暴露类型错误与 ToolResult 的 payload 字段名错误，执行 GPU 前
已改为 typed_payload 并修正类型；随后 src 全部 73 文件 mypy 通过，Ruff 通过。

原生 round1 退出 0：0017 clean 全过，mutant memcheck FINDING，删除 launch 被拒。
round2 退出 1：0018 clean 全过、racecheck 5/5 FINDING；0019 clean 全过、initcheck
FINDING；两者删 launch 均被拒。0020 clean 全过，但 mutant synccheck CLEAN，记录
`TARGET_FINDING_MISSING`，没有将 CLEAN 当作故障检出，也没有降低标准。

原始测试日志在 `.gpu-agent/diversity-review/` 的 offline-round1.log、
engine-new-oracles-round2.log、gpu-rotate-round1.log、gpu-remaining-round2.log。
两个独立输出目录均有 report.json 及 public 下不可变原始 artifacts。
四案例没有 API 调用；没有重跑旧 240/120 实验。

## 第三轮：归约故障未检出的编译证据与修正

### 现象、原因与证据

失败 mutant run 为 `82e7b75b8ad34b96a0bb07df5a3a2fac`，binary artifact
`f2a20d9e864b433690842c91c4fb9ea7`。对其原始二进制做只读反汇编：PTX 有两条
`bar.warp.sync 16777215`，但 CUDA 12.8 cuobjdump 的最终 sm_89 SASS 无 WARPSYNC。
因此原假设“常量错误 mask 必然保留可被 synccheck 检测的同步指令”不成立。
不是模型问题（没有调用模型），也不能据源码规则宣称 GPU 检测已经通过。

最初宿主 cuobjdump 不能解析 SM89，改用锁定镜像里的同版本工具。诊断过程中还
遇到镜像默认入口只允许 typed operation、artifact 0400 所有权及只读 /tmp 问题；
最终使用 cuobjdump 专用入口、uid/gid 1000、临时 /tmp 完成只读反汇编，没有
修改 artifact 权限或内容、没有运行被检查的二进制。最终命令如下：

```bash
docker run --rm --user 1000:1000 --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=32m --cap-drop ALL \
  --security-opt no-new-privileges --entrypoint /usr/local/cuda/bin/cuobjdump \
  --mount type=bind,src=/home/you/gpu-agent-diversity-20260929-round2/public/82e7b75b8ad34b96a0bb07df5a3a2fac/artifacts/f2a20d9e864b433690842c91c4fb9ea7,dst=/input/binary,readonly \
  gpu-agent-cuda:t07 --dump-sass /input/binary
```

反汇编留在 `.gpu-agent/diversity-review/warp-mutant-round2.ptx` 与
`warp-mutant-round2-container.sass`；后者内 BSYNC 是分支重汇合，不能当作 WARPSYNC。

### 应该与实际怎么修

只改 0020 clean/mutant：mask 从 kernel 内常量改为 launch 参数，clean 传全 mask，
mutant 少传高八位，核心共享内存归约与广播不变。这样设备编译器不能靠 kernel
内常量消去同一同步操作，代表主机传入参与配置错误。同步仍服务真实输出，不加
旁路探针、不加 case 特定提示，不修改 Sanitizer 判据。新 authored 源码更新清单
哈希；round2 原源码、哈希和失败原样保留。只复测 0020，不重跑已通过的 0017–0019。
此处是复测前记录，结果待追加。

### 第三轮实际结果与纠正

`validate-diversity --repository "$PWD" --output /home/you/gpu-agent-diversity-20260929-round3 --case case_0020`
退出 1；clean 八组数值和四工具全过，mutant 仍 CLEAN。run 为
`4a81eb786bab4ee7a5ec7899e4ee6298`，binary 为 `82957336ca074105a85abe119136e56d`。
同样只读反汇编仍无 WARPSYNC，故“仅把 mask 参数化能保留指令”这一假设被否定。
不再重复该方案。原始报告、源码和日志 gpu-warp-round3.log 保留。

## 第四轮：改用归约核心的条件块屏障

### 进一步诊断与方案变化

先做无 GPU 运行的代码生成检查：用 ballot 计算有效线程 mask、按 n 分支的另一种
写法仍无 WARPSYNC（warp-divergent-codegen.sass），没有为这版再付出一次 GPU
验收。决定放弃本案例的 mask fault，改成分组归约核心的条件 `__syncthreads`。
这仍是新案例设计阶段，不是对既有结果改判；新增故障类型与源 hash 均如实变更。

clean 所有线程先写 shared values，再全块同步，各组 leader 计算和写回，再同步
供组内广播。mutant 仅将第一屏障放到 `lane < 16` 条件下，导致同一块线程对屏障
参与不一致。它影响真实归约数据依赖，不是在输出完成后另加故障 kernel。
Oracle 仍是每 32 元素分组求和并广播，没有改期望数值来接受错误实现。

官方错误类别依据见 [Compute Sanitizer 手册](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#understanding-synccheck-reports)：条件不一致的
块屏障对应 Divergent thread(s) in block。该来源只作设计依据，实际验收仍以本机
锁定版本的工具输出为准。新源码预先反汇编见 block-reduce-codegen.sass，确实
包含带谓词的 BAR.SYNC.DEFER_BLOCKING 与后续无条件 BAR.SYNC，才启动 GPU。

### 实际修改与待验收项

只改 0020 两份 kernel 与新扩展清单中的该项，mutation_id 改为
predicate-block-barrier，expected_finding 改为对应的新故障类别。不是把原来的
CLEAN 改成通过，round2/3 依旧失败。命令输出目录为
`/home/you/gpu-agent-diversity-20260929-round4`，日志 gpu-reduce-round4.log。
0017–0019 不变且不重跑；结果完成后追加。

### 第四轮结果：验收编排暴露了新的缺口

退出 1，0020 clean 八种尺寸和四工具再次全过、ablation 被拒；mutant 普通运行
超时，导致新编排在 sanitizer 前就中止。原始 result 中 `timed_out=true`、
`runtime_status=TIMEOUT`、`tool_error=null`、exit=-9、耗时 30023.77ms，run
为 `dcc71ca13a2c41f2903fb1dedc860604`。尚不能仅凭超时认定同步故障检出。

## 第五轮：将 mutant 超时作为待工具确认的症状

### 原因、修法与边界

新验收入口把所有普通运行超时一律当成基础设施失败，忽略了死锁/错误同步的
mutant 可以有意在未插桩时挂住。修正只限新增验收入口：mutant 的无其他工具
错误的 TIMEOUT 要如实记录，并继续独立 sanitizer；仍必须完成全部目标工具
重复且每次命中确切类别才接受。clean、ablation 仍拒绝超时；sanitizer 超时或
未命中仍失败。不是将 TIMEOUT 视为修复或将超时当成 FINDING。

加入五种模拟情形测试，区分 mutant超时+命中、超时但CLEAN、基础设施错误、
clean超时、ablation超时；再加空/重复/未知 role 的拒绝测试。新增 `--role`
只复测失败角色，报告 `complete_acceptance_scope=false` 明确部分验收，不把
只有 mutant 的结果冒充全案例通过。

### 实际执行

```bash
"$PY" -m gpu_agent benchmark validate-diversity --repository "$PWD" \
  --output /home/you/gpu-agent-diversity-20260929-round5 --case case_0020 --role mutant
```

round4 的 clean/ablation 与 round5 的 mutant 源码相同，不重复前两项。新增
diverse-mutations.json 保存 provenance hash 的可重算原文，0020 mutation 的
provenance hash 同步新类型；这是元数据纠正，未修改 round4 已验收的 kernel
或其报告。元数据变更前后清单整体 hash 不同，不能把这些分轮测试当作一次冻结
正式评测。五轮原始证据均保留。结果待追加。

### 第五轮结果与边界

mutant 独立复测仍退出 1：普通执行 TIMEOUT，synccheck 120034.38ms 后 TIMEOUT，
空 sanitizer 日志，最终 TOOL_ERROR/completed=false；run
`cb2ef598399c44b4ac29e854efb97703`。编排正确拒绝，没有把超时算成检出。
runner-round5-tests.log 为 27 passed（0.41s），Ruff 全过，strict mypy 全部
73 文件通过。之前的定向 engine/replay 测试 23 passed、55 deselected（17.73s）。
它们不能代替这项实际失败。

## 第六轮：停止不收敛的同步案例设计，交付可检出的共享内存故障

### 为什么改变方案

用户要解决的是算法和核心路径代表性，不是必须每种新算法恰好对应一种工具。
已经得到确定证据：早期 warp mask 方案没有保留同步机器指令；条件块同步方案
则在本机锁定工具链挂住，没有完成原生证据。本轮不再换个 mask 重试、不加长
超时、不放宽完成条件，停止这一方向。同步扩展示例仍未解决，不作覆盖声明。

### 实际修法与局限

保持已通过 round4 的归约 clean 与 Oracle 不变，mutant 换成核心 shared tile
读取上界错误：32 项求和误写成 j<=32，第末组读到 values[128]。前三组还会把
相邻组首元素混入结果，这是实际数值计算错误，不是输出之后的旁路异常。
预期为 memcheck 的 Invalid __shared__ read。这个清单是新案例开发定义，随
变更更新 mutant/source/provenance，而不是偷偷改旧实验答案；所有失败版本在
旧 artifacts 中保留。无需重测 clean 或 ablation，只测改过的 mutant。

```bash
"$PY" -m gpu_agent benchmark validate-diversity --repository "$PWD" \
  --output /home/you/gpu-agent-diversity-20260929-round6 --case case_0020 --role mutant
```

范围明确从“新四案例各配四工具”改为“四种不同算法覆盖 memcheck/racecheck/
initcheck 三类缺陷”。旧 synccheck 案例和实验仍然保留；没有宣称它们解决了本批
新同步案例的失败。四算法扩展增加代表性，但不能证明广泛 CUDA 泛化。

### 第六轮结果与进一步证据

round6 退出 1，run `52b60566494a402dac4b56ea5b598e67`：ordinary 输出确实与
CPU Oracle 不同（oracle_passed=false），memcheck 却 CLEAN。检查二进制的
shared-reduce-round6.sass 能看到额外 LDS，资源报告 shared-reduce-resource-usage.log
显示 SHARED:512。源码级越界与运行时工具检出并不等价；当前证据不足以确定
未检出的底层原因，不把分配取整/工具覆盖限制猜测写成定论，也不改 CLEAN 的解释。

## 第七轮：归约算法保留，使用已证实工具可观测的全局访问故障

对新增案例的验收仍要求原生工具明确检出。为避免继续调试工具的 shared 边界
观测能力，停止这版 shared mutation：恢复正确求和上界，只把归约装载位置从
a[i] 改为 a[i+1]，形成全局输入末端的真实越界。0017 已证实本机同工具链能检出
此类分配边界错误；此处不是新增同一算法，而是归约算法中访问/聚合的数据流。
不足也必须说清：0017 与0020同属全局读取越界，不是两种独立故障家族。

clean 与 round4 完全一致，输入也不变；仅复测 mutant，目录
`/home/you/gpu-agent-diversity-20260929-round7`。如果仍未检出，不再更换归约
mutation 来凑数，应将 0020 留作未验收草案；本轮任务已有三种真实通过的新算法，
不把第四项作为虚假通过交付。

### 第七轮实际结果：通过

round7 退出 0，run `8ac14ea0e43344ebb5f43d1c707738d3`，memcheck FINDING，
类别与预先定义的 Invalid __global__ read 一致。没有修改工具解析器或通过条件。
与 round4 的 clean、ablation 组合核查前，逐项从不可变 run manifest 取 kernel
hash 对照最终清单：四项 clean/mutant 全部匹配。这个核查不是重跑或正式评测
拼接，而是按用户“不重跑正确项”要求，确认未变的测试对象能复用其开发验收证据。

最终对应关系（完整源码 hash 在 final-evidence-map.log 和各 manifest）：

```text
case_0017 clean    round1 054a2a9a64014bf2b57cac27e4a3c6c3
case_0017 mutant   round1 d9f325b1bbdb4f6c96e9510d49109eb0
case_0017 ablation round1 00bb08355191480c9212d83b7dbb13f8
case_0018 clean    round2 44d084bfc0384c8f971dbb5008a30d69
case_0018 mutant   round2 28dc4b4af3d447c7987de748f536b8fa
case_0018 ablation round2 ae978b489d6b4924b16ad6037bed677c
case_0019 clean    round2 aa098b9c434d4813a3351a54cceb14ff
case_0019 mutant   round2 81f304eeea60461cae84bf95ddfbc776
case_0019 ablation round2 bb00cf8d6ebc4342ad43edadfa14d3ab
case_0020 clean    round4 1ede72376fcc4bd98f527805040bd0d6
case_0020 ablation round4 c5fc80c643b94fd0ad32ed44143dd3de
case_0020 mutant   round7 8ac14ea0e43344ebb5f43d1c707738d3
```

### 最终离线检查、打包与未运行项

第四阶段实现后的定向检查：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -I -m pytest tests/unit/test_diverse_corpus.py tests/unit/test_verification_engine.py -k 'diverse or new_algorithm' -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -I -m pytest tests/unit/test_diverse_corpus.py tests/unit/test_verification_engine.py -k 'case_0020 or mutation_metadata' -q
/home/you/conda_env/agentic-gpu-debugger/bin/ruff check src tests/unit/test_diverse_corpus.py tests/unit/test_verification_engine.py tests/gpu/test_diverse_corpus.py
/home/you/conda_env/agentic-gpu-debugger/bin/mypy --strict src/gpu_agent
"$PY" -I -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir .gpu-agent/diversity-review/final-wheels .
git diff --check
```

第一组 31 passed、55 deselected（10.08s），涵盖四 Oracle engine/replay 和验收
入口测试；最后一次 0020 mutation 变更后只补测该案例及 metadata hash，3 passed、
83 deselected（2.90s）。两组有重合，不相加成独立测试数。早期兼容回归 97 passed
的范围见第二轮。Ruff、格式检查和 strict mypy 通过；git diff --check 通过。
最终 wheel 构建通过，SHA256 为
`1dc4aa6756384d2096f52531f341a7d6adac0abb4bdc72b80e38a1464319aebd`。
源码/清单/输入共 14 项新资源逐项核对 wheel 内字节与工作区相同，见
final-wheel-resources.log。未将较早试构建的 wheel 当作最终产物。

没有运行整个离线套件，没有运行新的 DeepSeek 诊断或付费五模式评测，没有
修改旧 corpus/holdout、注册 ledger 或冻结实验。tests/gpu/test_diverse_corpus.py
提供后续显式入口，本次实际 GPU 验收由同一 runner 的 CLI 调用完成，不再为了
pytest 的显示重复运行同一批 GPU。全部新执行 API 调用数为 0。

## 本主题结论与剩余边界

已经完成：四种新增算法、独立 float32 数值参考、受信源码映射、engine/replay
接通、真实 clean/故障/删除计算反例验收，以及新增文件打包。每项 clean 八尺寸
数值通过、四工具代表输入 CLEAN；三个 memcheck/initcheck mutant 共 3 次目标
检出，stencil racecheck 5/5 检出；四项删除核心计算均不能通过输出契约。

没有完成或不能声称：新同步故障案例（被放弃的版本没有验收通过）、新案例上的
模型修复成功率、全项目回归、广泛 CUDA 调试泛化能力。旧旁路案例保留为旧基线，
不篡改为新代表性证据。下一步若做新模型评测，要用这些新公开案例独立保存结果；
不能用本次 GPU 的人工已知 clean 参考通过冒充模型生成补丁通过。

本次未提交/推送；用户原来的检索改动、旧文档和本主题工作区修改均保留。
