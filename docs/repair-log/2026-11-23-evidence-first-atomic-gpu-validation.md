# 2026-11-23｜Evidence-first Context + Atomic Multi-edit：真实 CUDA GPU 验收

## 研究动机

此前一个自由决策 DeepSeek GPU Harness 运行中，出现重复读取源码、迟迟不提交补丁和最终 Budget Exhaustion。第一步不先宣称模型“学会了”，而是修正阻碍 Agent 决策的 **工具表示与上下文输入问题**：

1. Operator 注册 HardSet case 时，将当前 mutant CUDA 源码放入 `SessionSnapshot.workspace_source`（而不是让模型每次从目录列表开始）。
2. 引入 `repo_replace_many`：模型可一次提供**同一源码文件**的 2–4 个独立精确替换，控制器检验每个 anchor 唯一且不重叠，并在验证全部通过后一次原子写盘。
3. 对照参数 `--context-mode legacy|evidence_first` 与冻结 goal/SHA/预算、Checkpoints 绑定，不允许 Resume 时偷换设置。
4. 完全不改变 CUDA Docker、Numeric Oracle、独立 Strict Verifier 和权限边界。新工具仍是模型自主选择，不是规定必须执行的硬编码 patch 流程。

## 实际版本与环境

- Github 执行源码 SHA：`06a8923542f78ee47a4c367ffd0a40c083ee66d2`。
- 固定 codeload tar SHA256：`4696b2176644710da1602a008cf7cca5c2b534e36ab0b8bb3836b787bb59512a`。
- Tang 解压后重新初始化 Git，**用于隔离 Worktree 的本地 Git SHA**：`84e345343c9139ca3e545cbbff654fbdafdddbb3`。不应与 Github 原始 SHA 混淆。
- 执行机器：Tang RTX 4090 Laptop GPU，隔离 Docker CUDA 执行器。
- Python 3.11，复用先前已按 `requirements.lock` 安装并执行过 `pip check` 的独立 Venv。
- Tang 冻结源码路径：`/home/you/gpu-native-harness-validation-20261123/evidence-first-06a8923/source`。
- Tang 实验根目录：`/home/you/gpu-native-harness-validation-20261123/evidence-first-06a8923/real-gpu-evidence-first-h001`。
- 此次 Tang 静态 Ruff & Format 通过，针对 repo harness/worktree/public 的 **31 个 CPU 回归通过**。GitHub 专项 CI 冻结版本对应 **186 passed**。

## 实验实际结果：真 CUDA，脚本化决策

| 指标 | 实测 |
|---|---|
| 模型决策主体 | `EvidenceAwareScriptedDecision`，**脚本化测试替身** |
| 真实 LLM 物理请求 / Tokens | **0 / 0** |
| 工具动作 | **3** |
| 动作顺序 | `repo_replace_many → repo_run_public_checks → finish` |
| 首次有效代码修改 | turn 1 |
| 首次公开 CUDA 检查 | turn 2 |
| 首次完成 | turn 3 |
| 真实 Docker CUDA public checks | **1 次**，通过 |
| Numeric Oracle | `PUBLIC_CHECKS_PASSED` |
| Compute Sanitizer | memcheck/racecheck/initcheck/synccheck = **4 个 CLEAN** |
| public child RunStore | `2c145613d90449a481ff260c458be2a0`（COMPLETED） |
| parent RunStore | `e41cea126f74417aa3945183cc4d2e19` |
| 最终源码 candidate SHA256 | `88dd297b8a0bc53dc3b93b7d1a4678551346b4855d7139b4edaddecef0070ba4` |
| 独立隐藏验证 | **此会话未执行**，状态仍 `DIFF_STAGED_UNVERIFIED` |
| Journal | 14 events，`FINISHED` |

验证真实 RunStore 子运行：57 项 public artifacts，四个单独的 Sanitizer result.json 全部 `check_outcome=CLEAN`，没有 evaluator/private 证据混入。

### 与旧脚本化实验的区别

之前的**另一个脚本化动作序列**做了 6 次工具动作和 2 次真实 GPU 检查（第一次仅修分子失败、第二次修完整通过）。本次脚本化策略把两个已知相互独立的改动合并为一个原子请求，产生 3 次动作、1 次 GPU 公开检查，最终候选 SHA256 相同。

**不能把 6→3 当作真实 LLM 节省 50% 或 Agent 策略优越的因果结论。** 它只是证明新工具能够表达并执行更短的合法路径。另有同一个决定性模拟模型在 legacy/evidence_first 两个模式下的 CPU ablation：4 次→3 次工具动作，证明上下文差异对该模拟策略的影响；**不是 DeepSeek 实测**。

## 实验审计哈希

- `source.tar.gz` SHA256 = `4696b2176644710da1602a008cf7cca5c2b534e36ab0b8bb3836b787bb59512a`。
- `gpu-evidence-first-stdout.json` SHA256 = `adf9ecb3eda52eefa3c2cc0f5560901b86b4ef057b27683e86b54f8c55fcd652`。
- `result.json` SHA256 = `602b8c2f28586102264c25fd57d8c865c98edb8d8e983a623a0719846aea218c`。
- `session-events.jsonl` SHA256 = `dd86ee87b01017b15148a43fefd0e22d3669e32d812e658d57978e252c9ed7ec`。
- `session-checkpoint.json` SHA256 = `adf85624260f4a0314117e47c793811c8f344f2a80be469909e8a3dcbe230e56`。

机器可读索引：[Evidence-first GPU result](2026-11-23-evidence-first-gpu-result.json)。

## 重新运行

```bash
ROOT=/home/you/gpu-native-harness-validation-20261123/evidence-first-06a8923
PY=/home/you/gpu-native-harness-validation-20261123/repo-public-gpu-786bc357/venv/bin/python

cd "$ROOT/source"
# 已有目录不能覆盖；换一个不存在的新 output。
"$PY" -I -c 'import sys,runpy; root="/home/you/gpu-native-harness-validation-20261123/evidence-first-06a8923"; sys.path.insert(0,root+"/source/src"); sys.argv=[root+"/source/scripts/run_repository_hardset_evidence_first_gpu_smoke.py","--repository",root+"/source","--output",root+"/another-new-run","--allow-real-gpu"]; runpy.run_path(sys.argv[0],run_name="__main__")'
```

## 下一道实验关卡

在受审计 DeepSeek provider 可用时，冻结同一开发任务、模型权重/API 版本、最大物理调用、Token/GPU 预算与独立评测。用三组：规则 Workflow / 旧自由决策 Harness / evidence-first + 原子编辑 Harness。必须跨多个 task family 与独立种子，记录真实工具失败率、首次补丁轮次、相同任务的成功率/费用和最终 Strict Verifier 结果；不重抽失败，不从公开验证挑最优候选做隐藏测试污染。当前 Tang 的测试环境未注入真实 DeepSeek 模型 endpoint 与凭据，**没有执行新的物理模型请求**。
