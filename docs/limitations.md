# Limitations

`VERIFIED_FIXED` is evidence for one candidate, recorded toolchain, input suite, and
required checks—not proof for all inputs or hardware. Containers share the host kernel,
GPU driver, and physical GPU; this is not a hostile multi-tenant VM boundary. The vector
Oracle covers the registered protocol, not arbitrary CUDA applications. Dynamic parallelism,
multi-GPU behavior, performance correctness, full-chip hardware verification, and general
CUDA reduction are outside this release.

The frozen e80ce75 corpus has 16 public and eight private live-validated cases covering four
Sanitizer families; its 240+120 five-mode experiment is complete. Many cases share a vector-add
template, and some defects are side kernels rather than failures of the main numerical output.
These counts do not establish broad CUDA debugging generalization. E uses more model calls
than D and did not achieve a higher raw repair count in these runs.

The later public diversity extension adds rotate-add, tiled stencil, weighted histogram,
and grouped reduction with separate CPU oracles and faults in their numerical computation
paths. It does not retroactively replace the frozen corpus or demonstrate a new model repair
rate. See [scope and acceptance](case-diversity-CN.md) and the
[iteration record](repair-log/2026-09-29-case-diversity.md). Coverage remains bounded by the
two-array/one-array harness; multidimensional workloads and real-world codebases remain gaps.

Model-generated patches may compile but remain wrong; a clean numerical output alone is not
sufficient to claim a repair. Unknown provider usage stays unknown, and retries cannot prove
that an earlier timed-out request was not received or billed.

Independent manual scoring and the original full release gate remain incomplete. Later
tooling fixes and supplemental test results are documented separately; the baseline is not
silently rewritten or presented as evidence for a newer commit.
# 公开功能自检边界（2026-10-01）

repair v2依赖显式公开task.json与固定允许的算法checker，目前支持vector add、循环移位
加法、三点stencil、加权histogram和32元素分组归约。它不会从故障源码自动推断用户本意，
2026-10-03另加入二维五点stencil及128元素分段scan，共7类功能checker；二维布局仍通过
公开规则解释扁平数组，并非任意维度/多输入输出接口。上述能力边界仍然成立。
也不能执行模型提供的验证脚本；当前repair入口在模型/GPU启动前拒绝缺失或不合法的
公开规格/输入，而非执行到最后再返回自检不可用，更不宣称功能正确。
自检只覆盖调用者提供的公开输入，不能证明所有输入都正确，不能代替最后的独立验证。
本轮旧失败子集的通过不能当作新的完整D/E优势评测，更不能拼接原37个成功项报告新版本48/48。
