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

Model-generated patches may compile but remain wrong; a clean numerical output alone is not
sufficient to claim a repair. Unknown provider usage stays unknown, and retries cannot prove
that an earlier timed-out request was not received or billed.

Independent manual scoring and the original full release gate remain incomplete. Later
tooling fixes and supplemental test results are documented separately; the baseline is not
silently rewritten or presented as evidence for a newer commit.
