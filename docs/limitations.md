# Limitations

`VERIFIED_FIXED` is evidence for one candidate, recorded toolchain, input suite, and
required checks—not proof for all inputs or hardware. Containers share the host kernel,
GPU driver, and physical GPU; this is not a hostile multi-tenant VM boundary. The vector
Oracle covers the registered protocol, not arbitrary CUDA applications. Dynamic parallelism,
multi-GPU behavior, performance correctness, full-chip hardware verification, and general
CUDA reduction are outside this release.

The development corpus currently has four live-validated families. The required 16 public
and 8 private cases and the five-mode paid evaluation are not complete, so the release gate
must remain closed.
