# Architecture

The Python controller owns typed actions, budgets, immutable RunStore artifacts,
evidence projection, patch scope, private Oracle execution, and verdict policy.
Untrusted CUDA is compiled and executed only by a hash-locked runner in a read-only,
no-network Docker container with bounded tmpfs and the selected GPU. CUDA C++ supplies
the candidate kernel and a controller-selected vector protocol harness.

Public and evaluator stores are separate roots. The Agent sees public source snapshots,
sanitized findings, and registered documentation chunks; it never sees private suites,
expected values, checker implementations, Docker commands, or filesystem capabilities.
