# 2026-11-24 — isolated extended AST and public diagnostics

Parent P0 baseline: `52be179d46ab8be335a93abcf1d1ee986970bbc6`. Validated implementation: `b9f3eb789f52c3c4530d614b9fd7128ba5afc202`.

## Changes
- Independent opt-in flags: `--extended-syntax`, `--public-diagnostics`; checkpoint identity binds the mode.
- Full read of a previously observed truncated node, with source SHA and alias binding, maximum 1,200 UTF-8 bytes, bounded calls, and audited checkpoint replay. No general unbounded read or fuzzy edit.
- Condition, loop condition/update, initializer, assignment RHS expression slots; isolated expression parse plus full candidate parse. Overlapping spans and invalid/stale targets are rejected atomically.
- Mixed NOOP feedback reports unchanged and changed aliases and requests omitting unchanged entries. Parser failure can carry candidate line/column.
- Bounded numeric category, finite/zero flags, first failing group, tail flag. Expected outputs remain inside the controller, not model feedback. Multi-group mismatch is descriptive, not a diagnosis of warp synchronization.
- With public diagnostics enabled, the sanitizer stage proceeds after a numeric failure where execution is possible. Individual outcomes are recorded without overriding an already failed numeric verdict. This is a separate stage, not a new unrestricted sanitizer tool.

## Validation
On Tang in the existing isolated Python environment:
- `pytest -q tests/unit/test_gpu*harness* tests/unit/test_gpu_public_numeric_feedback.py`: 109 passed in 45.90s.
- Strict mypy: 122 source files, no issues.
- Ruff lint and format checks passed.
- Regression log SHA256: `c0e5c29962b78669c0cc8a3bd0ce57a95955f18ee5cc089aa69f04e2a3fc76d7`.
- Typecheck log SHA256: `f0589b4b402aba3c2eb319e964719fe30b143cab213cf5d528b089bc4d571edc`.
- Local artifacts: `/home/you/gpu-native-harness-validation-20261124/p1-extended-ast-diagnostics/`.

Tests exercise the actual tool host, parser, checkpoint and ledger, with scripted model/checker/backend fixtures as appropriate. They do not measure live LLM accuracy or real CUDA sanitizer behavior. General macro/include/function-signature edits, unbounded nodes, and arbitrary repository editing remain out of scope. Do not mix this branch with P0 Planner-only comparison until an explicitly controlled experiment. No change to hidden verification or reference-answer visibility.
