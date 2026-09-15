# Benchmark protocol

Public cases expose only a neutral case ID and `src/kernel.cu`. Mutation names,
expected tools, references, checker configuration, and split assignments remain
controller/evaluator metadata and are never included in an Agent request.

A case becomes countable only after the clean source and mutant run with identical
toolchain, harness, and input-set hashes. The clean execution must pass its Oracle
and every required check. The mutant must produce the expected target-tool finding;
a timeout, tool error, unsupported result, or missing finding is not confirmation.
Every fixed repetition is retained in `validation_run_ids` and detection outcomes.

The private corpus root is configured outside the repository and public RunStore.
A template ID cannot appear in both public and private splits. Case manifests publish
hashes and high-level provenance, never private inputs, expected outputs, seeds, or
raw evaluator artifacts.

The current development slice contains four real GPU-validated failure families.
The 16-public/8-private count is a release gate; unvalidated candidates are reported
as pending and must not be padded with mocked or skipped runs.
