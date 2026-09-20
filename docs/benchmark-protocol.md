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

The public definition now contains 16 source-distinct candidates, four per tool:

- memcheck: `case_0001`, `case_0005`–`case_0007`;
- racecheck: `case_0002`, `case_0008`–`case_0010`;
- initcheck: `case_0003`, `case_0011`–`case_0013`;
- synccheck: `case_0004`, `case_0014`–`case_0016`.

Only `case_0001`–`case_0004` have prior real-GPU family evidence. `case_0005`–
`case_0016` are human-reviewed **candidates**, not live-validated corpus members. Their
source, harness, input and mutation-provenance hashes pass static preflight, but each
must still pass ordinary and instrumented Oracle checks, clean memcheck precheck where
applicable, exact target finding, and every fixed repetition on the locked GPU stack.
Mocked, skipped, static-compile or preflight results never count as that evidence.

Adding the candidates changes the registry and mutation-provenance hashes. Current-revision
registration therefore requires fresh bound runs; old run IDs are not silently reused.
The 8-private count remains a release gate outside this public repository.
