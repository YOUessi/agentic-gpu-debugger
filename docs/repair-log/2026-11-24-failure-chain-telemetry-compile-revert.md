# 2026-11-24 — separate incomplete, build and numeric recovery diagnostics

## Frozen revisions

Provider replay: `ca5e8b3c03b658e48830d41be4d956fa8168315b`. Compile/revert/metrics: `f3e26ba319ffd4a1be35e19bfc3ef6e0875d36c3`. Numeric diagnostic arm: `b9f3eb789f52c3c4530d614b9fd7128ba5afc202` on the isolated P1 branch. No Planner or Extended AST × Diagnostics factorial sweep was run.

## A. Incomplete response evidence

Read the prior physical invocation artifacts, not aggregate token estimates:

| Prior call | Input | Output | Reasoning | Status |
|---|---:|---:|---:|---|
| h002 initial | 1829 | 4096 | 4095 | LLM_INCOMPLETE |
| h001 recovery live decision 1 | 2256 | 3971 | 3916 | COMPLETED |
| h001 recovery live decision 2 | 2474 | 4096 | 4095 | LLM_INCOMPLETE |

Historical provider response reasons are unavailable. Output cap is now operator-configurable (GPU_AGENT_MAX_OUTPUT_TOKENS, default 4096, 256–16384), carried through WorkerRequest, invocation telemetry, repository checkpoint/session identity and sweep plan. Allowlisted response status, incomplete reason, available output-item statuses/finish reasons, output text length/hash, and masked structural tail are recorded. No raw reasoning or generated values in termination telemetry. Incomplete classifications do not enter the invalid-format retry path.

Reconstructed the two old snapshots from frozen source, public tool evidence and checkpoint. For recovery, verified exact action observations, candidate diff, accepted hashes, active AST and query fingerprints. The old request had no payload hash, so these are reconstructed inputs, not a byte-attested historical request. Replay at fixed local Qwen3-8B-AWQ, same prompt v1, reasoning configuration, vLLM initialization seed 43; one physical request per condition, no tools/retry. 8192 tested only when the 4096 replay explicitly returned max_output_tokens.

| Snapshot | Cap | Input | Output | Reasoning | Wire result | Action |
|---|---:|---:|---:|---:|---|---|
| h002 initial | 4096 | 1829 | 4096 | 4095 | incomplete / max_output_tokens | none |
| h002 initial | 8192 | 1829 | 7700 | 7610 | completed | before probe size 129 |
| h001 recovery decision 2 | 4096 | 2474 | 2572 | 2501 | completed | after probe size 32 |

h002 4096 response had zero output_text characters. At 8192 it had 332 output_text characters and a valid typed decision. The two h002 requests have the same payload SHA256 `54ecbad68dea9269e6170b89a84a870c679d9c2ab61ad0c9d937a1ef4fe79440`. Recovery payload SHA256 `d27c5f5682826dc0940195914dc7de1d717199d39fef1a5ba98846d3c9b1f1df`; no 8192 recovery request was made because 4096 completed. No correctness or general causal success-rate claim from these three calls. Reasoning effort not changed; server seed does not guarantee request-level deterministic sampling.

| Artifact | SHA256 |
|---|---|
| h002-first-snapshot.json | e9e539f5f2af1d3045c86689f79e4cc35792d02b16947524c0ebc4edf0498ef6 |
| h001-recovery-second-snapshot.json | 9d9c7b79a5e7f340809921f12cda5e3dbbe80151c5fe53c16fb680f5c35be8a1 |
| h002-first-4096.json | a2adb9114d425b140dbca45ebe45b2132eaf4f7c9311cf33aa1def62e8114864 |
| h002-first-8192.json | d906440bcf25fc7eaf06612c20c30280e797dd3cf920b5ab7a5102287c8e09d6 |
| h001-recovery-second-4096.json | f0c4d2f273961c76be98d9de586abfeccd7d6ba1a46f4fd2599147f2f2f342d5 |

## B. Compile and rollback

Separate opt-in flags `--compile-gate`, `--allow-revert`. Default off, preserving previous executor prompt/payload. Separate compile quota (default 6, max 8), compile-only isolated backend and exact SHA-bound tool schema; independent full checker quota. When active, edited candidates require successful compilation before after-probe/full checker. Compile errors are extracted from bounded full compiler stderr, prioritizing first located kernel.cu error; source excerpt comes from candidate bytes. No arbitrary command or checkout. One rollback per session restores only the latest accepted AST before-image, authenticated by source hashes and immutable public audit, invalidates pinned AST and does not grant a public pass. Resume reconstructs compile/rollback ledger from completed public events and before-image/admission artifacts.

Real Tang NVCC replay, no live LLM calls:
- h001 exact prior failing candidate `a3a89df8bb0aa97a3631097129b36969bc8ea9c6b6c2dd43615689b8278a6488`: BUILD_FAILED at line 16, identifier "own" is undefined. Compile child `26a55bfb423f4485a5240517fd81a300`. Revert restores original `47e69f940ca6c8b9ab287b5ee5c9aed7973f4d1eeb7ac41b7e855a01609c9bce`; historical model good candidate `4d7d5fd3d3384cb610bda8128f3b580ed5202e269336b4d325e94b6f2e5d190c` BUILD_PASSED, child `eb0ba26dd97146ebba56587669ddeccf`. Parent run `aba5e5beaf364b30adcdbed0eaf55af2`.
- h003 exact prior failing candidate `acfe74458931f8fc7fff369c9f5a26425b6357214d8a87aa9df44a52319503b8`: BUILD_FAILED line 12, expression must be a modifiable lvalue. Child `5d994895df8e4ac0876efc37815c4fca`. Revert restores original `e8c6b2386175d3e77689311b9fc2dd54a7014365b45e2017a454a79fa5958bde`; historical model good candidate `dd2f7ba2889c6870416f3aabca6bcdf18b16a00fea0cbe1fd1124ea3b17519e4` BUILD_PASSED, child `d19044d780d947eb825a566e90c067bc`. Parent run `aff4f0b9cf174a788cb3e373d70dd91e`.

Both confirmed exact original bytes after rollback, no pinned state, no public pass, no checker quota consumed. Compile children only; no numeric dispatch. Correct historical actions are positive controls, not live model recovery. Replays `compile-revert-h001.json` SHA256 `3d40c7f8fde5e3cf59ab70d466a3211d6047eee14d4180cd65db015eb46733c4`; `compile-revert-h003.json` SHA256 `abdef2964306751dcb444918720fd353d842b228f13a18193d1409ed7e35cf9d`.

## C. Targeted public diagnostics

Same h001 numerator-only candidate `992264e8d9c65ba504357399a1d8229fe03a6396f215f0e226da5618ea9c9f73`, from previous model single-step result. Extended syntax disabled, public diagnostics enabled, isolated P1 SHA above. Real checker: NUMERIC_MISMATCH, size 32, FINITE_NUMERIC_MISMATCH, output_finite=true, all_zero=false, first_failing_group=0, tail=false. memcheck, racecheck, initcheck, synccheck all CLEAN; each sanitizer program output fails numeric comparison. Public check remains failed. No hidden expected outputs in model feedback. Parent `6dcd3543345b46f7b8ae48efdcab5c97`, checker `77c6a7054e0547e892fc81dc003f5265`. Summary `numeric-diagnostics-numerator-only.json` SHA256 `25487033baeab93526684591ec798669ade0e7f0a5f6a1821fb20f7a8f04ef26`. This verifies feedback availability, not model use or second candidate.

## D. Metrics and validation

Canonical metrics: first_ast_edit_accepted_turn, ast_edit_accepted_count, build_passed_candidate_count (distinct candidate hashes with attested build or numeric dispatch), numeric_dispatched_candidate_count, public_check_passed_candidate_count. Sweep effective_patch for repository arms now requires build evidence, not accepted AST edit. Legacy field names retained for reading historical records. Prior h001/h003 full trials had accepted edits 3/2 but zero build-passed candidates.

Tang final frozen implementation: 121 focused tests passed in 55.75s, strict mypy 122 source files, Ruff lint and format pass. Provider-specific earlier validation: 61 tests. Final regression SHA256 `a5ac7d8867cdb112e1165094658128f9f92e816bc8a3bb92d108f5a6a89056a6`, mypy SHA256 `f0589b4b402aba3c2eb319e964719fe30b143cab213cf5d528b089bc4d571edc`. HardSet V2 native-contract CI passed on implementation SHA; full CI still in progress at last check. Raw evidence: `/home/you/gpu-native-harness-validation-20261124/p2-failure-recovery/`. Local inference stopped; GPU memory 204 MiB, lock free.

## Next gate

Do not run generic factorial or Planner comparison yet. Confirm live model selection of rollback/compile after actual failures, and live numeric failure recovery with diagnostics-only versus basic feedback at the same explicit output cap. Extended AST needs a targeted case where the original statement-only tool cannot express a required condition/loop edit; existing parser/ToolHost fixtures do not establish real-model benefit. Keep all three failure types separate and use real accepted/build/numeric/pass counts. No independent strict verification or live successful second-candidate recovery claimed in this batch.
