# Reproducible demo

The recorded DeepSeek M1 run `ebce9bb96b9e45ad93b0fee942489ef7` was produced from
commit `8bd3cea`: diagnosis used five physical model calls, generated one patch, and
strict verification reported `VERIFIED_FIXED` with public 1/private 13 inputs. The
candidate hash was `b8d93e24…a2010`; input-suite hash was `9b596c90…a8625`.

Run `gpu-agent diagnose benchmarks/public/case_0001/public_input`, retain the printed
run ID, then pass that exact ID to `gpu-agent verify RUN_ID --generated-candidate
--strict` and `gpu-agent report RUN_ID`. Negative controls retain the OOB, return wrong
zeros, hard-code 257, or fail compilation; they produce NOT_FIXED or REGRESSION_DETECTED.
