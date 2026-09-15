# Reproducible public demo

The public OOB flow can be rerun from a clean checkout with the dedicated Conda environment,
the locked CUDA image, a local official-document index, and an explicitly configured provider.
Credentials remain in the controller environment and are never written to the repository or
mounted into candidate containers.

```bash
conda activate /home/you/conda_env/agentic-gpu-debugger
python -I -m pip install --no-deps -e .
python -I -m gpu_agent env --json
python -I -m gpu_agent diagnose benchmarks/public/case_0001/public_input
# Copy the printed run ID exactly:
python -I -m gpu_agent verify RUN_ID --generated-candidate --strict
python -I -m gpu_agent report RUN_ID
```

The previously recorded DeepSeek M1 run `ebce9bb96b9e45ad93b0fee942489ef7` belongs to commit
`8bd3cea`, not the current checkout. It used five physical model calls with `store=false` and
produced these full non-secret hashes:

- candidate: `b8d93e24288e21429491fcc8bf2fdbfdac48660bc709cd554fc45d28bb7a2010`
- patch: `1deb03011df3214d8f253cf3ba735fef971904d1043d40d4a14f445268f0bddc`
- input suite: `9b596c909f4469004a8efd05cdd8fc99761d6ca466c436f4f169c5500c5a8625`
- derived image: `sha256:cf603d6fefe333a7ec32ea5c93805658ab7fd402e88447fd16eff2c62d60bec3`
- CUDA base: `nvidia/cuda@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97`

That historical run is useful demonstration provenance, but the release gate correctly rejects
it for a different commit. There is currently no same-commit release run-report fixture; one may
only be generated after the 16+8 corpus, private alias map and cost-approved 360-unit batch pass.

Use the real candidate negative controls to demonstrate that a clean build is not enough:

```bash
python -I -m pytest tests/gpu/test_candidate_verification.py \
  -k 'zero or 257 or syntax or oob' --require-live -q
```

The controls retain the OOB, return wrong values, overfit length 257, or fail compilation. They
must produce `NOT_FIXED` or `REGRESSION_DETECTED`, never `VERIFIED_FIXED`.
