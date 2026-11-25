# Real CPU embedding validation — 2026-11-25

## Result

Real pretrained CPU embeddings were exercised through the project's actual `LocalSentenceEmbedder` and `SQLiteRepairStore`, at source commit `bf9e224d14e7a9382e2f40e450c0e35cda41e670`.

- Functional checks: **18/18 pass**.
- Fixed controlled retrieval sample: **4/5 semantic top-1 hits**, compared with **0/5 BM25-only top-1 hits** on exactly the same texts and queries.
- All five queries have zero lexical scores against every fixture. BM25 search therefore returns no items; its expected-document rank is null in every case.
- One compile-query miss is retained. The fixtures were not retuned or resampled after observing the result. This smoke demonstration is not a held-out benchmark, acceptance claim, or broad quality estimate.
- Two separate Python processes reproduced identical rankings and scores. Reloading the model and reopening the database also produced identical results.

| Expected item | Fixed query | Semantic top result | Hybrid score | Expected rank |
|---|---|---|---:|---:|
| compile | Correct malformed punctuation during translation | numeric | 0.0675423108 | absent |
| numeric | Normalize probabilities safely when exponential values explode | numeric | 0.1573775475 | 1 |
| ast | Locate updated syntax tree targets following a revision | ast | 0.1836079956 | 1 |
| infrastructure | Network outage blocks communication | infrastructure | 0.2714371605 | 1 |
| generic paraphrase control | doctor treats ill youngster | generic paraphrase control | 0.4777446818 | 1 |

Since lexical scores are zero, the displayed hybrid scores equal 0.65 times the nonnegative cosine similarity. The complete candidate scores, ranks, expected texts, and BM25 baseline are in the final JSON.

## Model and execution

- Official model: `sentence-transformers/paraphrase-MiniLM-L3-v2`.
- Fixed model revision: `4ca70771034acceecb2e72475f72050fcdde4ddc`.
- Model file: 69,569,488-byte safetensors weights, SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.
- The weight hash independently matches the official Hugging Face model page: https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/blob/main/model.safetensors .
- Direct Hugging Face access failed on Tang. The successful download used `hf-mirror.com`. Small files fetched via `main` were accepted only when `X-Repo-Commit` matched the pinned revision. `results/model-lock.json` locks every file by SHA-256 and size. All model files are regular files, with no symlinks, Python modules, or pickle weights.
- Installed in an isolated new venv: torch `2.5.1+cpu`, sentence-transformers `3.4.1`, transformers `4.48.3`. The source uses LangGraph `1.2.20`.
- Model device and every parameter were CPU. `CUDA_VISIBLE_DEVICES` was empty and `torch.version.cuda` was null.
- Embeddings have 384 finite dimensions and norm 1.0000000086449417. Repeated encoding had maximum absolute difference 0.0.
- Inference ran with offline flags and a socket connection blocker: zero connection attempts.

The functional checks also cover persisted 384-dimensional vectors, repository namespace isolation including a deliberately attractive item in another namespace, rejection of wrong-scope access, rejection of readonly writes and deletes, unchanged database bytes and mtime, model content identity, and removal of untrusted canary text from sanitized lessons while retaining the observed recovery episode.

## Reproduction on Tang

All installation and generated evidence are under this isolated directory. Existing project and runtime packages are read only; bytecode generation is disabled. The two runtime paths below match the existing CPU environment's package search path. They are appended after the isolated embedding environment, so torch and embedding dependencies come from the new venv.

```bash
EVAL_ROOT=/home/you/projects/agentic-gpu-graph-native-dev-20261125/embedding-validation
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "$EVAL_ROOT/venv/bin/python" \
  "$EVAL_ROOT/validate_real_embeddings.py" \
  --source /home/you/projects/agentic-gpu-graph-native-dev-20261125/source \
  --model "$EVAL_ROOT/model" \
  --runtime-site-packages /home/you/projects/agentic-gpu-graph-native-dev-20261125/cpu-venv/lib/python3.11/site-packages \
  --runtime-site-packages /home/you/gpu-native-harness-validation-20261123/qwen3-real-p0-59c159/venv/lib/python3.11/site-packages \
  --output "$EVAL_ROOT/results/reproduction.json"
```

`validate_real_embeddings.py` returns success for functional checks; retrieval quality is separately reported under `retrieval_metrics`. It does not treat 4/5 as perfect recall.

The small downloader can reconstruct or verify the model data from `results/model-lock.json`. Its default endpoint is the official Hugging Face site. The explicit `--endpoint https://hf-mirror.com` option selects the transport that worked on Tang; all locked hashes still apply. No custom remote code is executed.

`results/embedding-requirements.freeze.txt` records all packages installed in the isolated venv. Existing LangGraph and project dependencies were reused read only. No source checkout, existing virtualenv, frozen P8 material, GPU process, provider API, or service configuration was intentionally modified. No commit or push was performed.

## Evidence files

- `validate_real_embeddings.py`: fixed fixtures and real execution checks.
- `results/real-embedding-validation-final.json`: separated functional and retrieval results, with candidate scores and baseline ranks.
- `results/real-embedding-validation.json`: preserved original run, before separating the 18 functional checks from the one failed perfect-recall expectation.
- `results/cross-process-repeatability.json`: equality evidence and hashes for both runs.
- `results/model-lock.json` and `download_model.py`: content-locked model provenance and safe data downloader.
- `results/embedding-requirements.freeze.txt`: isolated dependency versions.

The local model and virtualenv binaries should not be committed. The script, small JSON evidence, lock, and this report are sufficient for review.
