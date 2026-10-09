# Contributing

Contributions should preserve the project's fail-closed evidence model. Do not
weaken source, toolchain, corpus, cost, or private-evaluator bindings merely to
make a test pass.

## Development setup

Use Python 3.11 or 3.12 in the dedicated environment described in `README.md`:

```bash
python -I -m pip install -r requirements.lock
python -I -m pip install --no-deps -e '.[dev]'
```

Run the zero-cost checks before submitting a change:

```bash
python -I -m ruff check src tests
python -I -m ruff format --check src tests
python -I -m mypy --strict src/gpu_agent
python -I -m pytest -q -m 'not gpu and not container and not live_llm and not release'
```

GPU, container, live-model, paid-evaluation, and release evidence must be run
only by an authorized operator in the documented environment. A skipped live
test is not acceptance evidence.

## Pull requests

- Keep changes focused and explain affected trust boundaries.
- Add regression tests for behavior changes.
- Never commit credentials, evaluator data, private corpus contents, run stores,
  model transcripts containing secrets, or generated evidence archives.
- Record user-visible changes in `CHANGELOG.md`.
- Sign off third-party code and retain its license and provenance.
