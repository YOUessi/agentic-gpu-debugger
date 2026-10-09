#!/usr/bin/env bash
# Operator helper for the uploaded-source integration. No family provisioning,
# GPU installation, driver changes, implicit registration, or model calls.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
help_message() {
  cat <<'HELP'
GPU seed batch helper (run from a NEWLY extracted source directory)

  bash gpu_batch.sh setup                 Check Python, install this checkout,
                                         initialize a LOCAL snapshot if no Git exists
  bash gpu_batch.sh preflight             Static four-seed source/hash checks only
  bash gpu_batch.sh run [case_0001]        Real isolated GPU validation, no registration
  bash gpu_batch.sh all                   Validate all four public seeds, no registration
  bash gpu_batch.sh report BATCH_ID       Show the persisted public batch report
  bash gpu_batch.sh export BATCH_ID FILE  Export only the summary-listed public runs

Environment:
  GPU_BATCH_PYTHON              Python executable in your existing 3.11/3.12 environment
                               (default: python)
  GPU_AGENT_CORPUS_FAMILY_ROOT  Existing trusted family/controller directory
  GPU_BATCH_DATA_ROOT          Optional; defaults to the configured public-store parent

setup never creates a corpus family/ledger and never modifies an existing Git history.
preflight can run before family configuration; it is NOT GPU acceptance.
run/all require an EXISTING trusted family, Docker GPU backend and valid toolchain.
Use the Chinese runbook before invoking explicit --register via the Python CLI.
HELP
}
case "$ACTION" in
  help|-h|--help) help_message; exit 0 ;;
  setup|preflight|run|all|report|export) ;;
  *) echo "Unknown action: $ACTION" >&2; help_message >&2; exit 2 ;;
esac
PY="${GPU_BATCH_PYTHON:-python}"
"$PY" - <<'PYTHON'
import sys
if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit("Use the project's existing Python 3.11 or 3.12 environment; no version guard is bypassed.")
PYTHON
cd -- "$ROOT"
if [[ "$ACTION" == setup ]]; then
  if [[ $# -ne 0 ]]; then echo 'setup takes no arguments.' >&2; exit 2; fi
  for git_var in GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_COMMON_DIR; do
    if [[ -v "$git_var" ]]; then
      echo "Refusing setup while $git_var is set; clear Git path overrides first." >&2
      exit 2
    fi
  done
  # Never commit into a containing repository by accident.
  if [[ ! -e .git ]]; then
    if git rev-parse --show-toplevel >/dev/null 2>&1; then
      echo 'This directory is nested inside another repository. Extract outside it.' >&2
      exit 2
    fi
    git init --quiet
    git -c core.hooksPath=/dev/null add --all
    git -c core.hooksPath=/dev/null -c commit.gpgsign=false \
        -c user.name='Local Source Snapshot' -c user.email='snapshot@localhost' \
        commit --quiet -m 'snapshot: public GPU seed batch source'
    echo 'Created a LOCAL source snapshot; no remote history was claimed or pushed.'
  elif [[ "$(git rev-parse --show-toplevel)" != "$ROOT" ]]; then
    echo 'The source directory is not the Git repository root.' >&2; exit 2
  fi
  if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    echo 'Worktree is not clean. Review and commit your changes manually; no reset performed.' >&2
    exit 2
  fi
  # Dependencies must already be present in the original development environment.
  "$PY" -m pip install --no-deps --no-build-isolation -e .
  "$PY" -c 'import gpu_agent; print("Loaded checkout:", gpu_agent.__file__)'
  echo 'Setup complete. Next: bash gpu_batch.sh preflight'
  exit 0
fi
case "$ACTION" in
  run|all)
    if [[ -z "${GPU_AGENT_CORPUS_FAMILY_ROOT:-}" ]]; then
      echo 'FAMILY_CONFIG_REQUIRED: export the existing trusted family path; no family is created automatically.' >&2
      exit 2
    fi ;;
esac
if [[ -n "${GPU_BATCH_DATA_ROOT:-}" ]]; then
  DATA_ROOT="$GPU_BATCH_DATA_ROOT"
elif [[ -n "${GPU_AGENT_CORPUS_FAMILY_ROOT:-}" ]]; then
  DATA_ROOT="$("$PY" - <<'PYTHON'
import os
from pathlib import Path
import json
from gpu_agent.benchmark.batch_security import public_store_path
from gpu_agent.store import read_regular, reject_symlinks
root = Path(os.environ['GPU_AGENT_CORPUS_FAMILY_ROOT']).absolute()
reject_symlinks(root)
# Read existing configuration only. The execution entry point validates the family.
config = json.loads(read_regular(root / 'family.json', 65536))
public = Path(config['public_store']).absolute()
if not public.is_dir():
    raise SystemExit('Existing configured public store is unavailable.')
if public_store_path(public.parent) != public:
    raise SystemExit("Configured public store layout is inconsistent.")
print(public.parent)
PYTHON
)"
else
  # Read-only preflight/report default; actual execution already requires a family.
  DATA_ROOT="$HOME/gpu-agent-experiments/seed-batch"
fi
case "$ACTION" in
  preflight)
    if [[ $# -ne 0 ]]; then echo 'preflight takes no arguments.' >&2; exit 2; fi
    exec "$PY" -m gpu_agent benchmark run-seeds --repository "$ROOT" --data-root "$DATA_ROOT" --preflight-only ;;
  run)
    if [[ $# -gt 1 ]]; then echo 'run accepts at most one case ID.' >&2; exit 2; fi
    exec "$PY" -m gpu_agent benchmark run-seeds --repository "$ROOT" --data-root "$DATA_ROOT" --case "${1:-case_0001}" ;;
  all)
    if [[ $# -ne 0 ]]; then echo 'all takes no arguments.' >&2; exit 2; fi
    exec "$PY" -m gpu_agent benchmark run-seeds --repository "$ROOT" --data-root "$DATA_ROOT" ;;
  report)
    if [[ $# -ne 1 ]]; then echo 'report requires BATCH_ID.' >&2; exit 2; fi
    exec "$PY" -m gpu_agent benchmark batch-report "$1" --data-root "$DATA_ROOT" ;;
  export)
    if [[ $# -ne 2 ]]; then echo 'export requires BATCH_ID FILE.zip.' >&2; exit 2; fi
    exec "$PY" -m gpu_agent benchmark export-batch "$1" --data-root "$DATA_ROOT" --output "$2" ;;
esac
