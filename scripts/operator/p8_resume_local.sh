#!/usr/bin/env bash
# Operator-only launcher. Run from the authorized Tang terminal.
# It does not change the frozen controller or resume a timed-out trial.
set -euo pipefail
umask 077

# The successful frozen remote runs used a proxy-free process environment.
# A desktop terminal may inherit a SOCKS proxy URL unsupported by HTTPX.
# Direct connectivity to the official DeepSeek endpoint was verified with
# an unauthenticated HTTP 401; no inference or credentials were involved.
# This changes only this launcher's child environment, never host settings.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
echo "Benchmark process network: direct; loopback bypass enabled."

mode="${1:---check}"
if [[ $# -gt 1 || ( "$mode" != --check && "$mode" != --run ) ]]; then
  echo "Usage: $0 [--check|--run]" >&2
  exit 64
fi

root="$HOME/gpu-native-harness-validation-20261124"
controller="$root/p8-formal-benchmark-controller"
python="$HOME/gpu-native-harness-validation-20261123/qwen3-real-p0-59c159/venv/bin/python"
repository="$root/p4-strict-and-binding/binding-source"
manifest="$controller/docs/benchmark/2026-11-25-formal-benchmark-manifest.json"
amendment="$controller/docs/benchmark/2026-11-25-p8-timeout-amendment.json"
output="$root/p8-formal-benchmark-results"
infra="$root/p8-infra-v2"
serving_python="$HOME/conda_env/llm-opt-serving-benchmark-v2/bin/python"
model="$HOME/models/Qwen3-8B-AWQ"
lock="$HOME/.cache/llm_opt_shared_gpu.lock"
environment="$infra/isolated-environment-extras.json"
paths="$HOME/.config/agentic-gpu-debugger/p8-operator/local-paths"

[[ -f "$paths" && -O "$paths" && ! -L "$paths" ]]
[[ "$(stat -c '%a' "$paths")" == 600 ]]
mapfile -t local_paths < "$paths"
[[ ${#local_paths[@]} -eq 2 ]]
provider_config="${local_paths[0]}"
evaluator_config="${local_paths[1]}"
[[ -n "$provider_config" && -r "$provider_config" && -f "$provider_config" ]]
[[ -n "$evaluator_config" && -r "$evaluator_config" && -f "$evaluator_config" ]]
[[ -x "$python" && -x "$serving_python" && -d "$model" ]]
[[ -d "$output" && -d "$infra" && -r "$environment" && -f "$lock" ]]
[[ "$(git -C "$controller" rev-parse HEAD)" == 8724f84875bf49c4c23e518d384ca23e3d52a45b ]]
git -C "$controller" diff --quiet
git -C "$controller" diff --cached --quiet
[[ "$(git -C "$repository" rev-parse HEAD)" == be3dafd6988972dfdb51cf2faee6984103565973 ]]
printf '%s  %s\n' 47d38ec2b9c363dd4acb29074f46aad6dd9913dac2bc35bc10bcd0d3ba050610 "$manifest" | sha256sum -c -
printf '%s  %s\n' 75f244c4608f8b11908c2c533401114ca81b54fe21540d2eba0396963d2514f9 "$amendment" | sha256sum -c -

# Only read the progress snapshot. Actual admission, evidence verification,
# GPU locking and per-block health gates remain the controller's responsibility.
"$python" - "$manifest" "$output/scheduler-v2/progress.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as file:
    manifest = json.load(file)
with open(sys.argv[2]) as file:
    progress = json.load(file)
states = progress["trial_states"]
budget = manifest["budget"]
assert budget["max_tool_steps"] == 16
assert budget["max_model_calls"] == 18
assert budget["max_output_tokens"] == 8192
assert budget["agent_watchdog_wall_seconds"] == 600
assert budget["strict_watchdog_wall_seconds"] == 600
assert states["r03-qwen-R-B-h001"]["state"] == "OUTCOME_UNKNOWN"
next_pending = next((key for key, value in states.items()
                     if value["state"] == "PENDING"), None)
print("States:", progress["states"])
print("Next pending:", next_pending)
print("Pinned budget: 600 s / 16 steps / 18 calls / 8192 output tokens")
PY

if [[ "$mode" == --check ]]; then
  echo "READ-ONLY LAUNCH CHECK PASSED; no inference or GPU evaluations started."
  exit 0
fi

# Explicit human initiation at an authorized interactive terminal.
if [[ ! -t 0 || ! -t 1 ]]; then
  echo "Run requires an authorized interactive operator terminal." >&2
  exit 64
fi
read -r -p 'Start the original pending benchmark trials? Type RUN: ' answer
[[ "$answer" == RUN ]] || { echo "Cancelled."; exit 0; }
log="$(mktemp "$infra/resume-v3-local-XXXXXXXX.log")"
echo "Controller log: $log"
cd "$controller"
"$python" scripts/run_frozen_benchmark_resume.py \
  --manifest "$manifest" \
  --repository "$repository" \
  --evidence-root "$root" \
  --output "$output" \
  --provider-config "$provider_config" \
  --evaluator-secret-file "$evaluator_config" \
  --qwen-model "$model" \
  --serving-python "$serving_python" \
  --gpu-lock "$lock" \
  --environment-report "$environment" \
  --infra-output "$infra" \
  --execute --allow-paid-calls --allow-real-gpu > "$log" 2>&1
