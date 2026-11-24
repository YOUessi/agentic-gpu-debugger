"""Frozen public AST microbenchmark: exactly one model edit per trial."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gpu_agent.harness.ast_microbench import run_ast_single_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("h001", "h002", "h003"), required=True)
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--allow-real-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_paid_calls:
        parser.error("explicit real-model call approval is required")
    repository = args.repository.resolve(strict=True)
    sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"], check=True,
        capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    if status:
        parser.error("single-step experiment requires a clean frozen source checkout")
    result = run_ast_single_step(
        repository=repository, output=args.output.absolute(), base_sha=sha,
        case_id=args.case, allow_paid_calls=True, allow_real_gpu=args.allow_real_gpu,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    # Failed edits are valid experimental observations, not lost trials.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
