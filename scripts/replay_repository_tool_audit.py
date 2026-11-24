"""Replay a registered AST run's public actions and exact visible observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_agent.harness.repo_audit import read_repository_tool_events
from gpu_agent.store import RunStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-store", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tool-count", type=int, required=True)
    args = parser.parse_args()
    records = read_repository_tool_events(
        RunStore(args.public_store),
        args.run_id,
        expected_count=args.tool_count,
        require_action=True,
        require_observation=True,
    )
    print(json.dumps(records, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
