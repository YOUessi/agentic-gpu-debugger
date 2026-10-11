"""Merge independently exported *frozen public* repair memory indexes."""

import argparse
from pathlib import Path

from gpu_agent.repair_memory import FrozenRepairMemory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse to overwrite an existing frozen memory index")
    indexes = [FrozenRepairMemory.load(path) for path in args.index]
    merged = FrozenRepairMemory.combine(indexes)
    merged.save(args.output)
    print(f"MEMORY_MERGED count={len(merged.records)} hash={merged.corpus_sha256}")


if __name__ == "__main__":
    main()
