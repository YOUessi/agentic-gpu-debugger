"""Export an explicitly selected and frozen public-only repair experience index."""

import argparse
from pathlib import Path

from gpu_agent.repair_memory import FrozenRepairMemory
from gpu_agent.store import RunStore


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public-store", type=Path, required=True)
    p.add_argument("--run-id", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("refuse to overwrite a frozen memory index")
    index = FrozenRepairMemory.from_public_runs(
        RunStore(args.public_store), args.run_id
    )
    index.save(args.output)
    print(f"MEMORY_EXPORTED count={len(index.records)} hash={index.corpus_sha256}")


if __name__ == "__main__":
    main()
