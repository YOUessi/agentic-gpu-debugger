"""Read-only accounting for public targeted retests; JSON to stdout, no API calls."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from gpu_agent.agent.provider import Invocation
from gpu_agent.usage_accounting import AccountingRates, summarize_calls


def read_file(root: Path, relative: str) -> bytes:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("invalid artifact path")
    target = root
    for part in path.parts:
        target /= part
        if target.is_symlink():
            raise ValueError("symlink is not allowed")
    if not target.is_file():
        raise ValueError("missing input file")
    return target.read_bytes()


def report(root: Path, rates: AccountingRates | None) -> dict[str, object]:
    root = root.absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("symlink root is not allowed")
    raw = read_file(root, "results.jsonl")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    all_calls: list[Invocation] = []
    units = []
    for record in records:
        rid = record["run_id"]
        if not isinstance(rid, str) or not re.fullmatch("[a-f0-9]{32}", rid):
            raise ValueError("invalid run ID")
        manifest = json.loads(read_file(root, f"public/{rid}/manifest.json"))
        if manifest["id"] != rid or manifest["kind"] != "diagnosis":
            raise ValueError("not a diagnosis run")
        grouped: dict[str, list[Invocation]] = {}
        for ref in manifest["artifact_refs"]:
            if not ref["name"].startswith("provider/"):
                continue
            if ref["visibility"] != "public" or ref["run_id"] != rid:
                raise ValueError("not a public provider artifact")
            if not ref["relative_path"].startswith(rid + "/artifacts/"):
                raise ValueError("artifact outside diagnosis")
            data = read_file(root, "public/" + ref["relative_path"])
            if len(data) != ref["byte_count"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
                raise ValueError("artifact hash or size mismatch")
            item = Invocation.model_validate_json(data)
            if item.run_id != rid or ref["name"] != (
                f"provider/{item.invocation_id}/{item.state}.json"
            ):
                raise ValueError("provider identity mismatch")
            grouped.setdefault(item.invocation_id, []).append(item)
        calls = []
        for history in grouped.values():
            terminal = [c for c in history if c.state != "STARTED"]
            started = [c for c in history if c.state == "STARTED"]
            if len(terminal) > 1 or len(started) != 1:
                raise ValueError("ambiguous invocation history")
            calls.append(terminal[0] if terminal else started[0])
        if len(calls) != record["physical_calls"]:
            raise ValueError("call count mismatch")
        all_calls.extend(calls)
        units.append({"run_id": rid, "accounting": summarize_calls(calls, rates)})
    return {
        "results_sha256": hashlib.sha256(raw).hexdigest(),
        "units": units,
        "total": summarize_calls(all_calls, rates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--rates", type=Path)
    args = parser.parse_args()
    rates = AccountingRates.model_validate_json(args.rates.read_bytes()) if args.rates else None
    print(json.dumps(report(args.root, rates), indent=2))


if __name__ == "__main__":
    main()
