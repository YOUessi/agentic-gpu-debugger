"""One fresh attempt per previously failed public unit; never a release evaluation.

No success filtering after execution, automatic reruns, dollar cap, or balance query.
The output directory must be new. Provider credentials come from the environment.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from gpu_agent.agent.prompts import PROMPT_VERSION
from gpu_agent.agent.provider import DevelopmentCallPolicy
from gpu_agent.execution.models import SanitizerTool
from gpu_agent.knowledge.retrieve import KnowledgeIndex
from gpu_agent.provenance import capture_repository_snapshot, runtime_code_fingerprint
from gpu_agent.service import ApplicationService


def save(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-results", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--knowledge-index", type=Path, required=True)
    parser.add_argument("--expected-corpus-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-mode", choices=("standard", "full"), default="full")
    args = parser.parse_args()
    repo = args.repository.resolve()
    snapshot = capture_repository_snapshot(repo)
    runtime_hash = runtime_code_fingerprint(repo)
    prior_raw = args.previous_results.read_bytes()
    prior = [json.loads(line) for line in prior_raw.splitlines() if line.strip()]
    failed = [
        row for row in prior if (row.get("verification") or {}).get("verdict") != "VERIFIED_FIXED"
    ]
    keys = [(r["case_id"], r["mode"], r["repeat"]) for r in failed]
    if not failed or len(keys) != len(set(keys)):
        raise ValueError("failed-unit selection is empty or ambiguous")
    index = KnowledgeIndex.load(args.knowledge_index)
    if index.corpus_hash != args.expected_corpus_hash:
        raise ValueError("knowledge corpus differs from frozen selection")
    index_bytes = args.knowledge_index.read_bytes()
    cases = {
        c["case_id"]: c
        for c in json.loads((repo / "benchmarks/corpus-registry.json").read_bytes())["cases"]
    }
    for row in failed:
        if row["case_id"] not in cases or row["mode"] not in {"A", "B", "C", "D", "E"}:
            raise ValueError("selection is not a known public unit")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "knowledge-index.json").write_bytes(index_bytes)
    (root / "knowledge-index.json").chmod(0o400)
    os.environ["GPU_AGENT_RUN_ROOT"] = str(root / "public")
    os.environ["GPU_AGENT_EVALUATOR_ROOT"] = str(root / "evaluator")
    os.environ["GPU_AGENT_KNOWLEDGE_INDEX"] = str(root / "knowledge-index.json")
    save(
        root / "selection.json",
        dict(
            purpose="selected-failure-development-regression-not-release-evidence",
            commit=snapshot.commit,
            runtime_code_hash=runtime_hash,
            prompt_version=PROMPT_VERSION,
            previous_results_sha256=hashlib.sha256(prior_raw).hexdigest(),
            knowledge_corpus_hash=index.corpus_hash,
            knowledge_file_sha256=hashlib.sha256(index_bytes).hexdigest(),
            cost_policy="record_only",
            units=keys,
            attempts_per_unit=1,
            verification_mode=args.verification_mode,
        ),
    )
    with (root / "results.jsonl").open("x") as stream:
        for ordinal, prior_row in enumerate(failed):
            capture_repository_snapshot(repo, expected_commit=snapshot.commit)
            if runtime_code_fingerprint(repo) != runtime_hash:
                raise ValueError("executing source changed")
            if (root / "knowledge-index.json").read_bytes() != index_bytes:
                raise ValueError("knowledge cache changed")
            case_id, mode, repeat = keys[ordinal]
            save(
                root / f"attempt-{ordinal:02d}.json",
                dict(
                    case_id=case_id,
                    mode=mode,
                    repeat=repeat,
                    previous_run_id=prior_row["run_id"],
                    started_at=time.time(),
                ),
            )
            print(f"START {ordinal + 1}/{len(failed)} {case_id} {mode} repeat={repeat}", flush=True)
            started = time.monotonic()
            service = ApplicationService.configured()
            service.allow_development_paid_calls(DevelopmentCallPolicy(max_llm_calls=40))
            case = cases[case_id]
            tools = tuple(
                dict.fromkeys((SanitizerTool.MEMCHECK, SanitizerTool(case["target_tool"])))
            )
            run = service.diagnose(
                repo / "benchmarks/public" / case_id / "public_input",
                mode=mode,
                required_tools=tools,
                expected_source_hash=case["mutant_source_hash"],
            )
            save(root / f"diagnosis-{ordinal:02d}.json", dict(run_id=run.id))
            diagnosis = service.diagnosis(run.id)
            candidates = service.candidates(run.id)
            verification, verification_id = None, None
            if candidates:
                print(f"GPU VERIFY run={run.id}", flush=True)
                verification, verification_id = service.verify_exact(
                    run.id, strict=args.verification_mode == "full"
                )
            calls = service.provider_invocations(run.id)
            row = dict(
                case_id=case_id,
                mode=mode,
                repeat=repeat,
                run_id=run.id,
                previous_run_id=prior_row["run_id"],
                diagnostic_outcome=diagnosis.diagnostic_outcome,
                limitations=diagnosis.limitations,
                candidate_ids=candidates,
                verification=verification.model_dump(mode="json") if verification else None,
                verification_run_id=verification_id,
                physical_calls=len(calls),
                known_total_tokens=sum(c.usage.total_tokens or 0 for c in calls if c.usage),
                unknown_usage_calls=sum(c.usage is None for c in calls),
                cost_usd=None,
                calls=[
                    dict(
                        kind=c.kind,
                        attempt=c.attempt,
                        state=c.state,
                        error_code=c.error_code,
                        diagnostics=c.output_diagnostics.model_dump(mode="json")
                        if c.output_diagnostics
                        else None,
                    )
                    for c in calls
                ],
                latency_seconds=time.monotonic() - started,
            )
            stream.write(json.dumps(row, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(
                f"DONE verdict={verification.verdict if verification else 'NO_CANDIDATE'} "
                f"calls={len(calls)}",
                flush=True,
            )
    save(root / "completed.json", dict(completed_units=len(failed), commit=snapshot.commit))


if __name__ == "__main__":
    main()
