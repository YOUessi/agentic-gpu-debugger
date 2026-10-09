"""One predeclared D/E development comparison; no retries or dollar limits.

Not a release evaluation or an estimate of broad generalization. Run from a clean
independent snapshot with its own Python environment and a new output directory.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

from gpu_agent.agent.prompts import PROMPT_VERSION
from gpu_agent.agent.provider import DevelopmentCallPolicy
from gpu_agent.knowledge.retrieve import KnowledgeIndex, parse_version
from gpu_agent.provenance import capture_repository_snapshot, runtime_code_fingerprint
from gpu_agent.service import ApplicationService


def schedule():
    units = [
        (case, mode)
        for case in ("case_0001", "case_0003", "case_0009", "case_0016")
        for mode in ("D", "E")
    ]
    random.Random(20260930).shuffle(units)
    return units


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())


def validate_knowledge(index, version):
    # This parameter is a toolchain compatibility selector, not a corpus hash.
    parse_version(version)
    for query in ("memcheck", "initcheck", "racecheck", "synccheck"):
        if not index.retrieve(query, version, 1).chunks:
            raise ValueError("knowledge preflight returned no compatible documentation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--knowledge-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repository.resolve()
    snapshot = capture_repository_snapshot(repo)
    runtime_hash = runtime_code_fingerprint(repo)
    index_raw = args.knowledge_index.read_bytes()
    index = KnowledgeIndex.load(args.knowledge_index)
    toolchain_version = os.environ.get("GPU_AGENT_KNOWLEDGE_VERSION", "")
    validate_knowledge(index, toolchain_version)
    cases = {
        c["case_id"]: c
        for c in json.loads((repo / "benchmarks/corpus-registry.json").read_bytes())["cases"]
    }
    units = schedule()
    for case_id, _ in units:
        for name, field in (("kernel.cu", "mutant_source_hash"), ("input.json", "input_set_hash")):
            content = (repo / "benchmarks/public" / case_id / "public_input" / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != cases[case_id][field]:
                raise ValueError("source or input differs from registry")
    root = args.output.absolute()
    root.mkdir(parents=True, exist_ok=False)
    (root / "knowledge-index.json").write_bytes(index_raw)
    (root / "knowledge-index.json").chmod(0o400)
    os.environ["GPU_AGENT_RUN_ROOT"] = str(root / "public")
    os.environ["GPU_AGENT_EVALUATOR_ROOT"] = str(root / "verification")
    os.environ["GPU_AGENT_KNOWLEDGE_INDEX"] = str(root / "knowledge-index.json")
    os.environ["GPU_AGENT_KNOWLEDGE_VERSION"] = toolchain_version
    save(
        root / "selection.json",
        dict(
            purpose="predeclared-de-investigation-development-check-not-release",
            commit=snapshot.commit,
            runtime_code_hash=runtime_hash,
            prompt_version=PROMPT_VERSION,
            knowledge_corpus_hash=index.corpus_hash,
            knowledge_toolchain_selector=toolchain_version,
            knowledge_file_sha256=hashlib.sha256(index_raw).hexdigest(),
            units=units,
            attempts_per_unit=1,
            verification_mode="full",
            max_llm_calls=40,
            cost_policy="record_only_no_price_assumption",
            configured_model=os.environ.get("OPENAI_MODEL"),
            request_timeout_seconds=os.environ.get("GPU_AGENT_LLM_TIMEOUT_SECONDS", "60"),
        ),
    )
    with (root / "results.jsonl").open("x") as stream:
        for ordinal, (case_id, mode) in enumerate(units):
            capture_repository_snapshot(repo, expected_commit=snapshot.commit)
            if runtime_code_fingerprint(repo) != runtime_hash:
                raise ValueError("executing code changed")
            if (root / "knowledge-index.json").read_bytes() != index_raw:
                raise ValueError("knowledge index changed")
            save(
                root / f"attempt-{ordinal:02d}.json",
                dict(case_id=case_id, mode=mode, started_at=time.time()),
            )
            print(f"START {ordinal + 1}/8 {case_id} {mode}", flush=True)
            service = ApplicationService.configured()
            service.allow_development_paid_calls(DevelopmentCallPolicy(max_llm_calls=40))
            started = time.monotonic()
            run = service.diagnose(
                repo / "benchmarks/public" / case_id / "public_input",
                mode=mode,
                expected_source_hash=cases[case_id]["mutant_source_hash"],
                expected_input_hash=cases[case_id]["input_set_hash"],
            )
            save(root / f"diagnosis-{ordinal:02d}.json", dict(run_id=run.id))
            result = service.diagnosis(run.id)
            candidates = service.candidates(run.id)
            verified, verification_id = None, None
            if candidates:
                print(f"VERIFY {case_id} {mode} {run.id}", flush=True)
                verified, verification_id = service.verify_exact(run.id, strict=True)
            calls = service.provider_invocations(run.id)
            actions = []
            for ref in service.store.load(run.id).artifact_refs:
                if ref.name.startswith("actions/") and ref.name.endswith("/decision.json"):
                    value = json.loads(service.store.read(ref))
                    actions.append(
                        {k: value[k] for k in ("action_type", "allowed", "reason_codes")}
                    )
            row = dict(
                case_id=case_id,
                mode=mode,
                repeat=0,
                run_id=run.id,
                diagnostic_outcome=result.diagnostic_outcome,
                limitations=result.limitations,
                candidates=candidates,
                verification=verified.model_dump(mode="json") if verified else None,
                verification_run_id=verification_id,
                physical_calls=len(calls),
                known_total_tokens=sum(c.usage.total_tokens or 0 for c in calls if c.usage),
                unknown_usage_calls=sum(c.usage is None for c in calls),
                cost_usd=None,
                calls=[
                    dict(
                        kind=c.kind,
                        state=c.state,
                        error_code=c.error_code,
                        invocation_id=c.invocation_id,
                        usage=c.usage.model_dump(mode="json") if c.usage else None,
                    )
                    for c in calls
                ],
                actions=actions,
                latency_seconds=time.monotonic() - started,
            )
            stream.write(json.dumps(row, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(
                f"DONE {case_id} {mode} {verified.verdict if verified else 'NO_CANDIDATE'} "
                f"calls={len(calls)}",
                flush=True,
            )
    save(root / "completed.json", dict(completed_units=len(units), commit=snapshot.commit))


if __name__ == "__main__":
    main()
