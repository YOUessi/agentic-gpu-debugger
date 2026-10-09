"""Predeclared 48-unit D/E public development comparison, no resampling or resume."""

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

from gpu_agent.agent.prompts import PROMPT_VERSION
from gpu_agent.agent.provider import DevelopmentCallPolicy, OpenAIProviderSettings
from gpu_agent.knowledge.retrieve import KnowledgeIndex, parse_version
from gpu_agent.provenance import capture_repository_snapshot, runtime_code_fingerprint
from gpu_agent.public_task import load_public_task
from gpu_agent.repair import RepairPolicy
from gpu_agent.service import ApplicationService

CASES = (
    "case_0001",
    "case_0003",
    "case_0009",
    "case_0016",
    "case_0017",
    "case_0018",
    "case_0019",
    "case_0020",
)


def schedule():
    units = [(case, mode, repeat) for case in CASES for mode in ("D", "E") for repeat in range(3)]
    random.Random(20260930).shuffle(units)
    return units


def save(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())


def preflight(repo, index, version, case_ids=CASES):
    parse_version(version)
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        if not index.retrieve(tool, version, 1).chunks:
            raise ValueError("compatible knowledge missing")
    cases = {}
    for name in ("corpus-registry.json", "diverse-registry.json"):
        for c in json.loads((repo / "benchmarks" / name).read_bytes())["cases"]:
            if c["case_id"] in cases:
                raise ValueError("duplicate case")
            cases[c["case_id"]] = c
    hashes = {}
    for case in case_ids:
        if not re.fullmatch(r"case_[0-9]{4}", case) or case not in cases:
            raise ValueError("unknown public case")
        hashes[case] = {}
        for file, key in (("kernel.cu", "mutant_source_hash"), ("input.json", "input_set_hash")):
            digest = hashlib.sha256(
                (repo / "benchmarks/public" / case / "public_input" / file).read_bytes()
            ).hexdigest()
            if digest != cases[case][key]:
                raise ValueError("registered input or source differs")
            hashes[case][file] = digest
        kernel = repo / "benchmarks/public" / case / "public_input/kernel.cu"
        if load_public_task(kernel, kernel.read_bytes()) is None:
            raise ValueError("public task required for functional self-check")
        hashes[case]["task.json"] = hashlib.sha256(
            kernel.with_name("task.json").read_bytes()
        ).hexdigest()
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--knowledge-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unit", action="append", help="Targeted regression only: case_id:mode:repeat")
    a = p.parse_args()
    repo, root = a.repository.resolve(), a.output.absolute()
    snapshot = capture_repository_snapshot(repo)
    runtime = runtime_code_fingerprint(repo)
    raw = a.knowledge_index.read_bytes()
    index = KnowledgeIndex.load(a.knowledge_index)
    version = os.environ["GPU_AGENT_KNOWLEDGE_VERSION"]
    units = schedule()
    if a.unit:
        units = []
        for value in a.unit:
            case, mode, repeat = value.split(":")
            unit_key = (case, mode, int(repeat))
            if (
                not re.fullmatch(r"case_[0-9]{4}", case)
                or mode not in {"D", "E"}
                or not 0 <= int(repeat) <= 2
                or unit_key in units
            ):
                raise ValueError("invalid or duplicate targeted unit")
            units.append(unit_key)
    hashes = preflight(repo, index, version, tuple(dict.fromkeys(c for c, _, _ in units)))
    settings = OpenAIProviderSettings.from_environment()
    if settings.timeout_seconds != 120 or not settings.api_key or not settings.model:
        raise ValueError("provider configuration does not match protocol")
    public_settings = settings.model_dump(mode="json", exclude={"api_key"})
    root.mkdir(parents=True, exist_ok=False)
    (root / "knowledge-index.json").write_bytes(raw)
    (root / "knowledge-index.json").chmod(0o400)
    os.environ["GPU_AGENT_KNOWLEDGE_INDEX"] = str(root / "knowledge-index.json")
    save(
        root / "selection.json",
        dict(
            purpose="selected-public-repair-validation"
            if a.unit
            else "development-repair-DE-not-release",
            commit=snapshot.commit,
            runtime_code_hash=runtime,
            prompt_version=PROMPT_VERSION,
            knowledge_hash=index.corpus_hash,
            knowledge_file_sha256=hashlib.sha256(raw).hexdigest(),
            inputs=hashes,
            units=units,
            provider=public_settings,
            max_candidates=3,
            max_llm_calls=40,
            max_task_seconds=600,
            cost_policy="record_only",
            attempts_per_unit=1,
        ),
    )
    with (root / "results.jsonl").open("x") as output:
        for ordinal, (case, mode, repeat) in enumerate(units):
            capture_repository_snapshot(repo, expected_commit=snapshot.commit)
            if runtime_code_fingerprint(repo) != runtime:
                raise ValueError("runtime changed")
            if (root / "knowledge-index.json").read_bytes() != raw:
                raise ValueError("knowledge changed")
            unit = root / f"unit-{ordinal:02d}"
            unit.mkdir()
            save(
                unit / "attempt.json",
                dict(case_id=case, mode=mode, repeat=repeat, started_at=time.time()),
            )
            os.environ["GPU_AGENT_RUN_ROOT"] = str(unit / "public")
            os.environ["GPU_AGENT_EVALUATOR_ROOT"] = str(unit / "verification")
            print(f"START {ordinal + 1}/{len(units)} {case} {mode} repeat={repeat}", flush=True)
            service = ApplicationService.configured()
            service.allow_development_paid_calls(DevelopmentCallPolicy(max_llm_calls=40))
            started = time.monotonic()
            try:
                run, verification = service.repair(
                    repo / "benchmarks/public" / case / "public_input",
                    mode=mode,
                    policy=RepairPolicy(max_candidates=3),
                )
            except Exception as exc:
                save(
                    unit / "interrupted.json",
                    dict(error_type=type(exc).__name__, automatic_retry=False),
                )
                raise  # preserve the started unit; never replay uncertain requests
            elapsed = time.monotonic() - started
            calls = service.provider_invocations(run.id)
            artifacts = {}
            for ref in run.artifact_refs:
                if ref.name in {"repair/summary.json", "agent/acquisition-usage.json"}:
                    artifacts[ref.name] = json.loads(service.store.read(ref))
            actions = [
                json.loads(service.store.read(r))["action_type"]
                for r in run.artifact_refs
                if r.name.startswith("actions/") and r.name.endswith("/decision.json")
            ]
            diagnosis = service.diagnosis(run.id)
            row = dict(
                ordinal=ordinal,
                case_id=case,
                mode=mode,
                repeat=repeat,
                run_id=run.id,
                diagnostic_outcome=diagnosis.diagnostic_outcome,
                limitations=diagnosis.limitations,
                verification=verification.model_dump(mode="json") if verification else None,
                artifacts=artifacts,
                proposed_actions=actions,
                latency_seconds=elapsed,
                diagnosis_and_selfcheck_seconds=(
                    run.events[-1].at - run.events[0].at
                ).total_seconds(),
                physical_calls=len(calls),
                cost_usd=None,
                known_total_tokens=sum(c.usage.total_tokens or 0 for c in calls if c.usage),
                unknown_usage_calls=sum(
                    c.usage is None or c.usage.total_tokens is None for c in calls
                ),
                calls=[
                    dict(
                        kind=c.kind,
                        state=c.state,
                        error_code=c.error_code,
                        elapsed_ms=c.elapsed_ms,
                        invocation_id=c.invocation_id,
                        usage=c.usage.model_dump(mode="json") if c.usage else None,
                    )
                    for c in calls
                ],
            )
            save(unit / "result.json", row)
            output.write(json.dumps(row) + "\n")
            output.flush()
            os.fsync(output.fileno())
            reason = artifacts.get("repair/summary.json", {}).get("stop_reason", "NO_CANDIDATE")
            outcome = verification.verdict if verification else reason
            print(
                f"DONE {ordinal + 1}/{len(units)} {outcome} calls={len(calls)}",
                flush=True,
            )
    save(root / "completed.json", dict(completed_units=len(units), commit=snapshot.commit))


if __name__ == "__main__":
    main()
