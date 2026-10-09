"""One explicitly authorized public development experiment: force a known failed first patch.

Only the initial patch PROPOSAL is supplied deterministically; planning,
diagnosis, GPU tools and subsequent revision use the configured real LLM.
This is neither a normal autonomous run nor a model-success-rate benchmark.
Never add private evaluator data to model feedback or export its details.
"""

import argparse
import difflib
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from gpu_agent.agent.provider import (
    DevelopmentCallPolicy,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
)
from gpu_agent.provenance import capture_repository_snapshot, runtime_code_fingerprint
from gpu_agent.public_task import PublicTask
from gpu_agent.repair import RepairPolicy
from gpu_agent.service import ApplicationService


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_exclusive(path: Path, content: object) -> None:
    with path.open("x") as out:
        json.dump(content, out, indent=2, ensure_ascii=False)
        out.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--knowledge-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-paid-calls", action="store_true")
    args = parser.parse_args()
    if not args.allow_paid_calls:
        raise ValueError("explicit --allow-paid-calls required")
    if os.environ.get("GPU_AGENT_REPAIR_MEMORY_INDEX"):
        raise ValueError("forced-failure experiment requires memory disabled")

    repo = args.repository.resolve(strict=True)
    output = args.output.absolute()
    if output.exists():
        raise ValueError("refuse to overwrite any experiment directory")
    snapshot = capture_repository_snapshot(repo)
    code_hash = runtime_code_fingerprint(repo)
    public_source = repo / "benchmarks/public/case_0022/public_input"
    source = (public_source / "kernel.cu").read_text()
    task = PublicTask.model_validate_json((public_source / "task.json").read_bytes())
    assert task.source_sha256 == digest(source.encode())
    old = "        __syncthreads();\n        tile[lane] = value;"
    assert source.count(old) == 1
    incorrect = source.replace(old, "        tile[lane] = value;\n        __syncthreads();", 1)
    assert digest(incorrect.encode()) == (
        "d1fac7bc60f231f4bc041b71ef7b04ece7f13ba8794aa90f7a9cdd504ebecce5"
    ), "forced error no longer matches frozen V3 candidate"
    diff = "".join(
        difflib.unified_diff(
            source.splitlines(True),
            incorrect.splitlines(True),
            fromfile="a/kernel.cu",
            tofile="b/kernel.cu",
        )
    )
    settings = OpenAIProviderSettings.from_environment()
    assert settings.endpoint and urlsplit(settings.endpoint).hostname == "api.deepseek.com"
    assert settings.model == "deepseek-v4-pro" and settings.api_key is not None
    knowledge_file = args.knowledge_index.resolve(strict=True)
    knowledge_bytes = knowledge_file.read_bytes()

    output.mkdir(mode=0o700)
    index_path = output / "knowledge-index.json"
    index_path.write_bytes(knowledge_bytes)
    index_path.chmod(0o400)
    os.environ["GPU_AGENT_RUN_ROOT"] = str(output / "public")
    os.environ["GPU_AGENT_EVALUATOR_ROOT"] = str(output / "evaluator")
    os.environ["GPU_AGENT_KNOWLEDGE_INDEX"] = str(index_path)
    save_exclusive(
        output / "predeclared.json",
        {
            "kind": "case0022_forced_first_failure_real_model_public_development",
            "started_at": datetime.now(UTC).isoformat(),
            "code_commit": snapshot.commit,
            "runtime_code_hash": code_hash,
            "model": settings.model,
            "case": "case_0022",
            "source_sha256": task.source_sha256,
            "forced_first_kernel_sha256": digest(incorrect.encode()),
            "forced_first_diff_sha256": digest(diff.encode()),
            "injected_initial_patch_only": True,
            "initial_patch_is_not_an_LLM_response": True,
            "initial_patch_counts_as_zero_paid_calls": True,
            "subsequent_planning_diagnosis_and_revision_are_real_llm": True,
            "no_repair_memory": True,
            "knowledge_index_sha256": digest(knowledge_bytes),
            "max_candidates": 3,
            "max_reinvestigations": 1,
            "max_llm_calls": 40,
            "total_task_deadline_seconds": 600,
            "unbounded_independent_sanitizer_call_count": True,
            "no_automatic_retries": True,
        },
    )
    service = ApplicationService.configured()
    service.allow_development_paid_calls(DevelopmentCallPolicy(max_llm_calls=40))
    original = OpenAIResponsesProvider.propose_patch
    injections: list[str] = []

    def forced_initial_patch(
        provider: OpenAIResponsesProvider,
        current_source: object,
        diagnosis: object,
        *,
        experience_hints: list[dict[str, str]] | None = None,
    ) -> str:
        content = getattr(current_source, "content", None)
        if content != source or injections:
            raise ValueError("unexpected initial patch proposal input or repeated injection")
        if experience_hints:
            raise ValueError("unexpected repair memory hint")
        injections.append(digest(diff.encode()))
        return diff

    started = time.monotonic()
    try:
        # Explicit developer harness injection; never patch production code on disk.
        OpenAIResponsesProvider.propose_patch = forced_initial_patch
        try:
            run, verdict = service.repair(
                public_source,
                mode="E",
                policy=RepairPolicy(
                    version="public-repair-v3",
                    max_candidates=3,
                    max_reinvestigations=1,
                    unbounded_sanitizer_calls=True,
                ),
            )
        finally:
            OpenAIResponsesProvider.propose_patch = original

        artifacts = {
            ref.name: json.loads(service.store.read(ref))
            for ref in run.artifact_refs
            if ref.name in ("repair/summary.json", "agent/final-budget.json")
            or ref.name.endswith("/feedback.json")
        }
        summary = artifacts.get("repair/summary.json", {})
        invocations = service.provider_invocations(run.id)
        usage = {
            "actual_physical_model_calls": len(invocations),
            "all_model_calls_completed": all(i.state == "COMPLETED" for i in invocations),
            "total_tokens": sum(i.usage.total_tokens or 0 for i in invocations if i.usage),
            "calls_with_missing_usage": sum(
                i.usage is None or i.usage.total_tokens is None for i in invocations
            ),
            "kinds": [i.kind for i in invocations],
            "prompt_versions": sorted({i.prompt_version for i in invocations}),
            "cost_usd": None,
        }
        saved_feedback = [
            {
                "artifact_name": name,
                "revision_history": value.get("revision_history", []),
                "diagnosis_scoped_to_latest_candidate": value.get(
                    "diagnosis_scoped_to_latest_candidate"
                ),
            }
            for name, value in sorted(artifacts.items())
            if name.endswith("/feedback.json")
        ]
        if len(injections) != 1:
            raise ValueError("initial forced patch was not injected exactly once")
        row = {
            "run_id": run.id,
            "finished_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "forced_first_kernel_sha256": digest(incorrect.encode()),
            "forced_first_proposal_count": len(injections),
            "verdict": verdict.verdict if verdict is not None else "NOT_VERIFIED",
            "reason_code": verdict.reason_code if verdict is not None else None,
            "stop_reason": summary.get("stop_reason"),
            "candidate_rounds": len(summary.get("rounds", [])),
            "public_rounds": [
                {
                    "round": item["round"],
                    "candidate_hash": item.get("candidate_hash"),
                    "checks": item.get("check", {}).get("checks"),
                    "status": item.get("check", {}).get("status"),
                }
                for item in summary.get("rounds", [])
            ],
            "reinvestigations": summary.get("reinvestigations", 0),
            "acquisition_usage": summary.get("acquisition_usage"),
            "feedback": saved_feedback,
            "usage": usage,
            "historical_v3_comparison": "single development replay; not causal evidence",
        }
        save_exclusive(output / "result.json", row)
        print(
            "FORCED_REPLAY_RESULT="
            + json.dumps(
                {
                    k: row[k]
                    for k in (
                        "run_id",
                        "verdict",
                        "stop_reason",
                        "candidate_rounds",
                        "reinvestigations",
                        "usage",
                    )
                }
            ),
            flush=True,
        )
        return 0 if row["verdict"] == "VERIFIED_FIXED" else 1
    except BaseException as exc:
        # Never print an upstream response body, secret, or private evaluator data.
        save_exclusive(
            output / "interrupted.json",
            {
                "exception_class": type(exc).__name__,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "forced_initial_patch_count": len(injections),
                "no_retry": True,
            },
        )
        print("FORCED_REPLAY_INTERRUPTED=" + type(exc).__name__, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
