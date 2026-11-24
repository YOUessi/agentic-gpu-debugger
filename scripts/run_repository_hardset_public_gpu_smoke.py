"""Opt-in *real GPU* smoke for registered h001 repository harness.

Six model-directed tool actions are SCRIPTED for connectivity testing. This is
NOT an autonomous LLM success or an independent hidden verification. The
controller still performs actual isolated CUDA builds and public checks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gpu_agent.harness.models import HarnessDecision, SessionSnapshot
from gpu_agent.harness.repo_runner import run_repository_coding_session

KERNEL = "benchmarks/hardset_v2/h001/mutant/kernel.cu"


class ScriptedPublicCudaAgent:
    def __init__(self) -> None:
        self.actions: list[dict[str, str | int]] = [
            {"tool": "repo_read_file", "path": KERNEL, "lines": 100},
            {
                "tool": "repo_replace_snippet",
                "path": KERNEL,
                "old_text": "expf(own) / denominator",
                "new_text": "expf(own - maximum) / denominator",
            },
            {"tool": "repo_run_public_checks"},
            {
                "tool": "repo_replace_snippet",
                "path": KERNEL,
                "old_text": "denominator += expf(x);",
                "new_text": "denominator += expf(x - maximum);",
            },
            {"tool": "repo_run_public_checks"},
            {
                "tool": "finish",
                "summary": "Exact registered h001 candidate passed public CUDA checks only",
            },
        ]
        self.snapshots: list[SessionSnapshot] = []

    def choose_harness_v1(self, snapshot: SessionSnapshot) -> HarnessDecision:
        if snapshot.registered_public_case != "h001":
            raise ValueError("this scripted smoke only supports registered h001")
        if not self.actions:
            raise ValueError("scripted public GPU smoke exhausted its actions")
        self.snapshots.append(snapshot)
        action = self.actions.pop(0)
        if action["tool"] not in snapshot.allowed_tools:
            raise ValueError("smoke action is not approved by the operator policy")
        return HarnessDecision.model_validate(
            {"action": action, "reason": "Scripted GPU connectivity acceptance, not an LLM"}
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-real-gpu",
        action="store_true",
        help="Explicitly authorize real Docker CUDA public checks (no paid model calls)",
    )
    options = parser.parse_args()
    if not options.allow_real_gpu:
        parser.error("pass --allow-real-gpu to authorize this real GPU smoke")
    repository = options.repository.resolve(strict=True)
    output = options.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error("output must be a new directory to preserve old evidence")
    base_sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    ).stdout.strip()
    model = ScriptedPublicCudaAgent()
    outcome = run_repository_coding_session(
        repository=repository,
        output=output,
        base_sha=base_sha,
        goal="Repair registered h001 softmax numerical instability with public GPU feedback",
        max_steps=8,
        max_model_calls=10,
        max_public_checks=2,
        public_case_id="h001",
        model_factory=lambda _: model,
    )
    # Preserve the *actual* complete result under output/result.json via runner.
    # This script is only a read-only summary; never relabel it VERIFIED_FIXED.
    observations = [
        snapshot.history[-1]
        for snapshot in model.snapshots
        if snapshot.history and snapshot.history[-1]["tool"] == "repo_run_public_checks"
    ]
    errors = [json.loads(entry["content"])["failure_code"] for entry in observations]
    summary = {
        "base_sha": base_sha,
        "status": outcome.status,
        "tool_calls": outcome.tool_calls,
        "model_physical_calls": outcome.model_calls,
        "public_checks_attempted": outcome.public_checks_attempted,
        "public_checks_passed": outcome.public_checks_passed,
        "public_failure_sequence": errors,
        "public_run_id": outcome.public_run_id,
        "independent_hidden_verification": False,
        "scripted_model_only": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return (
        0
        if (
            outcome.status == "DIFF_STAGED_UNVERIFIED"
            and outcome.public_checks_attempted == 2
            and outcome.public_checks_passed
            and outcome.model_calls == 0
            and errors == ["NUMERIC_MISMATCH", "PUBLIC_CHECKS_PASSED"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
