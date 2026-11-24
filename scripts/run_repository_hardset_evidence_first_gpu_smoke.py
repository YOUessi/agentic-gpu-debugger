"""Opt-in real GPU smoke for the evidence-first + atomic batch decision surface.

A DETERMINISTIC TEST MODEL chooses three typed actions from public observations:
one atomic Softmax repair -> real isolated GPU check -> finish. This proves
capability connectivity, NOT autonomous LLM improvement or reward learning.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gpu_agent.harness.metrics import summarize_repository_trajectory
from gpu_agent.harness.models import HarnessDecision, SessionSnapshot
from gpu_agent.harness.repo_runner import run_repository_coding_session

KERNEL = "benchmarks/hardset_v2/h001/mutant/kernel.cu"


class EvidenceAwareScriptedDecision:
    def __init__(self) -> None:
        self.snapshots: list[SessionSnapshot] = []

    def choose_harness_v1(self, snapshot: SessionSnapshot) -> HarnessDecision:
        self.snapshots.append(snapshot)
        if snapshot.registered_public_case != "h001" or snapshot.context_mode != "evidence_first":
            raise ValueError("smoke requires the registered evidence-first h001 context")
        if snapshot.public_checks_passed:
            action: dict[str, object] = {
                "tool": "finish",
                "summary": "Registered h001 passed public GPU checks; hidden verifier not run",
            }
        elif snapshot.current_candidate_sha256 is not None:
            action = {"tool": "repo_run_public_checks"}
        else:
            if not all(
                marker in snapshot.workspace_source
                for marker in ("denominator += expf(x);", "expf(own) / denominator")
            ):
                raise ValueError("initial public source evidence is missing")
            action = {
                "tool": "repo_replace_many",
                "path": KERNEL,
                "edits": [
                    {
                        "old_text": "denominator += expf(x);",
                        "new_text": "denominator += expf(x - maximum);",
                    },
                    {
                        "old_text": "expf(own) / denominator",
                        "new_text": "expf(own - maximum) / denominator",
                    },
                ],
            }
        if action["tool"] not in snapshot.allowed_tools:
            raise ValueError("action forbidden by the current controller policy")
        return HarnessDecision.model_validate(
            {"action": action, "reason": "Scripted routing checks the real GPU bridge"}
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-real-gpu",
        action="store_true",
        help="Authorize physical Docker/CUDA public checks, not paid model calls",
    )
    args = parser.parse_args()
    if not args.allow_real_gpu:
        parser.error("real GPU execution requires --allow-real-gpu")
    root = args.repository.resolve(strict=True)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error("refusing to overwrite any existing experiment directory")
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    ).stdout.strip()
    model = EvidenceAwareScriptedDecision()
    report = run_repository_coding_session(
        repository=root,
        output=output,
        base_sha=sha,
        goal="Repair the registered h001 softmax numeric instability; use public GPU checks",
        public_case_id="h001",
        max_public_checks=2,
        context_mode="evidence_first",
        max_steps=4,
        max_model_calls=6,
        model_factory=lambda _: model,
    )
    metrics = summarize_repository_trajectory(output)
    result = {
        "source_sha": sha,
        "public_case": metrics.public_case_id,
        "context_mode": metrics.context_mode,
        "status": report.status,
        "tool_calls": report.tool_calls,
        "model_physical_calls": report.model_calls,
        "public_checks_attempted": report.public_checks_attempted,
        "public_checks_passed": report.public_checks_passed,
        "first_accepted_edit_turn": metrics.first_accepted_edit_turn,
        "first_public_pass_turn": metrics.first_public_pass_turn,
        "first_finish_turn": metrics.first_accepted_finish_turn,
        "public_run_id": report.public_run_id,
        "candidate_sha256": report.public_checked_candidate_sha256,
        "model": "deterministic scripted test double, NOT an LLM",
        "independent_hidden_verification": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return (
        0
        if (
            report.status == "DIFF_STAGED_UNVERIFIED"
            and report.public_checks_attempted == 1
            and report.public_checks_passed
            and report.model_calls == 0
            and report.tool_calls == 3
            and metrics.first_accepted_edit_turn == 1
            and metrics.first_public_pass_turn == 2
            and metrics.first_accepted_finish_turn == 3
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
