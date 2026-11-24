"""Real isolated CUDA GPU smoke for Planner + Reflect + AST patch authority.

The model is an explicit deterministic TEST DOUBLE, not a real Qwen/DeepSeek.
Actual Docker/NVCC/public GPU numerical tests and all four sanitizers run.
Hidden strict verification is a separate operator-only process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from gpu_agent.harness.metrics import summarize_repository_trajectory
from gpu_agent.harness.models import HarnessDecision, TaskPlan, TaskReflection
from gpu_agent.harness.repo_runner import run_repository_coding_session

KERNEL = "benchmarks/hardset_v2/h001/mutant/kernel.cu"


class ScriptedPlanningProbe:
    def __init__(self) -> None:
        self.plan_calls = 0
        self.reflection_calls = 0

    def choose_repository_plan(self, snapshot):
        self.plan_calls += 1
        return TaskPlan.model_validate(
            {
                "milestones": [
                    {
                        "gate": gate,
                        "objective": f"Complete {gate} using only registered CUDA source evidence",
                        "success_evidence": f"Controller verifies the {gate} milestone",
                    }
                    for gate in ("inspect", "edit", "public_check", "finish")
                ],
                "failure_risk": "Source spans and syntax IDs must bind to current file SHA",
            }
        )

    def reflect_repository_task(self, snapshot):
        self.reflection_calls += 1
        return TaskReflection(
            observed_failure="No public errors in this deterministic connector test",
            avoid_next="Avoid repeating inspected AST nodes at the same revision",
            next_priority="Finish only if the exact current CUDA candidate passed",
        )

    def choose_harness_v1(self, snapshot):
        if snapshot.public_checks_passed:
            action = {
                "tool": "finish",
                "summary": "Public CUDA GPU checks passed, private strict verification not run",
            }
        elif snapshot.current_candidate_sha256 is not None:
            action = {"tool": "repo_run_public_checks"}
        elif any(item["tool"] == "repo_inspect_syntax" for item in snapshot.history):
            item = next(x for x in reversed(snapshot.history) if x["tool"] == "repo_inspect_syntax")
            mapping = json.loads(item["content"])
            nodes = mapping["nodes"]
            d = next(x for x in nodes if x["code"] == "denominator += expf(x);")
            n = next(x for x in nodes if x["code"] == "out[i] = expf(own) / denominator;")
            action = {
                "tool": "repo_replace_syntax",
                "path": KERNEL,
                "expected_source_sha256": mapping["source_sha256"],
                "edits": [
                    {
                        "node_id": d["id"],
                        "replacement_code": "denominator += expf(x - maximum);",
                    },
                    {
                        "node_id": n["id"],
                        "replacement_code": "out[i] = expf(own - maximum) / denominator;",
                    },
                ],
            }
        else:
            action = {"tool": "repo_inspect_syntax", "path": KERNEL, "symbol": "softmax32", "limit": 20}
        if action["tool"] not in snapshot.allowed_tools:
            raise ValueError("scripted probe tried a forbidden repository tool")
        return HarnessDecision.model_validate(
            {"action": action, "reason": "Deterministic test of the controller integration"}
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-real-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_real_gpu:
        parser.error("operator must explicitly authorize real isolated CUDA GPU checks")
    repo = args.repository.resolve(strict=True)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error("experiment output may not overwrite existing evidence")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    ).stdout.strip()
    source = (repo / KERNEL).read_bytes()
    model = ScriptedPlanningProbe()
    result = run_repository_coding_session(
        repository=repo,
        output=output,
        base_sha=sha,
        goal="Repair h001 stable Softmax using bounded CUDA AST tools and public checks",
        max_steps=6,
        max_model_calls=8,
        public_case_id="h001",
        max_public_checks=2,
        hierarchical_mode=True,
        syntax_mode=True,
        context_mode="evidence_first",
        model_factory=lambda _: model,
    )
    metrics = summarize_repository_trajectory(output)
    body = {
        "source_sha": sha,
        "original_kernel_sha256": hashlib.sha256(source).hexdigest(),
        "candidate_sha256": result.public_checked_candidate_sha256,
        "status": result.status,
        "public_checks_passed": result.public_checks_passed,
        "public_gpu_checks_attempted": result.public_checks_attempted,
        "tool_calls": result.tool_calls,
        "physical_model_calls": result.model_calls,
        "first_accepted_edit_turn": metrics.first_accepted_edit_turn,
        "first_public_pass_turn": metrics.first_public_pass_turn,
        "first_finish_turn": metrics.first_accepted_finish_turn,
        "hierarchical_mode": metrics.hierarchical_mode,
        "syntax_mode": metrics.syntax_mode,
        "plan_calls_test_double": model.plan_calls,
        "reflect_calls_test_double": model.reflection_calls,
        "independently_verified": False,
        "model": "DETERMINISTIC TEST DOUBLE; not a trained model",
    }
    print(json.dumps(body, indent=2, sort_keys=True))
    return (
        0
        if (
            result.status == "DIFF_STAGED_UNVERIFIED"
            and result.public_checks_passed
            and result.public_checks_attempted == 1
            and result.tool_calls == 4
            and result.model_calls == 0
            and model.plan_calls == 1
            and model.reflection_calls == 1
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
