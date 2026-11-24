"""One REAL CUDA ToolHost smoke: before probe -> AST -> after probe -> full check.

All model decisions here are a deterministic fixture, never a real LLM.
Docker/NVCC and numerical CPU Oracle execute physically on the GPU host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from gpu_agent.harness.metrics import summarize_repository_trajectory
from gpu_agent.harness.models import HarnessDecision
from gpu_agent.harness.repo_runner import run_repository_coding_session

KERNEL = "benchmarks/hardset_v2/h001/mutant/kernel.cu"


class PublicProbeFixture:
    """Scripted test double, not evidence of autonomous coding performance."""

    def choose_harness_v1(self, snapshot):
        history = list(snapshot.history)
        if snapshot.public_checks_passed:
            action = {
                "tool": "finish",
                "summary": "Public CUDA verification passed; independent hidden verdict not run",
            }
        elif not history:
            action = {"tool": "repo_probe_public", "phase": "before", "input_size": 33}
        elif history[-1]["tool"] == "repo_probe_public" and snapshot.public_probe_phase == "before":
            action = {"tool": "repo_inspect_syntax", "path": KERNEL, "symbol": "softmax32"}
        elif history[-1]["tool"] == "repo_inspect_syntax":
            syntax = json.loads(history[-1]["content"])
            nodes = syntax["nodes"]
            denominator = next(x for x in nodes if x["code"] == "denominator += expf(x);")
            numerator = next(x for x in nodes if x["code"] == "out[i] = expf(own) / denominator;")
            action = {
                "tool": "repo_replace_syntax",
                "path": KERNEL,
                "expected_source_sha256": syntax["source_sha256"],
                "edits": [
                    {
                        "node_id": denominator["id"],
                        "replacement_code": "denominator += expf(x - maximum);",
                    },
                    {
                        "node_id": numerator["id"],
                        "replacement_code": "out[i] = expf(own - maximum) / denominator;",
                    },
                ],
            }
        elif history[-1]["tool"] == "repo_replace_syntax":
            action = {"tool": "repo_probe_public", "phase": "after", "input_size": 33}
        elif history[-1]["tool"] == "repo_probe_public" and snapshot.public_probe_phase == "after":
            action = {"tool": "repo_run_public_checks"}
        else:
            raise ValueError("deterministic smoke encountered unknown Agent state")
        if action["tool"] not in snapshot.allowed_tools:
            raise ValueError("tool permission gate prohibited the fixture action")
        return HarnessDecision.model_validate(
            {"action": action, "reason": "Scripted public CUDA probe connectivity test"}
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-real-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_real_gpu:
        parser.error("real isolated GPU execution requires explicit operator approval")
    repo = args.repository.resolve(strict=True)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error("existing output cannot be overwritten")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
        timeout=20,
    ).stdout.strip()
    result = run_repository_coding_session(
        repository=repo,
        output=output,
        base_sha=sha,
        goal="Repair public registered h001 stable CUDA softmax and verify with public probes",
        public_case_id="h001",
        context_mode="evidence_first",
        syntax_mode=True,
        max_steps=6,
        max_model_calls=8,
        max_public_checks=2,
        model_factory=lambda _: PublicProbeFixture(),
    )
    metrics = summarize_repository_trajectory(output)
    checkpoint = json.loads((output / "session-checkpoint.json").read_text())
    observations = checkpoint["state"]["history"]
    probes = [json.loads(x["content"]) for x in observations if x["tool"] == "repo_probe_public"]
    summary = {
        "frozen_sha": sha,
        "physical_model_calls": result.model_calls,
        "public_probe_count": metrics.public_probes_attempted,
        "public_probe_results": [
            {
                "phase": x["phase"],
                "input_size": x["input_size"],
                "outcome": x["outcome"],
                "diagnostic_region": x["diagnostic_region"],
                "candidate_sha256": x["candidate_sha256"],
            }
            for x in probes
        ],
        "public_gpu_checks": result.public_checks_attempted,
        "public_checks_passed": result.public_checks_passed,
        "first_accepted_edit_turn": metrics.first_accepted_edit_turn,
        "first_public_check_turn": metrics.first_public_check_request_turn,
        "status": result.status,
        "tool_calls": result.tool_calls,
        "candidate_sha256": result.public_checked_candidate_sha256,
        "hidden_verification_ran": False,
        "model_type": "deterministic test double, NOT an autonomous LLM",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    expected = (
        result.status == "DIFF_STAGED_UNVERIFIED"
        and result.public_checks_passed
        and result.tool_calls == 6
        and result.model_calls == 0
        and result.public_probes_attempted == 2
        and metrics.public_probes_attempted == 2
        and len(probes) == 2
        and probes[0]["phase"] == "before"
        and probes[0]["outcome"] == "NUMERIC_MISMATCH"
        and probes[1]["phase"] == "after"
        and probes[1]["outcome"] == "PASS"
        and probes[0]["candidate_sha256"] != probes[1]["candidate_sha256"]
    )
    assert len(summary["candidate_sha256"] or "") == 64
    if expected:
        assert hashlib.sha256((output / "result.json").read_bytes()).hexdigest()
    return 0 if expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
