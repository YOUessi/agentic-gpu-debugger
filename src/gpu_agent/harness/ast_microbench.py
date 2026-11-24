"""One physical model decision with pinned public AST evidence, without Planner.

The operator collects all legal AST pages before asking the model for one edit.
No oracle, hidden verifier, expected answer, or reference patch enters the prompt.
A syntactically accepted, changed candidate is NOT evidence of semantic correctness.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gpu_agent.agent.models import AgentBudget, ProviderError
from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.agent.provider import (
    DevelopmentCallPolicy,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
)
from gpu_agent.harness.models import FinishTask, RepoInspectSyntax, RepoReplaceSyntax, RepoRunPublicChecks
from gpu_agent.harness.repo_runner import RepositoryDecisionModel
from gpu_agent.harness.repo_tools import RepositoryWorkspaceTools


def run_ast_single_step(
    *,
    repository: Path,
    output: Path,
    base_sha: str,
    case_id: str,
    allow_paid_calls: bool = False,
    allow_real_gpu: bool = False,
    model_factory: Callable[[RepositoryWorkspaceTools], RepositoryDecisionModel] | None = None,
) -> dict[str, Any]:
    """Keep one edit decision and the subsequent operator GPU check distinct."""
    if model_factory is None and not allow_paid_calls:
        raise ValueError("real model call requires explicit operator approval")
    kernel = f"benchmarks/hardset_v2/{case_id}/mutant/kernel.cu"
    goal = "Repair the registered public CUDA kernel to satisfy its functional requirement."
    with RepositoryWorkspaceTools(
        origin=repository, output=output, sha=base_sha, public_case_id=case_id,
        max_public_checks=1, context_mode="evidence_first",
        allow_batch_edits=False, syntax_mode=True,
    ) as host:
        offset = 0
        while True:
            obs = host.invoke(RepoInspectSyntax(path=kernel, offset=offset, limit=8))
            if obs.status != "OK":
                raise ValueError(f"operator AST prefill failed: {obs.content}")
            state = host.syntax_state
            if state is None:
                raise ValueError("operator AST evidence missing")
            if state.next_offset is None:
                break
            offset = state.next_offset
        snapshot = host.snapshot(goal, [], 1).model_copy(update={
            "allowed_tools": ("repo_replace_syntax",),
            "context_summary": (
                "Single-step diagnostic: select legal short aliases from syntax_state and "
                "write a changed CUDA replacement in exactly one atomic edit. The operator "
                "will run any public check after this response. No other model action is allowed."
            ),
        })
        if model_factory is None:
            settings = OpenAIProviderSettings.from_environment()
            provider = OpenAIResponsesProvider(
                settings,
                LLMCallGate(
                    AgentBudget(max_llm_calls=1, max_wall_time_seconds=600),
                    reserve_repair_stages=False,
                ),
                host.store,
                host.run_id,
                call_policy=DevelopmentCallPolicy(max_llm_calls=1),
            )
            model: RepositoryDecisionModel = provider
            model_name = settings.model
        else:
            model = model_factory(host)
            model_name = "DETERMINISTIC_TEST_DOUBLE"
        report: dict[str, Any] = {
            "schema_version": "gpu-ast-single-step-v1",
            "base_sha": base_sha,
            "case_id": case_id,
            "model": model_name,
            "planner": False,
            "public_run_id": host.run_id,
            "source_sha256": state.source_sha256,
            "prefilled_pages": len(state.visited_pages),
            "observed_nodes": len(state.observed_nodes),
            "decision_attempts": 1,
            "action": None,
            "edit_status": None,
            "edit_feedback": None,
            "provider_error": None,
            "legal_non_noop": False,
            "candidate_sha256": None,
            "checker_reached": False,
            "public_check_passed": False,
            "checker_feedback": None,
            "independent_hidden_verification": False,
        }
        (output / "single-step-snapshot.json").write_text(snapshot.model_dump_json(indent=2))
        try:
            decision = model.choose_harness_v1(snapshot)
        except ProviderError as exc:
            report["provider_error"] = exc.code
        else:
            report["action"] = decision.action.model_dump(mode="json")
            if not isinstance(decision.action, RepoReplaceSyntax):
                report["edit_status"] = "UNEXPECTED_TOOL"
            else:
                result = host.invoke(decision.action)
                report["edit_status"] = result.status
                report["edit_feedback"] = result.content
                candidate = (host.workspace.root / kernel).read_bytes()
                candidate_sha = hashlib.sha256(candidate).hexdigest()
                report["candidate_sha256"] = candidate_sha
                report["legal_non_noop"] = (
                    result.status == "OK" and candidate_sha != state.source_sha256
                    and bool(host.workspace.diff())
                )
                if report["legal_non_noop"] and allow_real_gpu:
                    checked = host.invoke(RepoRunPublicChecks())
                    report["checker_reached"] = host.public_checks_attempted == 1
                    report["public_check_passed"] = host.public_checks_passed
                    report["checker_feedback"] = checked.content
                    if host.public_checks_passed:
                        host.invoke(FinishTask(summary="Operator finalized a publicly checked candidate"))
        invocations = getattr(model, "invocations", None)
        records = invocations() if callable(invocations) else []
        report["model_physical_calls"] = len(records)
        report["provider_audit"] = [
            {"state": item.state, "error_code": item.error_code,
             "usage": item.usage.model_dump(mode="json") if item.usage else None}
            for item in records
        ]
        report["tool_calls_including_operator"] = host.tool_calls
        (output / "single-step-result.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    return report
