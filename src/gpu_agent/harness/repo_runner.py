"""Model-directed multi-file Git coding session with no untrusted host shell.

This runner stages code edits for review; it intentionally never claims that
staged changes passed CUDA execution or independent hidden verification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from gpu_agent.agent.models import AgentBudget
from gpu_agent.agent.policy import LLMCallGate, RepositoryCallBudget
from gpu_agent.agent.provider import (
    DevelopmentCallPolicy,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
)
from gpu_agent.harness.journal import EventKind, SessionJournal
from gpu_agent.harness.models import (
    HarnessDecision,
    RepoHarnessOutcome,
    SessionSnapshot,
)
from gpu_agent.harness.recovery import (
    RepositoryCheckpoint,
    validated_quiescent_repository_checkpoint,
    write_repository_checkpoint,
)
from gpu_agent.harness.repo_tools import RepositoryWorkspaceTools
from gpu_agent.harness.runner import AgentLoopState, run_tool_agent


class RepositoryDecisionModel(Protocol):
    def choose_harness_v1(self, snapshot: SessionSnapshot) -> HarnessDecision: ...


def run_repository_coding_session(
    *,
    repository: Path,
    output: Path,
    base_sha: str,
    goal: str,
    max_steps: int = 18,
    max_model_calls: int = 24,
    max_edited_files: int = 8,
    public_case_id: str | None = None,
    max_public_checks: int = 2,
    context_mode: Literal["legacy", "evidence_first"] = "evidence_first",
    allow_batch_edits: bool = True,
    hierarchical_mode: bool = False,
    syntax_mode: bool = False,
    compile_gate: bool = False,
    allow_revert: bool = False,
    max_compiles: int = 6,
    public_diagnostics: bool = False,
    ast_target_binding: bool = False,
    provider_settings: OpenAIProviderSettings | None = None,
    allow_paid_calls: bool = False,
    resume: bool = False,
    pause_after_turns: int | None = None,
    model_factory: (Callable[[RepositoryWorkspaceTools], RepositoryDecisionModel] | None) = None,
) -> RepoHarnessOutcome:
    """Execute a real Agent loop and stage independent Git worktree modifications.

    Only operator-registered HardSet public GPU tests may execute in Docker.
    Host Shell, arbitrary tests and the private verifier are never Agent tools.
    Physical model calls use the same killable SDK worker and on-disk audit as
    the CUDA single-file Harness.
    """
    if (model_factory is None or provider_settings is not None) and not allow_paid_calls:
        raise ValueError("real model use requires explicit allow_paid_calls")
    if not 1 <= max_steps <= 38 or not max_steps + 2 <= max_model_calls <= 40:
        raise ValueError("model calls must exceed step budget, within operator cap")
    role_budget = (
        RepositoryCallBudget.from_session(
            max_steps=max_steps, max_model_calls=max_model_calls, hierarchical=hierarchical_mode
        )
        if model_factory is None or provider_settings is not None
        else None
    )
    if (hierarchical_mode or syntax_mode) and public_case_id is None:
        raise ValueError("hierarchical and AST modes require registered public CUDA case")
    if not goal.strip() or len(goal) > 1500:
        raise ValueError("invalid task goal")
    settings = provider_settings or (
        OpenAIProviderSettings.from_environment() if model_factory is None else None
    )
    output_cap = settings.max_output_tokens if settings is not None else 4096
    checkpoint = None
    if resume:
        if not output.is_dir() or output.is_symlink():
            raise ValueError("resume requires an existing safe session output")
        checkpoint = validated_quiescent_repository_checkpoint(
            output,
            goal=goal,
            base_sha=base_sha,
            max_steps=max_steps,
            max_model_calls=max_model_calls,
            max_output_tokens=output_cap,
            max_edited_files=max_edited_files,
            public_case_id=public_case_id,
            max_public_checks=max_public_checks,
            context_mode=context_mode,
            allow_batch_edits=allow_batch_edits,
            hierarchical_mode=hierarchical_mode,
            syntax_mode=syntax_mode,
            compile_gate=compile_gate,
            allow_revert=allow_revert,
            max_compiles=max_compiles,
            public_diagnostics=public_diagnostics,
            ast_target_binding=ast_target_binding,
        )
        if syntax_mode and "syntax_state" not in checkpoint.model_fields_set:
            raise ValueError("legacy AST checkpoint cannot resume with short aliases")
        host = RepositoryWorkspaceTools.reopen(
            origin=repository,
            output=output,
            sha=base_sha,
            run_id=checkpoint.public_run_id,
            tool_calls=checkpoint.state["turns"],
            accepted_hashes=checkpoint.accepted_hashes,
            max_edited_files=max_edited_files,
            public_case_id=checkpoint.public_case_id,
            max_public_checks=checkpoint.max_public_checks,
            public_checks_attempted=checkpoint.public_checks_attempted,
            attempted_candidate_hashes=checkpoint.attempted_candidate_hashes,
            last_public_run_id=checkpoint.last_public_run_id,
            last_checked_diff_sha=checkpoint.last_checked_diff_sha,
            last_checked_candidate_sha=checkpoint.last_checked_candidate_sha,
            last_public_passed=checkpoint.last_public_passed,
            observed_query_fingerprints=checkpoint.observed_query_fingerprints,
            syntax_state=checkpoint.syntax_state,
            rejected_syntax_edits=checkpoint.rejected_syntax_edits,
            public_probe_signatures=checkpoint.public_probe_signatures,
            public_probe_run_ids=checkpoint.public_probe_run_ids,
            context_mode=checkpoint.context_mode,
            allow_batch_edits=checkpoint.allow_batch_edits,
            syntax_mode=checkpoint.syntax_mode,
            compile_gate=checkpoint.compile_gate,
            allow_revert=checkpoint.allow_revert,
            max_compiles=checkpoint.max_compiles,
            public_diagnostics=checkpoint.public_diagnostics,
            ast_target_binding=checkpoint.ast_target_binding,
        )
        if host.workspace.integrity()["diff_sha256"] != checkpoint.diff_sha256:
            raise ValueError("resumed workspace diff disagrees with checkpoint")
    else:
        if output.exists() or output.is_symlink():
            raise ValueError("workspace output cannot overwrite existing evidence")
        host = RepositoryWorkspaceTools(
            origin=repository,
            output=output,
            sha=base_sha,
            max_edited_files=max_edited_files,
            public_case_id=public_case_id,
            max_public_checks=max_public_checks,
            context_mode=context_mode,
            allow_batch_edits=allow_batch_edits,
            syntax_mode=syntax_mode,
            compile_gate=compile_gate,
            allow_revert=allow_revert,
            max_compiles=max_compiles,
            public_diagnostics=public_diagnostics,
            ast_target_binding=ast_target_binding,
        )

    with host as tools:
        if model_factory is None:
            assert settings is not None
            provider = OpenAIResponsesProvider(
                settings,
                LLMCallGate(
                    AgentBudget(
                        max_llm_calls=max_model_calls,
                        max_wall_time_seconds=600,
                    ),
                    # Coding Agent issues plan/tool requests only. The old
                    # Workflow D's diagnosis/patch reservations are unrelated.
                    reserve_repair_stages=False,
                ),
                tools.store,
                tools.run_id,
                call_policy=DevelopmentCallPolicy(max_llm_calls=max_model_calls),
                repository_budget=role_budget,
            )
            provider.ensure_available()
            model: RepositoryDecisionModel = provider
        else:
            model = model_factory(tools)
        journal = (
            SessionJournal.reopen(
                output / "session-events.jsonl",
                expected_head=checkpoint.journal_head_sha256,
            )
            if checkpoint is not None
            else SessionJournal(output / "session-events.jsonl")
        )
        try:
            if checkpoint is not None:
                journal.append(
                    "session_resumed",
                    {"previous_turns": checkpoint.state["turns"]},
                )
                events = checkpoint.event_counter
            else:
                journal.append(
                    "session_started",
                    {
                        "base_git_commit": base_sha,
                        "public_run_id": tools.run_id,
                        "goal_sha256": hashlib.sha256(goal.encode()).hexdigest(),
                        "public_case_id": public_case_id,
                        "max_public_checks": max_public_checks,
                        "context_mode": context_mode,
                        "allow_batch_edits": allow_batch_edits,
                        "hierarchical_mode": hierarchical_mode,
                        "syntax_mode": syntax_mode,
                        "compile_gate": compile_gate,
                        "allow_revert": allow_revert,
                        "max_compiles": max_compiles,
                        "public_diagnostics": public_diagnostics,
                        "ast_target_binding": ast_target_binding,
                        "max_steps": max_steps,
                        "max_model_calls": max_model_calls,
                        "public_check_audit_version": 1,
                        "max_output_tokens": output_cap,
                        "repository_call_budget": {
                            "executor_decisions": role_budget.executor_decisions,
                            "executor_format_retries": role_budget.executor_format_retries,
                            "planner_decisions": role_budget.planner_decisions,
                            "reflect_decisions": role_budget.reflect_decisions,
                            "planner_format_retries": role_budget.planner_format_retries,
                            "reflect_format_retries": role_budget.reflect_format_retries,
                            "physical_ceiling": role_budget.total,
                        }
                        if role_budget is not None
                        else None,
                    },
                )
                events = 0

            def save_event(event: dict[str, Any]) -> None:
                nonlocal events
                events += 1
                stage = str(event.get("stage", ""))
                kinds: dict[str, EventKind] = {
                    "model_started": "model_started",
                    "model": "model_selected",
                    "tool_started": "tool_started",
                    "tool": "tool_completed",
                }
                kind = kinds.get(stage)
                if kind:
                    journal.append(
                        kind,
                        {
                            "turn": event.get("turn", 0),
                            "tool": str(event.get("tool", ""))[:64],
                            "status": str(event.get("status", ""))[:24],
                        },
                    )
                tools.store.put(
                    tools.run_id,
                    f"repo-harness/v1/trace-{events:03d}.json",
                    json.dumps(
                        {
                            "turn": event.get("turn", 0),
                            "stage": stage,
                            "tool": event.get("tool", ""),
                            "status": event.get("status", ""),
                            "record_sha256": hashlib.sha256(
                                json.dumps(event, sort_keys=True).encode()
                            ).hexdigest(),
                        },
                        sort_keys=True,
                    ).encode(),
                    "public",
                )

            def persist_checkpoint(state: AgentLoopState) -> None:
                # Tool completion and trace have been fsynced before this
                # state becomes resumable. A crash in the gap fails closed.
                integrity = tools.workspace.integrity()
                write_repository_checkpoint(
                    output / "session-checkpoint.json",
                    RepositoryCheckpoint(
                        base_sha=base_sha,
                        goal_sha256=hashlib.sha256(goal.encode()).hexdigest(),
                        public_run_id=tools.run_id,
                        journal_head_sha256=journal.head_hash,
                        diff_sha256=str(integrity["diff_sha256"]),
                        event_counter=events,
                        max_steps=max_steps,
                        max_model_calls=max_model_calls,
                        max_output_tokens=output_cap,
                        max_edited_files=max_edited_files,
                        public_case_id=public_case_id,
                        max_public_checks=max_public_checks,
                        context_mode=context_mode,
                        allow_batch_edits=allow_batch_edits,
                        hierarchical_mode=hierarchical_mode,
                        syntax_mode=syntax_mode,
                        compile_gate=compile_gate,
                        allow_revert=allow_revert,
                        max_compiles=max_compiles,
                        public_diagnostics=public_diagnostics,
                        ast_target_binding=ast_target_binding,
                        public_checks_attempted=tools.public_checks_attempted,
                        public_probes_attempted=tools.public_probes_attempted,
                        public_probe_signatures=tools.public_probe_signatures,
                        public_probe_run_ids=tools.public_probe_run_ids,
                        attempted_candidate_hashes=tools.attempted_candidate_hashes,
                        last_public_run_id=tools.last_public_run_id,
                        last_checked_diff_sha=tools.last_checked_diff_sha,
                        last_checked_candidate_sha=tools.last_checked_candidate_sha,
                        last_public_passed=tools._last_public_passed,
                        observed_query_fingerprints=tools.observed_query_fingerprints,
                        syntax_state=tools.syntax_state,
                        rejected_syntax_edits=tools.rejected_syntax_edits,
                        accepted_hashes=tools.workspace.accepted_hashes,
                        state=dict(state),
                    ),
                )

            state = run_tool_agent(
                model,
                tools,
                goal=goal,
                max_steps=max_steps,
                on_event=save_event,
                on_checkpoint=persist_checkpoint,
                initial_state=cast(AgentLoopState, checkpoint.state)
                if checkpoint is not None
                else None,
                pause_after_turns=pause_after_turns,
                hierarchical_mode=hierarchical_mode,
            )
            inv = getattr(model, "invocations", None)
            records = inv() if callable(inv) else []
            token_counts = [
                item.usage.total_tokens
                for item in records
                if item.usage is not None and item.usage.total_tokens is not None
            ]
            tokens = sum(token_counts) if len(token_counts) == len(records) else None
            integrity = tools.workspace.integrity()
            succeeded = state.get("reason") == "WORKSPACE_DIFF_STAGED_UNVERIFIED"
            paused = state.get("reason") == "SESSION_PAUSED"
            if succeeded:
                # This can be restored from a checkpoint after a crash that
                # happened after FinishTask but before the terminal result.
                tools._finished = True
            outcome = RepoHarnessOutcome(
                status=(
                    "DIFF_STAGED_UNVERIFIED"
                    if succeeded
                    else "PAUSED_RESUMABLE"
                    if paused
                    else "INCONCLUSIVE"
                ),
                base_sha=base_sha,
                diff_sha256=str(integrity["diff_sha256"]),
                edited_files=tools.workspace.edited_files,
                turns=state.get("turns", 0),
                tool_calls=tools.tool_calls,
                model_calls=len(records),
                model_tokens=tokens,
                reason=state.get("reason", "UNKNOWN"),
                public_run_id=tools.run_id,
                context_mode=context_mode,
                allow_batch_edits=allow_batch_edits,
                hierarchical_mode=hierarchical_mode,
                syntax_mode=syntax_mode,
                public_checks_attempted=tools.public_checks_attempted,
                public_probes_attempted=tools.public_probes_attempted,
                public_checks_passed=tools.public_checks_passed,
                public_checked_candidate_sha256=tools.last_checked_candidate_sha,
            )
            if paused:
                # No final result/terminal journal event. RunStore remains
                # RUNNING and only an explicit, validated --resume can proceed.
                tools._suspended = True
                return outcome
            (output / "result.json").write_text(
                outcome.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            journal.append(
                "session_finished",
                {
                    "status": outcome.status,
                    "edited_files": len(outcome.edited_files),
                    "diff_sha256": outcome.diff_sha256,
                    "public_checks_attempted": outcome.public_checks_attempted,
                    "public_checks_passed": outcome.public_checks_passed,
                },
            )
            return outcome
        except Exception as exc:
            journal.append("session_failed", {"error_type": type(exc).__name__})
            raise
        finally:
            journal.close()
