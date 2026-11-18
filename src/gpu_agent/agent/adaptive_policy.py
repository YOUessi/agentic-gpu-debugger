"""Opt-in cost-aware evidence acquisition. The controller owns all safety decisions.

This is a *research variant*, not a verified improvement over the rule baseline:
it skips an LLM call for a mandatory memcheck, and may finish only when
the rule router and mandatory-evidence gate both agree. Ambiguous hazard
investigation still delegates to the LLM. No oracle access, external API,
model-authored tool permission, or override of the typed action guard.
"""

import os
from typing import Literal

from gpu_agent.agent.models import (
    AgentAction,
    AgentBudget,
    DocsArguments,
    PublicEvidence,
    RetrieveDocsAction,
)
from gpu_agent.agent.policy import missing_evidence
from gpu_agent.agent.rule_router import RuleRouter

PlannerPolicy = Literal["baseline", "adaptive-v1"]
SelectionReason = Literal["MANDATORY_MEMCHECK", "EVIDENCE_COMPLETE"]


def selected_planner_policy() -> PlannerPolicy:
    value = os.environ.get("GPU_AGENT_PLANNER_POLICY", "baseline")
    if value not in {"baseline", "adaptive-v1"}:
        raise ValueError("GPU_AGENT_PLANNER_POLICY must be baseline or adaptive-v1")
    return "adaptive-v1" if value == "adaptive-v1" else "baseline"


def predictable_action(
    evidence: PublicEvidence,
    budget: AgentBudget,
    router: RuleRouter,
) -> tuple[AgentAction, SelectionReason] | None:
    """Use deterministic actions only when no discretionary investigation is lost."""
    proposed = router.next_action(evidence, budget)
    if proposed.action_type == "run_memcheck" and "memcheck" not in evidence.sanitizer_outcomes:
        return proposed, "MANDATORY_MEMCHECK"
    if (
        proposed.action_type == "finish_diagnosis"
        and evidence.tool_findings
        and not missing_evidence(evidence)
    ):
        return proposed, "EVIDENCE_COMPLETE"
    return None


def evidence_scoped_rule_fallback(
    evidence: PublicEvidence,
    budget: AgentBudget,
    router: RuleRouter,
) -> AgentAction:
    """Keep rule authority but search CUDA terminology, not only sanitizer prose.

    NVIDIA tool findings often say 'Invalid __global__ write' while official
    documentation discusses 'out of bounds global memory'. Add a small,
    version-independent normalization of *observed* finding categories. Never
    use answers, case IDs, hidden tests or new external queries.
    """
    proposed = router.next_action(evidence, budget)
    if proposed.action_type != "retrieve_official_docs":
        return proposed
    names = " ".join(f.category.lower() for f in evidence.tool_findings)
    if "__global__" in names or "out of bounds" in names or "out-of-bounds" in names:
        query = "compute sanitizer memcheck out of bounds global memory"
    elif "race" in names or "hazard" in names or "shared" in names:
        query = "compute sanitizer racecheck shared memory"
    elif "uninitialized" in names or "initcheck" in names:
        query = "compute sanitizer initcheck uninitialized memory"
    elif "barrier" in names or "sync" in names:
        query = "compute sanitizer synccheck barrier synchronization"
    else:
        return proposed
    return RetrieveDocsAction(
        rationale="EVIDENCE_FALLBACK: match documented CUDA term to observed tool category",
        typed_arguments=DocsArguments(query=query, k=3),
    )
