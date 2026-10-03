"""Deterministic evidence-first fallback for bounded development investigations."""

from gpu_agent.agent.models import (
    AgentAction,
    AgentBudget,
    DocsArguments,
    FinishAction,
    InconclusiveAction,
    InitcheckAction,
    MemcheckAction,
    PublicEvidence,
    RacecheckAction,
    RetrieveDocsAction,
    SynccheckAction,
)
from gpu_agent.execution.models import SanitizerTool


class RuleRouter:
    """Choose only published, typed actions and stop when no budgeted evidence remains."""

    def next_action(self, evidence: PublicEvidence, budget: AgentBudget) -> AgentAction:
        if evidence.tool_findings:
            if not evidence.documentation and budget.rag_calls < budget.max_rag_calls:
                query = " ".join(f.category for f in evidence.tool_findings)[:500]
                return RetrieveDocsAction(
                    rationale="RULE_FALLBACK: retrieve official documentation for observed finding",
                    typed_arguments=DocsArguments(query=query, k=3),
                )
            return FinishAction(rationale="RULE_FALLBACK: evidence is sufficient for diagnosis")

        outcomes = evidence.sanitizer_outcomes
        if SanitizerTool.MEMCHECK not in outcomes:
            return MemcheckAction(rationale="RULE_FALLBACK: memory-safety precheck")
        if outcomes[SanitizerTool.MEMCHECK] != "CLEAN":
            return InconclusiveAction(rationale="RULE_FALLBACK: memcheck evidence is unavailable")
        if budget.sanitizer_calls >= budget.max_sanitizer_calls:
            return InconclusiveAction(rationale="RULE_FALLBACK: sanitizer budget exhausted")

        source = "\n".join(item.content for item in evidence.sources)
        ordered = [
            (SanitizerTool.SYNCCHECK, SynccheckAction)
            if "__sync" in source
            else (SanitizerTool.RACECHECK, RacecheckAction)
            if "__shared__" in source
            else (SanitizerTool.INITCHECK, InitcheckAction),
            (SanitizerTool.RACECHECK, RacecheckAction),
            (SanitizerTool.INITCHECK, InitcheckAction),
            (SanitizerTool.SYNCCHECK, SynccheckAction),
        ]
        seen: set[SanitizerTool] = set()
        for tool, action_type in ordered:
            if tool not in seen and tool not in outcomes:
                return action_type(rationale=f"RULE_FALLBACK: inspect with {tool.value}")
            seen.add(tool)
        return InconclusiveAction(rationale="RULE_FALLBACK: all applicable checks were clean")
