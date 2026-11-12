"""LangGraph state machine for D/E evidence acquisition.

The controller remains the sole authority for policy, budgets, public evidence and
tool execution. Graph state contains references/decisions, never credentials, private
oracle contents, executable shell commands or evaluator capabilities.

No checkpointer is deliberately configured here: replaying a side-effectful CUDA
tool or a paid LLM request without a durable idempotency ledger is unsafe.
RunStore remains the canonical append-only audit and the legacy controller is kept
for frozen evaluation comparisons.
"""

import json
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from gpu_agent.agent.models import (
    ACTION_ADAPTER,
    DiagnosisResult,
    ExecutedAction,
)
from gpu_agent.agent.policy import decide_action
from gpu_agent.agent.provider import ProviderError
from gpu_agent.benchmark.evaluation import EvaluationMode
from gpu_agent.contracts import CurrentPhase

if TYPE_CHECKING:
    from gpu_agent.agent.orchestrator import AgentOrchestrator


Route = Literal["replan", "tool", "diagnose", "inconclusive"]


class InvestigationState(TypedDict, total=False):
    """Bounded, controller-validated step state; not a conversation transcript."""

    run_id: str
    mode: Literal["D", "E"]
    feedback: list[str]
    replanned: bool
    action: dict[str, Any]
    route: Route
    diagnosis: dict[str, Any]
    observed_steps: int
    last_tool: str


def run_langgraph_investigation(
    controller: "AgentOrchestrator", run_id: str, mode: EvaluationMode
) -> DiagnosisResult:
    """Replace the imperative D/E loop without changing tool or verdict authority."""
    if mode not in {"D", "E"}:
        raise ValueError("LangGraph investigation supports only D/E acquisition")
    # Avoid a cyclic import while preserving the original evidence projection.
    from gpu_agent.agent.orchestrator import public_evidence

    def observe(state: InvestigationState) -> dict[str, Any]:
        budget = controller.budget
        controller.budget = budget.model_copy(
            update={
                "llm_calls": controller.provider.gate.snapshot().llm_calls,
                "remaining_seconds": controller.provider.gate.remaining(),
            }
        )
        if (
            controller.budget.agent_steps >= controller.budget.max_agent_steps
            or controller.budget.remaining_seconds <= 0
        ):
            raise ProviderError("AGENT_BUDGET_EXHAUSTED")
        # Read only the public projection. Evidence itself remains in RunStore.
        public_evidence(controller.store, run_id)
        return {"observed_steps": controller.budget.agent_steps}

    def plan(state: InvestigationState) -> dict[str, Any]:
        evidence = public_evidence(controller.store, run_id)
        feedback = state.get("feedback") or None
        if mode == "D":
            proposed = controller.rule_router.next_action(evidence, controller.budget)
        else:
            reservation = controller._reserve("planner_llm")
            try:
                proposed = controller.provider.plan(
                    evidence,
                    controller.budget,
                    feedback,
                    state=controller._planner_state(evidence, feedback),
                )
                controller.ledger.settle(reservation)
            except ProviderError:
                controller.ledger.settle(reservation, "FAILED")
                raise
        try:
            action = ACTION_ADAPTER.validate_python(proposed.model_dump())
        except ValidationError:
            raise ProviderError("ACTION_INVALID") from None
        manifest = controller.store.load(run_id)
        refs = [ref for ref in manifest.artifact_refs if ref.name == "evidence/bundle.json"]
        if not refs:
            raise ValueError("controller action has no evidence snapshot")
        controller.store.put(
            run_id,
            f"actions/{controller.budget.agent_steps}/step.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "action": action.model_dump(mode="json"),
                    "evidence_ref": refs[-1].model_dump(mode="json"),
                    "evidence": evidence.model_dump(mode="json"),
                    "budget": controller.budget.model_dump(mode="json"),
                    "seen": sorted(controller.seen),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            controller.store.visibility,
        )
        return {"action": action.model_dump(mode="json")}

    def authorize(state: InvestigationState) -> dict[str, Any]:
        action = ACTION_ADAPTER.validate_python(state["action"])
        evidence = public_evidence(controller.store, run_id)
        decision = decide_action(
            action, evidence, controller.budget, CurrentPhase.DIAGNOSING, controller.seen
        )
        controller.store.put(
            run_id,
            f"actions/{controller.budget.agent_steps}/decision.json",
            decision.model_dump_json().encode(),
            controller.store.visibility,
        )
        if not decision.allowed:
            controller.budget = controller.budget.model_copy(
                update={"agent_steps": controller.budget.agent_steps + 1}
            )
            if mode == "E" and not state.get("replanned", False):
                return {
                    "route": "replan",
                    "replanned": True,
                    "feedback": list(decision.reason_codes),
                }
            raise ProviderError(decision.reason_codes[0])
        controller.seen.add(action.action_type + action.typed_arguments.model_dump_json())
        controller.executed.append(
            ExecutedAction(
                action_type=action.action_type,
                typed_arguments=action.typed_arguments.model_dump(mode="json"),
            )
        )
        controller.budget = controller.budget.model_copy(
            update={"agent_steps": controller.budget.agent_steps + 1}
        )
        if action.action_type == "finish_diagnosis":
            route: Route = "diagnose"
        elif action.action_type == "declare_inconclusive":
            route = "inconclusive"
        else:
            route = "tool"
        return {"route": route, "feedback": []}

    def execute_tool(state: InvestigationState) -> dict[str, Any]:
        action = ACTION_ADAPTER.validate_python(state["action"])
        # Same typed registry and execution capability as the old controller.
        # LangGraph ToolNode must not receive an arbitrary shell/network tool.
        controller.registry[action.action_type](action)
        return {"last_tool": action.action_type}

    def diagnose(state: InvestigationState) -> dict[str, Any]:
        result = controller._diagnose(public_evidence(controller.store, run_id))
        return {"diagnosis": result.model_dump(mode="json")}

    def inconclusive(state: InvestigationState) -> dict[str, Any]:
        result = DiagnosisResult.inconclusive("MODEL_DECLARED_INCONCLUSIVE")
        return {"diagnosis": result.model_dump(mode="json")}

    def route(state: InvestigationState) -> Route:
        return state["route"]

    graph = StateGraph(InvestigationState)
    graph.add_node("observe", observe)
    graph.add_node("plan", plan)
    graph.add_node("authorize", authorize)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("diagnose", diagnose)
    graph.add_node("inconclusive", inconclusive)
    graph.add_edge(START, "observe")
    graph.add_edge("observe", "plan")
    graph.add_edge("plan", "authorize")
    graph.add_conditional_edges(
        "authorize",
        route,
        {
            "replan": "observe",
            "tool": "execute_tool",
            "diagnose": "diagnose",
            "inconclusive": "inconclusive",
        },
    )
    graph.add_edge("execute_tool", "observe")
    graph.add_edge("diagnose", END)
    graph.add_edge("inconclusive", END)

    # LangGraph's default recursion limit (25) is too low for 38 actions x nodes.
    # Neither graph recursion nor a checkpointer replaces controller budgets.
    compiled = graph.compile()
    initial: InvestigationState = {
        "run_id": run_id,
        "mode": cast(Literal["D", "E"], mode),
        "feedback": [],
        "replanned": False,
    }
    result = compiled.invoke(
        initial, config={"recursion_limit": (controller.budget.max_agent_steps + 3) * 5}
    )
    return DiagnosisResult.model_validate(result["diagnosis"])
