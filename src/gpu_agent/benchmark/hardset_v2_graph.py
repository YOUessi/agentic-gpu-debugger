"""Experimental LangGraph V2 Functional Probe loop shared by D and E.

Both policies can use ONLY the same controller-owned typed FunctionalProbeAction
with the same hard budget; the Agent has no shell, Oracle, evaluator secret or
private-store capability. The model planner is an injected adapter so this
module does not introduce a second, unaudited network/model transport.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.benchmark.hardset_v2_probe import (
    FunctionalProbeAction,
    FunctionalProbeObservation,
)

ResearchMode = Literal["D", "E"]


class FinishProbeAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action_type: Literal["finish_probe"] = "finish_probe"


ProbeChoice = FunctionalProbeAction | FinishProbeAction


class HardsetProbePlanOutput(BaseModel):
    """Restricted model-wire shape; no tool arguments except registered size."""

    model_config = ConfigDict(extra="forbid")
    action: Annotated[ProbeChoice, Field(discriminator="action_type")]


class PlannerSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    remaining_probes: int
    observations: tuple[FunctionalProbeObservation, ...]
    public_source: str = Field(default="", max_length=4 * 1024 * 1024)
    functional_requirement: str = Field(default="", max_length=2000)
    available_sizes: tuple[int, ...] = ()


class ControlledProbe(Protocol):
    @property
    def calls(self) -> int: ...

    @property
    def allowed_sizes(self) -> tuple[int, ...]: ...

    def invoke(self, action: FunctionalProbeAction) -> FunctionalProbeObservation: ...


Planner = Callable[[PlannerSnapshot], ProbeChoice]


class ProbeGraphState(TypedDict, total=False):
    case_id: str
    mode: ResearchMode
    observations: list[dict[str, Any]]
    next_action: dict[str, Any]
    route: Literal["probe", "done"]
    decision_steps: int
    completion_reason: str


RULE_SIZES = (32, 33, 65)


def run_hardset_probe_graph(
    probe: ControlledProbe,
    *,
    case_id: str,
    mode: ResearchMode,
    max_probes: int = 3,
    planner: Planner | None = None,
    public_source: str = "",
    functional_requirement: str = "",
) -> ProbeGraphState:
    """Execute bounded shared tool access; no candidate is generated or scored.

    D uses a reproducible fixed sequence. E must supply an explicitly
    authorized planner adapter; passing no planner is an error rather than an
    implicit API call. Results are returned as a public evidence projection.
    """
    if mode not in ("D", "E"):
        raise ValueError("unknown HardSet research mode")
    if not isinstance(max_probes, int) or isinstance(max_probes, bool) or not 1 <= max_probes <= 15:
        raise ValueError("invalid shared probe budget")
    if mode == "E" and planner is None:
        raise ValueError("Agent planner adapter is required")
    if mode == "D" and planner is not None:
        raise ValueError("deterministic baseline does not accept an LLM planner")

    def plan(state: ProbeGraphState) -> dict[str, Any]:
        observations = tuple(
            FunctionalProbeObservation.model_validate(x) for x in state.get("observations", [])
        )
        steps = state.get("decision_steps", 0) + 1
        if len(observations) >= max_probes:
            chosen: ProbeChoice = FinishProbeAction()
        elif mode == "D":
            seen = {x.size for x in observations}
            chosen = next(
                (FunctionalProbeAction(size=n) for n in RULE_SIZES if n not in seen),
                FinishProbeAction(),
            )
        else:
            assert planner is not None
            snapshot = PlannerSnapshot(
                case_id=case_id,
                remaining_probes=max_probes - len(observations),
                observations=observations,
                public_source=public_source,
                functional_requirement=functional_requirement,
                available_sizes=probe.allowed_sizes,
            )
            proposed = planner(snapshot)
            # Pydantic typed action is mandatory; a model must not pass a
            # dictionary containing paths, commands or arbitrary arguments.
            if not isinstance(proposed, (FunctionalProbeAction, FinishProbeAction)):
                raise ValueError("agent planner returned an unauthorized action")
            chosen = proposed
        return {
            "next_action": chosen.model_dump(mode="json"),
            "route": "done" if isinstance(chosen, FinishProbeAction) else "probe",
            "decision_steps": steps,
        }

    def execute(state: ProbeGraphState) -> dict[str, Any]:
        action = FunctionalProbeAction.model_validate(state["next_action"])
        # This is the SAME authority for both modes. It rejects duplicates,
        # unsupported sizes, over-budget probes and non-registered sources.
        observed = probe.invoke(action)
        previous = state.get("observations", [])
        return {"observations": [*previous, observed.model_dump(mode="json")]}

    def finish(state: ProbeGraphState) -> dict[str, str]:
        return {"completion_reason": "USER_OR_CONTROLLER_STOPPED"}

    graph = StateGraph(ProbeGraphState)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("finish", finish)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", lambda s: s["route"], {"probe": "execute", "done": "finish"}
    )
    graph.add_edge("execute", "plan")
    graph.add_edge("finish", END)
    return cast(
        ProbeGraphState,
        graph.compile().invoke(
            {"case_id": case_id, "mode": mode, "observations": [], "decision_steps": 0},
            config={"recursion_limit": (max_probes + 3) * 4},
        ),
    )
