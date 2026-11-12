"""Top-level public Repair graph.

Separate the public diagnosis/repair phase from strict final verification.
This graph never receives evaluator credentials, private test inputs or a
verification backend. The independent service retains final-verdict authority.
No checkpoint saver is configured until side effects are replay-safe.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from gpu_agent.benchmark.evaluation import EvaluationMode
from gpu_agent.contracts import RunManifest
from gpu_agent.repair import RepairPolicy
from gpu_agent.execution.models import SanitizerTool
from gpu_agent.verification.models import VerificationResult

if TYPE_CHECKING:
    from gpu_agent.service import ApplicationService


class PublicRepairState(TypedDict, total=False):
    run_id: str
    public_checks_passed: bool
    stop_reason: str


def run_langgraph_public_repair(
    service: "ApplicationService",
    source: Path,
    policy: RepairPolicy,
    mode: EvaluationMode,
) -> tuple[RunManifest, VerificationResult | None]:
    """Preserve the old service contract while making verification a graph boundary."""
    manifest: RunManifest | None = None
    verified: VerificationResult | None = None

    def investigate_and_repair(state: PublicRepairState) -> dict[str, str]:
        nonlocal manifest
        # _diagnose performs controlled build, LangGraph D/E evidence acquisition,
        # bounded candidate revision and public checks with the original scope guards.
        manifest = service._diagnose(
            source,
            mode=mode,
            required_tools=(SanitizerTool.MEMCHECK,),
            expected_source_hash=None,
            evaluation_unit=None,
            repair_policy=policy,
        )
        return {"run_id": manifest.id}

    def public_gate(state: PublicRepairState) -> dict[str, object]:
        run = service.store.load(state["run_id"])
        summaries = [r for r in run.artifact_refs if r.name == "repair/summary.json"]
        reason = (
            json.loads(service.store.read(summaries[-1]))["stop_reason"]
            if summaries
            else "NO_PUBLIC_REPAIR"
        )
        return {
            "public_checks_passed": reason == "PUBLIC_CHECKS_PASSED",
            "stop_reason": reason,
        }

    def should_verify(state: PublicRepairState) -> Literal["verify", "end"]:
        # Public check is the ONLY edge into the private evaluator.
        return "verify" if state["public_checks_passed"] else "end"

    def strict_verify(state: PublicRepairState) -> dict[str, str]:
        nonlocal verified
        # The evaluator capability never enters graph state or Agent tools.
        verified = service.verify_exact(state["run_id"], strict=True)[0]
        return {}

    graph = StateGraph(PublicRepairState)
    graph.add_node("investigate_and_repair", investigate_and_repair)
    graph.add_node("public_gate", public_gate)
    graph.add_node("strict_verify", strict_verify)
    graph.add_edge(START, "investigate_and_repair")
    graph.add_edge("investigate_and_repair", "public_gate")
    graph.add_conditional_edges(
        "public_gate",
        should_verify,
        {"verify": "strict_verify", "end": END},
    )
    graph.add_edge("strict_verify", END)
    graph.compile().invoke({}, config={"recursion_limit": 10})
    if manifest is None:
        raise RuntimeError("Repair graph ended without creating a diagnosis run")
    return manifest, verified
