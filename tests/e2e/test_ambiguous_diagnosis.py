"""Unknown symptoms gain evidence through the same bounded typed action set."""


def test_adaptive_tool_action_requires_observed_clean_memcheck():
    from gpu_agent.agent.models import AgentBudget, PublicEvidence, RacecheckAction
    from gpu_agent.agent.policy import decide_action
    from gpu_agent.contracts import CurrentPhase
    from gpu_agent.execution.models import SanitizerTool

    action = RacecheckAction()
    missing = decide_action(action, PublicEvidence(), AgentBudget(), CurrentPhase.DIAGNOSING, set())
    assert not missing.allowed and missing.reason_codes == ["MEMCHECK_PRECHECK_REQUIRED"]
    evidence = PublicEvidence(sanitizer_outcomes={SanitizerTool.MEMCHECK: "CLEAN"})
    allowed = decide_action(action, evidence, AgentBudget(), CurrentPhase.DIAGNOSING, set())
    assert allowed.allowed


def test_rule_fallback_adapts_to_new_shared_memory_evidence():
    from gpu_agent.agent.models import AgentBudget, PublicEvidence, PublicFinding, PublicSource
    from gpu_agent.agent.rule_router import RuleRouter
    from gpu_agent.execution.models import SanitizerTool
    from gpu_agent.knowledge.models import make_chunk

    router = RuleRouter()
    source = PublicSource(source_id="0" * 32, content="__shared__ float values[32];")
    evidence = PublicEvidence(sources=[source])
    assert router.next_action(evidence, AgentBudget()).action_type == "run_memcheck"

    evidence = evidence.model_copy(update={"sanitizer_outcomes": {SanitizerTool.MEMCHECK: "CLEAN"}})
    assert (
        router.next_action(evidence, AgentBudget(sanitizer_calls=1)).action_type == "run_racecheck"
    )

    finding = PublicFinding(artifact_id="1" * 32, category="Race reported")
    evidence = evidence.model_copy(update={"tool_findings": [finding]})
    assert (
        router.next_action(evidence, AgentBudget(sanitizer_calls=2)).action_type
        == "retrieve_official_docs"
    )
    chunk = make_chunk(
        source_id="compute-sanitizer",
        document_title="Compute Sanitizer",
        document_version="2025.1",
        section_title="Racecheck",
        source_url="https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html",
        retrieved_at="2026-09-15T00:00:00Z",
        text="Racecheck reports shared-memory hazards.",
        block_ordinal=0,
        compatibility={"cuda": ">=12,<13", "compute_sanitizer": ">=2025,<2026"},
    )
    evidence = evidence.model_copy(update={"documentation": [chunk]})
    assert router.next_action(evidence, AgentBudget()).action_type == "finish_diagnosis"


def test_unknown_source_checks_each_fallback_once_then_stops():
    from gpu_agent.agent.models import AgentBudget, PublicEvidence, PublicSource
    from gpu_agent.agent.rule_router import RuleRouter
    from gpu_agent.execution.models import SanitizerTool

    router = RuleRouter()
    evidence = PublicEvidence(
        sources=[PublicSource(source_id="0" * 32, content="opaque_kernel();")],
        sanitizer_outcomes={SanitizerTool.MEMCHECK: "CLEAN"},
    )
    actions = []
    for calls in range(1, 4):
        action = router.next_action(evidence, AgentBudget(sanitizer_calls=calls))
        actions.append(action.action_type)
        tool = SanitizerTool(action.action_type.removeprefix("run_"))
        evidence = evidence.model_copy(
            update={"sanitizer_outcomes": {**evidence.sanitizer_outcomes, tool: "CLEAN"}}
        )
    assert actions == ["run_initcheck", "run_racecheck", "run_synccheck"]
    assert (
        router.next_action(evidence, AgentBudget(sanitizer_calls=4)).action_type
        == "declare_inconclusive"
    )
