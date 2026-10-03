"""Tiny, explicitly opted-in planner smoke test against the configured real provider.

It performs at most GPU_AGENT_PLANNER_SMOKE_CALLS planner requests (default 3, hard cap
10) on the public case_0001 source, with no GPU, container, diagnosis or patch step.
Every outcome, including value-free output diagnostics, is printed so a failure names
the rejected schema fields. Run with:

    GPU_AGENT_PLANNER_SMOKE_CALLS=3 python -I -m pytest \
        tests/integration/test_live_planner_smoke.py -m live_llm -s --require-live
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live_llm

CASE = Path(__file__).resolve().parents[2] / "benchmarks/public/case_0001/public_input/kernel.cu"


def test_real_planner_outputs_validate_or_report_fields(store):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence, PublicSource
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import (
        DevelopmentCallPolicy,
        OpenAIProviderSettings,
        OpenAIResponsesProvider,
        ProviderError,
    )
    from gpu_agent.contracts import new_id

    settings = OpenAIProviderSettings.from_environment()
    if not (settings.api_key and settings.endpoint and settings.model):
        pytest.skip("OPENAI_BASE_URL, OPENAI_MODEL and OPENAI_API_KEY are not configured")
    requested = os.environ.get("GPU_AGENT_PLANNER_SMOKE_CALLS")
    if requested is None:
        pytest.skip("set GPU_AGENT_PLANNER_SMOKE_CALLS to opt in to paid planner calls")
    calls = min(max(int(requested), 1), 10)

    evidence = PublicEvidence(
        sources=[PublicSource(source_id=new_id(), content=CASE.read_text(encoding="utf-8"))]
    )
    accepted = 0
    for index in range(calls):
        run = store.create_run("planner-smoke")
        provider = OpenAIResponsesProvider(
            settings,
            LLMCallGate(),
            store,
            run.id,
            call_policy=DevelopmentCallPolicy(max_llm_calls=1),
        )
        try:
            action = provider.plan(evidence, AgentBudget())
            accepted += 1
            outcome = f"accepted {action.action_type}"
        except ProviderError as exc:
            outcome = f"rejected {exc.code}"
        for record in provider.invocations():
            diagnostics = record.output_diagnostics
            print(
                f"[{index}] attempt={record.attempt} state={record.state} "
                f"error={record.error_code} out_tokens="
                f"{record.usage.output_tokens if record.usage else None} "
                f"diagnostics={diagnostics.model_dump() if diagnostics else None}"
            )
        print(f"[{index}] {outcome}")
    assert accepted == calls, f"{calls - accepted}/{calls} planner calls were rejected"
