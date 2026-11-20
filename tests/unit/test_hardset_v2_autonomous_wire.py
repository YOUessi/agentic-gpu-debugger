"""Real-model protocol tests: audited HardSet diagnosis and patch, no network."""

import difflib
import hashlib
import json

import pytest
from pydantic import SecretStr

from gpu_agent.agent.models import AgentBudget, ProviderError
from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.agent.prompts import HARDSET_V2_REPAIR_PROMPT_VERSION, select_prompt
from gpu_agent.agent.provider import (
    DevelopmentCallPolicy,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
    SDKResult,
)
from gpu_agent.benchmark.hardset_v2_agent_models import HardsetAgentContext, HardsetDiagnosis
from gpu_agent.benchmark.hardset_v2_probe import FunctionalProbeObservation
from gpu_agent.store import RunStore

SOURCE = "float x = a[i];\nfloat y = x;\nout[i] = y;\n"
REPAIRED = "float x = a[i];\nfloat y = x - 1.0f;\nout[i] = y;\n"
DIFF = "".join(
    difflib.unified_diff(
        SOURCE.splitlines(keepends=True),
        REPAIRED.splitlines(keepends=True),
        fromfile="a/kernel.cu",
        tofile="b/kernel.cu",
    )
)
CITATION = "b" * 32


class Port:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def call(self, request):
        self.calls.append(request)
        return SDKResult(value=self.responses[min(len(self.calls) - 1, len(self.responses) - 1)])


def setup_provider(tmp_path, responses):
    store = RunStore(tmp_path / "public")
    run = store.create_run("hardset_real_agent_wire_unit")
    port = Port(responses)
    provider = OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.deepseek.com",
            model="unit-mock",
            api_key=SecretStr("not-a-real-secret"),
            supports_store_false=True,
        ),
        LLMCallGate(AgentBudget(max_llm_calls=10)),
        store,
        run.id,
        port=port,
        call_policy=DevelopmentCallPolicy(max_llm_calls=10),
    )
    return provider, store, run.id, port


def context():
    return HardsetAgentContext(
        case_id="h001",
        source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
        public_source=SOURCE,
        functional_requirement="Compute the correct result on all valid inputs",
        observations=(
            FunctionalProbeObservation(
                case_id="h001",
                size=65,
                outcome="NUMERIC_MISMATCH",
                input_sha256="a" * 64,
                evidence_ref_id=CITATION,
            ),
        ),
    )


def diagnosis():
    return {
        "root_cause": "The implementation omits the needed normalization",
        "suspected_lines": [2],
        "cited_public_probe_ids": [CITATION],
        "repair_strategy": "Correct the full computation for arbitrary input sizes",
        "confidence": "medium",
    }


def test_repair_agent_real_model_transport_carries_public_evidence_and_patch(tmp_path):
    provider, store, run_id, port = setup_provider(
        tmp_path,
        [diagnosis(), {"unified_diff": DIFF}],
    )
    ctx = context()
    result = provider.diagnose_hardset_v2(ctx)
    assert isinstance(result, HardsetDiagnosis)
    diff = provider.propose_patch_hardset_v2(ctx, result, [], [])
    assert diff == DIFF
    assert [r.kind for r in port.calls] == ["diagnose", "patch"]
    assert port.calls[0].payload["hardset_diagnose_contract"] == "hardset-v2"
    assert port.calls[1].payload["hardset_patch_contract"] == "hardset-v2"
    assert "evaluator_secret" not in json.dumps([r.payload for r in port.calls])
    assert "private_inputs" not in json.dumps([r.payload for r in port.calls])
    calls = provider.invocations()
    assert len(calls) == 2
    assert all(c.state == "COMPLETED" for c in calls)
    assert all(c.prompt_version == HARDSET_V2_REPAIR_PROMPT_VERSION for c in calls)
    refs = [r.name for r in store.load(run_id).artifact_refs]
    assert all("hidden" not in name and "private" not in name for name in refs)
    assert sum(name.startswith("provider/") for name in refs) >= 2


def test_model_cannot_forge_probe_citation_or_nonexistent_line(tmp_path):
    bad = {**diagnosis(), "cited_public_probe_ids": ["f" * 32]}
    provider, _, _, port = setup_provider(tmp_path, [bad])
    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        provider.diagnose_hardset_v2(context())
    assert len(port.calls) <= 2

    bad_line = {**diagnosis(), "suspected_lines": [9999]}
    provider, _, _, _ = setup_provider(tmp_path / "second", [bad_line])
    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        provider.diagnose_hardset_v2(context())


def test_model_patch_scope_validator_still_runs_inside_audited_worker(tmp_path):
    provider, _, _, port = setup_provider(tmp_path, [{"unified_diff": DIFF}])
    provider._diff_validator = lambda diff: (_ for _ in ()).throw(ValueError("invalid patch"))
    with pytest.raises(ProviderError):
        provider.propose_patch_hardset_v2(
            context(), HardsetDiagnosis.model_validate(diagnosis()), [], []
        )
    assert len(port.calls) <= 2


def test_new_prompt_variants_are_explicit_and_original_prompts_stay_default():
    original_version, original = select_prompt("diagnose", {"evidence": {}})
    new_version, new = select_prompt("diagnose", {"hardset_diagnose_contract": "hardset-v2"})
    patch_version, patch_instructions = select_prompt(
        "patch", {"hardset_patch_contract": "hardset-v2"}
    )
    assert original_version != new_version
    assert new_version == patch_version == HARDSET_V2_REPAIR_PROMPT_VERSION
    assert "cited_public_probe_ids" in new
    assert "unified_diff" in patch_instructions
    assert "independent" in patch_instructions.lower() or "private" in patch_instructions.lower()
    assert "tool_findings" in original
