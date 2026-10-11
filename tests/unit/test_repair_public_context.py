"""V3 prior hypotheses remain source-scoped context outside current evidence."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gpu_agent.agent.models import (
    AgentBudget,
    DiagnosisResult,
    EvidenceClaim,
    MemcheckAction,
    PublicEvidence,
    PublicFinding,
    PublicSource,
)
from gpu_agent.agent.policy import LLMCallGate, validate_diagnosis
from gpu_agent.agent.prompts import PROMPTS
from gpu_agent.agent.provider import (
    FakeProvider,
    OpenAIProviderSettings,
    OpenAIResponsesProvider,
    invoke_sdk,
)
from gpu_agent.execution.models import ExecutionModel, SourceLocation

ORIGINAL = (
    "__global__ void kernel(float *out, int n) {\n    int i = threadIdx.x;\n    out[i] = 0;\n}\n"
)
CANDIDATE = (
    "__global__ void kernel(float *out, int n) {\n    int i = threadIdx.x;\n"
    "    // first repair\n    if (i < n) out[i] = 1;\n}\n"
)
DIFF = (
    "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -3 +3 @@\n-    out[i] = 0;\n+    if (i < n) out[i] = 2;\n"
)
ORIGINAL_HASH = hashlib.sha256(ORIGINAL.encode()).hexdigest()
CANDIDATE_HASH = hashlib.sha256(CANDIDATE.encode()).hexdigest()
LEGACY_PROMPT_HASHES = {
    "plan": "218ea042704e935e7cbde3f79dc8240d6a5c9253cce00017b36d4b40ba891fed",
    "diagnose": "20d3a20925bbedbf1eeae381ca79dfaf5566e1cf26188d960f688ee0d4b3afb0",
    "patch": "28dcf39d1f7edbd6b9671ab46e447eaf3e4a18968ae4e72a9a2e7ecd2a739a2c",
}


def _previous_diagnosis():
    return DiagnosisResult(
        diagnostic_outcome="DIAGNOSED",
        root_cause="Previous source has an unguarded write.",
        source_locations=[SourceLocation(path="kernel.cu", line=3)],
        observed_facts=[EvidenceClaim(text="Previous source", citation_ids=["a" * 32])],
        tool_findings=[EvidenceClaim(text="Previous finding", citation_ids=["b" * 32])],
        documentation_evidence=[
            EvidenceClaim(text="Previously retrieved documentation", citation_ids=["previous-doc"])
        ],
    )


def _context():
    return {
        "repair_round": 1,
        "original_source_sha256": ORIGINAL_HASH,
        "candidate_source_sha256": CANDIDATE_HASH,
        "previous_diagnosis_source_sha256": ORIGINAL_HASH,
        "previous_diagnosis": _previous_diagnosis().model_dump(mode="json"),
        "public_checks": {"public_output": "FAILED"},
        "public_feedback": [{"code": "PUBLIC_OUTPUT_MISMATCH"}],
    }


def _evidence(*, repair=False):
    evidence = PublicEvidence(
        sources=[PublicSource(source_id="c" * 32, content=CANDIDATE)],
        observed_facts=[EvidenceClaim(text="Current candidate source", citation_ids=["c" * 32])],
        tool_findings=[
            PublicFinding(
                artifact_id="d" * 32,
                category="Current candidate finding",
                source_location=SourceLocation(path="kernel.cu", line=4),
            )
        ],
        sanitizer_outcomes={"memcheck": "FINDING"},
    )
    if repair:
        return PublicEvidence.model_validate(
            {**evidence.model_dump(mode="json"), "repair_context": _context()}
        )
    return evidence


def _current_diagnosis():
    return DiagnosisResult(
        diagnostic_outcome="DIAGNOSED",
        source_locations=[SourceLocation(path="kernel.cu", line=4)],
        observed_facts=[EvidenceClaim(text="Current source", citation_ids=["c" * 32])],
        tool_findings=[EvidenceClaim(text="Current finding", citation_ids=["d" * 32])],
    )


@pytest.mark.parametrize("explicit_none", [False, True])
def test_legacy_evidence_serializes_without_an_added_null_field(explicit_none):
    fields = {"repair_context": None} if explicit_none else {}
    evidence = PublicEvidence(
        sources=[PublicSource(source_id="c" * 32, content="one\ntwo\n")], **fields
    )
    expected = (
        '{"sources":[{"source_id":"cccccccccccccccccccccccccccccccc","path":"kernel.cu",'
        '"content":"one\\ntwo\\n","functional_requirement":null}],"observed_facts":[],'
        '"tool_findings":[],"documentation":[],"sanitizer_outcomes":{},"limitations":[]}'
    )
    assert evidence.model_dump_json() == expected
    assert evidence.model_dump(mode="json") == json.loads(expected)
    assert "repair_context" not in evidence.model_dump()

    class Envelope(ExecutionModel):
        evidence: PublicEvidence

    assert Envelope(evidence=evidence).model_dump_json() == '{"evidence":' + expected + "}"


def test_v3_context_reaches_plan_and_diagnose_without_entering_current_evidence():
    evidence = _evidence(repair=True)
    provider = FakeProvider([MemcheckAction()], _current_diagnosis(), DIFF)
    provider.plan(evidence, AgentBudget())
    provider.diagnose(evidence)
    for payload in provider.inputs:
        data = payload["evidence"]
        assert data["sources"][0]["content"] == CANDIDATE
        assert data["observed_facts"] == [
            {"text": "Current candidate source", "citation_ids": ["c" * 32]}
        ]
        assert [finding["artifact_id"] for finding in data["tool_findings"]] == ["d" * 32]
        assert data["documentation"] == []
        assert data["repair_context"] == {
            "version": "public-repair-v3",
            **_context(),
            "diagnostic_target": "current_candidate",
            "public_functional_failure": False,
        }


@pytest.mark.parametrize(
    "field,value",
    [
        ("repair_round", 0),
        ("repair_round", 21),
        ("original_source_sha256", "a" * 63),
        ("candidate_source_sha256", "g" * 64),
        ("previous_diagnosis_source_sha256", "A" * 64),
        ("diagnostic_target", "original_source"),
        ("version", "public-repair-v2"),
        ("raw_logs", "unprojected logs"),
    ],
)
def test_context_rejects_invalid_attribution_and_nonpublic_fields(field, value):
    with pytest.raises(ValidationError) as rejected:
        PublicEvidence.model_validate({"repair_context": {**_context(), field: value}})
    assert ("repair_context", field) in [issue["loc"] for issue in rejected.value.errors()]


def test_controller_functional_failure_survives_context_serialization():
    evidence = PublicEvidence.model_validate(
        {"repair_context": {**_context(), "public_functional_failure": True}}
    )
    assert json.loads(evidence.model_dump_json())["repair_context"]["public_functional_failure"]


@pytest.mark.parametrize("layer", ["observed_facts", "tool_findings", "documentation_evidence"])
def test_previous_diagnosis_citations_are_not_legal_current_evidence(layer):
    evidence = _evidence(repair=True)
    current = _current_diagnosis()
    assert validate_diagnosis(current, evidence)
    forged = current.model_copy(update={layer: getattr(_previous_diagnosis(), layer)})
    assert not validate_diagnosis(forged, evidence)
    assert not validate_diagnosis(
        current.model_copy(update={"source_locations": _previous_diagnosis().source_locations}),
        evidence,
    )


class _SDKPort:
    """Replace only the external SDK transport; keep provider and worker behavior real."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def _response(self, **kwargs):
        self.calls.append(kwargs)
        value = self.values.pop(0)
        envelope = {
            "id": "response-test",
            "model": "model-test",
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}
            ],
        }
        return SimpleNamespace(
            content=json.dumps(envelope),
            request_id="request-test",
            parse=lambda: SimpleNamespace(output_parsed=value),
        )

    def _client(self, **kwargs):
        return SimpleNamespace(
            responses=SimpleNamespace(
                with_raw_response=SimpleNamespace(parse=self._response, create=self._response)
            ),
            close=lambda: None,
        )

    def call(self, request):
        return invoke_sdk(request, client_factory=self._client)


def _provider(store, port, endpoint):
    run = store.create_run("public-repair-context-test")
    return OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint=endpoint,
            model="model-test",
            api_key="test-key",
            supports_store_false=True,
        ),
        LLMCallGate(),
        store,
        run.id,
        port=port,
    )


@pytest.mark.parametrize("endpoint", ["https://api.openai.com/v1", "https://api.deepseek.com"])
@pytest.mark.parametrize(
    "kind,contract",
    [
        ("plan", None),
        ("diagnose", None),
        ("patch", None),
        ("patch", "public-repair-v2"),
        ("plan", "public-repair-v3"),
        ("diagnose", "public-repair-v3"),
        ("patch", "public-repair-v3"),
    ],
)
def test_sdk_instructions_and_invocation_versions_select_v3_together(
    store, endpoint, kind, contract
):
    is_v3 = contract == "public-repair-v3"
    value = {
        "plan": {"action": {"action_type": "run_memcheck"}},
        "diagnose": _current_diagnosis().model_dump(mode="json"),
        "patch": {"unified_diff": DIFF},
    }[kind]
    port = _SDKPort([value])
    provider = _provider(store, port, endpoint)
    if kind == "plan":
        assert provider.plan(_evidence(repair=is_v3), AgentBudget()).action_type == "run_memcheck"
    elif kind == "diagnose":
        assert provider.diagnose(_evidence(repair=is_v3)) == _current_diagnosis()
    else:
        source = PublicSource(source_id="a" * 32, content=ORIGINAL)
        if contract is None:
            result = provider.propose_patch(source, _current_diagnosis())
        else:
            result = provider.revise_patch(
                source,
                _current_diagnosis(),
                {
                    "contract": contract,
                    "previous_candidate_source": CANDIDATE,
                    "diagnosis_source_sha256": CANDIDATE_HASH,
                    "original_source_sha256": ORIGINAL_HASH,
                    **({"diagnosis_source": CANDIDATE} if is_v3 else {}),
                },
            )
        assert result == DIFF
    assert len(port.calls) == 1
    sent = port.calls[0]
    payload = json.loads(sent["input"])["untrusted_data"]
    instructions = sent["instructions"]
    invocations = provider.invocations()
    assert len(invocations) == 1 and invocations[0].state == "COMPLETED"
    if is_v3:
        assert invocations[0].prompt_version == "public-repair-v3-2026-10-09-v5"
        assert instructions.startswith(PROMPTS[kind]) and instructions != PROMPTS[kind]
        assert "current_candidate" in instructions
        assert "previous_diagnosis_source_sha256" in instructions
        assert "hypothesis" in instructions
        assert "diagnosis_source_sha256" in instructions
        assert "previous_candidate_source" in instructions
        if kind != "patch":
            assert (
                payload["evidence"]["repair_context"]["candidate_source_sha256"] == CANDIDATE_HASH
            )
    else:
        assert invocations[0].prompt_version == "m3-2026-10-01-v12"
        assert hashlib.sha256(instructions.encode()).hexdigest() == LEGACY_PROMPT_HASHES[kind]
        assert "public-repair-v3" not in instructions
        if kind != "patch":
            assert "repair_context" not in payload["evidence"]
    if kind == "patch":
        assert payload["public_source"]["content"] == ORIGINAL
        assert payload["diagnosis"]["source_locations"][0]["line"] == 4


@pytest.mark.parametrize("endpoint", ["https://api.openai.com/v1", "https://api.deepseek.com"])
def test_patch_prompt_keeps_earlier_diagnosis_scoped_when_latest_candidate_differs(store, endpoint):
    latest_candidate = CANDIDATE.replace(
        "    // first repair\n", "    // second repair\n    // line locations changed\n"
    )
    port = _SDKPort([{"unified_diff": DIFF}])
    provider = _provider(store, port, endpoint)
    result = provider.revise_patch(
        PublicSource(source_id="a" * 32, content=ORIGINAL),
        _current_diagnosis(),
        {
            "contract": "public-repair-v3",
            "diagnosis_source": CANDIDATE,
            "diagnosis_source_sha256": CANDIDATE_HASH,
            "previous_candidate_source": latest_candidate,
            "candidate_source_sha256": hashlib.sha256(latest_candidate.encode()).hexdigest(),
            "original_source_sha256": ORIGINAL_HASH,
            "public_checks": {"public_output": "FAILED"},
        },
    )
    assert result == DIFF
    sent = port.calls[0]
    payload = json.loads(sent["input"])["untrusted_data"]
    feedback = payload["public_repair_feedback"]
    assert payload["diagnosis"]["source_locations"][0]["line"] == 4
    assert feedback["diagnosis_source"].splitlines()[3] == "    if (i < n) out[i] = 1;"
    assert feedback["previous_candidate_source"].splitlines()[3] == "    // line locations changed"
    assert feedback["diagnosis_source_sha256"] != feedback["candidate_source_sha256"]
    assert payload["public_source"]["content"] == ORIGINAL
    instructions = " ".join(sent["instructions"].split())
    assert (
        "Interpret diagnosis line numbers and source locations using "
        "public_repair_feedback.diagnosis_source, whose SHA256 must match "
        "public_repair_feedback.diagnosis_source_sha256."
    ) in instructions
    assert (
        "public_repair_feedback.previous_candidate_source belongs only to the most recent "
        "failed public checks and may differ from diagnosis_source."
    ) in instructions
    assert (
        "For patch, public_source is always the ORIGINAL source and the complete replacement "
        "diff must apply to it."
    ) in instructions


def test_rejected_old_citations_retry_with_current_evidence_and_v3_telemetry(store):
    port = _SDKPort(
        [
            _previous_diagnosis().model_dump(mode="json"),
            _current_diagnosis().model_dump(mode="json"),
        ]
    )
    provider = _provider(store, port, "https://api.openai.com/v1")
    assert provider.diagnose(_evidence(repair=True)) == _current_diagnosis()
    invocations = provider.invocations()
    assert [call.state for call in invocations] == ["FAILED", "COMPLETED"]
    assert {call.prompt_version for call in invocations} == {"public-repair-v3-2026-10-09-v5"}
    assert invocations[0].output_diagnostics.failure_class == "DOMAIN_REJECTED"
    for call in port.calls:
        evidence = json.loads(call["input"])["untrusted_data"]["evidence"]
        assert evidence["sources"][0]["content"] == CANDIDATE
        assert "previous_diagnosis_source_sha256" in call["instructions"]
    assert "Previous source has an unguarded write" not in invocations[0].model_dump_json()
