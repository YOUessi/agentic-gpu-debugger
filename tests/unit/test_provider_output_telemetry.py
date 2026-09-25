"""Planner wire contract and value-free output telemetry (no network, no SDK)."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

CANARY = "VALUE-CANARY-7f3a"


def test_extra_field_retry_explains_instance_contract_without_relaxing_validation():
    from gpu_agent.agent.provider import (
        PatchOutput,
        _OutputRejected,
        correction_hints,
        correction_text,
        parse_wire_text,
    )

    raw = json.dumps({"unified_diff": "diff\n", "$schema": CANARY})
    with pytest.raises(_OutputRejected) as rejected:
        parse_wire_text("patch", raw, PatchOutput)
    diagnostics = rejected.value.diagnostics
    assert diagnostics.failure_class == "SCHEMA_INVALID"
    assert any(i.loc == "$schema" and i.type == "extra_forbidden" for i in diagnostics.issues)
    feedback = correction_text("patch", correction_hints(diagnostics))
    assert "Remove the unrecognized fields" in feedback
    assert "not a JSON Schema definition" in feedback
    assert CANARY not in feedback
    assert CANARY not in diagnostics.model_dump_json()
    # Repetition is still rejected, not silently normalized to a passing output.
    with pytest.raises(_OutputRejected):
        parse_wire_text("patch", raw, PatchOutput)


def test_extra_field_instruction_does_not_replace_domain_retry():
    from gpu_agent.agent.provider import correction_text

    text = correction_text("patch", ["<patch>: hunk_context_not_found"])
    assert "correct JSON format" in text
    assert "not a JSON Schema definition" not in text


def test_parse_wire_text_classifies_without_leaking_values():
    from gpu_agent.agent.provider import PlannerOutput, _OutputRejected, parse_wire_text

    with pytest.raises(_OutputRejected) as not_json:
        parse_wire_text("plan", "not json " + CANARY, PlannerOutput)
    assert not_json.value.diagnostics.failure_class == "NOT_JSON"
    assert not_json.value.diagnostics.output_chars == len("not json " + CANARY)

    bad = json.dumps({"action": {"action_type": "run_shell", "rationale": CANARY}})
    with pytest.raises(_OutputRejected) as invalid:
        parse_wire_text("plan", bad, PlannerOutput)
    diagnostics = invalid.value.diagnostics
    assert diagnostics.failure_class == "SCHEMA_INVALID"
    assert diagnostics.issues and all(issue.loc for issue in diagnostics.issues)
    assert CANARY not in diagnostics.model_dump_json()
    assert diagnostics.output_sha256 and len(diagnostics.output_sha256) == 64


def test_constraint_names_are_recorded_for_bounded_fields():
    from gpu_agent.agent.provider import PlannerOutput, _OutputRejected, parse_wire_text

    text = json.dumps(
        {"action": {"action_type": "inspect_source", "typed_arguments": {"source_id": "x"}}}
    )
    with pytest.raises(_OutputRejected) as rejected:
        parse_wire_text("plan", text, PlannerOutput)
    issues = rejected.value.diagnostics.issues
    assert any(
        "source_id" in issue.loc and issue.constraint and "pattern" in issue.constraint
        for issue in issues
    )


def test_controller_owned_fields_are_dropped_and_reassigned():
    from gpu_agent.agent.models import PlannerOutput
    from gpu_agent.agent.provider import normalize_wire_value

    raw = {
        "action": {
            "action_id": "not-a-valid-id",
            "budget_snapshot": {"anything": 1},
            "action_type": "retrieve_official_docs",
            "rationale": "r" * 900,
            "typed_arguments": {"query": "q" * 900, "k": 12},
        }
    }
    wire = PlannerOutput.model_validate(normalize_wire_value("plan", raw))
    action = wire.to_action_output("c" * 32).action
    assert action.action_id == "c" * 32
    assert action.budget_snapshot is None
    assert len(action.rationale) == 300
    assert len(action.typed_arguments.query) == 500 and action.typed_arguments.k == 5


def test_planner_schema_exposes_only_supported_actions_and_no_controller_fields():
    from gpu_agent.agent.models import PlannerOutput
    from gpu_agent.agent.policy import SUPPORTED

    schema = json.dumps(PlannerOutput.model_json_schema())
    assert "action_id" not in schema and "budget_snapshot" not in schema
    for unsupported in ("run_program", "inspect_environment", "request_more_evidence"):
        assert unsupported not in schema
    for supported in SUPPORTED:
        assert supported in schema


def _deepseek_client(texts):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        text = texts.pop(0)
        envelope = {
            "id": "resp_ds",
            "model": "deepseek-test",
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        }
        return SimpleNamespace(content=json.dumps(envelope), request_id="req_ds")

    def factory(**_kwargs):
        return SimpleNamespace(
            responses=SimpleNamespace(with_raw_response=SimpleNamespace(create=create)),
            close=lambda: None,
        )

    return factory, calls


def _worker_request(attempt=0, hints=()):
    from gpu_agent.agent.provider import WorkerRequest

    return WorkerRequest(
        endpoint="https://api.deepseek.com",
        model="deepseek-test",
        api_key=SecretStr("secret-canary"),
        kind="plan",
        payload={"evidence": {}},
        client_request_id="a" * 32,
        timeout_seconds=30,
        attempt=attempt,
        correction_hints=list(hints),
    )


def test_deepseek_worker_reports_schema_locations_and_sends_reduced_schema():
    from gpu_agent.agent.provider import invoke_sdk

    bad = json.dumps({"action": {"action_type": "run_memcheck", "extra": CANARY}})
    factory, calls = _deepseek_client([bad])
    result = invoke_sdk(_worker_request(), factory)
    assert result.error_code == "LLM_INVALID_OUTPUT" and result.value is None
    assert result.diagnostics is not None
    assert result.diagnostics.failure_class == "SCHEMA_INVALID"
    assert any("extra" in issue.loc for issue in result.diagnostics.issues)
    assert CANARY not in result.model_dump_json()
    sent_schema = json.dumps(calls[0]["text"]["format"]["schema"])
    assert "action_id" not in sent_schema and "budget_snapshot" not in sent_schema


def test_deepseek_worker_accepts_minimal_planner_output():
    from gpu_agent.agent.provider import invoke_sdk

    good = json.dumps({"action": {"action_type": "run_memcheck", "rationale": "precheck"}})
    factory, _ = _deepseek_client([good])
    result = invoke_sdk(_worker_request(), factory)
    assert result.error_code is None and result.value is not None
    assert result.diagnostics is not None and result.diagnostics.failure_class is None


def test_retry_instructions_name_rejected_fields():
    from gpu_agent.agent.provider import invoke_sdk

    good = json.dumps({"action": {"action_type": "run_memcheck"}})
    factory, calls = _deepseek_client([good])
    invoke_sdk(_worker_request(attempt=1, hints=["action.extra: extra_forbidden"]), factory)
    assert "action.extra: extra_forbidden" in calls[0]["instructions"]


class _ScriptedPort:
    def __init__(self, results):
        self.results, self.requests = list(results), []

    def call(self, request):
        self.requests.append(request)
        return self.results.pop(0)


def test_call_persists_diagnostics_and_feeds_hints_into_the_single_retry(store):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import (
        OpenAIProviderSettings,
        OpenAIResponsesProvider,
        OutputDiagnostics,
        ResponseMetadata,
        SDKResult,
        Usage,
        ValidationIssue,
    )

    rejected = SDKResult(
        error_code="LLM_INVALID_OUTPUT",
        state="FAILED",
        metadata=ResponseMetadata(response_model="m", usage=Usage(output_tokens=250)),
        diagnostics=OutputDiagnostics(
            failure_class="SCHEMA_INVALID",
            output_chars=900,
            issues=[ValidationIssue(loc="action.rationale", type="string_too_long")],
        ),
    )
    accepted = SDKResult(
        value={"action": {"action_type": "run_memcheck", "rationale": "ok"}},
        metadata=ResponseMetadata(response_model="m", usage=Usage(output_tokens=40)),
    )
    port = _ScriptedPort([rejected, accepted])
    run = store.create_run("telemetry-test")
    provider = OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.openai.com/v1",
            model="m",
            api_key=SecretStr("k"),
            timeout_seconds=120,
        ),
        LLMCallGate(),
        store,
        run.id,
        port=port,
    )
    action = provider.plan(PublicEvidence(), AgentBudget())
    assert action.action_type == "run_memcheck"
    assert port.requests[1].correction_hints == ["action.rationale: string_too_long"]
    failed = [i for i in provider.invocations() if i.state == "FAILED"]
    assert failed and failed[0].output_diagnostics is not None
    assert failed[0].output_diagnostics.issues[0].loc == "action.rationale"
    assert all(60 < request.timeout_seconds <= 120 for request in port.requests)
    assert all(i.request_timeout_seconds == 120 for i in provider.invocations())


def _bounded_provider(store, port, max_llm_calls):
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import (
        DevelopmentCallPolicy,
        OpenAIProviderSettings,
        OpenAIResponsesProvider,
    )

    run = store.create_run("dev-cap-test")
    return OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.openai.com/v1", model="m", api_key=SecretStr("k")
        ),
        LLMCallGate(),
        store,
        run.id,
        port=port,
        call_policy=DevelopmentCallPolicy(max_llm_calls=max_llm_calls),
    )


def test_development_call_limit_counts_format_retries(store):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError, SDKResult

    port = _ScriptedPort([SDKResult(value={"invalid": True})])
    provider = _bounded_provider(store, port, max_llm_calls=1)
    with pytest.raises(ProviderError) as refused:
        provider.plan(PublicEvidence(), AgentBudget())
    assert refused.value.code == "AGENT_BUDGET_EXHAUSTED"
    assert len(port.requests) == len(provider.invocations()) == 1


def test_development_call_limit_records_usage_and_stops_before_next_request(store):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError, ResponseMetadata, SDKResult, Usage

    accepted = SDKResult(
        value={"action": {"action_type": "run_memcheck"}},
        metadata=ResponseMetadata(
            response_model="m", usage=Usage(input_tokens=100, output_tokens=10)
        ),
    )
    port = _ScriptedPort([accepted])
    provider = _bounded_provider(store, port, max_llm_calls=1)
    assert provider.plan(PublicEvidence(), AgentBudget()).action_type == "run_memcheck"
    assert len(port.requests) == 1
    assert provider.invocations()[0].usage.input_tokens == 100
    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        provider.plan(PublicEvidence(), AgentBudget())
    assert len(port.requests) == 1


def test_development_paid_calls_are_refused_on_evaluation_bound_services(oob_service):
    from gpu_agent.agent.provider import DevelopmentCallPolicy
    from gpu_agent.contracts import RepositorySnapshot, RunBinding

    service = oob_service[0]
    service._binding = RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="evaluation",
    )
    with pytest.raises(ValueError):
        service.allow_development_paid_calls(DevelopmentCallPolicy(max_llm_calls=40))


_KERNEL = "__global__ void k(float* out, int n) {\n    int i = threadIdx.x;\n    out[i] = 0;\n}\n"


def _patch_result(diff):
    from gpu_agent.agent.provider import ResponseMetadata, SDKResult

    return SDKResult(value={"unified_diff": diff}, metadata=ResponseMetadata(response_model="m"))


def _patch_provider(store, port):
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import OpenAIProviderSettings, OpenAIResponsesProvider

    run = store.create_run("patch-telemetry")
    return OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.openai.com/v1", model="m", api_key=SecretStr("k")
        ),
        LLMCallGate(),
        store,
        run.id,
        port=port,
    )


def test_rejected_patch_records_a_value_free_reason_and_retries_with_it(store):
    from gpu_agent.agent.models import DiagnosisResult, PublicSource

    stale = (
        "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -3 +3 @@\n-    out[i] = 1; // "
        + CANARY
        + "\n+    if (i < n) out[i] = 1;\n"
    )
    good = (
        "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -3 +3 @@\n-    out[i] = 0;\n"
        "+    if (i < n) out[i] = 0;\n"
    )
    port = _ScriptedPort([_patch_result(stale), _patch_result(good)])
    provider = _patch_provider(store, port)
    diff = provider.propose_patch(
        PublicSource(source_id="a" * 32, content=_KERNEL), DiagnosisResult.inconclusive("T")
    )
    assert "if (i < n)" in diff
    failed = [i for i in provider.invocations() if i.state == "FAILED"]
    assert failed[0].output_diagnostics.failure_class == "DOMAIN_REJECTED"
    assert [(x.loc, x.type) for x in failed[0].output_diagnostics.issues] == [
        ("<patch>", "hunk_context_not_found")
    ]
    assert CANARY not in failed[0].model_dump_json()
    assert port.requests[1].correction_hints == ["<patch>: hunk_context_not_found"]


_HEAD = "--- a/kernel.cu\n+++ b/kernel.cu\n"


@pytest.mark.parametrize(
    "body,code",
    [
        # case_0006 retry: JSON was valid but a hunk line broke a rule; name which one.
        ("@@ -3 +3 @@\n-    out[i] = 0;\n+    x;", "diff_missing_final_newline"),
        ("@@ -2,2 +2,2 @@\n     int i = threadIdx.x;\n\n", "hunk_blank_line_unprefixed"),
        (
            "@@ -3 +3 @@\n-    out[i] = 0;\n+    x;\n\\ No newline at end of file\n",
            "no_newline_marker",
        ),
        ("@@ -3 +3 @@\n*    out[i] = 0;\n", "hunk_line_no_prefix"),
    ],
)
def test_malformed_hunk_lines_get_distinct_reason_codes(body, code):
    from gpu_agent.patching import normalize_unified_diff_offsets, patch_rejection_code

    with pytest.raises(ValueError) as rejected:
        normalize_unified_diff_offsets(_KERNEL, _HEAD + body)
    assert patch_rejection_code(rejected.value) == code
    assert patch_rejection_code(ValueError("anything else " + CANARY)) == "patch_invalid"


def test_every_patch_reason_code_has_a_repair_hint():
    from gpu_agent.agent.provider import PATCH_REPAIR_HINTS
    from gpu_agent.patching import PATCH_REJECTION_CODES

    hinted = {
        "diff_missing_final_newline",
        "hunk_blank_line_unprefixed",
        "no_newline_marker",
        "hunk_line_no_prefix",
    }
    assert hinted <= set(PATCH_REJECTION_CODES.values()) and hinted <= set(PATCH_REPAIR_HINTS)


def test_domain_rejection_retry_keeps_the_json_envelope():
    """A content rejection must not instruct the model to replace its JSON envelope."""
    from gpu_agent.agent.provider import correction_text

    domain = correction_text("patch", ["<patch>: hunk_context_not_found"])
    assert "correct its format" not in domain
    assert '{"unified_diff": "..."}' in domain and "copied exactly" in domain
    schema = correction_text("plan", ["action.extra: extra_forbidden"])
    assert "correct its format" in schema and "action.extra: extra_forbidden" in schema


def test_worker_sends_domain_retry_text_for_patch(monkeypatch):
    from gpu_agent.agent.provider import WorkerRequest, invoke_sdk

    factory, calls = _deepseek_client([json.dumps({"unified_diff": "--- a/kernel.cu\n"})])
    request = WorkerRequest(
        endpoint="https://api.deepseek.com",
        model="deepseek-test",
        api_key=SecretStr("secret-canary"),
        kind="patch",
        payload={"public_source": {}},
        client_request_id="b" * 32,
        timeout_seconds=30,
        attempt=1,
        correction_hints=["<patch>: hunk_line_invalid"],
    )
    invoke_sdk(request, factory)
    assert "correct JSON format" in calls[0]["instructions"]
    assert "including the last line" in calls[0]["instructions"]


@pytest.mark.parametrize("reason", [CANARY, "include_changed " + CANARY, [CANARY], None])
def test_domain_reason_only_accepts_fixed_codes(reason):
    from gpu_agent.agent.provider import _domain_reason

    assert _domain_reason("patch", reason) == "patch_invalid"
    assert _domain_reason("diagnose", reason) == "diagnose_policy_rejected"
    assert _domain_reason("plan", reason) == "plan_policy_rejected"
    assert _domain_reason("diagnose", "include_changed") == "diagnose_policy_rejected"


def test_domain_retry_sanitizes_unknown_codes_even_in_mixed_hints():
    from gpu_agent.agent.provider import correction_text

    for hints in (["<patch>: " + CANARY], ["<patch>: " + CANARY, "unified_diff: too_short"]):
        text = correction_text("patch", hints)
        assert CANARY not in text
        assert "patch_invalid" in text
    diagnosis = correction_text("diagnose", ["<diagnose>: include_changed"])
    assert "diagnose_policy_rejected" in diagnosis
    assert "unified_diff" not in diagnosis
    assert "include_changed" not in diagnosis


def test_unknown_validator_error_never_reaches_telemetry_or_retry(store):
    from gpu_agent.agent.models import DiagnosisResult, ProviderError, PublicSource

    good = (
        "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -3 +3 @@\n-    out[i] = 0;\n"
        "+    if (i < n) out[i] = 0;\n"
    )
    port = _ScriptedPort([_patch_result(good), _patch_result(good)])
    provider = _patch_provider(store, port)

    def reject(_diff):
        raise ValueError(CANARY)

    provider._diff_validator = reject
    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        provider.propose_patch(
            PublicSource(source_id="a" * 32, content=_KERNEL), DiagnosisResult.inconclusive("T")
        )
    assert len(port.requests) == 2
    assert port.requests[1].correction_hints == ["<patch>: patch_invalid"]
    for call in provider.invocations():
        assert CANARY not in call.model_dump_json()
        assert call.output_diagnostics.issues[0].type == "patch_invalid"


@pytest.mark.parametrize(
    "text,code",
    [
        ("", "empty_output"),
        ('```json\n{"unified_diff": "x"\n```', "code_fence"),
        ('```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```', "code_fence"),
        ("--- a/kernel.cu\n+++ b/kernel.cu\n", "raw_diff"),
        ('Here is the fix: {"unified_diff": "x"}', "prose_before_json"),
        ('{"unified_diff": "--- a/kernel.cu\\n', "truncated_json"),
        ('{"unified_diff": "a\\qb"}', "invalid_escape"),
        ('{"unified_diff": "a\nb"}', "control_character"),
        ('{"unified_diff": "x"} {"unified_diff": "y"}', "extra_data"),
    ],
)
def test_unparsable_output_is_classified_by_shape_only(text, code):
    from gpu_agent.agent.models import PatchOutput
    from gpu_agent.agent.provider import _OutputRejected, parse_wire_text

    with pytest.raises(_OutputRejected) as rejected:
        parse_wire_text("patch", text + CANARY if code == "extra_data" else text, PatchOutput)
    diagnostics = rejected.value.diagnostics
    assert diagnostics.failure_class == "NOT_JSON"
    assert [(i.loc, i.type) for i in diagnostics.issues] == [("<json>", code)]
    assert CANARY not in diagnostics.model_dump_json()


def test_not_json_retry_asks_for_the_json_envelope_with_escaped_diff():
    from gpu_agent.agent.provider import correction_text

    text = correction_text("patch", ["<json>: raw_diff"])
    assert "not parsable JSON (raw_diff)" in text
    assert "no bare diff" in text and "\\n" in text
    assert "correct its format" not in text


def test_single_json_fence_is_removed_and_recorded():
    """case_0006/case_0015: a complete ```json fence around one object is unwrapped."""
    from gpu_agent.agent.models import PatchOutput
    from gpu_agent.agent.provider import parse_wire_text

    inner = json.dumps({"unified_diff": "--- a/kernel.cu\n+++ b/kernel.cu\n"})
    raw = "```json\n" + inner + "\n```"
    value, diagnostics = parse_wire_text("patch", raw, PatchOutput)
    assert value == json.loads(inner)
    assert diagnostics.normalizations == ["json_code_fence_removed"]
    # The recorded hash is of the raw output as received, not the unwrapped body.
    assert diagnostics.output_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    assert diagnostics.output_chars == len(raw)


def test_missing_final_diff_newline_is_added_and_recorded():
    from gpu_agent.agent.models import PatchOutput
    from gpu_agent.agent.provider import parse_wire_text

    raw = json.dumps({"unified_diff": "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -3 +3 @@\n-x\n+y"})
    value, diagnostics = parse_wire_text("patch", raw, PatchOutput)
    assert value["unified_diff"].endswith("+y\n")
    assert diagnostics.normalizations == ["diff_final_newline_added"]


def test_normalized_patch_still_faces_every_content_check(store):
    """Wrapper repair never makes a mismatched diff acceptable."""
    from gpu_agent.agent.models import DiagnosisResult, PublicSource
    from gpu_agent.agent.provider import ProviderError

    stale = "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -3 +3 @@\n-    out[i] = 9;\n+    x;"
    port = _ScriptedPort([_patch_result(stale), _patch_result(stale)])
    provider = _patch_provider(store, port)
    with pytest.raises(ProviderError, match="LLM_INVALID_OUTPUT"):
        provider.propose_patch(
            PublicSource(source_id="a" * 32, content=_KERNEL), DiagnosisResult.inconclusive("T")
        )


def test_plan_output_is_never_newline_repaired():
    from gpu_agent.agent.provider import normalize_output_value

    value = {"unified_diff": "x"}
    assert normalize_output_value("plan", value) == (value, [])
