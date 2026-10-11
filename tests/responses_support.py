"""Mock Responses provider wiring shared by native evaluation tests.

Every evaluation mode calls the same provider (docs/mode-contract.md), so native evaluation
fixtures bind a zero-cost MockResponsesProvider and its provider policy for all modes.
"""

import hashlib
import json


def _configure_responses_provider(
    executor,
    monkeypatch,
    *,
    response_model="eval-model",
    usage=True,
    mock_provider=True,
    full_script=False,
    invalid_plan_after=None,
    invalid_plan_calls=frozenset(),
    uncertain_plan_calls=frozenset(),
    invalid_kinds=frozenset(),
):
    from pydantic import SecretStr

    import gpu_agent.service as service_module
    from gpu_agent.agent.models import DiagnosisResult, EvidenceClaim, InconclusiveAction
    from gpu_agent.agent.prompts import PROMPT_VERSION
    from gpu_agent.agent.provider import (
        MockResponsesProvider,
        OpenAIProviderSettings,
        OpenAIResponsesProvider,
        ResponseMetadata,
        SDKResult,
        Usage,
    )
    from gpu_agent.benchmark.evaluation import PricingAttestation
    from gpu_agent.execution.models import SourceLocation

    # The scripted fake is stashed once; later reconfiguration reuses the same script.
    scripted = getattr(executor, "scripted_provider", None) or executor.service._provider
    executor.scripted_provider = scripted

    class Port:
        def __init__(self):
            self.plan_index = 0
            self.physical_plan_index = 0

        def call(self, request):
            physical_plan_index = self.physical_plan_index
            if request.kind == "plan":
                self.physical_plan_index += 1
            if request.kind in invalid_kinds:
                return SDKResult(
                    value={"invalid": True},
                    metadata=ResponseMetadata(
                        response_id="response-invalid",
                        provider_request_id="request-invalid",
                        response_model=response_model,
                        usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
                        http_status=200,
                    ),
                )
            if request.kind == "plan" and physical_plan_index in uncertain_plan_calls:
                return SDKResult(error_code="LLM_CONNECTION_ERROR", state="UNCERTAIN")
            if (
                full_script
                and request.kind == "plan"
                and (
                    physical_plan_index in invalid_plan_calls
                    or (invalid_plan_after is not None and self.plan_index >= invalid_plan_after)
                )
            ):
                value = {"invalid": True}
            elif full_script and request.kind == "plan":
                value = {"action": scripted.actions[self.plan_index].model_dump(mode="json")}
                self.plan_index += 1
            elif full_script and request.kind == "diagnose":
                evidence = request.payload["evidence"]
                if getattr(scripted, "force_limitation", False):
                    value = DiagnosisResult.inconclusive(scripted.limitation_canary).model_dump(
                        mode="json"
                    )
                else:
                    value = DiagnosisResult(
                        diagnostic_outcome="DIAGNOSED",
                        failure_family="out_of_bounds",
                        root_cause=(
                            "The thread index can exceed the input length."
                            + getattr(scripted, "response_canary", "")
                        ),
                        source_locations=[SourceLocation(path="kernel.cu", line=9)],
                        observed_facts=[
                            EvidenceClaim.model_validate(item)
                            for item in evidence["observed_facts"]
                        ],
                        tool_findings=[
                            EvidenceClaim(
                                text=item["category"],
                                citation_ids=[item["artifact_id"]],
                            )
                            for item in evidence["tool_findings"]
                        ],
                        documentation_evidence=[
                            EvidenceClaim(text=item["text"], citation_ids=[item["chunk_id"]])
                            for item in evidence["documentation"]
                        ],
                        model_inferences=[
                            "An index guard may prevent the reported write."
                            + getattr(scripted, "path_canary", "")
                        ],
                        recommended_change="Guard the write with i < n.",
                        confidence_label="high",
                        limitations=(
                            [scripted.limitation_canary]
                            if getattr(scripted, "limitation_canary", "")
                            else []
                        ),
                    ).model_dump(mode="json")
            elif full_script and request.kind == "patch":
                value = {"unified_diff": scripted.diff}
            else:
                value = {"action": InconclusiveAction().model_dump(mode="json")}
            return SDKResult(
                value=value,
                metadata=ResponseMetadata(
                    response_id="response-1",
                    provider_request_id="request-1",
                    response_model=response_model,
                    usage=(
                        Usage(input_tokens=3, output_tokens=2, total_tokens=5) if usage else None
                    ),
                    http_status=200,
                ),
            )

    settings = OpenAIProviderSettings(
        endpoint="https://api.openai.com/v1",
        model="eval-model",
        api_key=SecretStr("fixture-only"),
        supports_store_false=True,
    )
    provider_name = "mock-responses" if mock_provider else "openai-responses"
    rate_card = PricingAttestation._for_test(
        provider_name,
        "eval-model",
        executor.service.binding.repository.commit,
        "0" * 64,
    )
    policy = {
        "schema_version": 1,
        "provider": provider_name,
        "endpoint_host": "api.openai.com",
        "configured_model": "eval-model",
        "allowed_response_models": ["eval-model"],
        "prompt_version": PROMPT_VERSION,
        "pricing_hash": rate_card.rate_card_hash,
        "store_false_required": True,
    }
    policy_hash = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    binding = executor.service.binding.model_copy(update={"model_config_hash": policy_hash})
    executor.service._binding = binding
    executor.service._pricing_attestation = PricingAttestation._for_test(
        provider_name, "eval-model", binding.repository.commit, policy_hash
    )
    executor.service._provider = None
    holdout_service = getattr(executor, "holdout_service", None)
    if holdout_service is not None:
        holdout_service._binding = binding
        holdout_service._pricing_attestation = executor.service._pricing_attestation
        holdout_service._provider = None
    monkeypatch.setattr(service_module.OpenAIProviderSettings, "from_environment", lambda: settings)
    monkeypatch.setattr(
        service_module,
        "OpenAIResponsesProvider",
        lambda settings, gate, store, run_id, **kwargs: (
            MockResponsesProvider if mock_provider else OpenAIResponsesProvider
        )(settings, gate, store, run_id, port=Port(), **kwargs),
    )
    return binding, policy


def provider_artifacts(store):
    """Every physical provider request leaves an artifact; replay/resume must add none."""
    return sorted(
        ref.name
        for run_dir in store.root.iterdir()
        if run_dir.is_dir() and len(run_dir.name) == 32
        for ref in store.load(run_dir.name).artifact_refs
        if ref.name.startswith("provider/")
    )
