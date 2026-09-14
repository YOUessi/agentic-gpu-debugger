"""Official Responses adapter. One ledger event pair per physical request, no replay."""

import json
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, Protocol, TypeVar
from urllib.parse import urlsplit

import openai
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator

from gpu_agent.agent.models import (
    AgentAction,
    AgentActionOutput,
    AgentBudget,
    DiagnosisResult,
    PatchOutput,
    ProviderError,
    PublicEvidence,
    PublicSource,
)
from gpu_agent.agent.policy import CallKind, LLMCallGate
from gpu_agent.agent.prompts import PROMPT_VERSION, PROMPTS
from gpu_agent.contracts import new_id, now
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.store import RunStore

__all__ = [
    "ProviderError",
    "OpenAIProviderSettings",
    "OpenAIResponsesProvider",
    "FakeProvider",
    "LLMProvider",
]
Output = TypeVar("Output", bound=BaseModel)


class OpenAIProviderSettings(ExecutionModel):
    endpoint: str | None = None
    model: str | None = Field(default=None, min_length=1)
    api_key: SecretStr | None = Field(default=None, repr=False, exclude=True)
    timeout_seconds: float = Field(default=60, gt=0, le=60)
    supports_store_false: bool = False

    @field_validator("endpoint")
    @classmethod
    def safe_endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        url = urlsplit(value)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
        ):
            raise ValueError("endpoint must be an explicit HTTPS API root without credentials")
        return value.rstrip("/")

    @classmethod
    def from_environment(cls) -> "OpenAIProviderSettings":
        key = os.environ.get("OPENAI_API_KEY")
        return cls(
            endpoint=os.environ.get("OPENAI_BASE_URL") or None,
            model=os.environ.get("OPENAI_MODEL") or None,
            api_key=SecretStr(key) if key else None,
            supports_store_false=os.environ.get("GPU_AGENT_STORE_FALSE_SUPPORTED") == "1",
        )


class Usage(ExecutionModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None


class Invocation(ExecutionModel):
    invocation_id: str
    run_id: str
    kind: CallKind
    attempt: int
    state: Literal["STARTED", "COMPLETED", "FAILED", "UNCERTAIN"]
    started_at: datetime
    finished_at: datetime | None = None
    elapsed_ms: float | None = None
    configured_model: str
    response_model: str | None = None
    endpoint_host: str
    prompt_version: str = PROMPT_VERSION
    client_request_id: str
    provider_request_id: str | None = None
    response_id: str | None = None
    usage: Usage | None = None
    error_code: str | None = None
    http_status: int | None = None
    retryable: bool = False
    format_retry_of: str | None = None
    store_false_sent: bool = True


class LLMProvider(Protocol):
    gate: LLMCallGate
    provider_name: str
    model_name: str | None

    def ensure_available(self) -> None: ...
    def plan(self, evidence: PublicEvidence, budget: AgentBudget) -> AgentAction: ...
    def diagnose(self, evidence: PublicEvidence) -> DiagnosisResult: ...
    def propose_patch(self, public_source: PublicSource, diagnosis: DiagnosisResult) -> str: ...


class OpenAIResponsesProvider:
    provider_name = "openai-responses"

    def __init__(
        self,
        settings: OpenAIProviderSettings,
        gate: LLMCallGate,
        store: RunStore,
        run_id: str,
        *,
        client_factory: Callable[..., Any] = openai.OpenAI,
        diff_validator: Callable[[str], object] | None = None,
    ) -> None:
        self.settings, self.gate, self.store, self.run_id = settings, gate, store, run_id
        self.model_name = settings.model
        self._factory, self._diff_validator = client_factory, diff_validator

    def ensure_available(self) -> None:
        s = self.settings
        if s.api_key is None or len(s.api_key) == 0 or not s.endpoint or not s.model:
            raise ProviderError("LLM_UNAVAILABLE")
        official = urlsplit(s.endpoint).hostname == "api.openai.com"
        if not official and not s.supports_store_false:
            raise ProviderError("LLM_CAPABILITY_UNAVAILABLE")

    def _save(self, invocation: Invocation) -> None:
        self.store.put(
            self.run_id,
            f"provider/{invocation.invocation_id}/{invocation.state}.json",
            invocation.model_dump_json().encode(),
            "public",
        )

    def invocations(self) -> list[Invocation]:
        records: dict[str, Invocation] = {}
        for ref in self.store.load(self.run_id).artifact_refs:
            if ref.name.startswith("provider/"):
                record = Invocation.model_validate_json(self.store.read(ref))
                records[record.invocation_id] = record
        # A crash after STARTED cannot establish whether inference was sent or completed.
        return [
            r.model_copy(update={"state": "UNCERTAIN", "error_code": "INTERRUPTED_INVOCATION"})
            if r.state == "STARTED"
            else r
            for r in records.values()
        ]

    @staticmethod
    def _metadata(response: dict[str, Any]) -> dict[str, object]:
        usage = response.get("usage")
        parsed_usage = None
        if usage is not None:
            inputs = usage.get("input_tokens_details") or {}
            outputs = usage.get("output_tokens_details") or {}
            parsed_usage = Usage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
                cached_tokens=inputs.get("cached_tokens"),
                cache_write_tokens=inputs.get("cache_write_tokens"),
                reasoning_tokens=outputs.get("reasoning_tokens"),
            )
        return dict(
            response_id=response.get("id"),
            response_model=response.get("model"),
            usage=parsed_usage,
        )

    def _call(
        self,
        kind: CallKind,
        payload: dict[str, object],
        output_model: type[Output],
        validate: Callable[[Output], object] | None = None,
    ) -> Output:
        self.ensure_available()
        # OPENAI_LOG/debug host settings must not turn requests or headers into artifacts.
        for namespace in ("openai", "httpx2", "httpcore2"):
            names = [
                namespace,
                *(
                    name
                    for name in logging.Logger.manager.loggerDict
                    if name.startswith(namespace + ".")
                ),
            ]
            for name in names:
                logger = logging.getLogger(name)
                logger.disabled = True
                logger.setLevel(logging.CRITICAL + 1)
        # A recovered session must never replay an invocation of uncertain delivery.
        if any(i.state == "UNCERTAIN" for i in self.invocations()):
            raise ProviderError("LLM_UNCERTAIN_INVOCATION")
        previous: str | None = None
        for attempt in range(2):
            timeout = min(self.settings.timeout_seconds, self.gate.reserve(kind, attempt=attempt))
            invocation = Invocation(
                invocation_id=new_id(),
                run_id=self.run_id,
                kind=kind,
                attempt=attempt,
                state="STARTED",
                started_at=now(),
                configured_model=self.settings.model or "",
                client_request_id=new_id(),
                endpoint_host=urlsplit(self.settings.endpoint or "").hostname or "",
                format_retry_of=previous,
            )
            self._save(invocation)
            started = time.monotonic()
            response = None
            metadata: dict[str, object] = {}
            error: ProviderError | None = None
            client = None
            try:
                assert self.settings.api_key is not None
                client = self._factory(
                    api_key=self.settings.api_key.get_secret_value(),
                    base_url=self.settings.endpoint,
                    timeout=timeout,
                    max_retries=0,
                )
                raw = client.responses.with_raw_response.parse(
                    model=self.settings.model,
                    instructions=PROMPTS[kind]
                    + (
                        "\nPrevious output failed schema/scope validation; correct its format."
                        if attempt
                        else ""
                    ),
                    input=json.dumps({"untrusted_data": payload}, ensure_ascii=False),
                    text_format=output_model,
                    store=False,
                    max_output_tokens=4096,
                    timeout=timeout,
                    extra_headers={"X-Client-Request-Id": invocation.client_request_id},
                )
                # SDK structured parsing can raise before returning the Response. Retain
                # only allowlisted metadata from the successful HTTP envelope first.
                metadata["provider_request_id"] = raw.request_id
                envelope = json.loads(raw.content)
                metadata.update(self._metadata(envelope))
                if any(
                    c.get("type") == "refusal"
                    for item in envelope.get("output", [])
                    for c in item.get("content", [])
                ):
                    raise ProviderError("LLM_REFUSED", state="FAILED")
                if (
                    envelope.get("status") != "completed"
                    or envelope.get("error")
                    or envelope.get("incomplete_details")
                ):
                    raise ProviderError("LLM_INCOMPLETE", state="FAILED")
                response = raw.parse()
                value = output_model.model_validate(getattr(response, "output_parsed", None))
                if validate is not None:
                    validate(value)
            except openai.APITimeoutError:
                error = ProviderError("LLM_TIMEOUT", state="UNCERTAIN")
            except openai.APIConnectionError:
                error = ProviderError("LLM_CONNECTION_ERROR", state="UNCERTAIN")
            except openai.APIStatusError as exc:
                codes = {
                    400: "LLM_INVALID_REQUEST",
                    401: "LLM_AUTHENTICATION_FAILED",
                    403: "LLM_PERMISSION_DENIED",
                    404: "LLM_MODEL_OR_ENDPOINT_NOT_FOUND",
                    422: "LLM_INVALID_REQUEST",
                    429: "LLM_RATE_LIMITED",
                }
                error = ProviderError(
                    codes.get(exc.status_code, "LLM_PROVIDER_ERROR"),
                    state="FAILED",
                    retryable=exc.status_code in {409, 429} or exc.status_code >= 500,
                )
                metadata.update(provider_request_id=exc.request_id, http_status=exc.status_code)
            except (ValidationError, json.JSONDecodeError):
                error = ProviderError("LLM_INVALID_OUTPUT", state="FAILED")
            except ProviderError as exc:
                error = exc
            finally:
                if client is not None:
                    client.close()
            terminal = invocation.model_copy(
                update={
                    **metadata,
                    "state": error.state if error else "COMPLETED",
                    "finished_at": now(),
                    "elapsed_ms": (time.monotonic() - started) * 1000,
                    "error_code": error.code if error else None,
                    "retryable": error.retryable if error else False,
                }
            )
            self._save(terminal)
            if error is None:
                return value
            if error.code != "LLM_INVALID_OUTPUT" or attempt:
                raise error
            previous = invocation.invocation_id
        raise AssertionError("bounded request loop")

    def plan(self, evidence: PublicEvidence, budget: AgentBudget) -> AgentAction:
        return self._call(
            "plan",
            {
                "evidence": evidence.model_dump(mode="json"),
                "budget": budget.model_dump(mode="json"),
            },
            AgentActionOutput,
        ).action

    def diagnose(self, evidence: PublicEvidence) -> DiagnosisResult:
        return self._call(
            "diagnose", {"evidence": evidence.model_dump(mode="json")}, DiagnosisResult
        )

    def propose_patch(self, public_source: PublicSource, diagnosis: DiagnosisResult) -> str:
        def validate(value: PatchOutput) -> None:
            try:
                if not value.unified_diff.startswith(
                    ("--- a/kernel.cu\n", "diff --git a/kernel.cu ")
                ):
                    raise ValueError("invalid unified diff")
                if self._diff_validator is not None:
                    self._diff_validator(value.unified_diff)
            except ValueError:
                raise ProviderError("LLM_INVALID_OUTPUT", state="FAILED") from None

        return self._call(
            "patch",
            {
                "public_source": public_source.model_dump(mode="json"),
                "diagnosis": diagnosis.model_dump(mode="json"),
            },
            PatchOutput,
            validate,
        ).unified_diff


class FakeProvider:
    """Test-only scripted provider. It is never selected by configuration or CLI."""

    provider_name = "test-fake"
    model_name = "test-fake"

    def __init__(self, actions: list[AgentAction], diagnosis: DiagnosisResult, diff: str) -> None:
        self.actions, self.result, self.diff = actions, diagnosis, diff
        self.inputs: list[dict[str, object]] = []
        self.kinds: list[str] = []
        self.gate = LLMCallGate()

    def ensure_available(self) -> None:
        return None

    def _record(self, kind: CallKind, payload: dict[str, object]) -> None:
        self.gate.reserve(kind)
        self.kinds.append(kind)
        self.inputs.append(json.loads(json.dumps(payload)))

    def plan(self, evidence: PublicEvidence, budget: AgentBudget) -> AgentAction:
        self._record(
            "plan",
            {
                "evidence": evidence.model_dump(mode="json"),
                "budget": budget.model_dump(mode="json"),
            },
        )
        if not self.actions:
            raise ProviderError("FAKE_SCRIPT_EXHAUSTED")
        return self.actions.pop(0)

    def diagnose(self, evidence: PublicEvidence) -> DiagnosisResult:
        self._record("diagnose", {"evidence": evidence.model_dump(mode="json")})
        return self.result

    def propose_patch(self, public_source: PublicSource, diagnosis: DiagnosisResult) -> str:
        self._record(
            "patch",
            {
                "public_source": public_source.model_dump(mode="json"),
                "diagnosis": diagnosis.model_dump(mode="json"),
            },
        )
        return self.diff
