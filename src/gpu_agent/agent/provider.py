"""Official Responses adapter. One ledger event pair per physical request, no replay."""

import json
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime
from threading import Event
from typing import Any, Literal, Protocol, TypeVar
from urllib.parse import urlsplit

import openai
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator, model_validator

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


class WorkerRequest(ExecutionModel):
    endpoint: str
    model: str
    api_key: SecretStr = Field(repr=False, exclude=True)
    kind: CallKind
    payload: dict[str, object]
    client_request_id: str
    timeout_seconds: float = Field(gt=0, le=60)
    attempt: int = Field(ge=0, le=1)


class ResponseMetadata(ExecutionModel):
    response_id: str | None = Field(default=None, max_length=1024)
    provider_request_id: str | None = Field(default=None, max_length=1024)
    response_model: str | None = Field(default=None, max_length=1024)
    usage: Usage | None = None
    http_status: int | None = None


class SDKResult(ExecutionModel):
    value: dict[str, object] | None = None
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)
    error_code: (
        Literal[
            "LLM_TIMEOUT",
            "LLM_CONNECTION_ERROR",
            "LLM_INVALID_REQUEST",
            "LLM_AUTHENTICATION_FAILED",
            "LLM_PERMISSION_DENIED",
            "LLM_MODEL_OR_ENDPOINT_NOT_FOUND",
            "LLM_PROVIDER_ERROR",
            "LLM_RATE_LIMITED",
            "LLM_INVALID_OUTPUT",
            "LLM_REFUSED",
            "LLM_INCOMPLETE",
            "LLM_WORKER_ERROR",
        ]
        | None
    ) = None
    state: Literal["COMPLETED", "FAILED", "UNCERTAIN"] = "COMPLETED"
    retryable: bool = False

    @model_validator(mode="after")
    def consistent_envelope(self) -> "SDKResult":
        if self.error_code is None:
            if self.state != "COMPLETED" or self.value is None:
                raise ValueError("incomplete worker result")
        else:
            uncertain = self.error_code in {
                "LLM_TIMEOUT",
                "LLM_CONNECTION_ERROR",
                "LLM_WORKER_ERROR",
            }
            if self.value is not None or self.state != ("UNCERTAIN" if uncertain else "FAILED"):
                raise ValueError("inconsistent worker result")
        return self


class ResponsesPort(Protocol):
    def call(self, request: WorkerRequest) -> SDKResult: ...


def invoke_sdk(
    request: WorkerRequest, client_factory: Callable[..., Any] = openai.OpenAI
) -> SDKResult:
    """Official SDK wire contract; production invokes this only in the killable worker."""
    for namespace in ("openai", "httpx2", "httpcore2"):
        for name in [
            namespace,
            *(n for n in logging.Logger.manager.loggerDict if n.startswith(namespace + ".")),
        ]:
            logging.getLogger(name).disabled = True
            logging.getLogger(name).setLevel(logging.CRITICAL + 1)
    output_models: dict[CallKind, type[BaseModel]] = {
        "plan": AgentActionOutput,
        "diagnose": DiagnosisResult,
        "patch": PatchOutput,
    }
    output_model = output_models[request.kind]
    metadata: dict[str, object] = {}
    client = None
    value: dict[str, object] | None = None
    error: ProviderError | None = None
    try:
        client = client_factory(
            api_key=request.api_key.get_secret_value(),
            base_url=request.endpoint,
            timeout=request.timeout_seconds,
            max_retries=0,
        )
        raw = client.responses.with_raw_response.parse(
            model=request.model,
            instructions=PROMPTS[request.kind]
            + (
                "\nPrevious output failed schema/scope validation; correct its format."
                if request.attempt
                else ""
            ),
            input=json.dumps({"untrusted_data": request.payload}, ensure_ascii=False),
            text_format=output_model,
            store=False,
            max_output_tokens=4096,
            timeout=request.timeout_seconds,
            extra_headers={"X-Client-Request-Id": request.client_request_id},
        )
        metadata["provider_request_id"] = raw.request_id
        envelope = json.loads(raw.content)
        metadata.update(OpenAIResponsesProvider._metadata(envelope))
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
        value = output_model.model_validate(getattr(raw.parse(), "output_parsed", None)).model_dump(
            mode="json"
        )
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
    return SDKResult.model_validate(
        dict(
            value=value,
            metadata=metadata,
            error_code=error.code if error else None,
            state=error.state if error else "COMPLETED",
            retryable=error.retryable if error else False,
        )
    )


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
        port: ResponsesPort | None = None,
        cancel: Event | None = None,
        diff_validator: Callable[[str], object] | None = None,
    ) -> None:
        self.settings, self.gate, self.store, self.run_id = settings, gate, store, run_id
        self.model_name = settings.model
        from gpu_agent.agent.provider_process import ProviderProcessPort

        self._port = port if port is not None else ProviderProcessPort(cancel=cancel)
        self._cancel, self._diff_validator = cancel, diff_validator

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
        if self._cancel is not None and self._cancel.is_set():
            raise ProviderError("LLM_CANCELLED")
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
            metadata: dict[str, object] = {}
            error: ProviderError | None = None
            try:
                assert self.settings.api_key is not None
                result = self._port.call(
                    WorkerRequest(
                        endpoint=self.settings.endpoint or "",
                        model=self.settings.model or "",
                        api_key=self.settings.api_key,
                        kind=kind,
                        payload=payload,
                        client_request_id=invocation.client_request_id,
                        timeout_seconds=self.gate.timeout(timeout),
                        attempt=attempt,
                    )
                )
                metadata = result.metadata.model_dump()
                metadata["usage"] = result.metadata.usage
                if result.error_code is not None:
                    raise ProviderError(
                        result.error_code, state=result.state, retryable=result.retryable
                    )
                if result.state != "COMPLETED":
                    raise ProviderError("LLM_WORKER_ERROR", state="UNCERTAIN")
                value = output_model.model_validate(result.value)
                if validate is not None:
                    validate(value)
            except (ValidationError, json.JSONDecodeError):
                error = ProviderError("LLM_INVALID_OUTPUT", state="FAILED")
            except ProviderError as exc:
                error = exc
            terminal = invocation.model_copy(
                update={
                    **metadata,
                    "state": ("FAILED" if error.state == "NOT_STARTED" else error.state)
                    if error
                    else "COMPLETED",
                    "finished_at": now(),
                    "elapsed_ms": (time.monotonic() - started) * 1000,
                    "error_code": error.code if error else None,
                    "retryable": error.retryable if error else False,
                }
            )
            self._save(terminal)
            if error is None:
                return value
            if error.code != "LLM_INVALID_OUTPUT" or error.state != "FAILED" or attempt:
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
