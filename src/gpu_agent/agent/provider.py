"""Official Responses adapter. One ledger event pair per physical request, no replay."""

import hashlib
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
    PlannerOutput,
    PlannerState,
    ProviderError,
    PublicEvidence,
    PublicSource,
)
from gpu_agent.agent.policy import CallKind, LLMCallGate, validate_diagnosis
from gpu_agent.agent.prompts import PROMPT_VERSION, PROMPTS
from gpu_agent.contracts import new_id, now
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.patching import (
    PATCH_REJECTION_CODES,
    normalize_unified_diff_offsets,
    patch_rejection_code,
)
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


class ValidationIssue(ExecutionModel):
    """One schema failure location. Never contains model-produced values."""

    loc: str = Field(max_length=200)
    type: str = Field(max_length=100)
    constraint: str | None = Field(default=None, max_length=200)


class OutputDiagnostics(ExecutionModel):
    """Bounded, value-free telemetry for a rejected or accepted model output."""

    failure_class: (
        Literal["NO_OUTPUT_TEXT", "NOT_JSON", "SCHEMA_INVALID", "DOMAIN_REJECTED"] | None
    ) = None
    output_chars: int | None = Field(default=None, ge=0)
    output_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    issues: list[ValidationIssue] = Field(default_factory=list, max_length=10)


_CONSTRAINT_KEYS = ("max_length", "min_length", "pattern", "ge", "gt", "le", "lt", "expected")


def validation_issues(exc: ValidationError) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for error in exc.errors(include_input=False, include_url=False)[:10]:
        context = error.get("ctx") or {}
        constraint = ",".join(f"{key}={context[key]}" for key in _CONSTRAINT_KEYS if key in context)
        issues.append(
            ValidationIssue(
                loc=".".join(str(part) for part in error.get("loc", ()))[:200] or "<root>",
                type=str(error.get("type", "unknown"))[:100],
                constraint=constraint[:200] or None,
            )
        )
    return issues


PATCH_REPAIR_HINTS = {
    "hunk_context_not_found": "each hunk's context and '-' lines must be copied exactly, "
    "character for character, from the supplied kernel.cu",
    "hunk_context_mismatch": "each hunk's context and '-' lines must be copied exactly, "
    "character for character, from the supplied kernel.cu",
    "hunk_line_invalid": "every hunk line must start with ' ', '+' or '-' and end with a "
    "newline, including the last line",
    "hunk_header_invalid": "each hunk must start with a header like '@@ -12,3 +12,4 @@'",
    "diff_header_invalid": "the diff must start with '--- a/kernel.cu' then '+++ b/kernel.cu'",
    "include_changed": "do not add, remove or change #include lines",
    "fixed_input_length": "do not restrict the accepted input length to a fixed value",
    "file_outside_scope": "change only kernel.cu",
}


def _domain_reason(kind: "CallKind", reason: object) -> str:
    """Only controller-owned, value-free codes may enter domain telemetry or prompts."""
    fallback = "patch_invalid" if kind == "patch" else f"{kind}_policy_rejected"
    allowed = set(PATCH_REJECTION_CODES.values()) if kind == "patch" else {fallback}
    return reason if isinstance(reason, str) and reason in allowed else fallback


def correction_text(kind: "CallKind", hints: list[str]) -> str:
    """Retry instruction built only from fixed codes.

    A domain rejection means the JSON was well formed but its content was refused.
    Preserve the envelope explicitly rather than suggesting a format change.
    """
    prefix = f"<{kind}>: "
    hints = [
        prefix + _domain_reason(kind, hint.removeprefix(prefix))
        if hint.startswith(prefix)
        else hint
        for hint in hints
    ]
    domain = [hint.removeprefix(prefix) for hint in hints if hint.startswith(prefix)]
    if hints and len(domain) == len(hints):
        text = (
            "\nThe previous output had the correct JSON format but its content was rejected "
            f"({', '.join(domain)}). Keep exactly the same JSON object format"
        )
        if kind == "patch":
            text += ' {"unified_diff": "..."}'
            fixes = list(
                dict.fromkeys(PATCH_REPAIR_HINTS[c] for c in domain if c in PATCH_REPAIR_HINTS)
            )
            if fixes:
                text += "; " + "; ".join(fixes)
        return text + "."
    json_codes = [hint.removeprefix("<json>: ") for hint in hints if hint.startswith("<json>: ")]
    if hints and len(json_codes) == len(hints):
        text = (
            f"\nThe previous output was not parsable JSON ({', '.join(json_codes)}). Return "
            "exactly one JSON object matching the schema and nothing else: no prose, no code "
            "fences"
        )
        if kind == "patch":
            text += (
                ', no bare diff. Put the whole diff in the "unified_diff" string, escaping '
                'each newline as \\n and each double quote as \\"'
            )
        return text + "."
    text = "\nPrevious output failed schema/scope validation; correct its format."
    if hints:
        text += " Rejected fields: " + "; ".join(hints) + "."
    return text


class _DomainRejected(ProviderError):
    """Well-formed output refused by a domain check; `reason` is a fixed value-free code."""

    def __init__(self, reason: str) -> None:
        super().__init__("LLM_INVALID_OUTPUT", state="FAILED")
        self.reason = _domain_reason("patch", reason)


def correction_hints(diagnostics: "OutputDiagnostics | None") -> list[str]:
    if diagnostics is None:
        return []
    hints = [
        f"{issue.loc}: {issue.type}" + (f" ({issue.constraint})" if issue.constraint else "")
        for issue in diagnostics.issues
    ]
    if not hints and diagnostics.failure_class:
        hints = [diagnostics.failure_class]
    return [hint[:300] for hint in hints[:10]]


_CONTROLLER_OWNED_PLAN_KEYS = ("action_id", "budget_snapshot")


def normalize_wire_value(kind: "CallKind", value: object) -> object:
    """Drop controller-owned planner fields; the controller never trusts model identities."""
    if kind == "plan" and isinstance(value, dict) and isinstance(value.get("action"), dict):
        action = {
            key: item
            for key, item in value["action"].items()
            if key not in _CONTROLLER_OWNED_PLAN_KEYS
        }
        return {**value, "action": action}
    return value


WIRE_MODELS: "dict[CallKind, type[BaseModel]]" = {}


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
    output_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    output_diagnostics: OutputDiagnostics | None = None


class WorkerRequest(ExecutionModel):
    endpoint: str
    model: str
    api_key: SecretStr = Field(repr=False, exclude=True)
    kind: CallKind
    payload: dict[str, object]
    client_request_id: str
    timeout_seconds: float = Field(gt=0, le=60)
    attempt: int = Field(ge=0, le=1)
    correction_hints: list[str] = Field(default_factory=list, max_length=10)


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
    diagnostics: OutputDiagnostics | None = None

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


def _is_deepseek_endpoint(endpoint: str) -> bool:
    return urlsplit(endpoint).hostname == "api.deepseek.com"


class _OutputRejected(Exception):
    def __init__(self, diagnostics: OutputDiagnostics) -> None:
        super().__init__(diagnostics.failure_class)
        self.diagnostics = diagnostics


def _deepseek_output_text(envelope: dict[str, object]) -> str:
    output = envelope.get("output")
    if not isinstance(output, list):
        raise _OutputRejected(OutputDiagnostics(failure_class="NO_OUTPUT_TEXT"))
    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                text_parts.append(part["text"])
    if not text_parts:
        raise _OutputRejected(OutputDiagnostics(failure_class="NO_OUTPUT_TEXT"))
    return "".join(text_parts)


JSON_ERROR_CODES = (
    "empty_output",
    "code_fence",
    "raw_diff",
    "prose_before_json",
    "truncated_json",
    "invalid_escape",
    "control_character",
    "extra_data",
    "malformed_json",
)


def json_error_code(text: str, exc: json.JSONDecodeError) -> str:
    """Classify an unparsable output by shape only; the text itself is never recorded."""
    stripped = text.lstrip()
    if not stripped:
        return "empty_output"
    if stripped.startswith("```"):
        return "code_fence"
    if stripped.startswith(("--- a/", "diff --git", "@@ ")):
        return "raw_diff"
    if not stripped.startswith(("{", "[")):
        return "prose_before_json"
    if exc.msg.startswith("Extra data"):
        return "extra_data"
    if exc.msg.startswith("Invalid \\escape"):
        return "invalid_escape"
    if exc.msg.startswith("Invalid control character"):
        return "control_character"
    if exc.pos >= len(text.rstrip()) or exc.msg.startswith("Unterminated string"):
        return "truncated_json"
    return "malformed_json"


def parse_wire_text(
    kind: "CallKind", text: str, output_model: type[BaseModel]
) -> tuple[dict[str, object], OutputDiagnostics]:
    """Classify a raw model text; diagnostics carry only sizes, hashes and schema locations."""
    base = OutputDiagnostics(
        output_chars=len(text), output_sha256=hashlib.sha256(text.encode()).hexdigest()
    )
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _OutputRejected(
            base.model_copy(
                update={
                    "failure_class": "NOT_JSON",
                    "issues": [ValidationIssue(loc="<json>", type=json_error_code(text, exc))],
                }
            )
        ) from None
    try:
        value = output_model.model_validate(normalize_wire_value(kind, decoded))
    except ValidationError as exc:
        raise _OutputRejected(
            base.model_copy(
                update={"failure_class": "SCHEMA_INVALID", "issues": validation_issues(exc)}
            )
        ) from None
    return value.model_dump(mode="json"), base


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
    output_model = WIRE_MODELS[request.kind]
    metadata: dict[str, object] = {}
    client = None
    value: dict[str, object] | None = None
    error: ProviderError | None = None
    diagnostics: OutputDiagnostics | None = None
    try:
        client = client_factory(
            api_key=request.api_key.get_secret_value(),
            base_url=request.endpoint,
            timeout=request.timeout_seconds,
            max_retries=0,
        )
        correction = (
            correction_text(request.kind, request.correction_hints) if request.attempt else ""
        )
        instructions = PROMPTS[request.kind] + correction
        common = {
            "model": request.model,
            "input": json.dumps({"untrusted_data": request.payload}, ensure_ascii=False),
            "store": False,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "timeout": request.timeout_seconds,
            "extra_headers": {"X-Client-Request-Id": request.client_request_id},
        }
        deepseek = _is_deepseek_endpoint(request.endpoint)
        if deepseek:
            from openai.lib._pydantic import to_strict_json_schema

            raw = client.responses.with_raw_response.create(
                **common,
                instructions=instructions,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": output_model.__name__,
                        "schema": to_strict_json_schema(output_model),
                    }
                },
                reasoning={"effort": "none"},
            )
        else:
            raw = client.responses.with_raw_response.parse(
                **common,
                instructions=instructions,
                text_format=output_model,
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
        if deepseek:
            value, diagnostics = parse_wire_text(
                request.kind, _deepseek_output_text(envelope), output_model
            )
        else:
            parsed = getattr(raw.parse(), "output_parsed", None)
            if isinstance(parsed, BaseModel):
                parsed = parsed.model_dump(mode="json")
            try:
                value = output_model.model_validate(
                    normalize_wire_value(request.kind, parsed)
                ).model_dump(mode="json")
            except ValidationError as exc:
                raise _OutputRejected(
                    OutputDiagnostics(failure_class="SCHEMA_INVALID", issues=validation_issues(exc))
                ) from None
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
    except _OutputRejected as exc:
        error = ProviderError("LLM_INVALID_OUTPUT", state="FAILED")
        diagnostics = exc.diagnostics
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
            diagnostics=diagnostics,
        )
    )


WIRE_MODELS.update({"plan": PlannerOutput, "diagnose": DiagnosisResult, "patch": PatchOutput})


class LLMProvider(Protocol):
    gate: LLMCallGate
    provider_name: str
    model_name: str | None

    def ensure_available(self) -> None: ...
    def plan(
        self,
        evidence: PublicEvidence,
        budget: AgentBudget,
        feedback: list[str] | None = None,
        state: PlannerState | None = None,
    ) -> AgentAction: ...
    def diagnose(self, evidence: PublicEvidence) -> DiagnosisResult: ...
    def propose_patch(self, public_source: PublicSource, diagnosis: DiagnosisResult) -> str: ...


MAX_OUTPUT_TOKENS = 4096


class DevelopmentCallPolicy(ExecutionModel):
    """Explicit development opt-in; bound physical calls, record token usage."""

    schema_version: Literal[1] = 1
    evaluation: Literal[False] = False
    max_llm_calls: int = Field(default=40, ge=1, le=40, strict=True)


def provider_invocations(store: RunStore, run_id: str) -> list[Invocation]:
    """Read latest persisted state without constructing a provider or sending requests."""
    records: dict[str, Invocation] = {}
    for ref in store.load(run_id).artifact_refs:
        if ref.name.startswith("provider/"):
            record = Invocation.model_validate_json(store.read(ref))
            records[record.invocation_id] = record
    return [
        item.model_copy(update={"state": "UNCERTAIN", "error_code": "INTERRUPTED_INVOCATION"})
        if item.state == "STARTED"
        else item
        for item in records.values()
    ]


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
        idempotency_key: str | None = None,
        call_policy: DevelopmentCallPolicy | None = None,
    ) -> None:
        self.settings, self.gate, self.store, self.run_id = settings, gate, store, run_id
        self._call_policy = call_policy
        self.model_name = settings.model
        if settings.endpoint and _is_deepseek_endpoint(settings.endpoint):
            self.provider_name = "deepseek-responses"
        from gpu_agent.agent.provider_process import ProviderProcessPort

        self._port = port if port is not None else ProviderProcessPort(cancel=cancel)
        self._cancel, self._diff_validator = cancel, diff_validator
        self._idempotency_key = idempotency_key
        self._call_sequence = 0

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
            self.store.visibility,
        )

    def invocations(self) -> list[Invocation]:
        return provider_invocations(self.store, self.run_id)

    def _check_call_limit(self) -> None:
        if (
            self._call_policy is not None
            and len(self.invocations()) >= self._call_policy.max_llm_calls
        ):
            raise ProviderError("AGENT_BUDGET_EXHAUSTED")

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
        validate: Callable[[Output], Output] | None = None,
        *,
        wire_model: type[BaseModel] | None = None,
        convert: Callable[[Any], Output] | None = None,
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
        hints: list[str] = []
        sequence = self._call_sequence
        self._call_sequence += 1
        for attempt in range(2):
            self._check_call_limit()
            timeout = min(self.settings.timeout_seconds, self.gate.reserve(kind, attempt=attempt))
            invocation = Invocation(
                invocation_id=new_id(),
                run_id=self.run_id,
                kind=kind,
                attempt=attempt,
                state="STARTED",
                started_at=now(),
                configured_model=self.settings.model or "",
                client_request_id=(
                    hashlib.sha256(
                        f"{self._idempotency_key}:{sequence}:{kind}:{attempt}".encode()
                    ).hexdigest()[:32]
                    if self._idempotency_key is not None
                    else new_id()
                ),
                endpoint_host=urlsplit(self.settings.endpoint or "").hostname or "",
                format_retry_of=previous,
            )
            self._save(invocation)
            started = time.monotonic()
            metadata: dict[str, object] = {}
            error: ProviderError | None = None
            value: Output | None = None
            diagnostics: OutputDiagnostics | None = None
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
                        correction_hints=hints if attempt else [],
                    )
                )
                diagnostics = result.diagnostics
                metadata = result.metadata.model_dump()
                metadata["usage"] = result.metadata.usage
                if result.error_code is not None:
                    raise ProviderError(
                        result.error_code, state=result.state, retryable=result.retryable
                    )
                if result.state != "COMPLETED":
                    raise ProviderError("LLM_WORKER_ERROR", state="UNCERTAIN")
                try:
                    received = normalize_wire_value(kind, result.value)
                    if wire_model is not None and convert is not None:
                        value = convert(wire_model.model_validate(received))
                    else:
                        value = output_model.model_validate(received)
                except ValidationError as exc:
                    diagnostics = (diagnostics or OutputDiagnostics()).model_copy(
                        update={
                            "failure_class": "SCHEMA_INVALID",
                            "issues": validation_issues(exc),
                        }
                    )
                    raise
                if validate is not None:
                    try:
                        value = validate(value)
                    except ProviderError as rejected:
                        reason = _domain_reason(
                            kind, rejected.reason if isinstance(rejected, _DomainRejected) else None
                        )
                        diagnostics = (diagnostics or OutputDiagnostics()).model_copy(
                            update={
                                "failure_class": "DOMAIN_REJECTED",
                                "issues": [ValidationIssue(loc=f"<{kind}>", type=reason)],
                            }
                        )
                        raise
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
                    "output_hash": (
                        hashlib.sha256(value.model_dump_json().encode()).hexdigest()
                        if error is None and value is not None
                        else None
                    ),
                    "output_diagnostics": diagnostics,
                }
            )
            self._save(terminal)
            if error is None:
                assert value is not None
                return value
            if error.code != "LLM_INVALID_OUTPUT" or error.state != "FAILED" or attempt:
                raise error
            previous = invocation.invocation_id
            hints = correction_hints(diagnostics)
        raise AssertionError("bounded request loop")

    def plan(
        self,
        evidence: PublicEvidence,
        budget: AgentBudget,
        feedback: list[str] | None = None,
        state: PlannerState | None = None,
    ) -> AgentAction:
        payload: dict[str, object] = {
            "evidence": evidence.model_dump(mode="json"),
            "budget": budget.model_dump(mode="json"),
        }
        if feedback:
            # Controller reason codes for the one rejected proposal (bounded, no free text).
            payload["controller_feedback"] = {"rejected_previous_action": list(feedback)}
        if state is not None:
            # Controller-derived progress: missing evidence, executed actions, read ranges.
            payload["controller_state"] = state.model_dump(mode="json")
        return self._call(
            "plan",
            payload,
            AgentActionOutput,
            wire_model=PlannerOutput,
            convert=lambda wire: wire.to_action_output(new_id()),
        ).action

    def diagnose(self, evidence: PublicEvidence) -> DiagnosisResult:
        def validate(value: DiagnosisResult) -> DiagnosisResult:
            if not validate_diagnosis(value, evidence):
                raise ProviderError("LLM_INVALID_OUTPUT", state="FAILED")
            return value

        return self._call(
            "diagnose",
            {"evidence": evidence.model_dump(mode="json")},
            DiagnosisResult,
            validate,
        )

    def propose_patch(self, public_source: PublicSource, diagnosis: DiagnosisResult) -> str:
        def validate(value: PatchOutput) -> PatchOutput:
            try:
                if not value.unified_diff.startswith(
                    ("--- a/kernel.cu\n", "diff --git a/kernel.cu ")
                ):
                    raise ValueError("invalid unified diff")
                normalized = normalize_unified_diff_offsets(
                    public_source.content, value.unified_diff
                )
                if self._diff_validator is not None:
                    self._diff_validator(normalized)
            except ValueError as exc:
                raise _DomainRejected(patch_rejection_code(exc)) from None
            return value.model_copy(update={"unified_diff": normalized})

        return self._call(
            "patch",
            {
                "public_source": public_source.model_dump(mode="json"),
                "diagnosis": diagnosis.model_dump(mode="json"),
            },
            PatchOutput,
            validate,
        ).unified_diff


class MockResponsesProvider(OpenAIResponsesProvider):
    """Explicit zero-cost adapter used only with an injected in-process port."""

    provider_name = "mock-responses"

    def __init__(self, *args: object, port: ResponsesPort, **kwargs: object) -> None:
        super().__init__(*args, port=port, **kwargs)  # type: ignore[arg-type]
        self.provider_name = "mock-responses"


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

    def plan(
        self,
        evidence: PublicEvidence,
        budget: AgentBudget,
        feedback: list[str] | None = None,
        state: PlannerState | None = None,
    ) -> AgentAction:
        payload: dict[str, object] = {
            "evidence": evidence.model_dump(mode="json"),
            "budget": budget.model_dump(mode="json"),
        }
        if feedback:
            payload["controller_feedback"] = {"rejected_previous_action": list(feedback)}
        if state is not None:
            payload["controller_state"] = state.model_dump(mode="json")
        self._record("plan", payload)
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
