"""Replay one exact public repository payload under one reasoning intervention.

Uses the production SDK parsing path and identical structured-output schema.
Only the explicit reasoning effort (if supplied) is added to the request.
A separately started server can use a thinking-disabled chat template; that
is recorded as a server configuration intervention, not an API effort claim.
No tool execution, model retry, hidden evaluator, or raw reasoning retention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import openai
from openai.lib._pydantic import to_strict_json_schema

from gpu_agent.agent.prompts import select_prompt
from gpu_agent.agent.provider import OpenAIProviderSettings, WorkerRequest, invoke_sdk
from gpu_agent.harness.models import HarnessDecision, SessionSnapshot
from gpu_agent.store import read_regular


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--payload-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effort", choices=("default", "low", "none"), required=True)
    parser.add_argument("--server-thinking", choices=("default", "disabled"), required=True)
    parser.add_argument("--server-template-sha256", required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--allow-model-call", action="store_true")
    args = parser.parse_args()
    if not args.allow_model_call:
        parser.error("explicit model call approval required")
    if args.output.exists() or args.output.is_symlink():
        parser.error("immutable replay output already exists")
    raw = read_regular(args.payload, 256 * 1024)
    if sha(raw) != args.payload_sha256:
        parser.error("frozen payload hash differs")
    payload = json.loads(raw)
    if payload.get("gpu_repo_harness") != "v1":
        parser.error("registered repository payload required")
    snapshot = SessionSnapshot.model_validate_json(json.dumps(payload["snapshot"]))
    settings = OpenAIProviderSettings.from_environment()
    if settings.endpoint != "http://127.0.0.1:8127/v1" or settings.model != "qwen3-8b-awq":
        parser.error("only the frozen local Qwen endpoint is permitted")
    if settings.api_key is None:
        parser.error("local API key missing")

    observed: dict[str, Any] = {}
    tokenizer = None
    if args.tokenizer_json is not None:
        # Optional local re-tokenization is separately labelled; usage minus
        # reasoning includes specials and is not an exact output-text count.
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(args.tokenizer_json))

    class CaptureClient:
        def __init__(self, **kwargs: Any) -> None:
            self.client = openai.OpenAI(**kwargs)
            self.responses = self
            self.with_raw_response = self

        def close(self) -> None:
            self.client.close()

        def parse(self, **kwargs: Any) -> Any:
            if args.effort != "default":
                kwargs["reasoning"] = {"effort": args.effort}
            schema = to_strict_json_schema(kwargs["text_format"])
            observed.update(
                {
                    "reasoning_parameter_sent": kwargs.get("reasoning"),
                    "wire_schema_sha256": sha(json.dumps(schema, sort_keys=True).encode()),
                    "instructions_sha256": sha(kwargs["instructions"].encode()),
                    "wire_input_sha256": sha(kwargs["input"].encode()),
                    "wire_max_output_tokens": kwargs["max_output_tokens"],
                }
            )
            response = self.client.responses.with_raw_response.parse(**kwargs)
            envelope = json.loads(response.content)
            texts = [
                part["text"]
                for item in envelope.get("output", [])
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text" and isinstance(part.get("text"), str)
            ]
            if tokenizer is not None:
                observed["output_text_tokens_retokenized"] = (
                    len(tokenizer.encode("".join(texts), add_special_tokens=False).ids)
                    if texts
                    else 0
                )
            return response

    request = WorkerRequest(
        endpoint=settings.endpoint,
        model=settings.model,
        api_key=settings.api_key,
        kind="plan",
        payload=payload,
        client_request_id=sha(raw + args.effort.encode() + args.server_thinking.encode())[:32],
        timeout_seconds=settings.timeout_seconds,
        attempt=0,
        max_output_tokens=8192,
    )
    result = invoke_sdk(request, client_factory=CaptureClient)
    usage = result.metadata.usage
    legal = False
    tool = None
    if result.value is not None:
        action = HarnessDecision.model_validate(result.value).action
        tool = action.tool
        legal = tool in snapshot.allowed_tools
    report = {
        "payload_sha256": args.payload_sha256,
        "prompt_version": select_prompt("plan", payload)[0],
        "model": settings.model,
        "configured_max_output_tokens": 8192,
        "requested_effort": args.effort,
        "server_thinking": args.server_thinking,
        "server_template_sha256": args.server_template_sha256,
        **observed,
        "state": result.state,
        "error_code": result.error_code,
        "usage": usage.model_dump() if usage else None,
        "nonreasoning_output_tokens": (
            usage.output_tokens - usage.reasoning_tokens
            if usage and usage.output_tokens is not None and usage.reasoning_tokens is not None
            else None
        ),
        "response_termination": result.metadata.response_termination.model_dump(mode="json")
        if result.metadata.response_termination
        else None,
        "structured_action_legal": legal,
        "selected_tool": tool,
        "decision": result.value,
        "physical_calls": 1,
        "format_retries": 0,
        "tool_executed": False,
    }
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
