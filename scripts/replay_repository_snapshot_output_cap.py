"""One physical call per frozen snapshot and output cap; no tools or auto-retries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gpu_agent.agent.prompts import select_prompt
from gpu_agent.agent.provider import OpenAIProviderSettings, WorkerRequest
from gpu_agent.agent.provider_process import ProviderProcessPort
from gpu_agent.harness.models import SessionSnapshot
from gpu_agent.store import read_regular


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, choices=(4096, 8192), required=True)
    parser.add_argument("--allow-model-call", action="store_true")
    args = parser.parse_args()
    if not args.allow_model_call:
        parser.error("explicit model-call approval required")
    raw = read_regular(args.snapshot, 256 * 1024)
    if hashlib.sha256(raw).hexdigest() != args.snapshot_sha256:
        parser.error("snapshot SHA mismatch")
    # Keep exact serialized field order and values; validation is separate.
    snapshot = json.loads(raw)
    SessionSnapshot.model_validate_json(raw)
    payload = {"gpu_repo_harness": "v1", "snapshot": snapshot}
    settings = OpenAIProviderSettings.from_environment()
    if settings.api_key is None or settings.endpoint is None or settings.model is None:
        parser.error("model settings not configured")
    if settings.endpoint != "http://127.0.0.1:8127/v1":
        parser.error("this diagnostic runner permits only the registered local inference service")
    request = WorkerRequest(
        endpoint=settings.endpoint,
        model=settings.model,
        api_key=settings.api_key,
        kind="plan",
        payload=payload,
        client_request_id=hashlib.sha256(raw + str(args.max_output_tokens).encode()).hexdigest()[
            :32
        ],
        timeout_seconds=settings.timeout_seconds,
        attempt=0,
        max_output_tokens=args.max_output_tokens,
    )
    if args.output.exists():
        parser.error("immutable experiment output already exists")
    result = ProviderProcessPort().call(request)
    report = {
        "snapshot_sha256": args.snapshot_sha256,
        "payload_sha256": hashlib.sha256(
            json.dumps(payload, ensure_ascii=False).encode()
        ).hexdigest(),
        "prompt_version": select_prompt("plan", payload)[0],
        "configured_max_output_tokens": args.max_output_tokens,
        "model": settings.model,
        "physical_model_calls": 1,
        "format_retries": 0,
        "state": result.state,
        "error_code": result.error_code,
        "usage": result.metadata.usage.model_dump() if result.metadata.usage else None,
        "response_termination": (
            result.metadata.response_termination.model_dump(mode="json")
            if result.metadata.response_termination
            else None
        ),
        "decision": result.value,
        "tool_executed": False,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
