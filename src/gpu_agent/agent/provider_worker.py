"""Fixed trusted entry point. No credentials, requests or error bodies on disk/logs."""

import resource
import sys

from gpu_agent.agent.provider import SDKResult, WorkerRequest, invoke_sdk
from gpu_agent.agent.provider_process import MAX_INPUT, MAX_OUTPUT


def main() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    try:
        payload = sys.stdin.buffer.read(MAX_INPUT + 1)
        if len(payload) > MAX_INPUT:
            raise ValueError("oversized request")
        result = invoke_sdk(WorkerRequest.model_validate_json(payload))
        output = result.model_dump_json().encode()
        if len(output) > MAX_OUTPUT:
            raise ValueError("oversized result")
    except Exception:
        output = (
            SDKResult(error_code="LLM_WORKER_ERROR", state="UNCERTAIN").model_dump_json().encode()
        )
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
