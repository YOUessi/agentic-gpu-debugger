"""The real controller subprocess boundary enforces wall time and bounded IPC."""

import subprocess
import sys
import threading
import time

import pytest
from pydantic import SecretStr


@pytest.fixture
def worker_provider(store, monkeypatch):
    from gpu_agent.agent.models import AgentBudget
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import OpenAIProviderSettings, OpenAIResponsesProvider

    real_popen = subprocess.Popen
    children = []
    launches = []

    def make(code, *, timeout=0.25, remaining=600, cancel=None):
        def launch(argv, **kwargs):
            launches.append((argv, kwargs))
            assert argv == [sys.executable, "-I", "-m", "gpu_agent.agent.provider_worker"]
            assert "secret-canary" not in repr(argv) + repr(kwargs.get("env"))
            child = real_popen([sys.executable, "-I", "-c", code], **kwargs)
            children.append(child)
            return child

        monkeypatch.setattr(subprocess, "Popen", launch)
        run = store.create_run("deadline")
        provider = OpenAIResponsesProvider(
            OpenAIProviderSettings(
                endpoint="https://api.openai.com/v1",
                model="configured",
                api_key=SecretStr("secret-canary"),
                timeout_seconds=timeout,
            ),
            LLMCallGate(AgentBudget(max_wall_time_seconds=remaining)),
            store,
            run.id,
            cancel=cancel,
        )
        return provider, children, launches

    yield make
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=1)


@pytest.mark.parametrize(
    "behavior",
    [
        "import sys,time; sys.stdin.buffer.read(); time.sleep(10)",
        "import sys,time; sys.stdin.buffer.read()\nwhile True:\n"
        " sys.stdout.write('x'); sys.stdout.flush(); time.sleep(.02)",
        "import os,time; os.close(0); os.close(1); os.close(2); time.sleep(10)",
    ],
)
def test_absolute_deadline_cuts_blocking_and_trickling_workers(worker_provider, behavior):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError

    provider, children, launches = worker_provider(behavior)
    start = time.monotonic()
    with pytest.raises(ProviderError, match="LLM_TIMEOUT"):
        provider.plan(PublicEvidence(), AgentBudget())
    assert time.monotonic() - start < 1.5
    assert len(launches) == 1 and children[0].poll() is not None
    assert provider.invocations()[-1].state == "UNCERTAIN"
    assert provider.invocations()[-1].client_request_id
    assert provider.gate.snapshot().llm_calls == 1
    with pytest.raises(ProviderError, match="LLM_UNCERTAIN_INVOCATION"):
        provider.plan(PublicEvidence(), AgentBudget())
    assert len(launches) == 1


def test_deadline_respects_remaining_diagnosis_time(worker_provider):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError

    provider, children, _ = worker_provider("import time; time.sleep(10)", timeout=3, remaining=0.2)
    start = time.monotonic()
    with pytest.raises(ProviderError, match="LLM_TIMEOUT"):
        provider.plan(PublicEvidence(), AgentBudget())
    assert time.monotonic() - start < 1.5 and children[0].poll() is not None


@pytest.mark.parametrize(
    "code",
    [
        "import sys; sys.stderr.write('secret-canary'); sys.exit(7)",
        "print('malformed secret-canary')",
        "print('{}')",
        'print(\'{"error_code":"LLM_INVALID_OUTPUT","state":"UNCERTAIN"}\')',
        "import sys; sys.stdout.buffer.write(b'x' * (8 * 1024 * 1024 + 1))",
        "import sys; sys.stderr.buffer.write(b'x' * (8 * 1024 * 1024 + 1))",
    ],
)
def test_invalid_worker_output_is_bounded_private_and_not_retried(worker_provider, code):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError

    provider, children, launches = worker_provider(code, timeout=2)
    with pytest.raises(ProviderError, match="LLM_WORKER_ERROR"):
        provider.plan(PublicEvidence(), AgentBudget())
    assert len(launches) == 1 and children[0].poll() is not None
    record = provider.invocations()[-1]
    assert record.state == "UNCERTAIN" and "secret-canary" not in record.model_dump_json()


@pytest.mark.parametrize(
    "code",
    [
        "import time; time.sleep(10)",
        "import os,time; os.close(0); os.close(1); os.close(2); time.sleep(10)",
    ],
)
def test_controller_cancellation_kills_only_owned_worker(worker_provider, code):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError

    sibling = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(10)"], start_new_session=True
    )
    cancel = threading.Event()
    timer = threading.Timer(0.15, cancel.set)
    try:
        provider, children, launches = worker_provider(code, timeout=3, cancel=cancel)
        timer.start()
        with pytest.raises(ProviderError, match="LLM_CANCELLED"):
            provider.plan(PublicEvidence(), AgentBudget())
        assert children[0].poll() is not None and sibling.poll() is None
        assert len(launches) == 1 and provider.invocations()[-1].state == "UNCERTAIN"
    finally:
        timer.cancel()
        if timer.ident is not None:
            timer.join(timeout=1)
        sibling.kill()
        sibling.wait(timeout=1)


def test_secret_is_sent_only_through_stdin_and_response_is_decoded(worker_provider, monkeypatch):
    import os

    from gpu_agent.agent.models import AgentBudget, PublicEvidence

    signals = []
    real_killpg = os.killpg

    def killpg(pid, sig):
        signals.append(pid)
        real_killpg(pid, sig)

    monkeypatch.setattr(os, "killpg", killpg)
    code = """import json,sys
request=json.load(sys.stdin)
assert request['api_key']=='secret-canary'
assert request['endpoint']=='https://api.openai.com/v1'
assert request['kind']=='plan'
print(json.dumps({'value': {'action': {'action_type': 'run_memcheck'}}}))
"""
    provider, children, launches = worker_provider(code, timeout=2)
    action = provider.plan(PublicEvidence(), AgentBudget())
    assert action.action_type == "run_memcheck"
    assert len(launches) == 1 and children[0].poll() is not None
    assert "secret-canary" not in provider.invocations()[-1].model_dump_json()
    assert not signals  # Never signal a numeric PGID after its owner was reaped.


def test_cleanup_reap_failure_does_not_replace_original_timeout(worker_provider, monkeypatch):
    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError

    provider, children, _ = worker_provider("import time; time.sleep(10)")
    launch = subprocess.Popen

    def slow_reap(argv, **kwargs):
        child = launch(argv, **kwargs)
        real_wait = child.wait
        waits = []

        def wait(timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired(argv, timeout)
            return real_wait(timeout=timeout)

        child.wait = wait
        return child

    monkeypatch.setattr(subprocess, "Popen", slow_reap)
    with pytest.raises(ProviderError, match="LLM_TIMEOUT"):
        provider.plan(PublicEvidence(), AgentBudget())
    assert provider.invocations()[-1].state == "UNCERTAIN"
    children[0].wait(timeout=1)


def test_actual_worker_malformed_stdin_exits_without_error_body():
    result = subprocess.run(
        [sys.executable, "-I", "-m", "gpu_agent.agent.provider_worker"],
        input=b'{"api_key":"secret-canary"}',
        capture_output=True,
        timeout=5,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C"},
    )
    assert result.returncode == 0 and not result.stderr
    assert b"LLM_WORKER_ERROR" in result.stdout and b"secret-canary" not in result.stdout


@pytest.mark.parametrize("behavior", ["blocking", "trickling"])
def test_absolute_deadline_cuts_actual_sdk_mock_transport(worker_provider, monkeypatch, behavior):
    import os

    from gpu_agent.agent.models import AgentBudget, PublicEvidence
    from gpu_agent.agent.provider import ProviderError

    observed = []
    real_read = os.read

    def read(fd, size):
        chunk = real_read(fd, size)
        if b"wire-checked" in chunk:
            observed.append(True)
        return chunk

    monkeypatch.setattr(os, "read", read)
    code = """import json,sys,time,httpx2,openai
from gpu_agent.agent.provider import WorkerRequest,invoke_sdk
class Trickling(httpx2.SyncByteStream):
    def __iter__(self):
        while True:
            yield b' '
            time.sleep(.02)
def handler(request):
    body=json.loads(request.content)
    assert body['store'] is False and body['text']['format']['strict'] is True
    assert request.headers['authorization']=='Bearer secret-canary'
    sys.stdout.write('wire-checked'); sys.stdout.flush()
    if BEHAVIOR=='blocking': time.sleep(10)
    return httpx2.Response(200,stream=Trickling())
def factory(**kwargs):
    assert kwargs['max_retries']==0
    client=httpx2.Client(transport=httpx2.MockTransport(handler))
    return openai.OpenAI(**kwargs,http_client=client)
invoke_sdk(WorkerRequest.model_validate_json(sys.stdin.buffer.read()),factory)
""".replace("BEHAVIOR", repr(behavior))
    provider, children, launches = worker_provider(code, timeout=1.5)
    start = time.monotonic()
    with pytest.raises(ProviderError, match="LLM_TIMEOUT"):
        provider.plan(PublicEvidence(), AgentBudget())
    assert time.monotonic() - start < 3
    assert observed and len(launches) == 1 and children[0].poll() is not None
    assert provider.invocations()[-1].state == "UNCERTAIN"
    assert provider.gate.snapshot().llm_calls == 1
