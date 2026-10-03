import json
from types import SimpleNamespace

import pytest

from gpu_agent.repair import PublicCheck, RepairPolicy


@pytest.fixture(autouse=True)
def public_requirement(oob_service):
    import hashlib

    from gpu_agent.public_task import PublicTask

    _, _, source = oob_service
    task = PublicTask(
        source_sha256=hashlib.sha256((source / "kernel.cu").read_bytes()).hexdigest(),
        algorithm="vector-add-cpu-v1",
    )
    (source / "task.json").write_text(task.model_dump_json())


def summary(service, run):
    ref = next(r for r in run.artifact_refs if r.name == "repair/summary.json")
    return json.loads(service.store.read(ref))


def test_failed_public_check_revises_then_verifies_once(oob_service, monkeypatch):
    service, provider, source = oob_service
    fixed = provider.diff
    provider.diff = fixed.replace("i < n", "i <= n")
    received = []

    def revise(public_source, diagnosis, feedback):
        provider._record("patch", {"public_repair_feedback": feedback})
        assert "i <= n" in feedback["previous_candidate_source"]
        assert "i <= n" not in public_source.content
        assert "secret-canary" not in json.dumps(feedback)
        return fixed

    monkeypatch.setattr(provider, "revise_patch", revise)

    def check(store, run_id, sources, stdin, factory, gate, public_task):
        received.append(sources["kernel.cu"])
        assert stdin and store.visibility == "public"
        return PublicCheck(
            run_id="a" * 32,
            status="FAILED" if len(received) == 1 else "PASSED",
            checks={"memcheck": "FINDING" if len(received) == 1 else "CLEAN"},
            feedback=[],
        )

    monkeypatch.setattr("gpu_agent.repair.self_check", check)
    verified = []

    def verify(run_id, strict):
        verified.append(run_id)
        assert strict and len(received) == 2
        return SimpleNamespace(verdict="VERIFIED_FIXED"), "b" * 32

    monkeypatch.setattr(service, "verify_exact", verify)
    run, verdict = service.repair(source)
    assert verdict.verdict == "VERIFIED_FIXED" and verified == [run.id]
    assert len(service.candidates(run.id)) == 1
    report = summary(service, run)
    assert report["stop_reason"] == "PUBLIC_CHECKS_PASSED"
    assert len(report["rounds"]) == 2
    assert provider.kinds.count("patch") == 2
    final = next(r for r in run.artifact_refs if r.name == "agent/final-budget.json")
    assert json.loads(service.store.read(final))["llm_calls"] == 6


@pytest.mark.parametrize(
    "status,limit,reason,patches",
    [
        ("FAILED", 3, "REPEATED_CANDIDATE", 2),
        ("FAILED", 1, "CANDIDATE_LIMIT", 1),
        ("UNAVAILABLE", 3, "PUBLIC_CHECK_UNAVAILABLE", 1),
    ],
)
def test_stop_conditions_do_not_call_independent_verifier(
    oob_service, monkeypatch, status, limit, reason, patches
):
    service, provider, source = oob_service
    monkeypatch.setattr(
        "gpu_agent.repair.self_check",
        lambda *a: PublicCheck(run_id="a" * 32, status=status, checks={}, feedback=[]),
    )
    monkeypatch.setattr(
        service,
        "verify_exact",
        lambda *a, **k: pytest.fail("hidden verifier must not feed revisions"),
    )
    run, verdict = service.repair(source, policy=RepairPolicy(max_candidates=limit))
    assert verdict is None
    assert summary(service, run)["stop_reason"] == reason
    assert provider.kinds.count("patch") == patches


def test_self_check_uses_native_public_artifacts_and_cleans_up(oob_service):
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.repair import self_check

    service, _, source = oob_service
    parent = service.store.create_run("diagnosis")
    result = self_check(
        service.store,
        parent.id,
        {"kernel.cu": (source / "kernel.cu").read_bytes()},
        b"{}",
        service._backend_factory,
        LLMCallGate(),
    )
    # Fixture's GPU subprocess is simulated; this is not a real GPU result.
    assert result.status != "PASSED"
    child = service.store.load(result.run_id)
    assert child.parent_run_id == parent.id
    assert child.status == "COMPLETED"
    assert all(r.visibility == "public" for r in child.artifact_refs)
    assert any(r.name == "self-check.json" for r in child.artifact_refs)


def test_evaluator_store_rejected_before_execution(oob_service):
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.repair import self_check

    service, _, _ = oob_service
    with pytest.raises(ValueError, match="public store"):
        self_check(
            service.evaluator_store, "a" * 32, {}, b"", service._backend_factory, LLMCallGate()
        )


def test_legacy_diagnose_does_not_self_check(oob_service, monkeypatch):
    service, provider, source = oob_service
    monkeypatch.setattr(
        "gpu_agent.repair.self_check", lambda *a: pytest.fail("legacy path changed")
    )
    run = service.diagnose(source)
    assert provider.kinds.count("patch") == 1
    assert not any(r.name.startswith("repair/") for r in run.artifact_refs)


def test_revision_provider_keeps_feedback_and_total_call_limit(store):
    from pydantic import SecretStr

    from gpu_agent.agent.models import DiagnosisResult, PublicSource
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import (
        DevelopmentCallPolicy,
        OpenAIProviderSettings,
        OpenAIResponsesProvider,
        ProviderError,
        SDKResult,
    )

    diff = "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -1 +1 @@\n-int x;\n+int y;\n"

    class Port:
        requests = []

        def call(self, request):
            self.requests.append(request)
            return SDKResult(value={"unified_diff": diff})

    port = Port()
    run = store.create_run("diagnosis")
    provider = OpenAIResponsesProvider(
        OpenAIProviderSettings(
            endpoint="https://api.openai.com/v1", model="m", api_key=SecretStr("not-real")
        ),
        LLMCallGate(),
        store,
        run.id,
        port=port,
        call_policy=DevelopmentCallPolicy(max_llm_calls=2),
    )
    source = PublicSource(source_id="a" * 32, content="int x;\n")
    diagnosis = DiagnosisResult.inconclusive("TEST")
    provider.propose_patch(source, diagnosis)
    provider.revise_patch(
        source, diagnosis, {"contract": "public-repair-v1", "checks": {"racecheck": "FINDING"}}
    )
    assert "public_repair_feedback" not in port.requests[0].payload
    assert port.requests[1].payload["public_repair_feedback"]["checks"]["racecheck"] == "FINDING"
    with pytest.raises(ProviderError, match="AGENT_BUDGET_EXHAUSTED"):
        provider.revise_patch(source, diagnosis, {})
    assert len(port.requests) == 2


@pytest.mark.parametrize(
    "operation,status", [("build", "FAILED"), ("timeout", "UNAVAILABLE"), ("clean", "UNAVAILABLE")]
)
def test_native_self_check_classifies_build_and_clean(oob_service, operation, status):
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.execution.process import ProcessCapture
    from gpu_agent.repair import self_check

    service, _, _ = oob_service

    class Backend(service._backend_factory):
        def _container(self, path, op, timeout, *, stdin=b"", cancel=None):
            if op.startswith("build"):
                if operation == "build":
                    return ProcessCapture(1, b"", b"syntax error", False), b"", b""
                if operation == "timeout":
                    return ProcessCapture(None, b"", b"", True), b"", b""
                return ProcessCapture(0, b"", b"", False), b"binary", b""
            if op == "run":
                return ProcessCapture(0, b"ok", b"", False), b"", b""
            log = b"========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"
            if op == "racecheck":
                log = (
                    b"========= COMPUTE-SANITIZER\n"
                    b"========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
                )
            return ProcessCapture(0, b"ok", b"", False), b"", log

    parent = service.store.create_run("diagnosis")
    result = self_check(
        service.store,
        parent.id,
        {"kernel.cu": b"int main() {return 0;}\n"},
        b"",
        Backend,
        LLMCallGate(),
    )
    assert result.status == status
    if status == "PASSED":
        assert set(result.checks) == {
            "build",
            "runtime",
            "memcheck",
            "racecheck",
            "initcheck",
            "synccheck",
        }


def test_final_verification_failure_is_not_fed_back(oob_service, monkeypatch):
    service, provider, source = oob_service
    monkeypatch.setattr(
        "gpu_agent.repair.self_check",
        lambda *a: PublicCheck(run_id="a" * 32, status="PASSED", checks={}, feedback=[]),
    )
    monkeypatch.setattr(
        service,
        "verify_exact",
        lambda *a, **k: (
            SimpleNamespace(verdict="NOT_FIXED", private_canary="DO_NOT_FEED"),
            "b" * 32,
        ),
    )
    _, result = service.repair(source)
    assert result.verdict == "NOT_FIXED"
    assert provider.kinds.count("patch") == 1
    assert "DO_NOT_FEED" not in json.dumps(provider.inputs)


@pytest.mark.parametrize("verdict,exit_code", [("VERIFIED_FIXED", 0), ("NOT_FIXED", 1), (None, 1)])
def test_repair_cli_reports_actual_verdict(monkeypatch, verdict, exit_code):
    from typer.testing import CliRunner

    from gpu_agent.cli import app
    from gpu_agent.service import ApplicationService

    class Service:
        def allow_development_paid_calls(self, policy):
            assert policy.max_llm_calls == 40

        def repair(self, source, policy):
            assert policy.max_candidates == 3
            result = (
                SimpleNamespace(
                    verdict=verdict, model_dump_json=lambda **k: json.dumps({"verdict": verdict})
                )
                if verdict
                else None
            )
            return SimpleNamespace(id="a" * 32, artifact_refs=[]), result

        def diagnosis(self, run_id):
            return SimpleNamespace(limitations=["PUBLIC_CHECK_UNAVAILABLE"])

    monkeypatch.setattr(ApplicationService, "configured", lambda: Service())
    result = CliRunner().invoke(app, ["repair", "kernel.cu", "--allow-paid-calls"])
    assert result.exit_code == exit_code, result.output
    assert "run_id " + "a" * 32 in result.output
    if verdict:
        assert verdict in result.output


def test_repair_refuses_bound_evaluation_and_ablations(oob_service):
    service, provider, source = oob_service
    for mode in ("A", "B", "C"):
        with pytest.raises(ValueError, match="A-C ablations"):
            service.repair(source, mode=mode)
    service._binding = object()
    with pytest.raises(ValueError, match="non-evaluation"):
        service.repair(source)
    assert not provider.kinds
