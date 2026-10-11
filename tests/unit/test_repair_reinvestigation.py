"""Repair control-flow tests; container/model outputs here are explicitly synthetic."""

import hashlib
import json
from dataclasses import replace
from difflib import unified_diff
from types import SimpleNamespace

import pytest

from gpu_agent.agent.models import FinishAction, MemcheckAction, RetrieveDocsAction
from gpu_agent.public_task import PublicTask
from gpu_agent.repair import PublicCheck, RepairPolicy


@pytest.fixture(autouse=True)
def public_requirement(oob_service):
    _, _, source = oob_service
    task = PublicTask(
        source_sha256=hashlib.sha256((source / "kernel.cu").read_bytes()).hexdigest(),
        algorithm="vector-add-cpu-v1",
    )
    (source / "task.json").write_text(task.model_dump_json())


def artifact(store, run_id, name):
    refs = [r for r in store.load(run_id).artifact_refs if r.name == name]
    return json.loads(store.read(refs[-1]))


def scripted_checks(monkeypatch, outcomes):
    observed = []

    def check(store, parent, sources, stdin, factory, gate, task):
        index = min(len(observed), len(outcomes) - 1)
        status, checks = outcomes[index]
        observed.append(sources)
        run = store.create_run("repair_self_check", parent_run_id=parent)
        store.transition(run.id, "RUNNING", "PREPARING")
        for name, content in sources.items():
            store.put(run.id, f"sources/{name}", content, "public")
        result = PublicCheck(run_id=run.id, status=status, checks=checks, feedback=[])
        store.put(run.id, "self-check.json", result.model_dump_json().encode(), "public")
        store.put(run.id, "self-check-usage.json", b'{"sanitizer_calls":0}', "public")
        store.transition(run.id, "COMPLETED", None)
        return result

    monkeypatch.setattr("gpu_agent.repair.self_check", check)
    return observed


def append_candidate_actions(provider, *, docs=True):
    provider.actions.append(MemcheckAction())
    if docs:
        provider.actions.append(
            RetrieveDocsAction(typed_arguments={"query": "out of bounds", "k": 3})
        )
    provider.actions.append(FinishAction())


def public_output_backend(factory):
    """Give the one-element fixture a valid numeric wire format for fixed-check tests."""

    class Backend(factory):
        def _container(self, path, operation, timeout, *, stdin=b"", cancel=None):
            capture, binary, log = super()._container(
                path, operation, timeout, stdin=stdin, cancel=cancel
            )
            if operation == "run":
                capture = replace(capture, stdout=b'{"dtype":"float32","shape":[1],"values":[3]}')
            return capture, binary, log

    return Backend


def test_failed_candidate_gets_new_diagnosis_without_changing_original_base(
    oob_service, monkeypatch
):
    service, provider, source = oob_service
    fixed = provider.diff
    provider.diff = fixed.replace("i < n", "i <= n")
    append_candidate_actions(provider)
    initial_diagnose = provider.diagnose
    revised = []

    def diagnose(evidence):
        result = initial_diagnose(evidence)
        if evidence.repair_context is not None:
            assert "i <= n" in evidence.sources[0].content
            assert evidence.repair_context.diagnostic_target == "current_candidate"
            return result.model_copy(update={"root_cause": "The candidate still writes past n."})
        return result

    def revise(public_source, diagnosis, feedback):
        provider._record("patch", {"public_repair_feedback": feedback})
        assert "i <= n" not in public_source.content
        assert "i <= n" in feedback["previous_candidate_source"]
        assert diagnosis.root_cause == "The candidate still writes past n."
        assert (
            feedback["diagnosis_source_sha256"]
            == hashlib.sha256(feedback["previous_candidate_source"].encode()).hexdigest()
        )
        assert "secret-canary" not in json.dumps(feedback)
        revised.append(feedback)
        return fixed

    monkeypatch.setattr(provider, "diagnose", diagnose)
    monkeypatch.setattr(provider, "revise_patch", revise)
    checks = scripted_checks(monkeypatch, [("FAILED", {"memcheck": "FINDING"}), ("PASSED", {})])
    verifications = []

    def verify(run_id, strict):
        verifications.append(run_id)
        assert strict and len(checks) == 2
        return SimpleNamespace(verdict="VERIFIED_FIXED"), "f" * 32

    monkeypatch.setattr(service, "verify_exact", verify)
    run, verdict = service.repair(source, policy=RepairPolicy(version="public-repair-v3"))
    assert verdict.verdict == "VERIFIED_FIXED"
    assert verifications == [run.id] and len(revised) == 1
    report = artifact(service.store, run.id, "repair/summary.json")
    assert report["version"] == "public-repair-v3"
    assert report["reinvestigations"] == 1
    assert report["stop_reason"] == "PUBLIC_CHECKS_PASSED"
    child = [r for r in service.store.children(run.id) if r.kind == "repair_reinvestigation"]
    assert len(child) == 1 and child[0].status == "COMPLETED"
    assert service.diagnosis(run.id).root_cause != "The candidate still writes past n."
    parent_source = artifact(service.store, run.id, "evidence/bundle.json")["source_snapshot"][0]
    child_source = artifact(service.store, child[0].id, "evidence/bundle.json")["source_snapshot"][
        0
    ]
    assert parent_source["sha256"] != child_source["sha256"]
    assert all(r.visibility == "public" for r in child[0].artifact_refs)
    final = artifact(service.store, run.id, "agent/final-budget.json")
    assert final["llm_calls"] == 10
    assert final["sanitizer_calls"] == 2 and final["rag_calls"] == 2


@pytest.mark.parametrize(
    "status,checks,max_candidates,max_reinvestigations,expected",
    [
        ("UNAVAILABLE", {"interruption": "TOOL_ERROR"}, 3, 1, "PUBLIC_CHECK_UNAVAILABLE"),
        ("FAILED", {"racecheck": "FINDING"}, 1, 1, "CANDIDATE_LIMIT"),
        ("FAILED", {"racecheck": "FINDING"}, 3, 0, "REPEATED_CANDIDATE"),
        ("FAILED", {"build": "FAILED"}, 3, 1, "REPEATED_CANDIDATE"),
    ],
)
def test_limits_and_build_failures_do_not_start_new_investigation(
    oob_service, monkeypatch, status, checks, max_candidates, max_reinvestigations, expected
):
    service, provider, source = oob_service
    scripted_checks(monkeypatch, [(status, checks)])
    monkeypatch.setattr(
        service, "verify_exact", lambda *a, **k: pytest.fail("private verifier called")
    )
    run, verdict = service.repair(
        source,
        policy=RepairPolicy(
            version="public-repair-v3",
            max_candidates=max_candidates,
            max_reinvestigations=max_reinvestigations,
        ),
    )
    assert verdict is None
    assert provider.kinds.count("diagnose") == 1
    assert artifact(service.store, run.id, "repair/summary.json")["stop_reason"] == expected
    assert not any(r.kind == "repair_reinvestigation" for r in service.store.children(run.id))


def test_inconclusive_candidate_investigation_stops_without_revision(oob_service, monkeypatch):
    from gpu_agent.agent.models import InconclusiveAction

    service, provider, source = oob_service
    provider.actions.append(InconclusiveAction())
    scripted_checks(monkeypatch, [("FAILED", {"racecheck": "FINDING"})])
    monkeypatch.setattr(
        service, "verify_exact", lambda *a, **k: pytest.fail("private verifier called")
    )
    run, verdict = service.repair(source, policy=RepairPolicy(version="public-repair-v3"))
    assert verdict is None and provider.kinds.count("patch") == 1
    report = artifact(service.store, run.id, "repair/summary.json")
    assert report["stop_reason"] == "REINVESTIGATION_INCONCLUSIVE"
    assert report["reinvestigations"] == 1


@pytest.mark.parametrize("mode", ["D", "E"])
def test_functional_failure_can_be_diagnosed_with_clean_sanitizers(oob_service, monkeypatch, mode):
    from gpu_agent.execution.process import ProcessCapture
    from gpu_agent.knowledge.retrieve import KnowledgeIndex

    service, provider, source = oob_service
    # D queries the native finding category rather than E's scripted fixture query.
    # Use the production tokenizer so "write" matches the fixture's "writes".
    service.knowledge = KnowledgeIndex(service.knowledge.chunks, tokenizer_version="cuda-lex-v6")
    original_factory = service._backend_factory

    class CleanCandidateBackend(original_factory):
        def prepare(self, request):
            self.is_candidate = self.store.load(request.run_id).kind == "repair_reinvestigation"
            return super().prepare(request)

        def _container(self, path, operation, timeout, *, stdin=b"", cancel=None):
            if getattr(self, "is_candidate", False) and operation in {
                "run",
                "memcheck",
                "racecheck",
                "initcheck",
                "synccheck",
            }:
                return (
                    ProcessCapture(0, b'{"dtype":"float32","shape":[1],"values":[3]}', b"", False),
                    b"",
                    b"========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n",
                )
            return super()._container(path, operation, timeout, stdin=stdin, cancel=cancel)

    service._backend_factory = CleanCandidateBackend
    append_candidate_actions(provider, docs=False)
    fixed = provider.diff
    provider.diff = fixed.replace("i < n", "i <= n")
    seen = []

    def revise(public_source, diagnosis, feedback):
        provider._record("patch", feedback)
        seen.append(diagnosis)
        assert diagnosis.tool_findings == []
        assert any("functional" in fact.text.lower() for fact in diagnosis.observed_facts)
        return fixed

    monkeypatch.setattr(provider, "revise_patch", revise)
    scripted_checks(
        monkeypatch,
        [("FAILED", {"functional": "SHAPE_MISMATCH"}), ("PASSED", {})],
    )
    monkeypatch.setattr(
        service, "verify_exact", lambda *a, **k: (SimpleNamespace(verdict="NOT_FIXED"), "f" * 32)
    )
    run, verdict = service.repair(
        source, policy=RepairPolicy(version="public-repair-v3"), mode=mode
    )
    assert len(seen) == 1
    assert verdict.verdict == "NOT_FIXED"
    assert provider.kinds.count("patch") == 2
    assert artifact(service.store, run.id, "repair/summary.json")["reinvestigations"] == 1


@pytest.mark.parametrize("fault", ["tool", "cleanup", "cleanup_oserror"])
def test_candidate_infrastructure_failure_keeps_last_checked_candidate(
    oob_service, monkeypatch, fault
):
    from gpu_agent.execution.models import BackendInfrastructureError

    service, provider, source = oob_service
    original_factory = service._backend_factory
    append_candidate_actions(provider)

    class FailingCandidateBackend(original_factory):
        def prepare(self, request):
            self.is_candidate = self.store.load(request.run_id).kind == "repair_reinvestigation"
            return super().prepare(request)

        def run_sanitizer(self, request):
            if self.is_candidate and fault == "tool":
                raise BackendInfrastructureError("synthetic infrastructure failure")
            return super().run_sanitizer(request)

        def cleanup(self, handle):
            super().cleanup(handle)
            if self.is_candidate and fault == "cleanup":
                raise BackendInfrastructureError("synthetic cleanup failure")
            if self.is_candidate and fault == "cleanup_oserror":
                raise PermissionError("synthetic native rmtree failure")

    service._backend_factory = FailingCandidateBackend
    scripted_checks(monkeypatch, [("FAILED", {"racecheck": "FINDING"})])
    run, verdict = service.repair(source, policy=RepairPolicy(version="public-repair-v3"))
    report = artifact(service.store, run.id, "repair/summary.json")
    assert verdict is None
    assert report["stop_reason"] == "REINVESTIGATION_INCONCLUSIVE"
    assert len(service.candidates(run.id)) == 1
    assert provider.kinds.count("patch") == 1
    children = [r for r in service.store.children(run.id) if r.kind == "repair_reinvestigation"]
    assert len(children) == 1 and children[0].status == "COMPLETED"
    assert (
        "EXECUTION_INFRASTRUCTURE_UNAVAILABLE" in report["rounds"][0]["reinvestigation_limitations"]
    )


def test_later_revision_keeps_diagnosis_source_after_reinvestigation_limit(
    oob_service, monkeypatch
):
    from gpu_agent.agent.prompts import REPAIR_PROMPT_VERSION

    service, provider, source = oob_service
    original = (source / "kernel.cu").read_text()
    fixed = provider.diff
    provider.diff = fixed.replace("i < n", "i <= n")
    second_source = original.replace(
        "    out[i] = a[i] + b[i];",
        "    if (i < n - 1) {\n\n        out[i] = a[i] + b[i];\n    }",
    )
    second_diff = "".join(
        unified_diff(
            original.splitlines(True),
            second_source.splitlines(True),
            fromfile="a/kernel.cu",
            tofile="b/kernel.cu",
        )
    )
    append_candidate_actions(provider)
    checked = scripted_checks(
        monkeypatch,
        [
            ("FAILED", {"memcheck": "FINDING"}),
            ("FAILED", {"functional": "NUMERIC_MISMATCH"}),
            ("PASSED", {}),
        ],
    )
    received = []

    def revise(public_source, diagnosis, feedback):
        provider._record("patch", {"public_repair_feedback": feedback})
        assert public_source.content == original
        assert feedback["diagnosis_source"] == checked[0]["kernel.cu"].decode()
        assert (
            feedback["diagnosis_source_sha256"]
            == hashlib.sha256(checked[0]["kernel.cu"]).hexdigest()
        )
        history = feedback["revision_history"]
        assert len(history) == len(received) + 1
        assert [item["round"] for item in history] == list(range(1, len(history) + 1))
        assert all(item["status"] == "FAILED" for item in history)
        assert all(item["candidate_kernel_sha256"] for item in history)
        assert feedback["diagnosis_scoped_to_latest_candidate"] is (len(received) == 0)
        if len(received) == 1:
            assert history[0]["checks"]["memcheck"] == "FINDING"
            assert history[1]["checks"]["functional"] == "NUMERIC_MISMATCH"
            assert history[0]["candidate_kernel_sha256"] != history[1]["candidate_kernel_sha256"]
        received.append(feedback)
        if len(received) == 1:
            return second_diff
        assert feedback["previous_candidate_source"] == second_source
        assert feedback["diagnosis_source"] != feedback["previous_candidate_source"]
        return fixed

    monkeypatch.setattr(provider, "revise_patch", revise)
    monkeypatch.setattr(
        service, "verify_exact", lambda *a, **k: (SimpleNamespace(verdict="NOT_FIXED"), "f" * 32)
    )
    run, _ = service.repair(source, policy=RepairPolicy(version="public-repair-v3"))
    report = artifact(service.store, run.id, "repair/summary.json")
    assert len(received) == 2 and len(checked) == 3
    assert report["reinvestigations"] == 1
    assert report["rounds"][1]["decision"]["reason"] == "REINVESTIGATION_LIMIT"
    for number in (2, 3):
        assert (
            artifact(service.store, run.id, f"repair/{number}/candidate.json")["prompt_version"]
            == REPAIR_PROMPT_VERSION
        )


@pytest.mark.parametrize("error", ["infrastructure", "oserror"])
def test_self_check_cleanup_failure_is_unavailable_and_retains_report(oob_service, error):
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.execution.models import BackendInfrastructureError
    from gpu_agent.repair import self_check

    service, _, source = oob_service
    original_factory = public_output_backend(service._backend_factory)

    class CleanupFailureBackend(original_factory):
        def cleanup(self, handle):
            super().cleanup(handle)
            if error == "oserror":
                raise PermissionError("synthetic native rmtree failure")
            raise BackendInfrastructureError("synthetic cleanup failure")

    parent = service.store.create_run("diagnosis")
    result = self_check(
        service.store,
        parent.id,
        {"kernel.cu": (source / "kernel.cu").read_bytes()},
        b'{"n":1,"a":[1],"b":[2]}',
        CleanupFailureBackend,
        LLMCallGate(),
        PublicTask.model_validate_json((source / "task.json").read_bytes()),
    )
    assert result.status == "UNAVAILABLE"
    assert result.checks["interruption"] == "EXECUTION_INFRASTRUCTURE_UNAVAILABLE"
    assert service.store.load(result.run_id).status == "COMPLETED"
    assert artifact(service.store, result.run_id, "self-check.json") == result.model_dump()
    assert artifact(service.store, result.run_id, "self-check-usage.json")["sanitizer_calls"] > 0


def test_deadline_before_sanitizer_does_not_count_an_unstarted_call(oob_service, monkeypatch):
    from gpu_agent.agent.policy import LLMCallGate
    from gpu_agent.agent.provider import ProviderError
    from gpu_agent.repair import self_check

    service, _, source = oob_service
    original_factory = public_output_backend(service._backend_factory)

    class NoSanitizerBackend(original_factory):
        def run_sanitizer(self, request):
            pytest.fail("expired deadline must stop before the backend invocation")

    gate = LLMCallGate()
    normal_timeout = gate.timeout

    def timeout(seconds):
        if seconds == 60:
            raise ProviderError("LLM_BUDGET_EXHAUSTED")
        return normal_timeout(seconds)

    monkeypatch.setattr(gate, "timeout", timeout)
    parent = service.store.create_run("diagnosis")
    result = self_check(
        service.store,
        parent.id,
        {"kernel.cu": (source / "kernel.cu").read_bytes()},
        b'{"n":1,"a":[1],"b":[2]}',
        NoSanitizerBackend,
        gate,
        PublicTask.model_validate_json((source / "task.json").read_bytes()),
    )
    assert result.status == "UNAVAILABLE"
    assert result.checks["interruption"] == "LLM_BUDGET_EXHAUSTED"
    assert artifact(service.store, result.run_id, "self-check-usage.json")["sanitizer_calls"] == 0


def test_repair_cli_selects_v3_explicitly(monkeypatch):
    from typer.testing import CliRunner

    from gpu_agent.cli import app
    from gpu_agent.service import ApplicationService

    received = []

    class Service:
        def repair(self, source, policy):
            received.append(policy)
            return SimpleNamespace(id="a" * 32, artifact_refs=[]), SimpleNamespace(
                verdict="VERIFIED_FIXED", model_dump_json=lambda **k: "{}"
            )

    monkeypatch.setattr(ApplicationService, "configured", lambda: Service())
    result = CliRunner().invoke(
        app, ["repair", "kernel.cu", "--reinvestigate", "--max-reinvestigations", "2"]
    )
    assert result.exit_code == 0, result.output
    assert received[0].version == "public-repair-v3"
    assert received[0].max_reinvestigations == 2
