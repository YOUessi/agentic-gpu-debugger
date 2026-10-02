import json

from fastapi.testclient import TestClient

from gpu_agent.agent.models import DiagnosisResult
from gpu_agent.patching import PatchCandidate
from gpu_agent.service import ApplicationService
from gpu_agent.store import RunStore
from gpu_agent.verification.models import VerificationResult, VerificationVerdict
from gpu_agent.web.app import create_app


def _service(tmp_path):
    public = RunStore(tmp_path / "public")
    evaluator = RunStore(tmp_path / "evaluator" / "runs", visibility="evaluator")
    return ApplicationService(public, evaluator)


def _fixture_run(service: ApplicationService) -> str:
    store = service.store
    run = store.create_run("diagnosis")
    store.transition(run.id, "RUNNING", "DIAGNOSING")
    diagnosis = DiagnosisResult(
        diagnostic_outcome="DIAGNOSED",
        failure_family="shared_memory_race",
        root_cause="Two lanes update the same shared-memory location.",
        recommended_change="Separate the writes before the barrier.",
        confidence_label="high",
    )
    store.put(run.id, "diagnosis.json", diagnosis.model_dump_json().encode(), "public")
    store.put(
        run.id,
        "repair/summary.json",
        json.dumps({"stop_reason": "PUBLIC_CHECKS_PASSED", "rounds": []}).encode(),
        "public",
    )
    store.put(
        run.id,
        "actions/0/step.json",
        json.dumps({"action": {"action_type": "run_racecheck"}}).encode(),
        "public",
    )
    store.put(
        run.id,
        "actions/0/decision.json",
        json.dumps({"allowed": True, "action_type": "run_racecheck"}).encode(),
        "public",
    )
    store.put(run.id, "logs/example.txt", b"racecheck finding at kernel.cu:42", "public")
    store.transition(run.id, "RUNNING", "FINALIZING")
    store.transition(run.id, "COMPLETED", None)

    candidate_run = store.create_run("candidate", run.id)
    candidate = PatchCandidate(
        parent_run_id=run.id,
        base_source_hash="1" * 64,
        patched_source_hash="2" * 64,
        unified_diff="--- a/kernel.cu\n+++ b/kernel.cu\n",
        generated_by="agent",
        allowed_paths=["kernel.cu"],
    )
    store.put(
        candidate_run.id,
        "candidate.json",
        candidate.model_dump_json().encode(),
        "public",
    )
    store.transition(candidate_run.id, "RUNNING", "FINALIZING")
    store.transition(candidate_run.id, "COMPLETED", None)

    verification_run = store.create_run("verification", run.id)
    verification = VerificationResult(
        verdict=VerificationVerdict.VERIFIED_FIXED,
        failure_stage=None,
        reason_code="ALL_REQUIRED_CHECKS_PASSED",
        original_finding_present=False,
        public_oracle_passed=True,
        required_checks={"oracle": "PASSED", "racecheck": "CLEAN"},
        candidate_hash="2" * 64,
        public_passed_count=1,
    )
    store.put(
        verification_run.id,
        "verification/result.json",
        verification.model_dump_json().encode(),
        "public",
    )
    store.transition(verification_run.id, "RUNNING", "FINALIZING")
    store.transition(verification_run.id, "COMPLETED", None)
    return run.id


def test_dashboard_api_lists_and_expands_runs(tmp_path):
    service = _service(tmp_path)
    run_id = _fixture_run(service)
    client = TestClient(create_app(service, repository=tmp_path))

    health = client.get("/api/health")
    assert health.status_code == 200

    response = client.get("/api/runs", params={"kind": "diagnosis"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == run_id
    assert body["items"][0]["verification_verdict"] == "VERIFIED_FIXED"

    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["diagnosis"]["failure_family"] == "shared_memory_race"
    assert detail["candidate"]["generated_by"] == "agent"
    assert detail["verifications"][0]["verdict"] == "VERIFIED_FIXED"
    assert detail["actions"][0]["decision"]["allowed"] is True


def test_dashboard_api_stats_and_artifact_text(tmp_path):
    service = _service(tmp_path)
    run_id = _fixture_run(service)
    client = TestClient(create_app(service, repository=tmp_path))

    stats = client.get("/api/stats").json()
    assert stats["total_diagnoses"] == 1
    assert stats["verified_fixed"] == 1
    assert stats["failure_families"]["shared_memory_race"] == 1

    detail = client.get(f"/api/runs/{run_id}").json()
    artifact = next(item for item in detail["artifacts"] if item["name"] == "logs/example.txt")
    text = client.get(f"/api/runs/{run_id}/artifacts/{artifact['id']}").text
    assert "racecheck finding" in text


def test_dashboard_api_rejects_unknown_run(tmp_path):
    client = TestClient(create_app(_service(tmp_path), repository=tmp_path))
    response = client.get("/api/runs/" + "0" * 32)
    assert response.status_code == 404
