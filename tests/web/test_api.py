import hashlib
import json
import time

from fastapi.testclient import TestClient

from gpu_agent.agent.models import DiagnosisResult, EvidenceClaim
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
    evidence_ref = store.put(
        run.id,
        "logs/example.txt",
        b"racecheck finding at kernel.cu:42",
        "public",
    )
    chunk_id = "nvcuda-test-citation"
    store.put(
        run.id,
        f"docs/{chunk_id}.json",
        json.dumps(
            {
                "chunk_id": chunk_id,
                "document_title": "Compute Sanitizer Guide",
                "section_title": "Racecheck",
                "source_url": "https://docs.nvidia.com/compute-sanitizer/",
                "text": "Racecheck reports shared-memory hazards between CUDA threads.",
            }
        ).encode(),
        "public",
    )
    diagnosis = DiagnosisResult(
        diagnostic_outcome="DIAGNOSED",
        failure_family="shared_memory_race",
        root_cause="Two lanes update the same shared-memory location.",
        recommended_change="Separate the writes before the barrier.",
        confidence_label="high",
        tool_findings=[
            EvidenceClaim(
                text="Racecheck found a shared-memory hazard.",
                citation_ids=[evidence_ref.id],
            )
        ],
        documentation_evidence=[
            EvidenceClaim(text="NVIDIA documents this hazard class.", citation_ids=[chunk_id])
        ],
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
    assert detail["citations"]["nvcuda-test-citation"]["kind"] == "document"
    assert any(item["kind"] == "artifact" for item in detail["citations"].values())


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

def test_dashboard_api_lists_repair_ready_public_cases(tmp_path):
    repository = tmp_path / "repository"
    input_root = repository / "benchmarks" / "public" / "case_0021" / "public_input"
    input_root.mkdir(parents=True)
    source = b'#include "vector_api.h"\nextern "C" __global__ void kernel() {}\n'
    (input_root / "kernel.cu").write_bytes(source)
    (input_root / "input.json").write_text('{"n":1,"a":[1.0],"b":[2.0]}')
    (input_root / "task.json").write_text(
        json.dumps(
            {
                "version": "public-task-v1",
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "algorithm": "vector-add-cpu-v1",
            }
        )
    )
    (repository / "benchmarks" / "diverse-registry.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": "case_0021",
                        "template_id": "vector-add",
                        "mutation_id": "delete-guard",
                        "target_tool": "memcheck",
                        "expected_finding": "Invalid __global__ read",
                    }
                ],
            }
        )
    )

    client = TestClient(create_app(_service(tmp_path), repository=repository))
    response = client.get("/api/cases")
    assert response.status_code == 200
    body = response.json()
    assert body == [
        {
            "case_id": "case_0021",
            "algorithm": "vector-add-cpu-v1",
            "requirement": "For every i in [0,n), output[i] = a[i] + b[i].",
            "template_id": "vector-add",
            "mutation_id": "delete-guard",
            "target_tool": "memcheck",
            "expected_finding": "Invalid __global__ read",
            "repair_ready": True,
        }
    ]


def test_async_repair_api_returns_before_workflow_finishes(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    source = repository / "benchmarks" / "public" / "case_0021" / "public_input"
    source.mkdir(parents=True)
    service = _service(tmp_path)

    def fake_repair(selected, *, policy, mode, on_run_created):
        assert selected == source
        assert policy.max_candidates == 2
        assert mode == "E"
        run = service.store.create_run("diagnosis")
        on_run_created(run)
        service.store.transition(run.id, "RUNNING", "FINALIZING")
        completed = service.store.transition(run.id, "COMPLETED", None)
        return completed, None

    monkeypatch.setattr(service, "repair", fake_repair)
    client = TestClient(create_app(service, repository=repository))

    submitted = client.post(
        "/api/jobs/repair",
        json={
            "case_id": "case_0021",
            "mode": "E",
            "max_candidates": 2,
            "max_llm_calls": 8,
            "allow_paid_calls": False,
        },
    )
    assert submitted.status_code == 202
    job = submitted.json()
    assert job["status"] == "QUEUED"
    assert job["run_id"] is None

    deadline = time.monotonic() + 2
    current = job
    while current["status"] not in {"COMPLETED", "FAILED"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
        response = client.get(f"/api/jobs/{job['id']}")
        assert response.status_code == 200
        current = response.json()

    assert current["status"] == "COMPLETED"
    assert current["run_id"] is not None
    run = client.get(f"/api/runs/{current['run_id']}")
    assert run.status_code == 200
    assert run.json()["summary"]["status"] == "COMPLETED"
