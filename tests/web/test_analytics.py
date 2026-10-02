import json

from fastapi.testclient import TestClient

from gpu_agent.service import ApplicationService
from gpu_agent.store import RunStore
from gpu_agent.web.app import create_app


def _service(tmp_path):
    public = RunStore(tmp_path / "ops")
    evaluator = RunStore(tmp_path / "evaluator" / "runs", visibility="evaluator")
    return ApplicationService(public, evaluator)


def _analytics_store(tmp_path):
    store = RunStore(tmp_path / "analytics")

    batch = store.create_run("seed_batch")
    store.transition(batch.id, "RUNNING", "PREPARING")
    batch_payload = {
        "schema_version": 1,
        "batch_run_id": batch.id,
        "status": "COMPLETED",
        "register_requested": True,
        "started_at": "2026-10-03T00:00:00Z",
        "finished_at": "2026-10-03T00:01:00Z",
        "cases": [
            {
                "case_id": "case_0001",
                "target_tool": "memcheck",
                "repetitions": 1,
                "status": "REGISTERED",
                "clean": {
                    "run_id": "1" * 32,
                    "runtime_status": "SUCCESS",
                    "oracle_passed": True,
                    "sanitizer_outcomes": ["CLEAN"],
                },
                "mutant": {
                    "run_id": "2" * 32,
                    "runtime_status": "SUCCESS",
                    "oracle_passed": True,
                    "sanitizer_outcomes": ["FINDING"],
                    "target_detections": [True],
                },
            },
            {
                "case_id": "case_0002",
                "target_tool": "racecheck",
                "repetitions": 5,
                "status": "FAILED",
                "clean": None,
                "mutant": None,
                "reason_code": "TARGET_FINDING_MISSING",
            },
        ],
    }
    store.put(
        batch.id,
        "batch/summary.json",
        json.dumps(batch_payload).encode(),
        "public",
    )
    store.transition(batch.id, "RUNNING", "FINALIZING")
    store.transition(batch.id, "COMPLETED", None)

    records = [
        {
            "record_id": "a" * 32,
            "case_id": "case_0001",
            "template_id": "vector-add-index",
            "mode": "D",
            "repeat": 0,
            "status": "COMPLETED",
            "diagnosis": {
                "diagnostic_outcome": "DIAGNOSED",
                "failure_family": "out_of_bounds",
            },
            "oracle_passed": True,
            "verdict": "VERIFIED_FIXED",
            "usage": {
                "physical_calls": 2,
                "sanitizer_calls": 1,
                "total_tokens": 3000,
            },
            "latency_ms": 1000.0,
            "cost_usd": 0.01,
            "failure_reason": None,
        },
        {
            "record_id": "b" * 32,
            "case_id": "case_0002",
            "template_id": "vector-add-shared",
            "mode": "E",
            "repeat": 0,
            "status": "COMPLETED",
            "diagnosis": {
                "diagnostic_outcome": "DIAGNOSED",
                "failure_family": "shared_memory_race",
            },
            "oracle_passed": False,
            "verdict": "NOT_FIXED",
            "usage": {
                "physical_calls": 4,
                "sanitizer_calls": 2,
                "total_tokens": 5000,
            },
            "latency_ms": 3000.0,
            "cost_usd": 0.02,
            "failure_reason": "PUBLIC_ORACLE_FAILED",
        },
    ]
    evaluation = store.create_run("evaluation")
    store.transition(evaluation.id, "RUNNING", "PREPARING")
    evaluation_payload = {
        "schema_version": 1,
        "run_id": evaluation.id,
        "split": "development",
        "corpus_cutoff": 16,
        "expected_units": 2,
        "executed_units": 2,
        "modes": ["D", "E"],
        "repeats": 1,
        "records": records,
    }
    store.put(
        evaluation.id,
        "evaluation/manifest.json",
        json.dumps(evaluation_payload).encode(),
        "public",
    )
    store.transition(evaluation.id, "RUNNING", "FINALIZING")
    store.transition(evaluation.id, "COMPLETED", None)
    return store, batch.id, evaluation.id


def test_analytics_overview_and_details(tmp_path):
    service = _service(tmp_path)
    analytics_store, batch_id, evaluation_id = _analytics_store(tmp_path)
    client = TestClient(
        create_app(service, repository=tmp_path, analytics_store=analytics_store)
    )

    overview = client.get("/api/analytics/overview")
    assert overview.status_code == 200
    body = overview.json()
    assert body["batch_count"] == 1
    assert body["evaluation_count"] == 1
    assert body["batches"][0]["registered"] == 1
    assert body["batches"][0]["failed"] == 1
    assert body["evaluations"][0]["executed_units"] == 2
    assert body["evaluations"][0]["verified_fixed"] == 1
    assert body["evaluations"][0]["verified_rate"] == 0.5
    assert body["evaluations"][0]["llm_calls_mean"] == 3.0

    detail = client.get(
        f"/api/analytics/evaluations/{evaluation_id}",
        params={"mode": "E", "page": 1, "page_size": 20},
    )
    assert detail.status_code == 200
    eval_body = detail.json()
    assert eval_body["total"] == 1
    assert eval_body["records"][0]["case_id"] == "case_0002"
    assert eval_body["records"][0]["verdict"] == "NOT_FIXED"
    assert {item["mode"] for item in eval_body["mode_metrics"]} == {"D", "E"}

    batch = client.get(f"/api/analytics/batches/{batch_id}")
    assert batch.status_code == 200
    batch_body = batch.json()
    assert batch_body["register_requested"] is True
    assert batch_body["cases"][0]["target_detections"] == [True]
    assert batch_body["cases"][1]["reason_code"] == "TARGET_FINDING_MISSING"


def test_analytics_endpoints_do_not_fall_back_to_evaluator_store(tmp_path):
    service = _service(tmp_path)
    analytics_store = RunStore(tmp_path / "analytics")
    client = TestClient(
        create_app(service, repository=tmp_path, analytics_store=analytics_store)
    )

    overview = client.get("/api/analytics/overview").json()
    assert overview["batch_count"] == 0
    assert overview["evaluation_count"] == 0
    missing = client.get("/api/analytics/evaluations/" + "0" * 32)
    assert missing.status_code == 404


def test_analytics_overview_surfaces_invalid_public_projection(tmp_path):
    service = _service(tmp_path)
    analytics_store = RunStore(tmp_path / "analytics")
    broken = analytics_store.create_run("evaluation")
    analytics_store.transition(broken.id, "RUNNING", "PREPARING")
    analytics_store.put(
        broken.id,
        "evaluation/manifest.json",
        b"{not-json",
        "public",
    )
    analytics_store.transition(broken.id, "RUNNING", "FINALIZING")
    analytics_store.transition(broken.id, "COMPLETED", None)

    client = TestClient(
        create_app(service, repository=tmp_path, analytics_store=analytics_store)
    )
    body = client.get("/api/analytics/overview").json()

    assert body["evaluation_count"] == 0
    assert body["projection_errors"] == [
        f"{broken.id}:ANALYTICS_PROJECTION_INVALID"
    ]


def test_analytics_rejects_root_containing_evaluator_visibility_artifacts(tmp_path):
    service = _service(tmp_path)
    evaluator_root = tmp_path / "private-analytics"
    evaluator = RunStore(evaluator_root, visibility="evaluator")
    run = evaluator.create_run("evaluation")
    evaluator.transition(run.id, "RUNNING", "PREPARING")
    evaluator.put(
        run.id,
        "evaluation/manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run.id,
                "executed_units": 0,
                "records": [],
            }
        ).encode(),
        "evaluator",
    )
    evaluator.transition(run.id, "RUNNING", "FINALIZING")
    evaluator.transition(run.id, "COMPLETED", None)

    # Simulate a misconfigured path being reopened with the default public label.
    disguised = RunStore(evaluator_root)
    client = TestClient(
        create_app(service, repository=tmp_path, analytics_store=disguised)
    )
    response = client.get("/api/analytics/overview")

    assert response.status_code == 503
    assert response.json()["detail"] == "ANALYTICS_STORE_UNSAFE"
