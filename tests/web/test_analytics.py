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

    diagnosis = store.create_run("diagnosis")
    store.transition(diagnosis.id, "RUNNING", "DIAGNOSING")
    source_ref = store.put(
        diagnosis.id,
        "sources/kernel.cu",
        b'extern "C" __global__ void kernel(float *x) { x[threadIdx.x] = 1.0f; }\n',
        "public",
    )
    store.put(
        diagnosis.id,
        "diagnosis.json",
        json.dumps(
            {
                "diagnostic_outcome": "DIAGNOSED",
                "failure_family": "shared_memory_race",
                "root_cause": "Two threads update the same location without synchronization.",
                "recommended_change": "Separate writes and synchronize before reuse.",
                "confidence_label": "high",
                "observed_facts": [
                    {
                        "text": "CUDA source was captured for the evaluation run.",
                        "citation_ids": [source_ref.id],
                    }
                ],
                "tool_findings": [],
                "documentation_evidence": [],
            }
        ).encode(),
        "public",
    )
    store.transition(diagnosis.id, "RUNNING", "FINALIZING")
    store.transition(diagnosis.id, "COMPLETED", None)

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
            "lineage": {
                "kind": "native",
                "diagnosis_run_id": diagnosis.id,
                "candidate_run_id": "c" * 32,
                "verification_run_id": "d" * 32,
            },
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
    return store, batch.id, evaluation.id, diagnosis.id, source_ref.id


def test_analytics_overview_and_details(tmp_path):
    service = _service(tmp_path)
    analytics_store, batch_id, evaluation_id, diagnosis_id, source_artifact_id = _analytics_store(
        tmp_path
    )
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
    assert eval_body["records"][0]["diagnosis_run_id"] == diagnosis_id
    assert eval_body["records"][0]["candidate_run_id"] == "c" * 32
    assert eval_body["records"][0]["verification_run_id"] == "d" * 32
    assert {item["mode"] for item in eval_body["mode_metrics"]} == {"D", "E"}

    batch = client.get(f"/api/analytics/batches/{batch_id}")
    assert batch.status_code == 200
    batch_body = batch.json()
    assert batch_body["register_requested"] is True
    assert batch_body["cases"][0]["target_detections"] == [True]
    assert batch_body["cases"][1]["reason_code"] == "TARGET_FINDING_MISSING"

    run_detail = client.get(f"/api/analytics/runs/{diagnosis_id}")
    assert run_detail.status_code == 200
    run_body = run_detail.json()
    assert run_body["summary"]["id"] == diagnosis_id
    assert run_body["summary"]["failure_family"] == "shared_memory_race"
    assert "Two threads update the same location" in run_body["diagnosis"]["root_cause"]

    artifact = client.get(
        f"/api/analytics/runs/{diagnosis_id}/artifacts/{source_artifact_id}"
    )
    assert artifact.status_code == 200
    assert "__global__ void kernel" in artifact.text


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


def test_evaluation_comparison_detects_unit_regressions_and_improvements(tmp_path):
    service = _service(tmp_path)
    store = RunStore(tmp_path / "analytics")

    def add_eval(verdicts):
        run = store.create_run("evaluation")
        store.transition(run.id, "RUNNING", "PREPARING")
        records = []
        for index, verdict in enumerate(verdicts):
            records.append(
                {
                    "record_id": f"{index + 1:032x}",
                    "lineage": {
                        "diagnosis_run_id": f"{index + 10:032x}",
                    },
                    "case_id": f"case_{index + 1:04d}",
                    "template_id": f"template-{index + 1}",
                    "mode": "D",
                    "repeat": 0,
                    "status": "COMPLETED",
                    "diagnosis": {
                        "diagnostic_outcome": "DIAGNOSED",
                        "failure_family": "out_of_bounds",
                    },
                    "oracle_passed": verdict == "VERIFIED_FIXED",
                    "verdict": verdict,
                    "usage": {
                        "physical_calls": 2 + index,
                        "sanitizer_calls": 1,
                        "total_tokens": 1000 + 100 * index,
                    },
                    "latency_ms": 1000.0 + 100.0 * index,
                    "cost_usd": 0.01 + 0.001 * index,
                    "failure_reason": None,
                }
            )
        store.put(
            run.id,
            "evaluation/manifest.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run.id,
                    "split": "development",
                    "corpus_cutoff": 16,
                    "expected_units": len(records),
                    "executed_units": len(records),
                    "modes": ["D"],
                    "repeats": 1,
                    "records": records,
                }
            ).encode(),
            "public",
        )
        store.transition(run.id, "RUNNING", "FINALIZING")
        store.transition(run.id, "COMPLETED", None)
        return run.id

    baseline_id = add_eval(["VERIFIED_FIXED", "NOT_FIXED", "VERIFIED_FIXED"])
    candidate_id = add_eval(["NOT_FIXED", "VERIFIED_FIXED", "VERIFIED_FIXED"])
    client = TestClient(create_app(service, repository=tmp_path, analytics_store=store))

    response = client.get(
        "/api/analytics/evaluations/compare",
        params={"baseline": baseline_id, "candidate": candidate_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["comparable"] is True
    assert body["reasons"] == []
    assert body["matched_units"] == 3
    assert body["regressions"] == 1
    assert body["improvements"] == 1
    assert body["unchanged"] == 1
    assert body["regression_rows"][0]["case_id"] == "case_0001"
    assert body["regression_rows"][0]["baseline_diagnosis_run_id"] == f"{10:032x}"
    assert body["regression_rows"][0]["candidate_diagnosis_run_id"] == f"{10:032x}"
    assert len(body["mode_comparisons"]) == 1


def test_evaluation_comparison_refuses_different_populations(tmp_path):
    service = _service(tmp_path)
    store = RunStore(tmp_path / "analytics")

    def add_eval(split, cutoff):
        run = store.create_run("evaluation")
        store.transition(run.id, "RUNNING", "PREPARING")
        store.put(
            run.id,
            "evaluation/manifest.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run.id,
                    "split": split,
                    "corpus_cutoff": cutoff,
                    "expected_units": 1,
                    "executed_units": 1,
                    "modes": ["D"],
                    "repeats": 1,
                    "records": [
                        {
                            "record_id": "a" * 32,
                            "case_id": "case_0001",
                            "template_id": "template",
                            "mode": "D",
                            "repeat": 0,
                            "status": "COMPLETED",
                            "diagnosis": {
                                "diagnostic_outcome": "DIAGNOSED",
                                "failure_family": "out_of_bounds",
                            },
                            "verdict": "VERIFIED_FIXED",
                            "usage": {},
                        }
                    ],
                }
            ).encode(),
            "public",
        )
        store.transition(run.id, "RUNNING", "FINALIZING")
        store.transition(run.id, "COMPLETED", None)
        return run.id

    baseline_id = add_eval("development", 16)
    candidate_id = add_eval("holdout", 24)
    client = TestClient(create_app(service, repository=tmp_path, analytics_store=store))

    body = client.get(
        "/api/analytics/evaluations/compare",
        params={"baseline": baseline_id, "candidate": candidate_id},
    ).json()
    assert body["comparable"] is False
    assert "SPLIT_MISMATCH" in body["reasons"]
    assert "CORPUS_CUTOFF_MISMATCH" in body["reasons"]
    assert body["matched_units"] == 0
    assert body["regressions"] == 0
    assert body["overall_delta"]["verified_rate_delta"] is None


def test_evaluation_exports_full_public_records_and_sanitizes_csv_cells(tmp_path):
    service = _service(tmp_path)
    store = RunStore(tmp_path / "analytics")
    run = store.create_run("evaluation")
    store.transition(run.id, "RUNNING", "PREPARING")
    store.put(
        run.id,
        "evaluation/manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run.id,
                "split": "development",
                "corpus_cutoff": 1,
                "expected_units": 1,
                "executed_units": 1,
                "modes": ["D"],
                "repeats": 1,
                "records": [
                    {
                        "record_id": "a" * 32,
                        "case_id": "case_0001",
                        "template_id": "=SUM(A1:A2)",
                        "mode": "D",
                        "repeat": 0,
                        "status": "COMPLETED",
                        "diagnosis": {
                            "diagnostic_outcome": "DIAGNOSED",
                            "failure_family": "out_of_bounds",
                        },
                        "verdict": "VERIFIED_FIXED",
                        "usage": {"physical_calls": 2},
                    }
                ],
            }
        ).encode(),
        "public",
    )
    store.transition(run.id, "RUNNING", "FINALIZING")
    store.transition(run.id, "COMPLETED", None)
    client = TestClient(create_app(service, repository=tmp_path, analytics_store=store))

    json_export = client.get(f"/api/analytics/evaluations/{run.id}/export.json")
    assert json_export.status_code == 200
    assert "attachment;" in json_export.headers["content-disposition"]
    assert json_export.json()["total"] == 1

    csv_export = client.get(f"/api/analytics/evaluations/{run.id}/export.csv")
    assert csv_export.status_code == 200
    assert "attachment;" in csv_export.headers["content-disposition"]
    assert "template_id" in csv_export.text
    assert "'=SUM(A1:A2)" in csv_export.text
