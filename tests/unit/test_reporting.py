"""Report only public aggregate scope and recorded toolchain provenance."""

from gpu_agent.evidence.models import EvidenceBundle
from gpu_agent.evidence.repository import EvidenceRepository
from gpu_agent.reporting import render_report
from gpu_agent.verification.models import VerificationResult


def test_report_includes_recorded_scope_counts_and_tool_versions(store):
    run = store.create_run("diagnosis")
    source = store.put(run.id, "sources/kernel.cu", b"int main(){}", "public")
    EvidenceRepository(store).save(
        run.id,
        EvidenceBundle(
            source_snapshot=[source],
            environment={
                "backend": "IsolatedGPUBackend",
                "cuda_nvcc": "12.8.93",
                "compute_sanitizer": "2025.1.0.0",
                "target_arch": "sm_89",
                "image_id": "sha256:" + "a" * 64,
                "private_checker": "private-canary",
            },
        ),
    )
    child = store.create_run("verification", parent_run_id=run.id)
    result = VerificationResult(
        verdict="INCONCLUSIVE",
        failure_stage=None,
        reason_code="TEST",
        original_finding_present=None,
        public_oracle_passed=None,
        required_checks={},
        candidate_hash="b" * 64,
        binary_hashes=["c" * 64],
        public_passed_count=2,
    )
    store.put(child.id, "verification/result.json", result.model_dump_json().encode(), "public")
    report = render_report(store, run.id)
    for expected in [
        source.sha256,
        "12.8.93",
        "2025.1.0.0",
        "sm_89",
        "Verification: UNVERIFIED",
        "base_repo_digest: unavailable",
    ]:
        assert expected in report
    assert "private-canary" not in report and "private_checker" not in report
    assert "Private passed" not in report and "Input-set" not in report


def test_report_marks_missing_scope_and_provenance_unavailable(store):
    report = render_report(store, store.create_run("diagnosis").id)
    assert "Source SHA256: unavailable" in report
    assert "Input-set SHA256" not in report
    assert "cuda_nvcc: unavailable" in report
    assert "compute_sanitizer: unavailable" in report
