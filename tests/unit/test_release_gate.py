import pytest


@pytest.fixture
def manifest():
    from gpu_agent.benchmark.release import ReleaseManifest, TestCounts

    return ReleaseManifest(
        commit="a" * 40,
        toolchain_hash="b" * 64,
        corpus_hash="c" * 64,
        model_config_hash="d" * 64,
        test_counts=TestCounts(expected=10, executed=10, skipped_required=0, failed=0),
        public_case_count=16,
        private_case_count=8,
        evidence_run_ids={
            key: ["run"]
            for key in (
                "four_tools",
                "isolation",
                "private_oracle",
                "live_llm",
                "five_mode_evaluation",
            )
        },
        unresolved_items=[],
    )


def test_complete_manifest_passes(manifest):
    from gpu_agent.benchmark.release import ReleaseGate

    assert ReleaseGate().check(manifest).passed


@pytest.mark.parametrize(
    "update,reason",
    [
        (
            {"test_counts": {"expected": 10, "executed": 9, "skipped_required": 0, "failed": 0}},
            "TEST_COUNT_INCOMPLETE",
        ),
        (
            {"test_counts": {"expected": 10, "executed": 10, "skipped_required": 1, "failed": 0}},
            "REQUIRED_TEST_SKIPPED",
        ),
        ({"private_case_count": 0}, "CORPUS_COUNT_INSUFFICIENT"),
        ({"evidence_run_ids": {}}, "LIVE_EVIDENCE_MISSING"),
        ({"unresolved_items": ["paid evaluation pending"]}, "UNRESOLVED_ITEMS"),
    ],
)
def test_missing_release_evidence_fails(manifest, update, reason):
    from gpu_agent.benchmark.release import ReleaseGate, ReleaseManifest

    changed = ReleaseManifest.model_validate({**manifest.model_dump(), **update})
    result = ReleaseGate().check(changed)
    assert not result.passed and reason in result.reason_codes
