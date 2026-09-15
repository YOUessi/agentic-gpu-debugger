import hashlib
import json
import shutil
from datetime import UTC, datetime

import pytest

COMMIT = "a" * 40
TOOLCHAIN = "b" * 64
MODEL = "c" * 64


def _complete(store, kind, artifacts):
    run = store.create_run(kind)
    for name, value in artifacts.items():
        content = value if isinstance(value, bytes) else value.model_dump_json().encode()
        store.put(run.id, name, content, store.visibility)
    store.transition(run.id, "RUNNING", "FINALIZING")
    store.transition(run.id, "COMPLETED", None)
    return store.load(run.id)


def _junit(tmp_path, files):
    cases = "".join(
        f'<testcase classname="{path.removesuffix(".py").replace("/", ".")}" '
        f'name="test_release_surface_{index}" />'
        for index, path in enumerate(files)
    )
    path = tmp_path / "release.xml"
    path.write_text(f'<testsuite tests="{len(files)}">{cases}</testsuite>')
    return path


def _receipt(store, category, artifact_name="proof.json"):
    from gpu_agent.agent.models import DiagnosisResult
    from gpu_agent.agent.provider import Invocation
    from gpu_agent.benchmark.release import (
        FourToolsAcceptanceProof,
        IsolationAcceptanceProof,
        record_release_evidence,
    )
    from gpu_agent.verification.models import VerificationObservation

    source_kind = {
        "four_tools": "sanitizer_acceptance",
        "isolation": "isolation_acceptance",
        "private_oracle": "verification_audit",
        "live_llm": "diagnosis",
    }[category]
    source = store.create_run(source_kind)
    artifacts = {}
    if category == "four_tools":
        artifacts["acceptance/four-tools.json"] = FourToolsAcceptanceProof(
            clean_outcomes={tool: "CLEAN" for tool in (
                "memcheck", "racecheck", "initcheck", "synccheck"
            )},
            fault_outcomes={tool: "FINDING" for tool in (
                "memcheck", "racecheck", "initcheck", "synccheck"
            )},
        ).model_dump_json().encode()
    elif category == "isolation":
        artifacts["acceptance/isolation.json"] = IsolationAcceptanceProof(
            backend="isolated_gpu",
            network_disabled=True,
            read_only_root=True,
            bounded_resources=True,
            timeout_cleanup_verified=True,
            candidate_build_and_run_verified=True,
        ).model_dump_json().encode()
    elif category == "private_oracle":
        artifacts["observation.json"] = VerificationObservation(
            private_holdout_passed=True,
            required_evidence_missing=False,
        ).model_dump_json().encode()
    else:
        artifacts["diagnosis.json"] = DiagnosisResult(
            diagnostic_outcome="DIAGNOSED",
            failure_family="out_of_bounds",
            root_cause="index exceeded allocation",
        ).model_dump_json().encode()
        invocation = Invocation(
            invocation_id="invocation-1",
            run_id=source.id,
            kind="diagnose",
            attempt=0,
            state="COMPLETED",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            elapsed_ms=1,
            configured_model="frozen-model",
            response_model="frozen-model",
            endpoint_host="provider.invalid",
            client_request_id="request-1",
            response_id="response-1",
            store_false_sent=True,
        )
        artifacts["provider/invocation-1/COMPLETED.json"] = invocation.model_dump_json().encode()
    for name, content in artifacts.items():
        store.put(source.id, name, content, store.visibility)
    store.transition(source.id, "RUNNING", "FINALIZING")
    store.transition(source.id, "COMPLETED", None)
    receipt = record_release_evidence(
        store,
        source.id,
        category,
        current_commit=COMMIT,
        toolchain_hash=TOOLCHAIN,
    )
    return receipt.run_id


def _validation_run(store):
    from gpu_agent.benchmark.release import CorpusValidationProof, record_release_evidence

    source = store.create_run("benchmark_validation")
    proof = CorpusValidationProof(
        clean_oracle_passed=True,
        required_checks_clean=True,
        target_confirmed=True,
    ).model_dump_json().encode()
    store.put(source.id, "validation/result.json", proof, store.visibility)
    store.transition(source.id, "RUNNING", "FINALIZING")
    store.transition(source.id, "COMPLETED", None)
    record_release_evidence(
        store,
        source.id,
        "corpus_validation",
        current_commit=COMMIT,
        toolchain_hash=TOOLCHAIN,
    )
    return source.id


def _case(store, number, split, tool, validation_ids, *, template=None, mutation=None):
    from gpu_agent.benchmark.models import CaseManifest

    case = CaseManifest(
        id=f"case_{number:04d}",
        source_hash=f"{number + 1:064x}",
        harness_hash="d" * 64,
        mutation_id=mutation or f"mutation_{number:04d}",
        template_id=template or f"template_{number:04d}",
        split=split,
        oracle_id=f"oracle-{number}",
        target_tool=tool,
        expected_finding="target finding",
        validation_run_ids=validation_ids,
        toolchain_hash=TOOLCHAIN,
        input_set_hash=f"{number + 100:064x}",
    )
    _complete(store, "benchmark_case", {"case-manifest.json": case})
    return case


def _evaluation(store, cases, split, evaluator=None):
    from gpu_agent.benchmark.evaluation import (
        EvaluationAttempt,
        EvaluationBindings,
        EvaluationManifest,
        EvaluationSchedule,
        EvaluationScheduleItem,
        PublicEvaluationRecord,
    )

    run = store.create_run("evaluation")
    identities = [
        (
            "holdout-" + hashlib.sha256(f"case-{index}".encode()).hexdigest()[:16]
            if split == "holdout"
            else case.id,
            "opaque-" + hashlib.sha256(f"template-{index}".encode()).hexdigest()[:16]
            if split == "holdout"
            else case.template_id,
            case,
        )
        for index, case in enumerate(cases)
    ]
    items = []
    for case_id, template_id, _case_manifest in identities:
        for mode in "ABCDE":
            for repeat in range(3):
                items.append(
                    EvaluationScheduleItem(
                        ordinal=len(items),
                        case_id=case_id,
                        template_id=template_id,
                        mode=mode,
                        repeat=repeat,
                    )
                )
    schedule = EvaluationSchedule(
        selection="all",
        modes=list("ABCDE"),
        split=split,
        repeats=3,
        random_seed=7,
        bindings=EvaluationBindings(
            commit=COMMIT,
            prompt_version="v2",
            toolchain_hash=TOOLCHAIN,
            model_config_hash=MODEL,
            max_cost_usd=10,
            max_unit_cost_usd=0.01,
        ),
        items=items,
    )
    encoded = json.dumps(
        schedule.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    schedule_hash = hashlib.sha256(encoded).hexdigest()
    store.put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode(), "public")
    records = []
    for item in items:
        record = PublicEvaluationRecord(
            record_id=f"record-{split}-{item.ordinal}",
            case_id=item.case_id,
            template_id=item.template_id,
            mode=item.mode,
            repeat=item.repeat,
            input_hash="e" * 64,
            evidence_hash="f" * 64,
            executed_checks={},
            status="COMPLETED",
            diagnosis={},
            latency_ms=1,
            cost_usd=0,
        )
        records.append(record)
        attempt = EvaluationAttempt(
            run_id=run.id,
            ordinal=item.ordinal,
            schedule_hash=schedule_hash,
            idempotency_key=hashlib.sha256(f"{run.id}:{item.ordinal}".encode()).hexdigest(),
            reserved_cost_usd=0.01,
        )
        store.put(
            run.id,
            f"evaluation/attempts/{item.ordinal}.json",
            attempt.model_dump_json().encode(),
            "public",
        )
        store.put(
            run.id,
            f"evaluation/records/{item.ordinal}.json",
            record.model_dump_json().encode(),
            "public",
        )
    manifest = EvaluationManifest(
        run_id=run.id,
        commit=COMMIT,
        prompt_version="v2",
        toolchain_hash=TOOLCHAIN,
        model_config_hash=MODEL,
        schedule_hash=schedule_hash,
        expected_units=len(items),
        executed_units=len(items),
        modes=list("ABCDE"),
        split=split,
        repeats=3,
        random_seed=7,
        records=records,
    )
    store.put(run.id, "evaluation/manifest.json", manifest.model_dump_json().encode(), "public")
    store.transition(run.id, "RUNNING", "FINALIZING")
    store.transition(run.id, "COMPLETED", None)
    if split == "holdout":
        from gpu_agent.benchmark.release import (
            HoldoutIdentity,
            HoldoutIdentityMap,
            record_holdout_identity_map,
        )

        assert evaluator is not None
        identity_map = HoldoutIdentityMap(
            commit=COMMIT,
            toolchain_hash=TOOLCHAIN,
            evaluation_run_id=run.id,
            identities=[
                HoldoutIdentity(
                    evaluation_case_id=case_id,
                    evaluation_template_id=template_id,
                    private_case_id=case.id,
                    private_template_id=case.template_id,
                )
                for case_id, template_id, case in identities
            ],
        )
        record_holdout_identity_map(store, evaluator, identity_map)
    return run.id


@pytest.fixture(scope="module")
def release_state(tmp_path_factory):
    from gpu_agent.benchmark.release import record_pytest_report
    from gpu_agent.store import RunStore

    root = tmp_path_factory.mktemp("release-state")
    public = RunStore(root / "public")
    evaluator = RunStore(root / "evaluator", visibility="evaluator")
    public_validation = [_validation_run(public) for _ in range(2)]
    private_validation = [_validation_run(evaluator) for _ in range(2)]
    tools = ["memcheck", "racecheck", "initcheck", "synccheck"]
    public_cases = [
        _case(public, index, "public", tools[index // 4], public_validation)
        for index in range(16)
    ]
    private_cases = [
        _case(evaluator, 100 + index, "private", tools[index // 2], private_validation)
        for index in range(8)
    ]
    evidence = {
        category: [_receipt(evaluator if category == "private_oracle" else public, category)]
        for category in ("four_tools", "isolation", "private_oracle", "live_llm")
    }
    evidence["five_mode_evaluation"] = [
        _evaluation(public, public_cases, "development"),
        _evaluation(public, private_cases, "holdout", evaluator),
    ]
    record_pytest_report(
        public,
        _junit(
            root,
            [
                "tests/integration/test_isolation.py",
                "tests/gpu/test_failure_families.py",
                "tests/gpu/test_candidate_verification.py",
                "tests/e2e/test_oob_flow.py",
                "tests/unit/test_evaluation_modes.py",
                "tests/gpu/test_mutation_validation.py",
                "tests/e2e/test_release_acceptance.py",
            ],
        ),
        current_commit=COMMIT,
    )
    return public, evaluator, evidence


def _manifest(index, evidence):
    from gpu_agent.benchmark.release import ReleaseManifest

    return ReleaseManifest(
        commit=COMMIT,
        toolchain_hash=TOOLCHAIN,
        corpus_hash=index.corpus_hash,
        model_config_hash=MODEL,
        test_counts=index.test_counts,
        public_case_count=16,
        private_case_count=8,
        evidence_run_ids=evidence,
        unresolved_items=[],
    )


def test_complete_manifest_passes_only_against_derived_evidence(release_state):
    from gpu_agent.benchmark.release import ReleaseEvidenceIndex, ReleaseGate

    public, evaluator, evidence = release_state
    index = ReleaseEvidenceIndex.derive(public, evaluator, current_commit=COMMIT)
    assert ReleaseGate().check(_manifest(index, evidence), index).passed


def test_zero_tests_and_nonexistent_run_ids_are_not_evidence(tmp_path):
    from gpu_agent.benchmark.release import (
        ReleaseEvidenceIndex,
        ReleaseGate,
        ReleaseManifest,
        TestCounts,
    )
    from gpu_agent.store import RunStore

    public = RunStore(tmp_path / "public")
    evaluator = RunStore(tmp_path / "evaluator", visibility="evaluator")
    index = ReleaseEvidenceIndex.derive(public, evaluator, current_commit=COMMIT)
    manifest = ReleaseManifest(
        commit=COMMIT,
        toolchain_hash=TOOLCHAIN,
        corpus_hash="d" * 64,
        model_config_hash=MODEL,
        test_counts=TestCounts(expected=0, executed=0, skipped_required=0, failed=0),
        public_case_count=16,
        private_case_count=8,
        evidence_run_ids={key: ["nonexistent"] for key in ReleaseGate.REQUIRED_EVIDENCE},
        unresolved_items=[],
    )
    result = ReleaseGate().check(manifest, index)
    assert not result.passed
    assert "TEST_COUNT_ZERO" in result.reason_codes
    assert "EVIDENCE_RUN_MISMATCH" in result.reason_codes


def test_arbitrary_category_json_cannot_be_recorded_as_release_evidence(tmp_path):
    from gpu_agent.benchmark.release import record_release_evidence
    from gpu_agent.store import RunStore

    store = RunStore(tmp_path / "public")
    source = _complete(
        store,
        "isolation_acceptance",
        {"proof.json": b'{"category":"isolation"}'},
    )
    with pytest.raises(ValueError):
        record_release_evidence(
            store,
            source.id,
            "isolation",
            current_commit=COMMIT,
            toolchain_hash=TOOLCHAIN,
        )


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("commit", "0" * 40, "COMMIT_MISMATCH"),
        ("toolchain_hash", "0" * 64, "TOOLCHAIN_HASH_MISMATCH"),
        ("corpus_hash", "0" * 64, "CORPUS_HASH_MISMATCH"),
        ("model_config_hash", "0" * 64, "MODEL_CONFIG_HASH_MISMATCH"),
    ],
)
def test_manifest_bindings_must_match_derived_artifacts(release_state, field, value, reason):
    from gpu_agent.benchmark.release import ReleaseEvidenceIndex, ReleaseGate

    public, evaluator, evidence = release_state
    index = ReleaseEvidenceIndex.derive(public, evaluator, current_commit=COMMIT)
    manifest = _manifest(index, evidence).model_copy(update={field: value})
    result = ReleaseGate().check(manifest, index)
    assert not result.passed and reason in result.reason_codes


def test_cross_split_template_or_mutation_overlap_is_rejected(tmp_path):
    from gpu_agent.benchmark.release import ReleaseEvidenceIndex
    from gpu_agent.store import RunStore

    public = RunStore(tmp_path / "public")
    evaluator = RunStore(tmp_path / "evaluator", visibility="evaluator")
    public_ids = [_validation_run(public) for _ in range(2)]
    private_ids = [_validation_run(evaluator) for _ in range(2)]
    _case(public, 1, "public", "memcheck", public_ids, template="shared-template")
    _case(
        evaluator,
        2,
        "private",
        "racecheck",
        private_ids,
        template="shared-template",
    )
    index = ReleaseEvidenceIndex.derive(public, evaluator, current_commit=COMMIT)
    assert "CORPUS_SPLIT_OVERLAP" in index.integrity_errors


@pytest.mark.parametrize("missing", ["mode", "repeat", "case", "record"])
def test_evaluation_requires_every_scheduled_persisted_unit(release_state, missing, tmp_path):
    from gpu_agent.benchmark.release import ReleaseEvidenceIndex
    from gpu_agent.store import RunStore

    source_public, source_evaluator, _ = release_state
    shutil.copytree(source_public.root, tmp_path / "public")
    shutil.copytree(source_evaluator.root, tmp_path / "evaluator")
    public = RunStore(tmp_path / "public")
    evaluator = RunStore(tmp_path / "evaluator", visibility="evaluator")
    evaluation = next(
        public.load(path.name)
        for path in public.root.iterdir()
        if path.is_dir() and public.load(path.name).kind == "evaluation"
    )
    manifest_ref = next(
        ref for ref in evaluation.artifact_refs if ref.name == "evaluation/manifest.json"
    )
    blob = public.root / manifest_ref.relative_path
    data = json.loads(blob.read_bytes())
    if missing == "record":
        data["records"] = data["records"][:-1]
    elif missing == "mode":
        data["modes"] = data["modes"][:-1]
    elif missing == "repeat":
        data["repeats"] = 4
    else:
        data["records"] = [row for row in data["records"] if row["case_id"] != "case_0000"]
    blob.chmod(0o600)
    blob.write_text(json.dumps(data))
    blob.chmod(0o400)
    index = ReleaseEvidenceIndex.derive(public, evaluator, current_commit=COMMIT)
    assert "ARTIFACT_INTEGRITY_FAILURE" in index.integrity_errors
    assert not index.evaluation_complete


def test_release_test_report_requires_every_real_surface(tmp_path):
    from gpu_agent.benchmark.release import (
        ReleaseEvidenceIndex,
        ReleaseGate,
        ReleaseManifest,
        TestCounts,
        record_pytest_report,
    )
    from gpu_agent.store import RunStore

    public = RunStore(tmp_path / "public")
    evaluator = RunStore(tmp_path / "evaluator", visibility="evaluator")
    record_pytest_report(
        public,
        _junit(tmp_path, ["tests/e2e/test_release_acceptance.py"]),
        current_commit=COMMIT,
    )
    index = ReleaseEvidenceIndex.derive(public, evaluator, current_commit=COMMIT)
    manifest = ReleaseManifest(
        commit=COMMIT,
        toolchain_hash=TOOLCHAIN,
        corpus_hash=index.corpus_hash,
        model_config_hash=MODEL,
        test_counts=TestCounts(expected=1, executed=1, skipped_required=0, failed=0),
        public_case_count=0,
        private_case_count=0,
        evidence_run_ids={},
        unresolved_items=[],
    )
    result = ReleaseGate().check(manifest, index)
    assert not result.passed
    assert "RELEASE_SURFACE_INCOMPLETE" in result.reason_codes
