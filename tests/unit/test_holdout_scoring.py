"""Evaluator preflight rejects incomplete evidence without changing either store."""

import hashlib
import json
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import conftest
import pytest
from schedule_authority_support import schedule_client_for_test

from gpu_agent.benchmark.evaluation import EvaluationRunner, EvaluationSchedule
from gpu_agent.benchmark.executor import EvaluationExecutor
from gpu_agent.benchmark.holdout import (
    HoldoutController,
    ResolvedHoldoutEvaluationRecord,
    ValidatedHoldoutEvaluation,
)
from gpu_agent.benchmark.holdout_scoring import HoldoutLabelPackage, HoldoutScoringController
from gpu_agent.contracts import CurrentPhase, ExternalRunOrigin, RunStatus


def test_holdout_validation_api_has_no_caller_supplied_trust_bypass():
    from inspect import signature

    from gpu_agent.benchmark.executor import validate_evaluation_record

    forbidden = {"_proof", "_identity", "_batch_is_validated", "_trusted_case"}

    for entrypoint in (
        validate_evaluation_record,
        HoldoutController._prepared_execution,
        HoldoutController._resolve_prepared_execution,
        HoldoutController._validate_native_execution,
    ):
        assert forbidden.isdisjoint(signature(entrypoint).parameters)
    assert "resolve_scheduled_record" not in HoldoutController.__dict__


def test_validated_holdout_aggregate_is_not_a_serializable_execution_model():
    from gpu_agent.benchmark.evaluation import EvaluationBindings

    evaluation = ValidatedHoldoutEvaluation(
        evaluation_run_id="a" * 32,
        schedule=EvaluationSchedule(
            selection="all",
            modes=["A", "B", "C", "D", "E"],
            split="holdout",
            repeats=3,
            random_seed=7,
            corpus_cutoff=1,
            bindings=EvaluationBindings(
                commit="b" * 40,
                prompt_version="test",
                toolchain_hash="c" * 64,
                model_config_hash="d" * 64,
                max_cost_usd=1000,
                max_unit_cost_usd=1,
            ),
            items=[],
        ),
        schedule_hash="e" * 64,
        records=(),
        record_refs=(),
    )

    assert not hasattr(evaluation, "model_dump")
    assert not hasattr(evaluation, "model_dump_json")
    assert not hasattr(ResolvedHoldoutEvaluationRecord, "model_dump")
    assert not hasattr(ResolvedHoldoutEvaluationRecord, "model_dump_json")
    with pytest.raises(TypeError):
        json.dumps(evaluation)


@pytest.mark.parametrize("native_evaluation_executor", ["private_split"], indirect=True)
def test_validated_holdout_batch_derives_authority_once_but_single_resolve_revalidates(
    private_split_executor, native_evaluation_executor, monkeypatch
):
    import gpu_agent.benchmark.executor as executor_module
    from gpu_agent.benchmark.evaluation import EvaluationRunner

    executor = private_split_executor
    controller = executor.holdout_controller
    batch = executor.holdout_batch
    binding = executor.service.binding
    assert controller is not None and batch is not None and binding is not None
    manifest = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version or "",
        toolchain_hash=binding.toolchain_lock_hash,
        model_config_hash=binding.model_config_hash or "",
        binding=binding,
        max_cost_usd=1000,
        max_unit_cost_usd=1,
        random_seed=7,
        holdout_controller=controller,
        holdout_batch=batch,
    ).run("A", "holdout", 3)

    calls = {"batch": 0, "cases": 0}
    validate_batch = controller.validate_batch
    registered_cases = executor_module.registered_cases

    def counted_batch(*args, **kwargs):
        calls["batch"] += 1
        return validate_batch(*args, **kwargs)

    def counted_cases(*args, **kwargs):
        calls["cases"] += 1
        return registered_cases(*args, **kwargs)

    monkeypatch.setattr(controller, "validate_batch", counted_batch)
    monkeypatch.setattr(executor_module, "registered_cases", counted_cases)

    validated = controller.validated_evaluation(batch, manifest.run_id)
    assert len(validated.resolved_records) == 3
    # Four batch-level loads, plus one evaluator-owned truth lookup for each
    # verification record. The latter did not exist before private truth was wired.
    assert calls == {"batch": 1, "cases": 4 + len(validated.resolved_records)}

    controller.resolve_evaluation_record(batch, manifest.run_id, 0)
    controller.resolve_evaluation_record(batch, manifest.run_id, 1)
    # Standalone resolution deliberately revalidates the entire three-record
    # aggregate; callers resolving many records should reuse validated_evaluation.
    assert calls == {"batch": 3, "cases": 3 * (4 + len(validated.resolved_records))}


def test_holdout_lineage_discriminator_rejects_native_shape_with_holdout_kind():
    from pydantic import ValidationError

    from gpu_agent.benchmark.evaluation import PublicEvaluationRecord

    with pytest.raises(ValidationError):
        PublicEvaluationRecord.model_validate(
            {
                "record_id": "a" * 32,
                "corpus_cutoff": 1,
                "lineage": {
                    "kind": "holdout_commitment",
                    "corpus_cutoff": 1,
                    "diagnosis_run_id": "b" * 32,
                    "diagnosis_hash": "c" * 64,
                    "evidence_hash": "d" * 64,
                    "provider_invocation_hashes": [],
                },
                "case_id": "e" * 64,
                "template_id": "e" * 64,
                "mode": "D",
                "repeat": 0,
                "input_hash": "f" * 64,
                "evidence_hash": "d" * 64,
                "executed_checks": {},
                "status": "INCONCLUSIVE",
                "diagnosis": {},
                "latency_ms": 0,
            }
        )


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def evaluator_native_record(service, store, prepared, item, case):
    """Build the private adapter record from one real terminal evaluator run."""
    from gpu_agent.agent.models import AcquisitionUsage, AgentBudget
    from gpu_agent.agent.provider import Invocation
    from gpu_agent.benchmark.evaluation import (
        EvaluationRecord,
        NativeEvaluationLineage,
        PricingAttestation,
        evaluation_record_status,
    )
    from gpu_agent.evidence.repository import EvidenceRepository
    from gpu_agent.patching import PatchCandidate
    from gpu_agent.verification.models import VerificationResult, VerificationVerdict

    run = store.load(prepared.diagnosis_run_id)
    diagnosis = service.diagnosis(run.id)
    diagnosis_ref = next(ref for ref in run.artifact_refs if ref.name == "diagnosis.json")
    evidence_refs = [ref for ref in run.artifact_refs if ref.name == "evidence/bundle.json"]
    bundle = EvidenceRepository(store, evaluator=True).view(run.id)
    budget = AgentBudget.model_validate_json(
        store.read(next(ref for ref in run.artifact_refs if ref.name == "agent/final-budget.json"))
    )
    acquisition = AcquisitionUsage.model_validate_json(
        store.read(
            next(ref for ref in run.artifact_refs if ref.name == "agent/acquisition-usage.json")
        )
    )
    histories: dict[str, list[Invocation]] = {}
    terminal_hashes = []
    for ref in [ref for ref in run.artifact_refs if ref.name.startswith("provider/")]:
        invocation = Invocation.model_validate_json(store.read(ref))
        histories.setdefault(invocation.invocation_id, []).append(invocation)
        if invocation.state != "STARTED":
            terminal_hashes.append(ref.sha256)
    usage = {
        "physical_calls": budget.llm_calls,
        "sanitizer_calls": acquisition.sanitizer_calls,
        "retrieval_calls": acquisition.retrieval_calls,
        "sanitizer_attempts": budget.sanitizer_calls,
        "retrieval_attempts": budget.rag_calls,
        "build_calls": int(bundle.build_result is not None),
        "runtime_calls": int(bundle.execution_result is not None),
    }
    diagnostic_calls = (
        acquisition.sanitizer_calls
        + acquisition.retrieval_calls
        + int(bundle.build_result is not None)
        + int(bundle.execution_result is not None)
    )
    usage["diagnostic_tool_calls"] = diagnostic_calls
    usage["tool_calls"] = diagnostic_calls
    usage["total_sanitizer_calls"] = acquisition.sanitizer_calls
    finals = [history[-1] for history in histories.values()]
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        values = [getattr(value.usage, field) if value.usage else None for value in finals]
        usage[field] = (
            sum(value for value in values if value is not None)
            if values and len(values) == budget.llm_calls and None not in values
            else None
        )
    pricing = PricingAttestation.model_validate_json(
        store.read(
            next(ref for ref in run.artifact_refs if ref.name == "agent/pricing-attestation.json")
        )
    )
    input_tokens, output_tokens = usage["input_tokens"], usage["output_tokens"]
    cost = (
        0.0
        if usage["physical_calls"] == 0
        else pricing.cost(input_tokens, output_tokens)
        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
        else None
    )
    checks = {
        result.tool_result.typed_payload.tool: result.check_outcome
        for result in bundle.sanitizer_results
        if result.tool_result is not None
    }
    candidate = None
    candidate_run_id = None
    verification = None
    verification_run_id = None
    verification_ref = None
    finished = run.events[-1].at
    candidates = service.candidates(run.id)
    if candidates:
        candidate_run_id = candidates[0]
        candidate_run = store.load(candidate_run_id)
        candidate = PatchCandidate.model_validate_json(
            store.read(
                next(ref for ref in candidate_run.artifact_refs if ref.name == "candidate.json")
            )
        )
        service.verify(run.id, candidate_run_id)
        verification_run = next(
            child for child in store.children(run.id) if child.kind == "verification"
        )
        verification_run_id = verification_run.id
        verification_ref = next(
            ref for ref in verification_run.artifact_refs if ref.name == "verification/result.json"
        )
        verification = VerificationResult.model_validate_json(store.read(verification_ref))
        checks.update(
            {
                f"verification/{key}": value
                for key, value in verification.required_checks.items()
                if key != "private_oracle"
            }
        )
        usage["tool_calls"] = None
        usage["total_sanitizer_calls"] = None
        finished = verification_run.events[-1].at
    status, reason = evaluation_record_status(diagnosis)
    return EvaluationRecord.model_validate(
        {
            "record_id": run.id,
            "corpus_cutoff": prepared.binding.corpus_cutoff,
            "lineage": NativeEvaluationLineage(
                corpus_cutoff=prepared.binding.corpus_cutoff,
                diagnosis_run_id=run.id,
                diagnosis_hash=diagnosis_ref.sha256,
                evidence_hash=evidence_refs[-1].sha256,
                provider_invocation_hashes=terminal_hashes,
                candidate_run_id=candidate_run_id,
                verification_run_id=verification_run_id,
                public_verification_hash=(verification_ref.sha256 if verification_ref else None),
            ),
            "case_id": prepared.binding.private_case_id,
            "template_id": prepared.binding.private_template_id,
            "mode": item.mode,
            "repeat": item.repeat,
            "input_hash": case.source_hash,
            "evidence_hash": evidence_refs[-1].sha256,
            "executed_checks": checks,
            "status": status,
            "diagnosis": diagnosis.model_dump(mode="json"),
            "patch_hash": candidate.patched_source_hash if candidate else None,
            "oracle_passed": verification.public_oracle_passed if verification else None,
            "verdict": verification.verdict.value if verification else None,
            "regression_detected": bool(
                verification and verification.verdict == VerificationVerdict.REGRESSION_DETECTED
            ),
            "usage": usage,
            # The synthetic provider fixture may advance its deterministic clock
            # independently from the store event clock.  Latency is not the
            # subject of this persisted-lineage fixture, but the production
            # schema correctly rejects a negative duration.
            "latency_ms": max(0.0, (finished - run.events[0].at).total_seconds() * 1000),
            "cost_usd": cost,
            "failure_reason": reason,
        }
    )


def overwrite_fixture_artifact(path, content):
    mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode | stat.S_IWUSR)
    try:
        path.write_bytes(content)
    finally:
        path.chmod(mode)


class ScoringFixture:
    def __init__(self, executor, root, patch):
        self.public, self.evaluator = executor.service.store, executor.corpus
        self.binding = executor.service.binding
        self.repository = root / "repository"
        self.holdout = HoldoutController(
            self.public,
            self.evaluator,
            binding=self.binding,
            _schedule_verifier=executor._schedule_verifier,
        )
        self.batch = self.holdout.prepare()
        from gpu_agent.benchmark.executor import registered_cases
        from gpu_agent.service import ApplicationService

        evaluator_service = ApplicationService(
            self.evaluator,
            self.evaluator,
            provider=executor.service._provider,
            backend_factory=executor.service._backend_factory,
            knowledge=executor.service.knowledge,
            knowledge_version=executor.service.knowledge_version,
            _binding=self.binding,
            _evaluation_schedule_verifier=executor._schedule_verifier,
        )
        evaluator_service._pricing_attestation = executor.service._pricing_attestation
        holdout_executor = EvaluationExecutor(
            executor.service,
            executor.corpus,
            executor.sources,
            holdout_service=evaluator_service,
            holdout_controller=self.holdout,
            holdout_batch=self.batch,
            _corpus_family=executor._corpus_family,
            _schedule_verifier=executor._schedule_verifier,
        )
        reservations = {}

        def use_reserved_diagnosis(_verifier, unit):
            return self.evaluator.load(reservations[unit.model_dump_json()].diagnosis_run_id)

        patch.setattr(
            self.evaluator,
            "validate_and_create_evaluation_child",
            use_reserved_diagnosis,
        )
        cases = registered_cases(
            self.evaluator,
            self.binding,
            executor._corpus_family,
            cutoff=self.batch.corpus_cutoff,
        )
        runner = EvaluationRunner(
            self.public,
            holdout_executor,
            schedule_client=schedule_client_for_test(holdout_executor),
            commit=self.binding.repository.commit,
            prompt_version=self.binding.prompt_version,
            toolchain_hash=self.binding.toolchain_lock_hash,
            model_config_hash=self.binding.model_config_hash,
            binding=self.binding,
            max_cost_usd=120,
            max_unit_cost_usd=1,
            random_seed=7,
            holdout_controller=self.holdout,
            holdout_batch=self.batch,
        )
        # Exercise the real signed native producer once per unit. The runner's
        # recovery loop rescans every previous record on every iteration; that
        # quadratic recovery behavior is covered separately, not under test here.
        from gpu_agent.benchmark.schedule_authority import (
            activate_schedule,
            bind_reserved_schedule,
            rebuild_reserved_schedule,
            reserve_evaluation_cutoff,
            seal_schedule,
        )

        run = self.public.create_run("evaluation", binding=self.binding)
        reserve_evaluation_cutoff(
            executor._corpus_family,
            self.public,
            run.id,
            self.binding,
            selection="all",
            modes=["A", "B", "C", "D", "E"],
            split="holdout",
            repeats=3,
            random_seed=7,
            max_cost_usd=120,
            max_unit_cost_usd=1,
            holdout_proof=self.holdout.validate_batch(self.batch),
            holdout_aliases=self.batch.aliases,
        )
        schedule = rebuild_reserved_schedule(
            executor._corpus_family, self.public, run.id, self.binding
        )
        bind_reserved_schedule(executor._corpus_family, self.public, run.id, schedule, self.binding)
        runner._put(run.id, "evaluation/schedule.json", schedule.model_dump_json().encode())
        seal_schedule(
            executor._corpus_family,
            self.public,
            run.id,
            schedule,
            self.binding,
            schedule_client_for_test(executor),
            executor._schedule_verifier,
        )
        activate_schedule(self.public, executor._schedule_verifier, run.id)
        records = []
        verified_records = {}
        for item in schedule.items:
            attempt = runner._attempt(run.id, schedule, item)
            runner._put(
                run.id,
                f"evaluation/attempts/{item.ordinal}.json",
                attempt.model_dump_json().encode(),
            )
            prepared = self.holdout.reserve_execution(
                self.batch,
                evaluation_run_id=run.id,
                item=item,
                attempt=attempt,
            )
            reservations[prepared.evaluation_unit.model_dump_json()] = prepared
            case = cases[prepared.binding.private_case_id]
            diagnosis = evaluator_service.diagnose(
                executor.sources[case.id],
                mode=item.mode,
                required_tools=(case.target_tool,),
                expected_source_hash=case.source_hash,
                evaluation_unit=prepared.evaluation_unit,
            )
            native = evaluator_native_record(
                evaluator_service,
                self.evaluator,
                prepared,
                item,
                case,
            )
            assert diagnosis.id == prepared.diagnosis_run_id
            record = self.holdout.complete_execution(prepared, native)
            recovered = self.holdout.recover_execution(self.batch, item, attempt)
            if recovered is None or recovered != record:
                raise ValueError("fixture holdout record does not recover exactly")
            verified_records[item.ordinal] = recovered
            runner._put(
                run.id,
                f"evaluation/records/{item.ordinal}.json",
                recovered.model_dump_json().encode(),
            )
            records.append(recovered)

        def validate_persisted_holdout(_executor, record, item, attempt):
            cached = verified_records.get(item.ordinal)
            if (
                attempt.ordinal != item.ordinal
                or attempt.run_id != run.id
                or cached is None
                or cached != record
            ):
                raise ValueError("fixture holdout record does not recover exactly")

        patch.setattr(
            EvaluationExecutor,
            "validate_scheduled_record",
            validate_persisted_holdout,
        )
        manifest = runner._terminal(run.id, schedule, records, None, RunStatus.COMPLETED)
        assert manifest.executed_units == 120 and manifest.stopped_reason is None
        self.evaluation_run_id = manifest.run_id
        self.mapping_run_id = self.batch.evaluator_run_id
        self.records = manifest.records
        self.refs = [self.ref(f"evaluation/records/{i}.json") for i in range(120)]
        self.schedule = json.loads(self.public.read(self.ref("evaluation/schedule.json")))
        judgments, record_set = [], []
        for ordinal, (record, ref) in enumerate(zip(self.records, self.refs, strict=True)):
            blind = record.blind()
            blind_hash = digest(canonical(blind))
            record_set.append([ordinal, ref.sha256, blind_hash])
            judgments.append(
                {
                    "blind_id": blind["blind_id"],
                    "public_record_hash": ref.sha256,
                    "blind_payload_hash": blind_hash,
                    "labels": {"claim_support": {"PRIVATE-JUDGMENT-CANARY": True}},
                    "score": {
                        "family_correct": True,
                        "root_cause_correct": False,
                        "location_correct": True,
                        "inconclusive_correct": False,
                    },
                    "should_be_inconclusive": False,
                    "private_holdout_passed": True,
                }
            )
        self.package = HoldoutLabelPackage.model_validate(
            {
                "evaluation_run_id": manifest.run_id,
                "private_binding_run_id": self.mapping_run_id,
                "schedule_hash": digest(canonical(self.schedule)),
                "aliases_hash": self.batch.public_alias_hash,
                "corpus_cutoff": 8,
                "record_set_hash": digest(canonical(record_set)),
                "rubric_hash": digest(b"Evaluator rubric v1\n"),
                "judgments": judgments,
            }
        )
        self.labels_path = root / "labels.json"
        self.controller = HoldoutScoringController(
            self.public,
            self.evaluator,
            binding=self.binding,
            _schedule_verifier=executor._schedule_verifier,
        )
        self.write_package(self.package.model_dump(mode="json"))

    def ref(self, name):
        return next(
            r for r in self.public.load(self.evaluation_run_id).artifact_refs if r.name == name
        )

    def write_package(self, payload):
        self.labels_path.write_bytes(canonical(payload))
        self.labels_path.chmod(0o600)
        return self.labels_path

    def preflight(self):
        return self.controller.preflight(
            self.evaluation_run_id, self.mapping_run_id, self.labels_path, self.repository
        )

    def score(self, labels_path=None):
        return self.controller.score(
            self.evaluation_run_id,
            self.mapping_run_id,
            labels_path or self.labels_path,
            self.repository,
        )

    def changed_package(self):
        payload = self.package.model_dump(mode="json")
        for judgment in payload["judgments"]:
            judgment["score"]["root_cause_correct"] = True
        path = self.labels_path.with_name("changed-labels.json")
        path.write_bytes(canonical(payload))
        path.chmod(0o600)
        return path

    def snapshot(self):
        return {
            (str(store.root), str(path.relative_to(store.root))): path.read_bytes()
            if path.is_file()
            else None
            for store in (self.public, self.evaluator)
            for path in store.root.rglob("*")
        }

    def alter_artifact(self, name, change):
        ref = self.ref(name)
        path = self.public.root / ref.relative_path
        payload = json.loads(path.read_bytes())
        change(payload)
        content = canonical(payload)
        overwrite_fixture_artifact(path, content)
        manifest_path = self.public.root / self.evaluation_run_id / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        for item in manifest["artifact_refs"]:
            if item["id"] == ref.id:
                item["sha256"] = digest(content)
                item["byte_count"] = len(content)
        manifest_path.write_bytes(canonical(manifest))

    def seed(self, ordinal, **changes):
        # Independent expected score bytes and binding, not the preparer under test.
        record, ref = self.records[ordinal], self.refs[ordinal]
        mapping_ref = next(
            r
            for r in self.evaluator.load(self.mapping_run_id).artifact_refs
            if r.name == "holdout/private-alias-map.json"
        )
        mapping = json.loads(self.evaluator.read(mapping_ref))
        identity = next(i for i in mapping["identities"] if i["alias"] == record.case_id)
        import hmac

        run_id = hmac.new(
            bytes.fromhex(mapping["nonce_hex"]),
            f"holdout-score-v1:{record.record_id}".encode(),
            hashlib.sha256,
        ).hexdigest()[:32]
        judgment = self.package.judgments[ordinal]
        private_score = canonical(
            {
                "schema_version": 2,
                "corpus_cutoff": 8,
                "public_record_hash": ref.sha256,
                **judgment.model_dump(
                    mode="json", exclude={"blind_id", "blind_payload_hash", "public_record_hash"}
                ),
            }
        )
        binding = {
            "evaluator_score_run_id": run_id,
            "public_evaluation_run_id": self.evaluation_run_id,
            "public_record_id": record.record_id,
            "public_record_hash": ref.sha256,
            "private_case_id": identity["private_case_id"],
            "private_template_id": identity["private_template_id"],
            "private_score_hash": digest(private_score),
            "corpus_cutoff": 8,
        }
        run = self.evaluator.create_run(
            changes.get("kind", "holdout_score"),
            parent_run_id=changes.get("parent", self.mapping_run_id),
            _run_id=changes.get("run_id", run_id),
        )
        self.evaluator.transition(run.id, RunStatus.RUNNING, "FINALIZING")
        self.evaluator.put(
            run.id,
            "holdout/private-score.json",
            changes.get("private_score", private_score),
            "evaluator",
        )
        binding.update(changes.get("binding", {}))
        self.evaluator.put(
            run.id,
            "holdout/record-binding.json",
            json.dumps(binding, separators=(",", ":")).encode(),
            "evaluator",
        )
        if not changes.get("nonterminal"):
            self.evaluator.transition(run.id, RunStatus.COMPLETED, None)
        return run.id


@pytest.fixture(scope="module")
def scoring_base(tmp_path_factory):
    root = tmp_path_factory.mktemp("holdout-scoring")
    with pytest.MonkeyPatch.context() as patch:
        store = conftest.store.__wrapped__(root)
        service = conftest.oob_service.__wrapped__(store, root)
        signer = conftest.test_schedule_commit_client.__wrapped__(root)
        executor = conftest.native_evaluation_executor.__wrapped__(
            service, root, patch, SimpleNamespace(param="private_eight"), signer
        )
        from test_evaluation_modes import _configure_responses_provider

        from gpu_agent.agent.models import InconclusiveAction

        service[1].actions = [InconclusiveAction()]
        # Scoring mechanics only: every mode's shared diagnosis model declares INCONCLUSIVE,
        # so the 120-unit fixture never needs a GPU-backed private verification.
        service[1].force_limitation = True
        service[1].limitation_canary = "NOT_ENOUGH_EVIDENCE"
        _configure_responses_provider(executor, patch, full_script=True)
        yield ScoringFixture(executor, root, patch)


@pytest.fixture
def scoring_fixture(scoring_base):
    fixture = scoring_base
    before = fixture.snapshot()
    fixture.write_package(fixture.package.model_dump(mode="json"))
    yield fixture
    # Restore intentional corruption and remove only test-created child directories.
    for store in (fixture.public, fixture.evaluator):
        existing = {relative for (root, relative) in before if root == str(store.root)}
        for path in list(store.root.iterdir()):
            if path.name not in existing:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        for (root, relative), content in before.items():
            if root == str(store.root) and content is not None:
                path = Path(root) / relative
                if not path.exists() or path.read_bytes() != content:
                    overwrite_fixture_artifact(path, content)


def test_exact_package_zero_mutation_plan(scoring_fixture):
    f = scoring_fixture
    before = f.snapshot()
    plan = f.preflight()
    assert [item.ordinal for item in plan.items] == list(range(120))
    assert all(item.existing_binding is None for item in plan.items)
    assert {item.alias for item in plan.items} == set(f.batch.aliases)
    first = plan.items[0]
    private = json.loads(first.prepared_score.private_score_content)
    assert private["score"] == {
        "family_correct": True,
        "root_cause_correct": False,
        "location_correct": True,
        "inconclusive_correct": False,
    }
    assert private["labels"]["claim_support"] == {"PRIVATE-JUDGMENT-CANARY": True}
    assert first.prepared_score.binding.public_record_hash == f.refs[0].sha256
    assert f.snapshot() == before


@pytest.mark.parametrize(
    "fault",
    [
        "119",
        "121",
        "duplicate",
        "evaluation_run_id",
        "private_binding_run_id",
        "schedule_hash",
        "aliases_hash",
        "corpus_cutoff",
        "record_set_hash",
        "rubric_hash",
        "public_record_hash",
        "blind_payload_hash",
        "blind_id",
        "extra",
        "judgment_extra",
    ],
)
def test_package_exact_binding_rejections_zero_mutation(scoring_fixture, fault):
    f = scoring_fixture
    payload = f.package.model_dump(mode="json")
    if fault == "119":
        payload["judgments"].pop()
    elif fault in {"121", "duplicate"}:
        if fault == "duplicate":
            payload["judgments"].pop()
        payload["judgments"].append(payload["judgments"][0])
    elif fault in {"public_record_hash", "blind_payload_hash", "blind_id"}:
        payload["judgments"][0][fault] = "f" * 64
    elif fault == "extra":
        payload["private_identity"] = "CANARY"
    elif fault == "judgment_extra":
        payload["judgments"][0]["private_identity"] = "CANARY"
    else:
        payload[fault] = 9 if fault == "corpus_cutoff" else "f" * len(payload[fault])
    f.write_package(payload)
    before = f.snapshot()
    with pytest.raises(ValueError):
        f.preflight()
    assert f.snapshot() == before


@pytest.mark.parametrize("fault", ["ordinal", "mode", "repeat", "cartesian"])
def test_exact_product_mutations_reach_exact_product_branch(scoring_fixture, monkeypatch, fault):
    f = scoring_fixture
    schedule_payload = dict(f.schedule)
    schedule_payload["items"] = [dict(item) for item in f.schedule["items"]]
    if fault == "ordinal":
        schedule_payload["items"][-1]["ordinal"] = 0
    elif fault == "mode":
        schedule_payload["modes"] = ["A", "B", "C", "D", "D"]
    elif fault == "repeat":
        schedule_payload["repeats"] = 4
    else:
        schedule_payload["items"][-1].update(
            {
                key: schedule_payload["items"][0][key]
                for key in ("case_id", "template_id", "mode", "repeat")
            }
        )
    evaluation = ValidatedHoldoutEvaluation(
        evaluation_run_id=f.evaluation_run_id,
        schedule=EvaluationSchedule.model_validate(schedule_payload),
        schedule_hash=f.package.schedule_hash,
        records=tuple(f.records),
        record_refs=tuple(f.refs),
    )
    monkeypatch.setattr(
        f.controller.holdout,
        "validated_evaluation",
        lambda *_args, **_kwargs: evaluation,
    )
    before = f.snapshot()

    with pytest.raises(
        ValueError,
        match="holdout evaluation requires the exact 8 x 5 x 3 Cartesian product",
    ):
        f.preflight()

    assert f.snapshot() == before


@pytest.mark.parametrize(
    "fault",
    [
        "status",
        "selection",
        "stopped_reason",
        "119_attempts",
        "121_attempts",
        "119_records",
        "121_records",
        "noncanonical_attempt",
        "noncanonical_record",
        "missing_ordinal",
        "duplicate_record_id",
        "manifest_count",
        "manifest_order",
        "aliases",
    ],
)
def test_evaluation_root_exact_cartesian_zero_mutation(scoring_fixture, fault):
    f = scoring_fixture
    root_path = f.public.root / f.evaluation_run_id / "manifest.json"
    root = json.loads(root_path.read_bytes())
    if fault == "status":
        root["status"] = "FAILED"
    elif fault.startswith(("119_", "121_", "noncanonical_")):
        kind = "attempts" if "attempt" in fault else "records"
        selected = next(
            r for r in root["artifact_refs"] if r["name"] == f"evaluation/{kind}/119.json"
        )
        if fault.startswith("119"):
            root["artifact_refs"].remove(selected)
        elif fault.startswith("noncanonical"):
            selected["name"] = f"evaluation/{kind}/0119.json"
        else:
            selected = dict(selected)
            selected["name"] = f"evaluation/{kind}/120.json"
            root["artifact_refs"].append(selected)
    else:
        if fault in {"stopped_reason", "manifest_count", "manifest_order"}:
            name = "evaluation/manifest.json"

            def change(payload):
                if fault == "stopped_reason":
                    payload["stopped_reason"] = "COST_UNKNOWN"
                elif fault == "manifest_count":
                    payload["executed_units"] = 119
                else:
                    payload["records"].reverse()
        elif fault == "duplicate_record_id":
            name = "evaluation/records/119.json"

            def change(payload):
                payload["record_id"] = f.records[0].record_id
        else:
            name = "evaluation/schedule.json"

            def change(payload):
                if fault == "selection":
                    payload["selection"] = "D"
                elif fault == "missing_ordinal":
                    payload["items"].pop()
                elif fault == "aliases":
                    payload["items"][-1]["case_id"] = "f" * 64

        f.alter_artifact(name, change)
    if fault == "status" or fault.startswith(("119_", "121_", "noncanonical_")):
        root_path.write_bytes(canonical(root))
    before = f.snapshot()
    with pytest.raises(ValueError):
        f.preflight()
    assert f.snapshot() == before


@pytest.mark.parametrize("count", [1, 37, 119])
def test_existing_score_exact_subsets_zero_mutation(scoring_fixture, count):
    f = scoring_fixture
    for ordinal in range(count):
        f.seed(ordinal)
    before = f.snapshot()
    plan = f.preflight()
    assert [i.ordinal for i in plan.items if i.existing_binding is not None] == list(range(count))
    assert f.snapshot() == before


@pytest.mark.parametrize(
    "fault",
    [
        "score",
        "kind",
        "duplicate",
        "run_id",
        "parent",
        "binding",
        "nonterminal",
        "bytes",
    ],
)
def test_existing_score_conflict_zero_mutation(scoring_fixture, fault):
    f = scoring_fixture
    kwargs = {}
    if fault in {"score", "bytes"}:
        kwargs["private_score"] = b'{"conflicting":true}'
    elif fault == "kind":
        kwargs["kind"] = "unexpected"
    elif fault in {"run_id", "duplicate"}:
        kwargs["run_id"] = "f" * 32
        if fault == "duplicate":
            f.seed(99)
    elif fault == "parent":
        kwargs["parent"] = f.evaluator.create_run("other", binding=f.binding).id
    elif fault == "binding":
        kwargs["binding"] = {"private_case_id": "wrong-private-case"}
    else:
        kwargs["nonterminal"] = True
    f.seed(99, **kwargs)
    before = f.snapshot()
    with pytest.raises(ValueError, match="existing holdout score conflicts"):
        f.preflight()
    assert f.snapshot() == before


@pytest.mark.parametrize(
    "target",
    ["attempt", "record", "manifest", "child_binding", "child_private_score"],
)
def test_cross_run_artifact_substitution_zero_mutation(scoring_fixture, target):
    f = scoring_fixture
    if target.startswith("child_"):
        store, run_id = f.evaluator, f.seed(99)
        name = {
            "child_binding": "holdout/record-binding.json",
            "child_private_score": "holdout/private-score.json",
        }[target]
    else:
        store, run_id = f.public, f.evaluation_run_id
        name = {
            "attempt": "evaluation/attempts/99.json",
            "record": "evaluation/records/99.json",
            "manifest": "evaluation/manifest.json",
        }[target]
    original = next(ref for ref in store.load(run_id).artifact_refs if ref.name == name)
    donor = store.create_run("artifact_donor", binding=f.binding)
    borrowed = store.put(donor.id, name, store.read(original), original.visibility)
    assert borrowed.run_id != run_id
    assert store.read(borrowed) == store.read(original)
    manifest_path = store.root / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifact_refs"] = [
        borrowed.model_dump(mode="json") if ref["name"] == name else ref
        for ref in manifest["artifact_refs"]
    ]
    overwrite_fixture_artifact(manifest_path, canonical(manifest))
    before = f.snapshot()
    with pytest.raises(ValueError):
        f.preflight()
    assert f.snapshot() == before


def test_clean_looking_divergent_rubric_is_rejected(tmp_path):
    import subprocess

    from gpu_agent.benchmark.holdout_scoring import _tracked_rubric_hash
    from gpu_agent.provenance import capture_repository_snapshot

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        ).stdout

    git("init", "-q")
    (tmp_path / "evaluation").mkdir()
    rubric = tmp_path / "evaluation/rubric.md"
    rubric.write_bytes(b"committed evaluator rubric\n")
    git("add", "evaluation/rubric.md")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "rubric")
    assert _tracked_rubric_hash(tmp_path) == digest(b"committed evaluator rubric\n")
    git("update-index", "--assume-unchanged", "evaluation/rubric.md")
    rubric.write_bytes(b"PRIVATE-DIVERGENT-RUBRIC-CANARY\n")
    assert git("status", "--porcelain") == b""
    assert capture_repository_snapshot(tmp_path).clean is True
    with pytest.raises(ValueError) as error:
        _tracked_rubric_hash(tmp_path)
    assert "PRIVATE-DIVERGENT-RUBRIC-CANARY" not in str(error.value)
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize("target", ["mapping", "evaluation"])
def test_missing_lock_zero_mutation(scoring_fixture, target):
    f = scoring_fixture
    store, run_id = (
        (f.evaluator, f.mapping_run_id) if target == "mapping" else (f.public, f.evaluation_run_id)
    )
    lock = store.root / run_id / ".lock"
    displaced = f.labels_path.parent / f"preserved-{target}.lock"
    lock.rename(displaced)
    before = f.snapshot()
    try:
        with pytest.raises(ValueError):
            f.preflight()
        assert f.snapshot() == before
    finally:
        lock.unlink(missing_ok=True)
        displaced.rename(lock)


@pytest.mark.parametrize(
    "fault", ["relative", "public_store", "evaluator_store", "repository", "mode"]
)
def test_package_private_path_zero_mutation(scoring_fixture, fault):
    f = scoring_fixture
    original = f.labels_path
    path = {
        "relative": Path("labels.json"),
        "public_store": f.public.root / "labels.json",
        "evaluator_store": f.evaluator.root / "labels.json",
        "repository": f.repository / "labels.json",
        "mode": original,
    }[fault]
    f.labels_path = path
    if fault == "mode":
        path.chmod(0o644)
    elif fault != "relative":
        f.write_package(f.package.model_dump(mode="json"))
    before = f.snapshot()
    try:
        with pytest.raises(ValueError):
            f.preflight()
        assert f.snapshot() == before
    finally:
        f.labels_path = original
        original.chmod(0o600)
        if fault not in {"mode", "relative"}:
            path.unlink()


def test_package_judgment_order_does_not_change_ordinal_plan(scoring_fixture):
    f = scoring_fixture
    payload = f.package.model_dump(mode="json")
    payload["judgments"].reverse()
    f.write_package(payload)
    before = f.snapshot()
    plan = f.preflight()
    assert [i.judgment.blind_id for i in plan.items] == [r.blind()["blind_id"] for r in f.records]
    assert f.snapshot() == before


def test_package_rubric_must_be_committed(tmp_path):
    import subprocess

    from gpu_agent.benchmark.holdout_scoring import _tracked_rubric_hash

    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation/rubric.md").write_bytes(b"ignored evaluator rubric\n")
    (tmp_path / ".gitignore").write_text("evaluation/rubric.md\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore"], check=True)
    with pytest.raises(ValueError):
        _tracked_rubric_hash(tmp_path)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", "evaluation/rubric.md"], check=True)
    with pytest.raises(ValueError):
        _tracked_rubric_hash(tmp_path)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "rubric",
        ],
        check=True,
    )
    assert _tracked_rubric_hash(tmp_path) == digest(b"ignored evaluator rubric\n")


@pytest.mark.parametrize("kind", ["blob", "missing", "tree"])
def test_rubric_uses_exact_bound_commit(tmp_path, kind):
    import subprocess

    from gpu_agent.benchmark.holdout_scoring import _tracked_rubric_hash

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        ).stdout

    git("init", "-q")
    (tmp_path / "evaluation").mkdir()
    rubric = tmp_path / "evaluation/rubric.md"
    (tmp_path / "tracked.txt").write_bytes(b"fixture\n")
    if kind == "blob":
        rubric.write_bytes(b"bound committed rubric\n")
    elif kind == "tree":
        rubric.mkdir()
        (rubric / "child").write_bytes(b"not a rubric blob\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "bound")
    bound_commit = git("rev-parse", "HEAD").decode().strip()
    if kind != "blob":
        with pytest.raises(ValueError, match="rubric") as error:
            _tracked_rubric_hash(tmp_path, bound_commit)
        assert str(tmp_path) not in str(error.value)
        return
    assert _tracked_rubric_hash(tmp_path, bound_commit) == digest(b"bound committed rubric\n")
    rubric.write_bytes(b"new HEAD rubric\n")
    git("add", ".")
    git(
        "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "new HEAD"
    )
    assert _tracked_rubric_hash(tmp_path) == digest(b"new HEAD rubric\n")
    with pytest.raises(ValueError, match="rubric"):
        _tracked_rubric_hash(tmp_path, bound_commit)


def test_package_repository_recapture_zero_mutation(scoring_fixture, monkeypatch):
    import gpu_agent.benchmark.holdout_scoring as scoring

    f = scoring_fixture
    capture = scoring.capture_repository_snapshot
    calls = 0
    rubric = f.repository / "evaluation/rubric.md"
    original = rubric.read_bytes()

    def change_before_recapture(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            rubric.write_bytes(b"changed after native validation\n")
        return capture(*args, **kwargs)

    monkeypatch.setattr(scoring, "capture_repository_snapshot", change_before_recapture)
    before = f.snapshot()
    try:
        with pytest.raises(ValueError, match="clean|changed"):
            f.preflight()
        assert calls == 2
        assert f.snapshot() == before
    finally:
        rubric.write_bytes(original)


def test_existing_lock_guard_is_read_only(store):
    from gpu_agent.benchmark.holdout_scoring import _require_existing_lock

    run = store.create_run("evaluation")
    lock = store.root / run.id / ".lock"
    lock.unlink(missing_ok=True)
    with pytest.raises(ValueError, match="lock"):
        _require_existing_lock(store, run.id)
    assert not lock.exists()
    store.transition(run.id, RunStatus.RUNNING, "FINALIZING")
    content = lock.read_bytes()
    _require_existing_lock(store, run.id)
    assert lock.read_bytes() == content


def _session_id(f):
    return digest(f"holdout-scoring-v1:{f.evaluation_run_id}:{f.mapping_run_id}".encode())[:32]


def _assert_scoring_complete(f, result):
    run = f.evaluator.load(_session_id(f))
    assert result.scoring_run_id == run.id
    assert result.scored_count == 120
    assert run.kind == "holdout_scoring" and run.parent_run_id is None
    assert run.external_origin.run_id == f.evaluation_run_id
    assert run.external_origin.visibility == "public" and run.binding == f.binding
    assert [(e.status.value, e.phase.value if e.phase else None) for e in run.events] == [
        ("QUEUED", None),
        ("RUNNING", "PREPARING"),
        ("RUNNING", "FINALIZING"),
        ("COMPLETED", None),
    ]
    artifacts = {ref.name: f.evaluator.read(ref) for ref in run.artifact_refs}
    assert set(artifacts) == {
        f"holdout-scoring/{name}.json"
        for name in ("input-binding", "bindings", "metrics", "result")
    }
    bindings = json.loads(artifacts["holdout-scoring/bindings.json"])
    assert [b["public_record_id"] for b in bindings] == [r.record_id for r in f.records]
    assert len({b["evaluator_score_run_id"] for b in bindings}) == 120
    assert result.metrics_hash == digest(artifacts["holdout-scoring/metrics.json"])
    assert json.loads(artifacts["holdout-scoring/result.json"]) == result.model_dump(mode="json")
    assert (
        json.loads(artifacts["holdout-scoring/input-binding.json"])["package_hash"]
        == result.package_hash
    )
    metrics = json.loads(artifacts["holdout-scoring/metrics.json"])
    assert metrics["overall"]["record_count"] == 120
    assert metrics["overall"]["case_count"] == 8
    assert set(metrics["by_mode"]) == set("ABCDE")
    assert all(group["record_count"] == 24 for group in metrics["by_mode"].values())
    assert all(group["record_count"] == 15 for group in metrics["by_case"].values())
    children = [
        f.evaluator.load(path.name)
        for path in f.evaluator.root.iterdir()
        if path.is_dir() and len(path.name) == 32
    ]
    assert len([r for r in children if r.kind == "holdout_scoring"]) == 1
    scores = [r for r in children if r.parent_run_id == f.mapping_run_id]
    assert len(scores) == 120
    for child in scores:
        assert child.kind == "holdout_score" and child.status == RunStatus.COMPLETED
        assert [e.status.value for e in child.events] == ["QUEUED", "RUNNING", "COMPLETED"]
    return artifacts


def test_session_lifecycle_and_completed_retry_is_read_only(scoring_fixture, monkeypatch):
    f = scoring_fixture
    result = f.score()
    _assert_scoring_complete(f, result)
    before = f.snapshot()

    def forbidden(*args, **kwargs):
        raise AssertionError("completed retry attempted mutation")

    monkeypatch.setattr(f.evaluator, "put", forbidden)
    monkeypatch.setattr(f.evaluator, "put_if_absent_exact", forbidden)
    monkeypatch.setattr(f.evaluator, "transition", forbidden)
    monkeypatch.setattr(f.controller.holdout, "_bind_prepared_score", forbidden)
    assert f.score() == result
    assert f.snapshot() == before
    with pytest.raises(ValueError, match="holdout scoring package conflicts"):
        f.score(f.changed_package())
    assert f.snapshot() == before


@pytest.mark.parametrize(
    "boundary",
    [
        "created",
        "running",
        "input-binding",
        "score-0",
        "score-1",
        "score-119",
        "bindings",
        "metrics",
        "result",
        "finalizing",
        "completed",
    ],
)
def test_session_crash_recovery_at_every_durable_boundary(scoring_fixture, monkeypatch, boundary):
    f = scoring_fixture
    session_id = _session_id(f)

    class Crash(RuntimeError):
        pass

    def fail_if(name):
        if name == boundary:
            raise Crash(name)

    with monkeypatch.context() as patch:
        create = f.evaluator.create_run
        transition = f.evaluator.transition
        put = f.evaluator.put_if_absent_exact
        bind = f.controller.holdout._bind_prepared_score
        ordinal = 0

        def crash_create(*args, **kwargs):
            run = create(*args, **kwargs)
            if run.id == session_id:
                fail_if("created")
            return run

        def crash_transition(run_id, status, phase):
            run = transition(run_id, status, phase)
            if run_id == session_id:
                fail_if(
                    "completed"
                    if status == RunStatus.COMPLETED
                    else "finalizing"
                    if str(phase) == "FINALIZING"
                    else "running"
                )
            return run

        def crash_put(run_id, name, *args, **kwargs):
            ref = put(run_id, name, *args, **kwargs)
            if run_id == session_id:
                fail_if(Path(name).stem)
            return ref

        def crash_bind(*args, **kwargs):
            nonlocal ordinal
            result = bind(*args, **kwargs)
            current = ordinal
            ordinal += 1
            fail_if(f"score-{current}")
            return result

        patch.setattr(f.evaluator, "create_run", crash_create)
        patch.setattr(f.evaluator, "transition", crash_transition)
        patch.setattr(f.evaluator, "put_if_absent_exact", crash_put)
        patch.setattr(f.controller.holdout, "_bind_prepared_score", crash_bind)
        with pytest.raises(Crash, match=boundary):
            f.score()
    result = f.score()
    _assert_scoring_complete(f, result)
    assert result.package_hash == digest(f.labels_path.read_bytes())


def _scoring_worker(f, labels, barrier, output):
    # fork workers inherit the native test verifier; stores still use real process locks.
    barrier.wait(timeout=30)
    try:
        output.put(("ok", f.score(labels).model_dump(mode="json")))
    except Exception as exc:
        output.put(("error", str(exc)))


@pytest.mark.parametrize("processes", [False, True], ids=["threads", "processes"])
@pytest.mark.parametrize("different", [False, True], ids=["same_package", "package_race"])
def test_session_concurrent_packages(scoring_fixture, processes, different):
    import multiprocessing
    import queue
    import threading

    f = scoring_fixture
    if processes:
        context = multiprocessing.get_context("fork")
        barrier, output, worker = context.Barrier(2), context.Queue(), context.Process
    else:
        barrier, output, worker = threading.Barrier(2), queue.Queue(), threading.Thread
    labels = [f.labels_path, f.changed_package() if different else f.labels_path]
    workers = [worker(target=_scoring_worker, args=(f, path, barrier, output)) for path in labels]
    for task in workers:
        task.start()
    results = [output.get(timeout=600) for _ in workers]
    for task in workers:
        task.join(timeout=30)
        assert not task.is_alive()
        if processes:
            assert task.exitcode == 0
    success = [value for status, value in results if status == "ok"]
    errors = [value for status, value in results if status == "error"]
    assert len(success) == (1 if different else 2), results
    assert errors == (["holdout scoring package conflicts"] if different else [])
    assert all(item == success[0] for item in success)
    winning_path = next(
        path for path in labels if digest(path.read_bytes()) == success[0]["package_hash"]
    )
    result = f.score(winning_path)
    artifacts = _assert_scoring_complete(f, result)
    expected_root = json.loads(winning_path.read_bytes())["judgments"][0]["score"][
        "root_cause_correct"
    ]
    for binding in json.loads(artifacts["holdout-scoring/bindings.json"]):
        child = f.evaluator.load(binding["evaluator_score_run_id"])
        ref = next(r for r in child.artifact_refs if r.name == "holdout/private-score.json")
        assert json.loads(f.evaluator.read(ref))["score"]["root_cause_correct"] == expected_root


@pytest.mark.parametrize("unsafe", ["symlink", "permissions", "hardlink"])
def test_session_claim_rejects_unsafe_lock(scoring_fixture, unsafe):
    f = scoring_fixture
    path = f.evaluator.root / f".holdout-scoring-{_session_id(f)}.lock"
    other = f.labels_path.with_name("lock-target")
    other.write_bytes(b"do not touch")
    other.chmod(0o600)
    if unsafe == "symlink":
        path.symlink_to(other)
    elif unsafe == "hardlink":
        path.hardlink_to(other)
    else:
        path.touch(mode=0o644)
    with pytest.raises(ValueError, match="claim"):
        f.score()
    assert not (f.evaluator.root / _session_id(f)).exists()
    assert other.read_bytes() == b"do not touch"


def test_prepared_persistence_checks_identity_and_matches_standalone(scoring_fixture):
    f = scoring_fixture
    item = f.preflight().items[0]
    prepared = item.prepared_score
    before = f.snapshot()
    stale = (
        prepared.model_copy(update={"run_id": "f" * 32}),
        prepared.model_copy(
            update={
                "binding": prepared.binding.model_copy(update={"evaluator_score_run_id": "f" * 32})
            }
        ),
        prepared.model_copy(
            update={"binding": prepared.binding.model_copy(update={"private_score_hash": "f" * 64})}
        ),
        prepared.model_copy(update={"private_score_content": b"{}"}),
    )
    for candidate in stale:
        with pytest.raises(ValueError):
            _persist_prepared_item(f, item, candidate)
        assert f.snapshot() == before
    binding = _persist_prepared_item(f, item, prepared, reload=True)
    expected = f.package.judgments[0]
    assert (
        f.controller.holdout.bind_score(
            f.batch,
            f.records[0].case_id,
            f.refs[0],
            labels=expected.labels,
            score=expected.score,
            should_be_inconclusive=expected.should_be_inconclusive,
            private_holdout_passed=expected.private_holdout_passed,
        )
        == binding
    )
    assert binding == prepared.binding


def _persist_prepared_item(f, item, prepared, *, reload=False):
    return f.controller.holdout._bind_prepared_score(
        f.batch,
        prepared,
        alias=item.alias,
        public_record_ref=item.public_record_ref,
        public_record=f.records[item.ordinal],
        private_case_id=item.prepared_score.binding.private_case_id,
        private_template_id=item.prepared_score.binding.private_template_id,
        _reload=reload,
    )


def test_prepared_persistence_rejects_complete_stale_binding_before_mutation(scoring_fixture):
    f = scoring_fixture
    plan = f.preflight()
    item, other = plan.items[:2]
    prepared = item.prepared_score
    private = json.loads(prepared.private_score_content)
    private["public_record_hash"] = "f" * 64
    stale_score = canonical(private)
    stale = (
        prepared.model_copy(
            update={
                "binding": prepared.binding.model_copy(
                    update={"public_evaluation_run_id": "f" * 32}
                )
            }
        ),
        prepared.model_copy(
            update={
                "run_id": other.prepared_score.run_id,
                "binding": prepared.binding.model_copy(
                    update={
                        "evaluator_score_run_id": other.prepared_score.run_id,
                        "public_record_id": other.prepared_score.binding.public_record_id,
                    }
                ),
            }
        ),
        prepared.model_copy(
            update={
                "private_score_content": stale_score,
                "binding": prepared.binding.model_copy(
                    update={
                        "public_record_hash": "f" * 64,
                        "private_score_hash": digest(stale_score),
                    }
                ),
            }
        ),
        prepared.model_copy(
            update={
                "binding": prepared.binding.model_copy(update={"private_case_id": "stale-case"})
            }
        ),
        prepared.model_copy(
            update={
                "binding": prepared.binding.model_copy(
                    update={"private_template_id": "stale-template"}
                )
            }
        ),
    )
    before = f.snapshot()
    for candidate in stale:
        with pytest.raises(ValueError, match="prepared holdout score is invalid"):
            _persist_prepared_item(f, item, candidate)
        assert f.snapshot() == before


def test_prepared_persistence_rejects_stale_child_lineage_before_mutation(scoring_fixture):
    f = scoring_fixture
    item = f.preflight().items[0]
    run_id = f.seed(0)
    manifest_path = f.evaluator.root / run_id / "manifest.json"
    original = manifest_path.read_bytes()
    changes = {
        "binding": lambda payload: payload["binding"].update(prompt_version="stale"),
        "parent": lambda payload: payload.update(parent_run_id="f" * 32),
        "external_origin": lambda payload: payload.update(
            external_origin={"run_id": "f" * 32, "visibility": "public"}
        ),
    }
    for change in changes.values():
        payload = json.loads(original)
        change(payload)
        overwrite_fixture_artifact(manifest_path, canonical(payload))
        before = f.snapshot()
        with pytest.raises(ValueError, match="holdout score transaction is invalid"):
            _persist_prepared_item(f, item, item.prepared_score)
        assert f.snapshot() == before
    overwrite_fixture_artifact(manifest_path, original)


_SESSION_ARTIFACTS = tuple(
    f"holdout-scoring/{name}.json" for name in ("input-binding", "bindings", "metrics", "result")
)


def _seed_scoring_session(f, status, phase, names):
    run = f.evaluator.create_run(
        "holdout_scoring",
        binding=f.binding,
        external_origin=ExternalRunOrigin(run_id=f.evaluation_run_id, visibility="public"),
        _run_id=_session_id(f),
    )
    if status != RunStatus.QUEUED:
        f.evaluator.transition(
            run.id,
            RunStatus.RUNNING,
            CurrentPhase.EXECUTING if phase == CurrentPhase.EXECUTING else CurrentPhase.PREPARING,
        )
    for ordinal, name in enumerate(names):
        f.evaluator.put(run.id, name, f'{{"ordinal":{ordinal}}}'.encode(), "evaluator")
    if phase == CurrentPhase.FINALIZING or status == RunStatus.COMPLETED:
        f.evaluator.transition(run.id, RunStatus.RUNNING, CurrentPhase.FINALIZING)
    if status == RunStatus.COMPLETED:
        f.evaluator.transition(run.id, RunStatus.COMPLETED, None)
    return f.evaluator.load(run.id)


@pytest.mark.parametrize(
    "status,phase,names",
    [
        (RunStatus.QUEUED, None, ()),
        (RunStatus.RUNNING, CurrentPhase.PREPARING, ()),
        (RunStatus.RUNNING, CurrentPhase.PREPARING, _SESSION_ARTIFACTS[:1]),
        (RunStatus.RUNNING, CurrentPhase.PREPARING, _SESSION_ARTIFACTS[:2]),
        (RunStatus.RUNNING, CurrentPhase.PREPARING, _SESSION_ARTIFACTS[:3]),
        (RunStatus.RUNNING, CurrentPhase.PREPARING, _SESSION_ARTIFACTS),
        (RunStatus.RUNNING, CurrentPhase.FINALIZING, _SESSION_ARTIFACTS),
        (RunStatus.COMPLETED, None, _SESSION_ARTIFACTS),
    ],
)
def test_session_recovery_accepts_only_valid_durable_boundaries(
    scoring_fixture, status, phase, names
):
    f = scoring_fixture
    expected = _seed_scoring_session(f, status, phase, names)
    before = f.snapshot()
    assert f.controller._session(_session_id(f), f.evaluation_run_id) == expected
    assert f.snapshot() == before


@pytest.mark.parametrize(
    "status,phase,names,manifest_fault",
    [
        (RunStatus.QUEUED, None, _SESSION_ARTIFACTS[:1], None),
        (
            RunStatus.RUNNING,
            CurrentPhase.PREPARING,
            (*_SESSION_ARTIFACTS[:1], "holdout-scoring/unexpected.json"),
            None,
        ),
        (RunStatus.RUNNING, CurrentPhase.PREPARING, _SESSION_ARTIFACTS[1:2], None),
        (
            RunStatus.RUNNING,
            CurrentPhase.PREPARING,
            (_SESSION_ARTIFACTS[0], _SESSION_ARTIFACTS[2]),
            None,
        ),
        (RunStatus.RUNNING, CurrentPhase.EXECUTING, (), None),
        (RunStatus.RUNNING, CurrentPhase.FINALIZING, _SESSION_ARTIFACTS[:1], None),
        (RunStatus.COMPLETED, None, _SESSION_ARTIFACTS[:3], None),
        (
            RunStatus.COMPLETED,
            None,
            (*_SESSION_ARTIFACTS, "holdout-scoring/unexpected.json"),
            None,
        ),
        (RunStatus.RUNNING, CurrentPhase.PREPARING, _SESSION_ARTIFACTS[:1], "duplicate"),
        (RunStatus.RUNNING, CurrentPhase.PREPARING, _SESSION_ARTIFACTS[:1], "foreign"),
    ],
)
def test_session_recovery_rejects_invalid_inventory_before_mutation(
    scoring_fixture, status, phase, names, manifest_fault
):
    f = scoring_fixture
    run = _seed_scoring_session(f, status, phase, names)
    if manifest_fault is not None:
        manifest_path = f.evaluator.root / run.id / "manifest.json"
        payload = json.loads(manifest_path.read_bytes())
        if manifest_fault == "duplicate":
            payload["artifact_refs"].append(payload["artifact_refs"][0])
        else:
            payload["artifact_refs"][0]["run_id"] = "f" * 32
        overwrite_fixture_artifact(manifest_path, canonical(payload))
    before = f.snapshot()
    with pytest.raises(ValueError, match="holdout scoring session conflicts"):
        f.controller._session(_session_id(f), f.evaluation_run_id)
    assert f.snapshot() == before


def test_grouped_metrics_load_once_and_match_independent_aggregate(scoring_fixture, monkeypatch):
    from gpu_agent.benchmark.holdout import EvaluatorRecordBinding
    from gpu_agent.benchmark.metrics import aggregate, aggregate_grouped

    f = scoring_fixture
    bindings = []
    for ordinal in (0, 1):
        child = f.evaluator.load(f.seed(ordinal))
        ref = next(r for r in child.artifact_refs if r.name == "holdout/record-binding.json")
        bindings.append(EvaluatorRecordBinding.model_validate_json(f.evaluator.read(ref)))
    context = dict(
        public_store=f.public,
        evaluator_store=f.evaluator,
        run_binding=f.binding,
        schedule_verifier=f.controller.holdout._schedule_verifier,
    )
    expected = aggregate(bindings, **context)
    loads = []
    validations = []
    original = HoldoutController._load_metric_record
    original_validation = HoldoutController.validated_evaluation

    def counted(self, binding, *args, **kwargs):
        loads.append(binding.public_record_id)
        return original(self, binding, *args, **kwargs)

    def counted_validation(self, batch, evaluation_run_id):
        validations.append(evaluation_run_id)
        return original_validation(self, batch, evaluation_run_id)

    monkeypatch.setattr(HoldoutController, "_load_metric_record", counted)
    monkeypatch.setattr(HoldoutController, "validated_evaluation", counted_validation)
    result = aggregate_grouped(bindings, **context)
    assert result.overall == expected
    assert loads == [binding.public_record_id for binding in bindings]
    assert validations == [f.evaluation_run_id]


def test_private_commitment_resolution_returns_exact_native_transaction(scoring_fixture):
    """A blind public ordinal resolves to one exact terminal evaluator record."""
    from gpu_agent.benchmark.evaluation import NativeEvaluationLineage

    f = scoring_fixture
    ordinal = 73
    resolved = f.holdout.resolve_evaluation_record(f.batch, f.evaluation_run_id, ordinal)

    assert resolved.public_record == f.records[ordinal]
    assert resolved.execution_binding.ordinal == ordinal
    assert resolved.execution_binding.public_evaluation_run_id == f.evaluation_run_id
    assert resolved.execution_binding.alias == f.records[ordinal].case_id
    assert resolved.native_record.case_id == resolved.execution_binding.private_case_id
    assert resolved.native_record.template_id == resolved.execution_binding.private_template_id
    assert isinstance(resolved.native_record.lineage, NativeEvaluationLineage)
    assert (
        resolved.native_record.lineage.diagnosis_run_id
        == resolved.execution_binding.diagnosis_run_id
    )


def test_persisted_metric_record_uses_validated_native_diagnosis(scoring_fixture):
    """Metrics consume evaluator-native diagnosis, not the blind public projection."""
    from gpu_agent.benchmark.holdout import EvaluatorRecordBinding

    f = scoring_fixture
    run_id = f.seed(0)
    run = f.evaluator.load(run_id)
    binding = EvaluatorRecordBinding.model_validate_json(
        f.evaluator.read(
            next(ref for ref in run.artifact_refs if ref.name == "holdout/record-binding.json")
        )
    )
    loaded = f.controller.holdout._load_metric_record(binding)
    resolved = f.holdout.resolve_evaluation_record(f.batch, f.evaluation_run_id, 0)

    assert loaded.diagnosis == resolved.native_record.diagnosis
    assert loaded.diagnosis


def test_private_scoring_canaries_never_enter_public_store(scoring_fixture):
    f = scoring_fixture
    f.seed(0)
    mapping = json.loads(
        f.evaluator.read(
            next(
                ref
                for ref in f.evaluator.load(f.mapping_run_id).artifact_refs
                if ref.name == "holdout/private-alias-map.json"
            )
        )
    )
    public_bytes = b"\n".join(
        path.read_bytes() for path in sorted(f.public.root.rglob("*")) if path.is_file()
    )
    execution_ids = [
        path.name.encode()
        for path in f.evaluator.root.iterdir()
        if path.is_dir()
        and len(path.name) == 32
        and f.evaluator.load(path.name).kind in {"holdout_execution", "diagnosis"}
    ]

    for canary in (
        b"PRIVATE-JUDGMENT-CANARY",
        mapping["nonce_hex"].encode(),
        str(f.evaluator.root).encode(),
        *execution_ids,
    ):
        assert canary not in public_bytes
