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

from gpu_agent.benchmark.evaluation import EvaluationRunner
from gpu_agent.benchmark.executor import EvaluationExecutor
from gpu_agent.benchmark.holdout import HoldoutController
from gpu_agent.benchmark.holdout_scoring import HoldoutLabelPackage, HoldoutScoringController
from gpu_agent.contracts import RunStatus


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def overwrite_fixture_artifact(path, content):
    mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode | stat.S_IWUSR)
    try:
        path.write_bytes(content)
    finally:
        path.chmod(mode)


class ScoringFixture:
    def __init__(self, executor, root):
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
        holdout_executor = EvaluationExecutor(
            executor.service,
            executor.corpus,
            executor.sources,
            holdout_controller=self.holdout,
            holdout_batch=self.batch,
            _corpus_family=executor._corpus_family,
            _schedule_verifier=executor._schedule_verifier,
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
        for item in schedule.items:
            attempt = runner._attempt(run.id, schedule, item)
            runner._put(
                run.id,
                f"evaluation/attempts/{item.ordinal}.json",
                attempt.model_dump_json().encode(),
            )
            record = holdout_executor.execute_scheduled(run.id, item.ordinal)
            runner._put(
                run.id,
                f"evaluation/records/{item.ordinal}.json",
                record.public().model_dump_json().encode(),
            )
            records.append(record)
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
        _configure_responses_provider(executor, patch, full_script=True)
        yield ScoringFixture(executor, root)


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
        "duplicate_ordinal",
        "duplicate_record_id",
        "manifest_count",
        "manifest_order",
        "modes",
        "repeats",
        "aliases",
        "cartesian",
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
                elif fault == "modes":
                    payload["modes"] = ["A", "B", "C", "D", "D"]
                elif fault == "repeats":
                    payload["repeats"] = 4
                elif fault == "missing_ordinal":
                    payload["items"].pop()
                elif fault == "duplicate_ordinal":
                    payload["items"][-1]["ordinal"] = 0
                elif fault == "aliases":
                    payload["items"][-1]["case_id"] = "f" * 64
                elif fault == "cartesian":
                    payload["items"][-1].update(
                        {
                            key: payload["items"][0][key]
                            for key in ("case_id", "template_id", "mode", "repeat")
                        }
                    )

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
