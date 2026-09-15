import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpu_agent.benchmark.evaluation import (
    EvaluationBindings,
    EvaluationSchedule,
    EvaluationScheduleItem,
    HoldoutScheduleProof,
)
from gpu_agent.benchmark.ledger import CorpusFamily
from gpu_agent.benchmark.schedule_authority import (
    EvaluationScheduleAuthority,
    EvaluationScheduleVerifier,
)
from gpu_agent.contracts import CurrentPhase, RepositorySnapshot, RunBinding, RunStatus
from gpu_agent.store import RunStore


def _binding() -> RunBinding:
    return RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="evaluation",
        toolchain_lock_hash="c" * 64,
        prompt_version="prompt-v1",
        model_config_hash="d" * 64,
        corpus_ledger_namespace_hash="e" * 64,
    )


def _schedule() -> EvaluationSchedule:
    bindings = EvaluationBindings(
        commit="a" * 40,
        prompt_version="prompt-v1",
        toolchain_hash="c" * 64,
        model_config_hash="d" * 64,
        max_cost_usd=6,
        max_unit_cost_usd=1,
    )
    items = [
        EvaluationScheduleItem(
            ordinal=ordinal,
            case_id=case_id,
            template_id=template_id,
            mode=mode,
            repeat=repeat,
            split="development",
        )
        for ordinal, (repeat, case_id, template_id, mode) in enumerate(
            (repeat, case_id, template_id, mode)
            for repeat in range(3)
            for case_id, template_id in {"case_1": "template_1", "case_2": "template_2"}.items()
            for mode in ("A",)
        )
    ]
    return EvaluationSchedule(
        selection="A",
        modes=["A"],
        split="development",
        repeats=3,
        random_seed=7,
        bindings=bindings,
        items=items,
    )


def _setup(tmp_path, monkeypatch):
    public = RunStore(tmp_path / "evaluation-runs")
    corpus_public = RunStore(tmp_path / "corpus-public")
    family = CorpusFamily.provision(
        tmp_path / "controller",
        public_store=corpus_public.root,
        evaluator_store=tmp_path / "corpus-private",
        repository=tmp_path / "repository",
    )
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    binding = _binding().model_copy(update={"corpus_ledger_namespace_hash": family.namespace_hash})
    run = public.create_run("evaluation", binding=binding)
    public.transition(run.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)
    schedule = _schedule()
    public.put(
        run.id,
        "evaluation/schedule.json",
        schedule.model_dump_json().encode(),
        "public",
    )
    return public, family, run.id, binding, schedule


def test_schedule_requires_controller_committed_transaction(tmp_path, monkeypatch):
    public, family, run_id, binding, schedule = _setup(tmp_path, monkeypatch)
    authority = EvaluationScheduleAuthority.for_family(family, public)
    prepared = authority.prepare(run_id, schedule, binding)

    with pytest.raises(ValueError, match="not committed"):
        EvaluationScheduleVerifier.for_family(family, public).verify(run_id)

    committed = authority.commit(prepared)
    authority.persist_receipt(public, committed)
    verified = EvaluationScheduleVerifier.for_family(family, public).verify(run_id)
    assert verified.transaction_id == committed.transaction_id
    assert verified.case_templates == {"case_1": "template_1", "case_2": "template_2"}


def test_schedule_transaction_is_concurrent_exact_and_copy_bound(tmp_path, monkeypatch):
    public, family, run_id, binding, schedule = _setup(tmp_path, monkeypatch)
    authority = EvaluationScheduleAuthority.for_family(family, public)

    with ThreadPoolExecutor(max_workers=2) as pool:
        transactions = list(
            pool.map(lambda _: authority.seal(public, run_id, schedule, binding), range(2))
        )
    assert transactions[0] == transactions[1]

    copied = public.create_run("evaluation", binding=binding)
    public.transition(copied.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)
    for name in ("evaluation/schedule.json", "evaluation/schedule-receipt.json"):
        source = next(ref for ref in public.load(run_id).artifact_refs if ref.name == name)
        public.put(copied.id, name, public.read(source), "public")
    with pytest.raises(ValueError, match="not committed"):
        EvaluationScheduleVerifier.for_family(family, public).verify(copied.id)


def test_schedule_verifier_rejects_subset_and_has_no_writer_capability(tmp_path, monkeypatch):
    public, family, run_id, binding, schedule = _setup(tmp_path, monkeypatch)
    authority = EvaluationScheduleAuthority.for_family(family, public)
    authority.seal(public, run_id, schedule, binding)
    verifier = EvaluationScheduleVerifier.for_family(family, public)
    assert not hasattr(verifier, "prepare")
    assert not hasattr(verifier, "commit")
    assert not any("key" in name or "sign" in name for name in vars(verifier))

    run = public.load(run_id)
    ref = next(item for item in run.artifact_refs if item.name == "evaluation/schedule.json")
    payload = json.loads(public.read(ref))
    payload["items"] = payload["items"][:-1]
    # A copied, incomplete schedule cannot be authorized under a fresh run.
    forged = public.create_run("evaluation", binding=binding)
    public.transition(forged.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)
    public.put(forged.id, "evaluation/schedule.json", json.dumps(payload).encode(), "public")
    with pytest.raises(ValueError):
        verifier.verify(forged.id)


def test_holdout_schedule_must_cover_entire_committed_alias_batch(tmp_path, monkeypatch):
    public, family, _, binding, development = _setup(tmp_path, monkeypatch)
    aliases = ["1" * 64, "2" * 64]
    alias_run = public.create_run("holdout_aliases", binding=binding)
    public.transition(alias_run.id, RunStatus.RUNNING, CurrentPhase.PREPARING)
    alias_ref = public.put(
        alias_run.id,
        "holdout/aliases.json",
        json.dumps(
            {"schema_version": 1, "aliases": aliases},
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        "public",
    )
    public.transition(alias_run.id, RunStatus.COMPLETED, None)
    proof = HoldoutScheduleProof(public_run_id=alias_run.id, aliases_hash=alias_ref.sha256)
    holdout = development.model_copy(
        update={
            "split": "holdout",
            "holdout_proof": proof,
            "items": [
                item.model_copy(
                    update={
                        "case_id": aliases[index % 2],
                        "template_id": aliases[index % 2],
                        "split": "holdout",
                        "holdout_proof": proof,
                    }
                )
                for index, item in enumerate(development.items)
            ],
        }
    )
    run = public.create_run("evaluation", binding=binding)
    public.transition(run.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)
    public.put(run.id, "evaluation/schedule.json", holdout.model_dump_json().encode(), "public")
    authority = EvaluationScheduleAuthority.for_family(family, public)
    authority.seal(public, run.id, holdout, binding)
    assert (
        EvaluationScheduleVerifier.for_family(family, public).verify(run.id).holdout_aliases
        == aliases
    )

    subset = holdout.model_copy(
        update={
            "items": [
                item.model_copy(update={"ordinal": ordinal})
                for ordinal, item in enumerate(
                    item for item in holdout.items if item.case_id == aliases[0]
                )
            ]
        }
    )
    forged = public.create_run("evaluation", binding=binding)
    public.transition(forged.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)
    public.put(forged.id, "evaluation/schedule.json", subset.model_dump_json().encode(), "public")
    with pytest.raises(ValueError, match="complete alias batch"):
        authority.prepare(forged.id, subset, binding)
