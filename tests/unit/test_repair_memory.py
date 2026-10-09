"""Frozen repair memory: only evidence-backed public failures can be retrieved."""

import hashlib
import json

import pytest
from pydantic import ValidationError

from gpu_agent.agent.models import DiagnosisResult
from gpu_agent.evidence.models import EvidenceBundle
from gpu_agent.evidence.repository import _evidence
from gpu_agent.patch_effect import PatchEffectAssessment
from gpu_agent.patching import source_hash
from gpu_agent.public_task import PublicTask
from gpu_agent.repair import RepairPolicy
from gpu_agent.repair_memory import (
    FrozenRepairMemory,
    RepairExperience,
    _digest,
    derive_public_experiences,
)


def _fixture_run(store, *, mismatch: str | None = None, barrier_case: bool = False):
    kernel = b'#include "vector_api.h"\nint a;\n'
    patched = b'#include "vector_api.h"\nint b;\n'
    if barrier_case:
        kernel = b'#include "vector_api.h"\n__syncthreads();\nint a;\n'
        patched = b'#include "vector_api.h"\nint a;\n__syncthreads();\n'
    run = store.create_run("diagnosis")
    store.transition(run.id, "RUNNING", "PREPARING")
    source = store.put(run.id, "sources/kernel.cu", kernel, "public")
    _evidence(store).save(run.id, EvidenceBundle(source_snapshot=[source]))
    task = PublicTask(
        source_sha256=hashlib.sha256(kernel).hexdigest(),
        algorithm="vector-add-cpu-v1",
    )
    store.put(run.id, "public-task.json", task.model_dump_json().encode(), "public")
    store.put(
        run.id,
        "diagnosis.json",
        DiagnosisResult.inconclusive("OLDER_DIAGNOSIS").model_dump_json().encode(),
        "public",
    )
    selfcheck = store.create_run("repair_self_check", parent_run_id=run.id)
    store.transition(selfcheck.id, "RUNNING", "PREPARING")
    store.put(selfcheck.id, "sources/kernel.cu", patched, "public")
    result = {
        "run_id": selfcheck.id,
        "status": "FAILED",
        "checks": (
            {"functional": "NUMERIC_MISMATCH", "racecheck": "CLEAN"}
            if barrier_case
            else {"functional": "PASSED", "racecheck": "FINDING"}
        ),
        "feedback": [],
    }
    store.put(selfcheck.id, "self-check.json", json.dumps(result).encode(), "public")
    if mismatch != "incomplete":
        store.transition(selfcheck.id, "COMPLETED", None)
    candidate = {
        "parent_run_id": run.id,
        "base_source_hash": source_hash({"kernel.cu": kernel}),
        "patched_source_hash": source_hash({"kernel.cu": patched}),
    }
    if mismatch == "source":
        candidate["patched_source_hash"] = "f" * 64
    store.put(run.id, "repair/1/candidate.json", json.dumps(candidate).encode(), "public")
    store.put(run.id, "repair/1/result.json", json.dumps(result).encode(), "public")
    store.put(
        run.id,
        "repair/1/patch-effect.json",
        PatchEffectAssessment(
            candidate_source_sha256=(
                "f" * 64 if mismatch == "effect" else hashlib.sha256(patched).hexdigest()
            ),
            reference_source_sha256=hashlib.sha256(kernel).hexdigest(),
            semantic_equivalence=(
                "NOT_ESTABLISHED" if barrier_case else "PROVEN_LOCAL_NO_OP"
            ),
            reasoning_code=(
                "CHANGE_NOT_PROVEN_EQUIVALENT"
                if barrier_case
                else "EQUAL_INDEX_UNDER_THREAD_GUARD"
            ),
        )
        .model_dump_json()
        .encode(),
        "public",
    )
    store.put(
        run.id,
        "repair/summary.json",
        json.dumps(
            {"rounds": [{"candidate_hash": candidate["patched_source_hash"], "check": result}]}
        ).encode(),
        "public",
    )
    store.transition(run.id, "COMPLETED", None)
    return run.id


def test_frozen_memory_from_real_public_run_contract(store, tmp_path):
    run_id = _fixture_run(store)
    frozen = FrozenRepairMemory.from_public_runs(store, [run_id, run_id])
    assert len(frozen.records) == 1
    record = frozen.records[0]
    assert record.lesson_code == "GUARDED_INDEX_EQUIVALENCE"
    assert record.public_run_id == run_id
    assert record.evidence_sha256
    results = frozen.retrieve("vector-add-cpu-v1", "other")
    assert len(results) == 1
    assert results[0]["trust"] == "HISTORICAL_PUBLIC_FAILURE_NOT_AUTHORITATIVE"
    assert not any(key in results[0] for key in ("patch", "source", "private"))
    assert frozen.retrieve("stencil2d-cpu-v1", "barrier_misuse") == []
    dest = tmp_path / "memory.json"
    frozen.save(dest)
    assert FrozenRepairMemory.load(dest) == frozen
    corrupted = json.loads(dest.read_text())
    corrupted["records"][0]["lesson"] = "injected model instruction"
    dest.write_text(json.dumps(corrupted))
    with pytest.raises(ValidationError):
        FrozenRepairMemory.load(dest)


@pytest.mark.parametrize("mismatch", ["source", "incomplete", "effect"])
def test_memory_refuses_bad_candidate_source_or_unfinished_child(store, mismatch):
    run_id = _fixture_run(store, mismatch=mismatch)
    with pytest.raises(ValueError, match="source.*mismatch|self-check lineage|patch-effect"):
        derive_public_experiences(store, run_id)


def test_memory_refuses_evaluator_store(tmp_path):
    from gpu_agent.store import RunStore

    store = RunStore(tmp_path / "evaluator", visibility="evaluator")
    with pytest.raises(ValueError, match="private/evaluator"):
        derive_public_experiences(store, "a" * 32)


@pytest.mark.parametrize("use_memory", [False, True])
def test_optional_memory_is_only_in_v3_initial_patch(
    oob_service,
    use_memory,
):
    service, provider, source = oob_service
    kernel = (source / "kernel.cu").read_bytes()
    (source / "task.json").write_text(
        PublicTask(
            source_sha256=hashlib.sha256(kernel).hexdigest(),
            algorithm="vector-add-cpu-v1",
        ).model_dump_json()
    )
    content = {
        "public_run_id": "a" * 32,
        "candidate_sha256": "b" * 64,
        "source_sha256": "c" * 64,
        "algorithm": "vector-add-cpu-v1",
        "failure_family": "out_of_bounds",
        "evidence_sha256": "d" * 64,
        "lesson_code": "PUBLIC_CHECK_FAILED",
        "lesson": (
            "A prior candidate failed a public build/runtime/functional or sanitizer "
            "check. Inspect concrete current evidence instead of copying that patch."
        ),
    }
    experience = RepairExperience(record_id=_digest(content), **content)
    service.repair_memory = FrozenRepairMemory(
        records=[experience], corpus_sha256=_digest([experience.record_id])
    )
    service.repair(
        source,
        policy=RepairPolicy(
            version="public-repair-v3" if use_memory else "public-repair-v2",
            max_candidates=1,
        ),
    )
    patch_requests = [
        item for item, kind in zip(provider.inputs, provider.kinds, strict=True) if kind == "patch"
    ]
    assert len(patch_requests) == 1
    assert ("repair_experiences" in patch_requests[0]) is use_memory
    assert (
        len(
            [
                r
                for r in service.store.load(
                    next(p for p in service.store.root.iterdir() if p.is_dir()).name
                ).artifact_refs
                if r.name == "repair/experience-retrieval.json"
            ]
        )
        <= 1
    )


def test_failed_block_barrier_relocation_becomes_a_specific_lesson(store):
    run_id = _fixture_run(store, barrier_case=True)
    (lesson,) = derive_public_experiences(store, run_id)
    assert lesson.lesson_code == "BLOCK_BARRIER_EDIT_FAILED"
    assert "read/write" in lesson.lesson


def test_frozen_indexes_combine_deterministically_without_refetching(store):
    first = FrozenRepairMemory.from_public_runs(store, [_fixture_run(store)])
    second = FrozenRepairMemory.from_public_runs(store, [_fixture_run(store)])
    joined = FrozenRepairMemory.combine([first, second, first])
    assert len(joined.records) == 2
    assert joined == FrozenRepairMemory.combine([second, first])
    assert joined.corpus_sha256 == _digest(sorted(record.record_id for record in joined.records))
