"""Frozen, public-only repair experiences; heuristics are not CUDA authorities.

Records are deterministically distilled from *failed public self-check* outcomes.
No evaluator store, private inputs, model-authored lessons, arbitrary URLs, or
on-the-fly updates are allowed. Rebuild/freeze the index between experiments.
"""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from gpu_agent.agent.models import DiagnosisResult
from gpu_agent.contracts import ArtifactRef, RunStatus
from gpu_agent.evidence.repository import _evidence
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.patch_effect import PatchEffectAssessment
from gpu_agent.patching import source_hash
from gpu_agent.public_task import PublicTask
from gpu_agent.store import RunStore, read_regular

LessonCode = Literal[
    "GUARDED_INDEX_EQUIVALENCE",
    "NUMERIC_PASS_RACE_REMAINS",
    "FUNCTIONAL_MISMATCH",
    "PUBLIC_CHECK_FAILED",
    "BLOCK_BARRIER_EDIT_FAILED",
]

LESSONS: dict[str, str] = {
    "GUARDED_INDEX_EQUIVALENCE": (
        "A historical patch replaced an array index with threadIdx.x under a "
        "matching equality guard, leaving the expression locally equivalent. "
        "That candidate failed its public checks; reassess the actual fault."
    ),
    "NUMERIC_PASS_RACE_REMAINS": (
        "A prior candidate produced correct public numeric output but racecheck "
        "still reported a finding. Functional success does not imply race freedom."
    ),
    "FUNCTIONAL_MISMATCH": (
        "A prior patch did not preserve public numerical semantics even though "
        "other checks might have passed. Recheck the exact computation."
    ),
    "PUBLIC_CHECK_FAILED": (
        "A prior candidate failed a public build/runtime/functional or sanitizer "
        "check. Inspect concrete current evidence instead of copying that patch."
    ),
    "BLOCK_BARRIER_EDIT_FAILED": (
        "A previous public patch added, removed or moved a block barrier and still "
        "failed a functional or sanitizer check. Reassess shared-memory read/write "
        "ordering and block-uniform barrier participation; adding or moving a "
        "barrier alone is not proof of correctness."
    ),
}


_BARRIER_STMT = re.compile(r"^\s*__syncthreads\s*\(\s*\)\s*;\s*(?://.*)?$")


def _barrier_neighbors(source: bytes) -> list[tuple[str, str]]:
    """Record the nearest operations on both sides of every block barrier."""
    lines = [
        line.strip()
        for line in source.decode("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    ]
    return [
        (lines[i - 1] if i else "", lines[i + 1] if i + 1 < len(lines) else "")
        for i, line in enumerate(lines)
        if _BARRIER_STMT.fullmatch(line) is not None
    ]


def _edited_block_barrier(before: bytes, after: bytes) -> bool:
    """Detect insertion/removal or changed operation order around block barriers.

    This is a historical warning about an edit and failed check, never a
    deterministic attribution that this edit caused the observed failure.
    """
    return _barrier_neighbors(before) != _barrier_neighbors(after)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


class RepairExperience(ExecutionModel):
    record_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    public_run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    algorithm: str = Field(min_length=1, max_length=128)
    failure_family: str = Field(min_length=1, max_length=64)
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    lesson_code: LessonCode
    lesson: str = Field(min_length=1, max_length=700)

    @model_validator(mode="after")
    def integrity(self) -> "RepairExperience":
        if self.lesson != LESSONS[self.lesson_code]:
            raise ValueError("repair experience lesson text must be controller-derived")
        content = self.model_dump(mode="json", exclude={"record_id"})
        if self.record_id != _digest(content):
            raise ValueError("repair experience content hash mismatch")
        return self


def _one(store: RunStore, run_id: str, name: str) -> ArtifactRef | None:
    found = [ref for ref in store.load(run_id).artifact_refs if ref.name == name]
    return found[-1] if len(found) == 1 else None


def derive_public_experiences(store: RunStore, run_id: str) -> list[RepairExperience]:
    """Admit only hash-checked, source-bound public artifacts from unbound diagnosis runs."""
    if store.visibility != "public":
        raise ValueError("private/evaluator memories are forbidden")
    manifest = store.load(run_id)
    if (
        manifest.kind != "diagnosis"
        or manifest.status != RunStatus.COMPLETED
        or manifest.binding is not None
        or manifest.external_origin is not None
    ):
        raise ValueError("only completed unbound public diagnosis runs may teach the Agent")
    task_ref = _one(store, run_id, "public-task.json")
    diagnosis_ref = _one(store, run_id, "diagnosis.json")
    summary_ref = _one(store, run_id, "repair/summary.json")
    if not task_ref or not diagnosis_ref or not summary_ref:
        return []
    task = PublicTask.model_validate_json(store.read(task_ref))
    bundle = _evidence(store).view(run_id)
    originals = {PurePosixPath(ref.name).name: store.read(ref) for ref in bundle.source_snapshot}
    if hashlib.sha256(originals.get("kernel.cu", b"")).hexdigest() != task.source_sha256:
        raise ValueError("public task source is not bound to the original snapshot")
    original_manifest_hash = source_hash(originals)
    diagnosis = DiagnosisResult.model_validate_json(store.read(diagnosis_ref))
    summary = json.loads(store.read(summary_ref))
    experiences: list[RepairExperience] = []
    for index, round_info in enumerate(summary.get("rounds", []), 1):
        checked = round_info.get("check", {})
        if checked.get("status") != "FAILED":
            continue
        candidate_ref = _one(store, run_id, f"repair/{index}/candidate.json")
        result_ref = _one(store, run_id, f"repair/{index}/result.json")
        if candidate_ref is None or result_ref is None:
            raise ValueError("failed repair lacks public candidate or check evidence")
        candidate = json.loads(store.read(candidate_ref))
        check = json.loads(store.read(result_ref))
        if (
            check != checked
            or candidate.get("patched_source_hash") != round_info.get("candidate_hash")
            or candidate.get("base_source_hash") != original_manifest_hash
            or candidate.get("parent_run_id") != run_id
        ):
            raise ValueError("repair experience candidate or public check source mismatch")
        check_run_id = check.get("run_id")
        if not isinstance(check_run_id, str):
            raise ValueError("repair experience missing public self-check run")
        child = store.load(check_run_id)
        if (
            child.kind != "repair_self_check"
            or child.status != RunStatus.COMPLETED
            or child.parent_run_id != run_id
            or child.binding is not None
        ):
            raise ValueError("repair experience self-check lineage is invalid")
        check_ref = _one(store, check_run_id, "self-check.json")
        if check_ref is None or json.loads(store.read(check_ref)) != check:
            raise ValueError("repair experience public self-check report mismatch")
        checked_sources = {
            PurePosixPath(ref.name).name: store.read(ref)
            for ref in child.artifact_refs
            if ref.name.startswith("sources/")
        }
        if source_hash(checked_sources) != candidate["patched_source_hash"]:
            raise ValueError("repair experience self-check source hash mismatch")
        checks = checked.get("checks", {})
        effect_ref = _one(store, run_id, f"repair/{index}/patch-effect.json")
        effect = (
            PatchEffectAssessment.model_validate_json(store.read(effect_ref))
            if effect_ref is not None
            else None
        )
        if effect is not None:
            if (
                effect.candidate_source_sha256
                != hashlib.sha256(checked_sources["kernel.cu"]).hexdigest()
                or effect.reference_source_sha256 != task.source_sha256
            ):
                raise ValueError("repair patch-effect source provenance mismatch")
        if effect is not None and effect.semantic_equivalence == "PROVEN_LOCAL_NO_OP":
            lesson_code: LessonCode = "GUARDED_INDEX_EQUIVALENCE"
        elif checks.get("racecheck") == "FINDING" and checks.get("functional") == "PASSED":
            lesson_code = "NUMERIC_PASS_RACE_REMAINS"
        elif _edited_block_barrier(
            originals["kernel.cu"], checked_sources["kernel.cu"]
        ) and (
            checks.get("functional") not in (None, "PASSED")
            or any(
                checks.get(tool) == "FINDING"
                for tool in ("memcheck", "racecheck", "initcheck", "synccheck")
            )
        ):
            lesson_code = "BLOCK_BARRIER_EDIT_FAILED"
        elif checks.get("functional") not in (None, "PASSED"):
            lesson_code = "FUNCTIONAL_MISMATCH"
        else:
            lesson_code = "PUBLIC_CHECK_FAILED"
        record = {
            "public_run_id": run_id,
            "candidate_sha256": candidate["patched_source_hash"],
            "source_sha256": task.source_sha256,
            "algorithm": task.algorithm,
            "failure_family": diagnosis.failure_family,
            "evidence_sha256": result_ref.sha256,
            "lesson_code": lesson_code,
            "lesson": LESSONS[lesson_code],
        }
        experiences.append(RepairExperience(record_id=_digest(record), **record))
    return experiences


class FrozenRepairMemory(ExecutionModel):
    version: Literal["public-repair-memory-v1"] = "public-repair-memory-v1"
    records: list[RepairExperience]
    corpus_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def frozen_integrity(self) -> "FrozenRepairMemory":
        ids = [r.record_id for r in self.records]
        if ids != sorted(set(ids)) or self.corpus_sha256 != _digest(ids):
            raise ValueError("repair memory index is not canonical and frozen")
        return self

    @classmethod
    def combine(cls, indexes: list["FrozenRepairMemory"]) -> "FrozenRepairMemory":
        """Merge independently frozen public indexes, preserving deterministic identity."""
        unique: dict[str, RepairExperience] = {}
        for index in indexes:
            validated = cls.model_validate(index)
            for record in validated.records:
                prior = unique.get(record.record_id)
                if prior is not None and prior != record:
                    raise ValueError("conflicting frozen repair experience identity")
                unique[record.record_id] = record
        ids = sorted(unique)
        return cls(records=[unique[i] for i in ids], corpus_sha256=_digest(ids))

    @classmethod
    def from_public_runs(cls, store: RunStore, run_ids: list[str]) -> "FrozenRepairMemory":
        records = []
        for run_id in sorted(set(run_ids)):
            records.extend(derive_public_experiences(store, run_id))
        unique = {r.record_id: r for r in records}
        ids = sorted(unique)
        return cls(records=[unique[i] for i in ids], corpus_sha256=_digest(ids))

    def retrieve(
        self,
        algorithm: str,
        failure_family: str,
        *,
        k: int = 3,
    ) -> list[dict[str, str]]:
        if not 1 <= k <= 5:
            raise ValueError("invalid retrieval count")
        ranks = sorted(
            self.records,
            key=lambda record: (
                -(
                    5 * (record.algorithm == algorithm)
                    + 3 * (record.failure_family == failure_family)
                ),
                record.record_id,
            ),
        )
        eligible = [
            record
            for record in ranks
            if record.algorithm == algorithm or record.failure_family == failure_family
        ][:k]
        return [
            {
                "record_id": rec.record_id,
                "source_public_run_id": rec.public_run_id,
                "evidence_sha256": rec.evidence_sha256,
                "lesson_code": rec.lesson_code,
                "lesson": rec.lesson,
                "trust": "HISTORICAL_PUBLIC_FAILURE_NOT_AUTHORITATIVE",
            }
            for rec in eligible
        ]

    def save(self, destination: Path) -> None:
        destination.write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "FrozenRepairMemory":
        return cls.model_validate_json(read_regular(path.absolute(), 4 * 1024 * 1024))
