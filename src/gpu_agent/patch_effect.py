"""Conservative static *advice* on candidate patch effect; never a verifier."""

import difflib
import hashlib
import re
from typing import Literal

from pydantic import Field

from gpu_agent.agent.models import DiagnosisResult
from gpu_agent.execution.models import ExecutionModel

_GUARDED_INDEX = re.compile(r"\bif\s*\(\s*threadIdx\.x\s*==\s*(\d+)\s*\)")


class PatchEffectAssessment(ExecutionModel):
    version: Literal[1] = 1
    candidate_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reference_source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    changed_original_lines: list[int] = Field(default_factory=list)
    diagnosed_location_touched: bool | None = None
    semantic_equivalence: Literal["PROVEN_LOCAL_NO_OP", "NOT_ESTABLISHED"] = "NOT_ESTABLISHED"
    reasoning_code: str = "NO_STATIC_PROOF"


def analyze_patch_effect(
    original: str,
    patched: str,
    diagnosis: DiagnosisResult,
    *,
    diagnosis_source: str,
) -> PatchEffectAssessment:
    """Recognize only exact guarded-index equivalence, without blocking a patch.

    The checker deliberately abstains from broad C++ equivalence claims, and
    treats a diagnosis from another candidate's hash as a different source version.
    """
    old_lines = original.splitlines()
    new_lines = patched.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    changes: list[tuple[int, int, list[str], list[str]]] = []
    modified: list[int] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changes.append((i1, j1, old_lines[i1:i2], new_lines[j1:j2]))
        modified.extend(range(i1 + 1, i2 + 1))

    source_matches = (
        hashlib.sha256(diagnosis_source.encode()).hexdigest()
        == hashlib.sha256(original.encode()).hexdigest()
    )
    touched = (
        any(
            location.path == "kernel.cu" and location.line in modified
            for location in diagnosis.source_locations
        )
        if source_matches
        else None
    )
    equivalence: Literal["PROVEN_LOCAL_NO_OP", "NOT_ESTABLISHED"] = "NOT_ESTABLISHED"
    code = "CHANGE_NOT_PROVEN_EQUIVALENT"
    if len(changes) == 1:
        _, _, before, after = changes[0]
        if len(before) == len(after) == 1:
            old, new = before[0], after[0]
            guards = list(_GUARDED_INDEX.finditer(old))
            new_guard = _GUARDED_INDEX.search(new)
            if len(guards) == 1 and new_guard is not None:
                index = guards[0].group(1)
                if (
                    new_guard.group(1) == index
                    and old.replace(f"[{index}]", "[threadIdx.x]", 1) == new
                ):
                    equivalence = "PROVEN_LOCAL_NO_OP"
                    code = "EQUAL_INDEX_UNDER_THREAD_GUARD"
    if not changes:
        equivalence, code = "PROVEN_LOCAL_NO_OP", "NO_SOURCE_CHANGE"
    return PatchEffectAssessment(
        candidate_source_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        reference_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        changed_original_lines=modified,
        diagnosed_location_touched=touched,
        semantic_equivalence=equivalence,
        reasoning_code=code,
    )
