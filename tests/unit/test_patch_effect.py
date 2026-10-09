"""Patch-effect hints are conservative and never substitute for GPU verification."""

from gpu_agent.agent.models import DiagnosisResult
from gpu_agent.execution.models import SourceLocation
from gpu_agent.patch_effect import analyze_patch_effect


def _diagnosis(line=1):
    return DiagnosisResult(
        diagnostic_outcome="DIAGNOSED",
        failure_family="shared_memory_race",
        root_cause="Shared-memory write collisions.",
        source_locations=[SourceLocation(path="kernel.cu", line=line)],
        confidence_label="high",
    )


def test_guarded_slot_replacement_is_proven_local_noop():
    original = "if (threadIdx.x == 0) block_summary = slots[0];\n"
    bad = "if (threadIdx.x == 0) block_summary = slots[threadIdx.x];\n"
    result = analyze_patch_effect(original, bad, _diagnosis(), diagnosis_source=original)
    assert result.semantic_equivalence == "PROVEN_LOCAL_NO_OP"
    assert result.reasoning_code == "EQUAL_INDEX_UNDER_THREAD_GUARD"
    assert result.diagnosed_location_touched is True


def test_real_fix_and_cross_candidate_location_abstain():
    source = (
        "__shared__ volatile float slots[16];\n"
        "const unsigned int slot = threadIdx.x & 15U;\n"
        "if (threadIdx.x == 0) block_summary = slots[0];\n"
    )
    fixed = source.replace("slots[16]", "slots[32]").replace("threadIdx.x & 15U", "threadIdx.x")
    result = analyze_patch_effect(source, fixed, _diagnosis(2), diagnosis_source=source)
    assert result.semantic_equivalence == "NOT_ESTABLISHED"
    assert result.diagnosed_location_touched is True
    older = analyze_patch_effect(source, fixed, _diagnosis(2), diagnosis_source=source + "//v2")
    assert older.diagnosed_location_touched is None


def test_ambiguous_guard_and_other_changes_do_not_trigger():
    source = "if (threadIdx.x == 0) block_summary = slots[0];\n"
    candidates = [
        "if (threadIdx.x == 1) block_summary = slots[threadIdx.x];\n",
        "if (threadIdx.x == 0) block_summary = slots[threadIdx.x] + 1;\n",
        "block_summary = slots[threadIdx.x];\n",
    ]
    for candidate in candidates:
        result = analyze_patch_effect(source, candidate, _diagnosis(), diagnosis_source=source)
        assert result.semantic_equivalence == "NOT_ESTABLISHED"
