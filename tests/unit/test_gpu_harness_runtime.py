"""The self-built LangGraph harness internals: compaction, permissions, edits, journal.

No paid model or native GPU required; these invariants run on Python 3.11/3.12.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gpu_agent.harness.context import compact_public_context, source_with_line_numbers
from gpu_agent.harness.editing import exact_replace_diff
from gpu_agent.harness.journal import (
    SessionJournal,
    read_verified_journal,
    recovery_disposition,
)
from gpu_agent.harness.models import HarnessDecision, ReplaceSnippet
from gpu_agent.harness.registry import (
    TOOL_REGISTRY,
    ToolPermissionPolicy,
    registered_tool_names,
)
from gpu_agent.patching import (
    SourceSnapshot,
    apply_generated_candidate,
    materialize_candidate,
)

REPO = Path(__file__).resolve().parents[2]


def test_compaction_carries_public_failure_without_repeating_huge_source() -> None:
    old = [{"tool": "read_file", "status": "OK", "content": "x" * 3800}] * 10
    old += [
        {
            "tool": "functional_probe",
            "status": "FAILED",
            "content": '{"size":33,"outcome":"PROGRAM_FAILED"}',
        },
        {
            "tool": "apply_patch",
            "status": "REJECTED",
            "content": "ValueError: hunk context does not match source",
        },
    ]
    recent = [{"tool": "read_file", "status": "OK", "content": "y" * 3000} for _ in range(5)]
    source = "__global__ void kernel() {\n    expf(x);\n}\n"
    view = compact_public_context(old + recent, source)
    assert view.total_actions == 17
    assert view.compaction_applied
    assert len(view.summary) <= 2600
    assert "PROGRAM_FAILED" in view.summary
    assert "hunk context" in view.summary
    assert "deduplicated successful reads/searches: 9" in view.summary
    assert len(view.recent) == 5
    assert all(len(event["content"]) <= 920 for event in view.recent)
    assert "   2     expf(x);" in view.source
    assert len(view.source) <= 12_000


def test_large_source_is_bounded_but_read_file_can_recover_rest() -> None:
    source = "float x = 1.0f;\n" * 5_000
    excerpt = source_with_line_numbers(source)
    assert len(excerpt) <= 12_000
    assert "[Source excerpt truncated." in excerpt
    assert excerpt.startswith("   1 float x = 1.0f;")


@pytest.mark.parametrize("recent", [0, 13])
def test_compaction_invalid_windows_are_rejected(recent: int) -> None:
    with pytest.raises(ValueError):
        compact_public_context([], "", recent_count=recent)


def test_tool_policy_is_controller_owned_and_cannot_be_model_approved() -> None:
    policy = ToolPermissionPolicy(
        allow_gpu=False,
        allow_edit=True,
        approval_required=frozenset({"apply_patch", "replace_snippet"}),
    )
    assert policy.check("list_files") == "ALLOW"
    assert policy.check("functional_probe") == "DENY"
    assert policy.check("run_public_checks") == "DENY"
    assert policy.check("apply_patch") == "APPROVAL_REQUIRED"
    assert "apply_patch" not in policy.available_tools
    assert "run_public_checks" not in policy.available_tools
    assert policy.check("host_shell") == "DENY"
    approved = ToolPermissionPolicy(
        approval_required=frozenset({"replace_snippet"}),
        approved_tools=frozenset({"replace_snippet"}),
    )
    assert approved.check("replace_snippet") == "ALLOW"
    assert "host_shell" not in registered_tool_names()
    assert len({spec.name for spec in TOOL_REGISTRY}) == len(TOOL_REGISTRY)


def test_exact_source_replace_still_passes_existing_patchscope(tmp_path: Path) -> None:
    original = (REPO / "benchmarks/hardset_v2/h001/mutant/kernel.cu").read_bytes()
    original_path = tmp_path / "kernel.cu"
    original_path.write_bytes(original)
    manifest = {"kernel.cu": hashlib.sha256(original).hexdigest()}
    snapshot = SourceSnapshot(parent_run_id="a" * 32, root=tmp_path, hashes=manifest)

    first = exact_replace_diff(
        original,
        original,
        "denominator += expf(x);",
        "denominator += expf(x - maximum);",
    )
    first_patch = apply_generated_candidate(snapshot, first)
    first_source = materialize_candidate(snapshot, first_patch)["kernel.cu"]
    assert b"denominator += expf(x - maximum);" in first_source
    assert b"out[i] = expf(own) / denominator;" in first_source

    second = exact_replace_diff(
        original,
        first_source,
        "out[i] = expf(own) / denominator;",
        "out[i] = expf(own - maximum) / denominator;",
    )
    final_patch = apply_generated_candidate(snapshot, second)
    candidate = materialize_candidate(snapshot, final_patch)["kernel.cu"]
    clean = (REPO / "benchmarks/hardset_v2/h001/clean/kernel.cu").read_bytes()
    assert candidate == clean


def test_replace_snippet_only_accepts_kernel_scoped_exact_edits() -> None:
    decision = HarnessDecision.model_validate(
        {
            "action": {
                "tool": "replace_snippet",
                "old_text": "expf(x)",
                "new_text": "expf(x - maximum)",
            },
            "reason": "This expression requires max subtraction.",
        }
    )
    assert isinstance(decision.action, ReplaceSnippet)
    with pytest.raises(ValueError, match="exactly once"):
        exact_replace_diff(b"x();\nx();\n", b"x();\nx();\n", "x()", "y()")
    with pytest.raises(ValueError, match="no effect"):
        exact_replace_diff(b"x();\n", b"x();\n", "x()", "x()")
    with pytest.raises(ValueError, match="NUL"):
        exact_replace_diff(b"x();\n", b"x();\n", "x()", "y\x00()")


def test_session_journal_persists_hash_chain_and_recovery_state(tmp_path: Path) -> None:
    path = tmp_path / "session-events.jsonl"
    with SessionJournal(path) as journal:
        journal.append("session_started", {"case_id": "h001"})
        journal.append("model_started", {"turn": 1})
        assert (
            recovery_disposition(read_verified_journal(path))
            == "UNCERTAIN_EXTERNAL_EFFECT_RECONCILE_MANUALLY"
        )
        journal.append("model_selected", {"turn": 1, "tool": "read_file"})
        journal.append("tool_started", {"turn": 1})
        assert "UNCERTAIN" in recovery_disposition(read_verified_journal(path))
        journal.append("tool_completed", {"turn": 1, "tool": "read_file"})
        journal.append("session_finished", {"status": "INCONCLUSIVE"})
    records = read_verified_journal(path)
    assert len(records) == 6
    assert recovery_disposition(records) == "FINISHED"
    assert path.stat().st_mode & 0o077 == 0

    with pytest.raises(FileExistsError):
        SessionJournal(path)
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'"h001"', b'"h002"', 1))
    with pytest.raises(ValueError, match="chain mismatch"):
        read_verified_journal(path)


def test_incomplete_or_secret_bearing_journal_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    with SessionJournal(path) as journal:
        with pytest.raises(ValueError, match="private data"):
            journal.append("model_started", {"api_key": "forbidden"})
        journal.append("session_started", {"case_id": "h001"})
    with path.open("ab") as output:
        output.write(b'{"broken": true')
    with pytest.raises(ValueError, match="incomplete"):
        read_verified_journal(path)


def test_workspace_snapshot_exposes_bounded_source_and_controller_permissions(
    tmp_path: Path,
) -> None:
    from gpu_agent.harness.hardset_tools import HardsetWorkspaceTools

    history = [{"tool": "read_file", "status": "OK", "content": "source"} for _ in range(10)]
    history += [
        {
            "tool": "functional_probe",
            "status": "FAILED",
            "content": "size 33: PROGRAM_FAILED",
        }
    ]
    policy = ToolPermissionPolicy(allow_gpu=False, allow_edit=False)
    host = HardsetWorkspaceTools(
        REPO,
        tmp_path / "harness-no-gpu",
        "h001",
        permission_policy=policy,
    )
    snapshot = host.snapshot("Fix CUDA numerical instability", history, 6)
    assert snapshot.workspace_source.startswith("   1 ")
    assert "expf(" in snapshot.workspace_source
    assert "functional_probe" not in snapshot.allowed_tools
    assert "apply_patch" not in snapshot.allowed_tools
    assert "replace_snippet" not in snapshot.allowed_tools
    assert "read_file" in snapshot.allowed_tools
    assert snapshot.total_public_actions == 11
    assert len(snapshot.history) == 5
    assert "deduplicated successful reads" in snapshot.context_summary
    assert snapshot.remaining_steps == 6
    assert snapshot.remaining_public_checks == 3


def test_runtime_emits_pre_and_post_tool_events_in_order() -> None:
    from gpu_agent.harness.models import (
        FinishTask,
        HarnessDecision,
        SessionSnapshot,
        ToolObservation,
    )
    from gpu_agent.harness.runner import run_tool_agent

    class Host:
        def snapshot(self, goal: str, history: list[dict[str, str]], steps: int) -> SessionSnapshot:
            return SessionSnapshot(
                goal=goal,
                case_id="h001",
                source_hash="a" * 64,
                functional_requirement="Fix this numeric CUDA kernel",
                remaining_steps=steps,
                history=tuple(history),
                allowed_tools=("finish",),
            )

        def invoke(self, action: FinishTask) -> ToolObservation:
            return ToolObservation(tool=action.tool, status="OK", content="finished and checked")

    class Model:
        def choose_harness_v1(self, snapshot: SessionSnapshot) -> HarnessDecision:
            return HarnessDecision(
                action=FinishTask(summary="Public CUDA verification passed"),
                reason="The operator tool has the evidence",
            )

    timeline: list[dict[str, object]] = []
    result = run_tool_agent(Model(), Host(), goal="Fix CUDA task", on_event=timeline.append)
    assert result["reason"] == "PUBLIC_CANDIDATE_ACCEPTED"
    assert [event["stage"] for event in timeline] == [
        "model_started",
        "model",
        "tool_started",
        "tool",
    ]
    assert timeline[0]["turn"] == timeline[3]["turn"] == 1
    assert len(timeline[3]["observation_sha256"]) == 64
