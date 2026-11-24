"""Attested public-check lifecycle, shared by complete and failed trajectories.

'Executed' means the numeric workload was dispatched to the backend, not that a
GPU kernel necessarily completed. Admission, dispatch, terminal result and pass
are different events. Old ledgers without admission/dispatch markers are never
silently upgraded to proof of execution.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from gpu_agent.benchmark.hardset_v2_agent_models import HardsetPublicCheck
from gpu_agent.benchmark.public_compile import PublicCompileResult
from gpu_agent.store import RunStore

_ADMITTED = re.compile(r"^repo-harness/v1/check-([0-9]{3})-admitted\.json$")


@dataclass(frozen=True)
class PublicCheckLifecycle:
    requested: tuple[int, ...]
    admitted: tuple[int, ...]
    executed: tuple[int, ...]
    completed: tuple[int, ...]
    passed: tuple[int, ...]
    build_passed_candidates: tuple[str, ...]


def public_check_lifecycle(
    store: RunStore,
    run_id: str,
    journal: list[dict[str, Any]],
    tool_events: tuple[dict[str, Any], ...],
) -> PublicCheckLifecycle:
    requests = {
        record["payload"]["turn"]
        for record in journal
        if record["kind"] == "tool_started"
        and record["payload"].get("tool") == "repo_run_public_checks"
    }
    # Older finished journals still have tool requests; tool artifacts are
    # an additional durable source if a process failed before journaling.
    requests.update(
        turn
        for turn, event in enumerate(tool_events, 1)
        if event["tool"] == "repo_run_public_checks"
    )
    modern = journal[0]["payload"].get("public_check_audit_version") == 1
    parent = store.load(run_id)
    admitted: dict[str, int] = {}
    for ref in parent.artifact_refs:
        match = _ADMITTED.fullmatch(ref.name)
        if match is None:
            continue
        turn = int(match.group(1))
        item = json.loads(store.read(ref))
        candidate = item.get("candidate_sha256")
        if (
            turn not in requests
            or item.get("turn") != turn
            or not isinstance(candidate, str)
            or re.fullmatch(r"[0-9a-f]{64}", candidate) is None
            or candidate in admitted
            or turn in admitted.values()
            or item.get("case_id") != journal[0]["payload"].get("public_case_id")
        ):
            raise ValueError("invalid public checker admission evidence")
        admitted[candidate] = turn
    completed: set[int] = set()
    passed: set[int] = set()
    executed: set[int] = set()
    seen_children: set[str] = set()
    built: set[str] = set()
    compile_events = {
        json.loads(event["observation"])["run_id"]: event
        for event in tool_events
        if event["tool"] == "repo_compile_candidate" and event["status"] in ("OK", "FAILED")
    }
    for child in store.children(run_id):
        if child.kind == "hardset_v2_public_compile":
            event = compile_events.get(child.id)
            if event is None:
                # Incomplete compile-only run is not evidence of a successful build.
                continue
            refs = [ref for ref in child.artifact_refs if ref.name == "compile/public-summary.json"]
            if len(refs) != 1:
                raise ValueError("compile summary missing")
            compile_result = PublicCompileResult.model_validate_json(store.read(refs[0]))
            if (
                compile_result != PublicCompileResult.model_validate_json(event["observation"])
                or compile_result.run_id != child.id
                or compile_result.candidate_sha256 != event["action"]["expected_source_sha256"]
                or child.status.value != (
                    "COMPLETED" if compile_result.status == "BUILD_PASSED" else "FAILED"
                )
            ):
                raise ValueError("compile candidate evidence mismatch")
            if compile_result.status == "BUILD_PASSED":
                built.add(compile_result.candidate_sha256)
            continue
        if child.kind != "hardset_v2_public_candidate":
            continue
        refs = {ref.name: ref for ref in child.artifact_refs}
        summary_ref = refs.get("hardset/public-selfcheck.json")
        candidate_ref = refs.get("hardset/public-check-candidate.json")
        dispatch_ref = refs.get("hardset/public-numeric-dispatch.json")
        build_ref = refs.get("hardset/public-build-passed.json")
        summary = (
            HardsetPublicCheck.model_validate_json(store.read(summary_ref))
            if summary_ref is not None
            else None
        )
        candidate = (
            json.loads(store.read(candidate_ref)).get("candidate_sha256")
            if candidate_ref is not None
            else summary.candidate_sha256
            if summary
            else None
        )
        if candidate is None:
            # Unidentified child: cannot attribute it to a tool request.
            continue
        if candidate not in admitted:
            if modern:
                raise ValueError("public checker child lacks matching admission")
            continue
        if candidate in seen_children:
            raise ValueError("duplicate public checker candidate child")
        seen_children.add(candidate)
        turn = admitted[candidate]
        if build_ref is not None:
            if json.loads(store.read(build_ref)).get("candidate_sha256") != candidate:
                raise ValueError("public build provenance mismatch")
            built.add(candidate)
        if dispatch_ref is not None:
            dispatch = json.loads(store.read(dispatch_ref))
            if dispatch.get("candidate_sha256") != candidate or dispatch.get("input_size") != 32:
                raise ValueError("public numeric dispatch provenance mismatch")
            executed.add(turn)
            # Numeric dispatch only occurs after successful NVCC build.
            built.add(candidate)
        if summary is not None:
            if (
                summary.candidate_sha256 != candidate
                or summary.public_run_id != child.id
                or child.status.value != ("COMPLETED" if summary.passed else "FAILED")
            ):
                raise ValueError("public checker completion provenance mismatch")
            completed.add(turn)
            if summary.passed:
                passed.add(turn)
    if not modern:
        # Preserve historical admission/completion estimates for legacy ledgers.
        # Do not invent numeric dispatch markers for them.
        legacy = {
            turn
            for turn, event in enumerate(tool_events, 1)
            if event["tool"] == "repo_run_public_checks" and event["status"] in ("OK", "FAILED")
        }
        completed |= legacy
        passed |= {
            turn
            for turn, event in enumerate(tool_events, 1)
            if event["tool"] == "repo_run_public_checks" and event["status"] == "OK"
        }
    else:
        legacy = set()
        for turn, event in enumerate(tool_events, 1):
            if event["tool"] == "repo_run_public_checks" and event["status"] in ("OK", "FAILED"):
                if turn not in completed or (event["status"] == "OK") != (turn in passed):
                    raise ValueError("completed checker tool disagrees with child evidence")
    return PublicCheckLifecycle(
        requested=tuple(sorted(requests)),
        admitted=tuple(sorted(set(admitted.values()) | legacy)),
        executed=tuple(sorted(executed)),
        completed=tuple(sorted(completed)),
        passed=tuple(sorted(passed)),
        build_passed_candidates=tuple(sorted(built)),
    )
