"""Read the public repository action ledger without executing source or tools.

New registered AST sessions retain exact model-visible public observations.
Older sessions retain action/observation hashes; never invent missing feedback.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from gpu_agent.harness.models import HarnessDecision
from gpu_agent.store import RunStore

_TOOL_EVENT = re.compile(r"^repo-harness/v1/tool-([0-9]{3})\.json$")


def read_repository_tool_events(
    store: RunStore,
    run_id: str,
    *,
    expected_count: int,
    require_action: bool = False,
    require_observation: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Read verified public artifacts in order; fail closed on gaps or bad hashes."""
    run = store.load(run_id)
    if run.kind != "gpu_repository_agent":
        raise ValueError("not a repository tool ledger")
    records: dict[int, dict[str, Any]] = {}
    for ref in run.artifact_refs:
        match = _TOOL_EVENT.fullmatch(ref.name)
        if match is None:
            continue
        turn = int(match.group(1))
        if ref.visibility != "public" or turn in records:
            raise ValueError("duplicate or nonpublic repository tool event")
        body = json.loads(store.read(ref))
        if not isinstance(body, dict):
            raise ValueError("invalid repository tool event")
        tool = body.get("tool")
        status = body.get("status")
        digest = body.get("observation_sha256")
        if (
            not isinstance(tool, str)
            or status not in ("OK", "FAILED", "REJECTED")
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("invalid repository tool event identity")
        action = body.get("action")
        if action is not None:
            parsed = HarnessDecision.model_validate(
                {"action": action, "reason": "Replay recorded public action"}
            )
            if parsed.action.tool != tool:
                raise ValueError("repository tool action does not match recorded tool")
        elif require_action:
            raise ValueError("repository tool action missing")
        observation = body.get("observation")
        if observation is not None:
            if (
                not isinstance(observation, str)
                or len(observation) > 4000
                or hashlib.sha256(observation.encode()).hexdigest() != digest
            ):
                raise ValueError("public observation does not match recorded hash")
        elif require_observation:
            raise ValueError("full public observation unavailable for this legacy event")
        records[turn] = body
    if sorted(records) != list(range(1, expected_count + 1)):
        raise ValueError("repository tool ledger is not contiguous")
    return tuple(records[turn] for turn in sorted(records))
