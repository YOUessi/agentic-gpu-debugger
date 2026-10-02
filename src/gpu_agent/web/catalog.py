"""Read-only projection of RunStore data for the web console."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from gpu_agent.contracts import ArtifactRef, RunManifest
from gpu_agent.store import RunStore
from gpu_agent.web.models import (
    ArtifactSummary,
    CitationTarget,
    RunDetail,
    RunStats,
    RunSummary,
)

_RUN_ID = re.compile(r"^[a-f0-9]{32}$")
_ACTION = re.compile(r"^actions/(\d+)/step\.json$")
_DECISION = re.compile(r"^actions/(\d+)/decision\.json$")
_REPAIR_RESULT = re.compile(r"^repair/(\d+)/result\.json$")


class RunCatalog:
    def __init__(self, store: RunStore) -> None:
        if store.visibility != "public":
            raise ValueError("web catalog only exposes the public RunStore")
        self.store = store

    def _manifests(self) -> list[RunManifest]:
        result: list[RunManifest] = []
        for path in sorted(self.store.root.iterdir()):
            if path.is_dir() and _RUN_ID.fullmatch(path.name):
                result.append(self.store.load(path.name))
        return result

    def _inventory(self) -> tuple[list[RunManifest], dict[str, list[RunManifest]]]:
        manifests = self._manifests()
        children: dict[str, list[RunManifest]] = {}
        for run in manifests:
            if run.parent_run_id is not None:
                children.setdefault(run.parent_run_id, []).append(run)
        return manifests, children

    @staticmethod
    def _ref(run: RunManifest, name: str) -> ArtifactRef | None:
        refs = [ref for ref in run.artifact_refs if ref.name == name]
        return refs[-1] if refs else None

    def _json(self, run: RunManifest, name: str) -> dict[str, Any] | None:
        ref = self._ref(run, name)
        if ref is None:
            return None
        value = json.loads(self.store.read(ref))
        return value if isinstance(value, dict) else None

    def _verification_results(
        self,
        run_id: str,
        children: list[RunManifest] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for child in children if children is not None else self.store.children(run_id):
            if child.kind != "verification":
                continue
            ref = self._ref(child, "verification/result.json")
            if ref is None:
                continue
            value = json.loads(self.store.read(ref))
            if isinstance(value, dict):
                results.append({"run_id": child.id, **value})
        return results

    def _candidate(
        self,
        run_id: str,
        children: list[RunManifest] | None = None,
    ) -> dict[str, Any] | None:
        for child in children if children is not None else self.store.children(run_id):
            if child.kind != "candidate":
                continue
            ref = self._ref(child, "candidate.json")
            if ref is None:
                continue
            value = json.loads(self.store.read(ref))
            if isinstance(value, dict):
                return {"run_id": child.id, **value}
        return None

    def summary(
        self,
        run: RunManifest,
        children: list[RunManifest] | None = None,
    ) -> RunSummary:
        diagnosis = self._json(run, "diagnosis.json")
        repair = self._json(run, "repair/summary.json")
        verifications = (
            self._verification_results(run.id, children) if run.kind == "diagnosis" else []
        )
        verdict = verifications[-1]["verdict"] if verifications else None
        last_event = run.events[-1].at if run.events else None
        return RunSummary(
            id=run.id,
            kind=run.kind,
            parent_run_id=run.parent_run_id,
            status=run.status.value,
            phase=run.current_phase.value if run.current_phase else None,
            last_event_at=last_event,
            artifact_count=len(run.artifact_refs),
            diagnosis_outcome=diagnosis.get("diagnostic_outcome") if diagnosis else None,
            failure_family=diagnosis.get("failure_family") if diagnosis else None,
            confidence=diagnosis.get("confidence_label") if diagnosis else None,
            repair_stop_reason=repair.get("stop_reason") if repair else None,
            verification_verdict=verdict,
        )

    def list_runs(
        self,
        *,
        page: int,
        page_size: int,
        query: str | None = None,
        status: str | None = None,
        kind: str | None = None,
    ) -> tuple[list[RunSummary], int]:
        manifests, children = self._inventory()
        rows = [self.summary(run, children.get(run.id, [])) for run in manifests]
        if query:
            needle = query.lower()
            rows = [
                row
                for row in rows
                if needle in row.id.lower()
                or needle in row.kind.lower()
                or needle in (row.failure_family or "").lower()
            ]
        if status:
            rows = [row for row in rows if row.status == status]
        if kind:
            rows = [row for row in rows if row.kind == kind]
        rows.sort(
            key=lambda row: row.last_event_at.timestamp() if row.last_event_at else 0.0,
            reverse=True,
        )
        total = len(rows)
        start = (page - 1) * page_size
        return rows[start : start + page_size], total

    def stats(self) -> RunStats:
        manifests, children = self._inventory()
        diagnoses = [run for run in manifests if run.kind == "diagnosis"]
        rows = [self.summary(run, children.get(run.id, [])) for run in diagnoses]
        families = Counter(row.failure_family for row in rows if row.failure_family)
        fixed = sum(row.verification_verdict == "VERIFIED_FIXED" for row in rows)
        attention = sum(
            row.status == "FAILED"
            or row.repair_stop_reason not in {None, "PUBLIC_CHECKS_PASSED"}
            or row.verification_verdict in {"NOT_FIXED", "REGRESSION_DETECTED", "INCONCLUSIVE"}
            for row in rows
        )
        return RunStats(
            total_diagnoses=len(rows),
            active=sum(row.status == "RUNNING" for row in rows),
            diagnosed=sum(row.diagnosis_outcome == "DIAGNOSED" for row in rows),
            verified_fixed=fixed,
            needs_attention=attention,
            failure_families=dict(families),
        )

    @staticmethod
    def _citation_ids(diagnosis: dict[str, Any] | None) -> set[str]:
        if diagnosis is None:
            return set()
        ids: set[str] = set()
        for section in ("observed_facts", "tool_findings", "documentation_evidence"):
            claims = diagnosis.get(section, [])
            if not isinstance(claims, list):
                continue
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                citations = claim.get("citation_ids", [])
                if isinstance(citations, list):
                    ids.update(item for item in citations if isinstance(item, str))
        return ids

    def _citations(
        self,
        run: RunManifest,
        diagnosis: dict[str, Any] | None,
    ) -> dict[str, CitationTarget]:
        wanted = self._citation_ids(diagnosis)
        if not wanted:
            return {}
        resolved: dict[str, CitationTarget] = {}
        by_artifact_id = {ref.id: ref for ref in run.artifact_refs}
        for citation_id in wanted:
            ref = by_artifact_id.get(citation_id)
            if ref is not None:
                resolved[citation_id] = CitationTarget(
                    citation_id=citation_id,
                    artifact_id=ref.id,
                    artifact_name=ref.name,
                    kind="artifact",
                    label=ref.name,
                )
        for ref in run.artifact_refs:
            if not ref.name.startswith("docs/") or not ref.name.endswith(".json"):
                continue
            try:
                payload = json.loads(self.store.read(ref))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            chunk_id = payload.get("chunk_id")
            if not isinstance(chunk_id, str) or chunk_id not in wanted:
                continue
            title = str(payload.get("document_title", "NVIDIA documentation"))
            section = str(payload.get("section_title", ""))
            label = f"{title} · {section}" if section else title
            text = payload.get("text")
            source_url = payload.get("source_url")
            resolved[chunk_id] = CitationTarget(
                citation_id=chunk_id,
                artifact_id=ref.id,
                artifact_name=ref.name,
                kind="document",
                label=label,
                preview=text[:360] if isinstance(text, str) else None,
                source_url=source_url if isinstance(source_url, str) else None,
            )
        return resolved

    def detail(self, run_id: str) -> RunDetail:
        run = self.store.load(run_id)
        children = self.store.children(run_id)
        actions: list[dict[str, Any]] = []
        action_steps: set[int] = set()
        repair_rounds: list[dict[str, Any]] = []
        for ref in run.artifact_refs:
            match = _ACTION.fullmatch(ref.name)
            if match:
                step = json.loads(self.store.read(ref))
                index = int(match.group(1))
                action_steps.add(index)
                decision = self._json(run, f"actions/{match.group(1)}/decision.json")
                actions.append(
                    {
                        "step": index,
                        "proposal": step,
                        "decision": decision,
                    }
                )
            repair_match = _REPAIR_RESULT.fullmatch(ref.name)
            if repair_match:
                number = int(repair_match.group(1))
                result = json.loads(self.store.read(ref))
                candidate = self._json(run, f"repair/{number}/candidate.json")
                feedback = self._json(run, f"repair/{number}/feedback.json")
                repair_rounds.append(
                    {
                        "round": number,
                        "candidate": candidate,
                        "result": result,
                        "feedback": feedback,
                    }
                )
        for ref in run.artifact_refs:
            decision_match = _DECISION.fullmatch(ref.name)
            if decision_match is None:
                continue
            index = int(decision_match.group(1))
            if index in action_steps:
                continue
            decision = json.loads(self.store.read(ref))
            action_type = (
                decision.get("action_type", "unknown")
                if isinstance(decision, dict)
                else "unknown"
            )
            actions.append(
                {
                    "step": index,
                    "proposal": {"action": {"action_type": action_type}},
                    "decision": decision if isinstance(decision, dict) else None,
                }
            )
        actions.sort(key=lambda item: item["step"])
        repair_rounds.sort(key=lambda item: item["round"])
        diagnosis = self._json(run, "diagnosis.json")
        artifacts = [
            ArtifactSummary(
                id=ref.id,
                name=ref.name,
                byte_count=ref.byte_count,
                sha256=ref.sha256,
            )
            for ref in run.artifact_refs
        ]
        return RunDetail(
            summary=self.summary(run, children),
            events=[event.model_dump(mode="json") for event in run.events],
            diagnosis=diagnosis,
            repair_summary=self._json(run, "repair/summary.json"),
            repair_rounds=repair_rounds,
            candidate=self._candidate(run_id, children),
            verifications=self._verification_results(run_id, children),
            actions=actions,
            citations=self._citations(run, diagnosis),
            artifacts=artifacts,
        )

    def artifact_text(self, run_id: str, artifact_id: str, max_bytes: int) -> str:
        run = self.store.load(run_id)
        ref = next((item for item in run.artifact_refs if item.id == artifact_id), None)
        if ref is None:
            raise ValueError("artifact is not registered on this run")
        data = self.store.read(ref)
        if len(data) > max_bytes:
            data = data[:max_bytes] + b"\n\n[truncated by web console]\n"
        return data.decode("utf-8", errors="replace")
