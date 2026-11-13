"""Repository-backed public case catalog for the operator console."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from gpu_agent.public_task import DESCRIPTIONS, PublicTask
from gpu_agent.store import read_regular
from gpu_agent.web.models import CaseSummary

_CASE_ID = re.compile(r"^case_\d{4}$")


class PublicCaseCatalog:
    def __init__(self, repository: Path) -> None:
        self.repository = repository.absolute()
        self.public_root = self.repository / "benchmarks" / "public"

    def resolve_case(self, case_id: str) -> Path:
        if not _CASE_ID.fullmatch(case_id):
            raise ValueError("invalid public case ID")
        selected = (self.public_root / case_id / "public_input").absolute()
        if selected.parent.parent != self.public_root.absolute() or not selected.is_dir():
            raise ValueError("public case is unavailable")
        return selected

    @staticmethod
    def _registry_cases(path: Path) -> dict[str, dict[str, Any]]:
        if not path.is_file():
            return {}
        payload = json.loads(read_regular(path, 4 * 1024 * 1024))
        items = payload.get("cases", []) if isinstance(payload, dict) else []
        result: dict[str, dict[str, Any]] = {}
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("case_id"), str):
                result[item["case_id"]] = item
        return result

    def list_cases(self) -> list[CaseSummary]:
        registries: dict[str, dict[str, Any]] = {}
        for name in ("corpus-registry.json", "diverse-registry.json"):
            registries.update(self._registry_cases(self.repository / "benchmarks" / name))

        cases: list[CaseSummary] = []
        if not self.public_root.is_dir():
            return cases
        for directory in sorted(self.public_root.iterdir()):
            if not directory.is_dir() or not _CASE_ID.fullmatch(directory.name):
                continue
            input_root = directory / "public_input"
            kernel = input_root / "kernel.cu"
            task_path = input_root / "task.json"
            if not kernel.is_file() or not task_path.is_file():
                continue
            try:
                source = read_regular(kernel, 4 * 1024 * 1024)
                task = PublicTask.model_validate_json(read_regular(task_path, 64 * 1024))
            except (OSError, ValueError):
                continue
            if task.source_sha256 != hashlib.sha256(source).hexdigest():
                continue
            metadata = registries.get(directory.name, {})
            cases.append(
                CaseSummary(
                    case_id=directory.name,
                    algorithm=task.algorithm,
                    requirement=DESCRIPTIONS[task.algorithm],
                    template_id=str(metadata.get("template_id", "")) or None,
                    mutation_id=str(metadata.get("mutation_id", "")) or None,
                    target_tool=str(metadata.get("target_tool", "")) or None,
                    expected_finding=str(metadata.get("expected_finding", "")) or None,
                    repair_ready=(input_root / "input.json").is_file(),
                )
            )
        return cases
