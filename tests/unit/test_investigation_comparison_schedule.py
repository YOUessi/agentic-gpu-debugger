import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_agent.knowledge.models import KnowledgeVersionUnavailableError


def test_predeclared_comparison_is_balanced_and_repeatable():
    path = Path(__file__).resolve().parents[2] / "scripts/compare_investigation_modes.py"
    spec = importlib.util.spec_from_file_location("comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    units = module.schedule()
    assert units == module.schedule()
    assert len(units) == len(set(units)) == 8
    assert set(units) == {
        (case, mode)
        for case in ("case_0001", "case_0003", "case_0009", "case_0016")
        for mode in ("D", "E")
    }
    with pytest.raises(KnowledgeVersionUnavailableError):
        module.validate_knowledge(None, "a" * 64)
    queries = []

    class Index:
        def retrieve(self, query, version, k):
            queries.append(query)
            return SimpleNamespace(chunks=["compatible document"])

    module.validate_knowledge(Index(), "cuda=12.8.1;compute-sanitizer=2025.1.0.0")
    assert queries == ["memcheck", "initcheck", "racecheck", "synccheck"]
