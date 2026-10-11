"""Protect default pytest imports from cross-directory module collisions."""

from collections import defaultdict
from pathlib import Path


def test_test_module_basenames_are_unique():
    root = Path(__file__).resolve().parents[1]
    modules = defaultdict(list)
    for path in root.rglob("test_*.py"):
        modules[path.stem].append(str(path.relative_to(root)))
    duplicates = {name: paths for name, paths in modules.items() if len(paths) > 1}
    assert not duplicates, f"pytest default import mode has ambiguous modules: {duplicates}"
