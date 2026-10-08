"""Freeze a small diverse public V2/V3 exploratory comparison."""

import importlib.util
from pathlib import Path


def test_comparison_schedule_is_predeclared_and_unique():
    module_path = Path(__file__).resolve().parents[2] / "scripts/compare_repair_versions.py"
    import sys

    scripts = str(module_path.parent)
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("repair_compare", module_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(scripts)
    scheduled = mod.schedule()
    assert scheduled == mod.schedule()
    assert set(scheduled) == {
        ("case_0021", "V2"),
        ("case_0021", "V3"),
        ("case_0022", "V2"),
        ("case_0022", "V3"),
    }
    assert len(scheduled) == 4
