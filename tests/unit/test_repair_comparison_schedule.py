import importlib.util
from collections import Counter
from pathlib import Path


def test_predeclared_48_units_and_registered_inputs():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "compare_repair", root / "scripts/compare_repair_modes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    units = module.schedule()
    assert len(units) == len(set(units)) == 48
    assert units == module.schedule()
    assert Counter((c, m) for c, m, _ in units) == {(c, m): 3 for c in module.CASES for m in "DE"}

    class Index:
        def retrieve(self, *args):
            return type("Results", (), {"chunks": [1]})()

    hashes = module.preflight(root, Index(), "cuda=12.8.1;compute-sanitizer=2025.1.0.0")
    assert set(hashes) == set(module.CASES)
    new = module.preflight(
        root, Index(), "cuda=12.8.1;compute-sanitizer=2025.1.0.0", ("case_0021", "case_0022")
    )
    assert set(new) == {"case_0021", "case_0022"}
    assert all("task.json" in entry for entry in new.values())
