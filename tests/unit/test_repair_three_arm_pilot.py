"""Three-arm public GPU pilot requires frozen disjoint case/arm assignments."""

import importlib.util
from pathlib import Path


def test_three_arm_schedule_is_deterministic_and_covers_all_units():
    path = Path(__file__).resolve().parents[2] / "scripts/compare_repair_three_arms.py"
    spec = importlib.util.spec_from_file_location("three_arm_public_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        (case, arm) for case in ("case_0019", "case_0020")
        for arm in ("V2", "V3_OLD", "V3_NEW")
    }
    assert module.schedule() == module.schedule()
    assert len(module.schedule()) == len(expected) == 6
    assert set(module.schedule()) == expected
    assert "case_0022" not in str(module.schedule())
    assert module.LEGACY_SHA != module.CURRENT_SHA


def test_pilot_sha256_is_stable():
    path = Path(__file__).resolve().parents[2] / "scripts/compare_repair_three_arms.py"
    spec = importlib.util.spec_from_file_location("three_arm_public_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.digest(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
