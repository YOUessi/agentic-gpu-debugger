"""Exercise the image helper's file boundary without executing candidate code."""

import os
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("kind", ["symlink", "fifo", "oversized", "bounded"])
def test_runner_export_is_a_bounded_regular_file(tmp_path, kind):
    helper = runpy.run_path(str(Path(__file__).resolve().parents[2] / "containers/runner.py"))
    target = tmp_path / "export"
    if kind == "symlink":
        target.symlink_to(tmp_path / "secret")
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        target.write_bytes(b"abcde" if kind == "oversized" else b"abc")
    if kind in {"symlink", "fifo"}:
        with pytest.raises((OSError, ValueError)):
            helper["bounded_file"](target, 4)
    else:
        data, truncated = helper["bounded_file"](target, 4)
        assert data == (b"abcd" if kind == "oversized" else b"abc")
        assert truncated is (kind == "oversized")
