from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize("mode", ["runtime", "marker", "collection"])
def test_required_live_skip_fails_instead_of_passing(pytester, monkeypatch, mode):
    # pytester's child interpreter does not inherit the parent's -I flag.
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeconftest(Path(__file__).parents[1].joinpath("conftest.py").read_text())
    pytester.makeini("[pytest]\nmarkers = gpu: real hardware required\n")
    source = {
        "runtime": "import pytest\n@pytest.mark.gpu\ndef test_gpu(): pytest.skip('no GPU')\n",
        "marker": (
            "import pytest\n@pytest.mark.gpu\n@pytest.mark.skip(reason='no GPU')\n"
            "def test_gpu(): pass\n"
        ),
        "collection": "import pytest\npytest.skip('no GPU', allow_module_level=True)\n",
    }
    pytester.makepyfile(source[mode])
    ordinary = pytester.runpytest_subprocess("-q")
    assert ordinary.ret in {0, 5}
    required = pytester.runpytest_subprocess("-q", "--require-live")
    assert required.ret in {1, 2}
    assert "Required live" in required.stdout.str()
