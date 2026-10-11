"""Watcher lifecycle changes must not masquerade as manifest modifications."""

import os

import pytest

import gpu_agent.store as store_module
from gpu_agent.file_changes import FileChanges
from gpu_agent.store import RunStore


def test_load_survives_real_watcher_rotation(tmp_path):
    store = RunStore(tmp_path / "runs")
    run = store.create_run("diagnosis")
    store._manifest_cache.clear()
    watcher = store._file_changes
    watcher._start()
    # The next real watch crosses the cache's 4096-entry lifecycle boundary.
    watcher._versions.update({n: 0 for n in range(10000, 14096)})
    assert store.load(run.id) == run


def test_unstable_watcher_is_bounded_and_rejected(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs")
    run = store.create_run("diagnosis")
    store._manifest_cache.clear()
    calls = []

    def changing(_self, _fd):
        calls.append(1)
        return len(calls), 1, 0

    monkeypatch.setattr(FileChanges, "version", changing)
    with pytest.raises(ValueError, match="manifest changed while reading"):
        store.load(run.id)
    assert len(calls) <= 6


def test_actual_manifest_change_during_read_still_rejected(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs")
    run = store.create_run("diagnosis")
    store._manifest_cache.clear()
    original = store_module._read_regular_at
    path = store.root / run.id / "manifest.json"

    def modified(fd, name, limit):
        result = original(fd, name, limit)
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(store_module, "_read_regular_at", modified)
    with pytest.raises(ValueError, match="manifest changed while reading"):
        store.load(run.id)
