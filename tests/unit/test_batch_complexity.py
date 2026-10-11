"""An evaluation batch performs O(n) native record validations, not O(n^2)."""

import os
import struct

import pytest
from schedule_authority_support import schedule_client_for_test


def test_directory_inventory_fork_keeps_parent_events(tmp_path):
    from gpu_agent.store_inventory import DirectoryInventory

    inventory = DirectoryInventory()
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    assert inventory.names(descriptor) == []
    ready_read, ready_write = os.pipe()
    result_read, result_write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(ready_write)
        os.close(result_read)
        os.read(ready_read, 1)
        observed = inventory.names(descriptor)
        os.write(result_write, b"y" if observed == ["a" * 32] else b"n")
        os._exit(0)
    os.close(ready_read)
    os.close(result_write)
    try:
        (tmp_path / ("a" * 32)).mkdir()
        os.write(ready_write, b"y")
        assert os.read(result_read, 1) == b"y"
        # Child reads must not consume the parent's inotify event queue.
        assert inventory.names(descriptor) == ["a" * 32]
    finally:
        os.close(ready_write)
        os.close(result_read)
        os.close(descriptor)
        os.waitpid(child, 0)


@pytest.mark.parametrize("watch_failure", ["overflow", "unavailable"])
def test_directory_inventory_rescans_after_watcher_failure(tmp_path, monkeypatch, watch_failure):
    from gpu_agent.store_inventory import DirectoryInventory

    first, second = "a" * 32, "b" * 32
    (tmp_path / first).mkdir()
    inventory = DirectoryInventory()
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert inventory.names(descriptor) == [first]
        (tmp_path / second).mkdir()
        if watch_failure == "overflow":
            read = os.read
            pending = [struct.pack("iIII", -1, 0x00004000, 0, 0)]

            def overflow_once(fd, size):
                if fd == inventory._fd and pending:
                    return pending.pop()
                return read(fd, size)

            monkeypatch.setattr(os, "read", overflow_once)
        else:
            inventory._identity = None

            def unavailable(fd):
                raise OSError("watch resources exhausted")

            monkeypatch.setattr(inventory, "_start", unavailable)
        assert inventory.names(descriptor) == [first, second]
    finally:
        os.close(descriptor)


def test_runner_validates_each_record_a_bounded_number_of_times(
    native_evaluation_executor, monkeypatch
):
    from gpu_agent.benchmark.evaluation import EvaluationRunner
    from gpu_agent.benchmark.executor import EvaluationExecutor

    executor = native_evaluation_executor
    native = EvaluationExecutor.validate_scheduled_record
    calls = []

    def counted(self, record, item, attempt):
        calls.append(item.ordinal)
        return native(self, record, item, attempt)

    monkeypatch.setattr(EvaluationExecutor, "validate_scheduled_record", counted)
    binding = executor.service.binding
    result = EvaluationRunner(
        executor.service.store,
        executor,
        schedule_client=schedule_client_for_test(executor),
        commit=binding.repository.commit,
        prompt_version=binding.prompt_version,
        toolchain_hash=binding.toolchain_lock_hash,
        model_config_hash=binding.model_config_hash,
        binding=binding,
        max_cost_usd=1000,
        max_unit_cost_usd=1,
        random_seed=7,
    ).run("D", "development", 3)
    assert result.executed_units == 3 and result.stopped_reason is None
    # Per unit: the returned record and its persisted projection, then one terminal pass.
    # Previously every unit re-derived all earlier records (1 + 2 + ... + n).
    assert sorted(calls) == sorted([0, 1, 2] * 3)


def test_store_reads_parse_each_manifest_version_once(store):
    from gpu_agent.store import RunStore

    run = store.create_run("inventory")
    refs = [store.put(run.id, f"items/{index}.json", b"{}", "public") for index in range(50)]
    parses = []
    native_load = RunStore.load

    def counted_load(self, run_id):
        parses.append(run_id)
        return native_load(self, run_id)

    RunStore.load = counted_load
    try:
        for ref in refs:
            store.read(ref)
    finally:
        RunStore.load = native_load
    assert len(parses) <= 1


def test_child_inventory_scan_and_manifest_bytes_grow_linearly(tmp_path, monkeypatch):
    import os

    import gpu_agent.store as module
    from gpu_agent.store import RunStore

    read = module._read_regular_at
    listdir = os.listdir
    measures = []
    for count in (16, 32):
        store = RunStore(tmp_path / str(count))
        parent = store.create_run("inventory")
        totals = {"reads": 0, "bytes": 0, "scans": 0}

        def counted_read(fd, name, limit, totals=totals):
            value = read(fd, name, limit)
            if name == "manifest.json":
                totals["reads"] += 1
                totals["bytes"] += len(value)
            return value

        def counted_listdir(path, totals=totals):
            totals["scans"] += 1
            return listdir(path)

        with monkeypatch.context() as patch:
            patch.setattr(module, "_read_regular_at", counted_read)
            patch.setattr(os, "listdir", counted_listdir)
            for index in range(count):
                store.create_run("child", parent.id)
                with store.evaluation_run_lease(parent.id) as lease:
                    assert len(lease.children()) == index + 1
            # A second store/process writes a new child: the first must see the event.
            external = RunStore(store.root)
            extra = external.create_run("external", parent.id)
            with store.evaluation_run_lease(parent.id) as lease:
                assert extra.id in {child.id for child in lease.children()}
        measures.append(totals)
        assert totals["scans"] == 1
        assert totals["reads"] <= count + 3
    assert measures[1]["bytes"] <= 2.2 * measures[0]["bytes"]
    # This bounds disk reads/JSON parsing, not all metadata stats or parent writes.


def test_cached_manifest_detects_replacement_and_same_size_edit(store):
    import os

    run = store.create_run("first")
    first = store.load(run.id)
    first.kind = "caller-edited"
    assert store.load(run.id).kind == "first"
    path = store.root / run.id / "manifest.json"
    before = path.stat()
    path.write_bytes(path.read_bytes().replace(b'"first"', b'"other"'))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert store.load(run.id).kind == "other"
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(path.read_bytes().replace(b'"other"', b'"third"'))
    os.replace(replacement, path)
    assert store.load(run.id).kind == "third"


def test_directory_inventory_detects_removal_and_move(store):
    import os

    parent = store.create_run("parent")
    child = store.create_run("child", parent.id)
    with store.evaluation_run_lease(parent.id) as lease:
        assert [item.id for item in lease.children()] == [child.id]
        original = store.root / child.id
        moved = store.root / "temporarily-moved"
        os.rename(original, moved)
        assert lease.children() == []
        os.rename(moved, original)
        assert [item.id for item in lease.children()] == [child.id]


def test_manifest_cache_observes_writes_with_identical_stat_timestamps(store, monkeypatch):
    from types import SimpleNamespace

    run = store.create_run("first")
    path = store.root / run.id / "manifest.json"
    fixed = path.stat()
    real_stat = os.stat

    def frozen_times(name, *args, **kwargs):
        info = real_stat(name, *args, **kwargs)
        if name == "manifest.json":
            return SimpleNamespace(
                st_dev=info.st_dev,
                st_ino=info.st_ino,
                st_mode=info.st_mode,
                st_size=info.st_size,
                st_mtime_ns=fixed.st_mtime_ns,
                st_ctime_ns=fixed.st_ctime_ns,
            )
        return info

    monkeypatch.setattr(os, "stat", frozen_times)
    assert store.load(run.id).kind == "first"
    path.write_bytes(path.read_bytes().replace(b'"first"', b'"other"'))
    assert store.load(run.id).kind == "other"


def test_manifest_cache_unavailable_watcher_reads_fresh(store, monkeypatch):
    run = store.create_run("first")
    assert store.load(run.id).kind == "first"
    monkeypatch.setattr(store._file_changes, "version", lambda fd: None)
    path = store.root / run.id / "manifest.json"
    path.write_bytes(path.read_bytes().replace(b'"first"', b'"other"'))
    assert store.load(run.id).kind == "other"
