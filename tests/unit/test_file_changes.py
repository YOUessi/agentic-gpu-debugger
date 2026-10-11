import os
import struct

from gpu_agent.file_changes import FileChanges


def test_overflow_invalidates_generation(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{}")
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    watcher = FileChanges()
    try:
        before = watcher.version(fd)
        assert before is not None
        read = os.read
        pending = [struct.pack("iIII", -1, 0x4000, 0, 0)]

        def overflow_once(descriptor, size):
            if descriptor == watcher._fd and pending:
                return pending.pop()
            return read(descriptor, size)

        monkeypatch.setattr(os, "read", overflow_once)
        after = watcher.version(fd)
        assert after is not None and after != before
    finally:
        os.close(fd)


def test_unavailable_watcher_returns_no_generation(tmp_path, monkeypatch):
    watcher = FileChanges()

    def unavailable():
        raise OSError("unavailable")

    monkeypatch.setattr(watcher, "_start", unavailable)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert watcher.version(fd) is None
    finally:
        os.close(fd)


def test_child_does_not_drain_parent_write_events(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("first")
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    watcher = FileChanges()
    before = watcher.version(fd)
    assert before is not None
    path.write_text("other")
    child = os.fork()
    if child == 0:
        value = watcher.version(fd)
        os._exit(0 if value is not None else 1)
    try:
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert watcher.version(fd) != before
    finally:
        os.close(fd)
