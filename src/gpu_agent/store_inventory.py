"""Incremental Linux directory inventory; overflow always forces a fresh scan.

Only names are cached here. Callers still pin directories and validate manifests.
No threads, polling sleeps, persistent indexes or new authority files are needed.
"""

import ctypes
import os
import re
import struct
import weakref
from collections.abc import Callable
from threading import RLock

_ADD = 0x00000100 | 0x00000080  # IN_CREATE | IN_MOVED_TO
_REMOVE = 0x00000200 | 0x00000040  # IN_DELETE | IN_MOVED_FROM
_RESET = 0x00004000 | 0x00008000 | 0x00000400 | 0x00000800
_HEADER = struct.Struct("iIII")


class DirectoryInventory:
    def __init__(self) -> None:
        self._fd = -1
        self._identity: tuple[int, int] | None = None
        self._names: set[str] = set()
        self._close: Callable[[], object] | None = None
        self._lock = RLock()
        self._pid = os.getpid()

    def _start(self, root_fd: int) -> None:
        if self._close is not None:
            self._close()
        self._fd = -1
        libc = ctypes.CDLL(None, use_errno=True)
        initialize = libc.inotify_init1
        initialize.argtypes = [ctypes.c_int]
        initialize.restype = ctypes.c_int
        watch = libc.inotify_add_watch
        watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        watch.restype = ctypes.c_int
        fd = initialize(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "directory watcher unavailable")
        if watch(fd, os.fsencode(f"/proc/self/fd/{root_fd}"), _ADD | _REMOVE | _RESET) < 0:
            error = ctypes.get_errno()
            os.close(fd)
            raise OSError(error, "directory watch unavailable")
        self._fd = fd
        self._close = weakref.finalize(self, os.close, fd)
        info = os.fstat(root_fd)
        self._identity = (info.st_dev, info.st_ino)
        # Subscribe before scanning: concurrent creates/deletes are then replayed.
        self._names = {name for name in os.listdir(root_fd) if re.fullmatch(r"[a-f0-9]{32}", name)}

    def names(self, root_fd: int) -> list[str]:
        if self._pid != os.getpid():
            # fork inherits both the lock state and the same kernel event queue.
            # Never acquire an inherited lock or drain another process's events.
            self._lock = RLock()
            self._identity = None
            self._pid = os.getpid()
        with self._lock:
            return self._names_locked(root_fd)

    def _names_locked(self, root_fd: int) -> list[str]:
        info = os.fstat(root_fd)
        try:
            if self._fd < 0 or self._identity != (info.st_dev, info.st_ino):
                self._start(root_fd)
            while True:
                try:
                    events = os.read(self._fd, 65536)
                except BlockingIOError:
                    return sorted(self._names)
                offset = 0
                while offset < len(events):
                    _, mask, _, length = _HEADER.unpack_from(events, offset)
                    offset += _HEADER.size
                    name = os.fsdecode(events[offset : offset + length].split(b"\0", 1)[0])
                    offset += length
                    if mask & _RESET:
                        self._start(root_fd)
                        break
                    if not re.fullmatch(r"[a-f0-9]{32}", name):
                        continue
                    if mask & _REMOVE:
                        self._names.discard(name)
                    if mask & _ADD:
                        self._names.add(name)
        except (AttributeError, OSError):
            # Other platforms / resource exhaustion retain the original checked scan.
            self._identity = None
            return sorted(
                name for name in os.listdir(root_fd) if re.fullmatch(r"[a-f0-9]{32}", name)
            )
