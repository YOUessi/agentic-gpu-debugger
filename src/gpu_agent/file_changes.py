"""In-process Linux write generations; failure disables caching, never validation."""

import ctypes
import os
import struct
import weakref
from collections.abc import Callable
from threading import RLock

_EVENT = struct.Struct("iIII")
_MASK = 0x2 | 0x4 | 0x8 | 0x400 | 0x800  # modify, attrib, close-write, delete/move self


class FileChanges:
    def __init__(self) -> None:
        self._fd = -1
        self._pid = os.getpid()
        self._epoch = 0
        self._versions: dict[int, int] = {}
        self._close: Callable[[], object] | None = None
        self._lock = RLock()

    def _start(self) -> None:
        if self._close:
            self._close()
        self._fd = -1
        self._versions.clear()
        self._epoch += 1
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            raise OSError("file watcher unavailable")
        self._fd = fd
        self._close = weakref.finalize(self, os.close, fd)

    def version(self, directory_fd: int) -> tuple[int, int, int] | None:
        forked = self._pid != os.getpid()
        if forked:
            self._lock = RLock()
            self._pid = os.getpid()
        with self._lock:
            try:
                if forked or self._fd < 0 or len(self._versions) > 4096:
                    self._start()
                libc = ctypes.CDLL(None, use_errno=True)
                watch = libc.inotify_add_watch
                watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
                watch.restype = ctypes.c_int
                wd = watch(
                    self._fd, os.fsencode(f"/proc/self/fd/{directory_fd}/manifest.json"), _MASK
                )
                if wd < 0:
                    raise OSError("manifest watcher unavailable")
                self._versions.setdefault(wd, 0)
                while True:
                    try:
                        data = os.read(self._fd, 65536)
                    except BlockingIOError:
                        return self._epoch, wd, self._versions[wd]
                    offset = 0
                    while offset < len(data):
                        changed, mask, _, length = _EVENT.unpack_from(data, offset)
                        offset += _EVENT.size + length
                        if mask & 0x4000:  # queue overflow invalidates every cached generation
                            self._epoch += 1
                        else:
                            self._versions[changed] = self._versions.get(changed, 0) + 1
            except (AttributeError, OSError):
                self._epoch += 1
                return None
