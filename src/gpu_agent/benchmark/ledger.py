"""Atomic privacy-preserving uniqueness reservations shared by corpus stores."""

import fcntl
import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path

from gpu_agent.store import read_regular, reject_symlinks, sync_directory


class CorpusLedger:
    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        reject_symlinks(self.root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.key_path = self.root / "identity.key"
        init_path = self.root / ".init-lock"
        init_fd = os.open(init_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(init_fd, fcntl.LOCK_EX)
            try:
                fd = os.open(
                    self.key_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                )
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(os.urandom(32))
                    stream.flush()
                    os.fsync(stream.fileno())
                sync_directory(self.root)
        finally:
            os.close(init_fd)
        self.key = read_regular(self.key_path, 32)
        if len(self.key) != 32 or self.key_path.stat().st_mode & 0o077:
            raise ValueError("corpus ledger key is unavailable or has unsafe permissions")
        self.namespace_hash = hashlib.sha256(b"gpu-agent-corpus-ledger-v1\0" + self.key).hexdigest()

    def _identity(self, domain: bytes, value: bytes) -> str:
        return hmac.new(self.key, domain + b"\0" + value, hashlib.sha256).hexdigest()

    def reserve(
        self, case_identity: bytes, template_identity: bytes, source_pair: bytes
    ) -> tuple[str, str, str]:
        case_hash = self._identity(b"case", case_identity)
        template_hash = self._identity(b"template", template_identity)
        pair_hash = self._identity(b"source-pair", source_pair)
        lock_path = self.root / ".lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            state_path = self.root / "reservations.json"
            if state_path.exists():
                state = json.loads(read_regular(state_path, 16 * 1024 * 1024))
            else:
                state = {
                    "schema_version": 1,
                    "case_hashes": [],
                    "template_hashes": [],
                    "source_pair_hashes": [],
                }
            if (
                state.get("schema_version") != 1
                or not isinstance(state.get("case_hashes"), list)
                or not isinstance(state.get("template_hashes"), list)
                or not isinstance(state.get("source_pair_hashes"), list)
            ):
                raise ValueError("corpus ledger is malformed")
            if (
                case_hash in state["case_hashes"]
                or template_hash in state["template_hashes"]
                or pair_hash in state["source_pair_hashes"]
            ):
                raise ValueError("corpus identity or source pair is already reserved")
            state["case_hashes"].append(case_hash)
            state["template_hashes"].append(template_hash)
            state["source_pair_hashes"].append(pair_hash)
            raw = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
            temporary_fd, temporary = tempfile.mkstemp(prefix=".ledger-", dir=self.root)
            try:
                with os.fdopen(temporary_fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, state_path)
                sync_directory(self.root)
            finally:
                Path(temporary).unlink(missing_ok=True)
            return case_hash, template_hash, pair_hash
        finally:
            os.close(fd)
