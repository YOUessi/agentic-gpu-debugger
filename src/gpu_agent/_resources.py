"""Resolve immutable project data from a checkout or an installed wheel."""

from pathlib import Path, PurePosixPath
from sysconfig import get_path


def runtime_resource(relative: str) -> Path:
    """Return a bounded project resource path without accepting path traversal."""
    name = PurePosixPath(relative)
    if name.is_absolute() or not name.parts or any(part in {"", ".", ".."} for part in name.parts):
        raise ValueError("invalid runtime resource path")
    bounded = Path(*name.parts)
    checkout = Path(__file__).resolve().parents[2] / bounded
    if checkout.is_file() or checkout.is_dir():
        return checkout
    return Path(get_path("data")) / "share" / "agentic-gpu-debugger" / bounded
