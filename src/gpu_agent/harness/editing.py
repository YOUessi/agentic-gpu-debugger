"""Exact-source editing helpers: the model cannot bypass PatchScope by editing bytes.

Even snippet edits compile into a full unified diff from the frozen original.
The existing source-bound candidate validator remains the only write authority.
"""

from __future__ import annotations

import difflib


def exact_replace_diff(
    original: bytes,
    current: bytes,
    old: str,
    new: str,
    *,
    max_source_bytes: int = 4 * 1024 * 1024,
) -> str:
    if not old:
        raise ValueError("replacement target is empty")
    if "\x00" in old or "\x00" in new:
        raise ValueError("NUL is not allowed in CUDA source edits")
    original_text = original.decode("utf-8")
    current_text = current.decode("utf-8")
    if current_text.count(old) != 1:
        raise ValueError("exact edit target must occur exactly once in current kernel")
    updated = current_text.replace(old, new, 1)
    if updated == current_text:
        raise ValueError("candidate edit has no effect")
    if len(updated.encode("utf-8")) > max_source_bytes:
        raise ValueError("edited source exceeds bounded workspace size")
    if not original_text.endswith("\n") or not updated.endswith("\n"):
        raise ValueError("CUDA patch source must end with a newline")
    diff = "".join(
        difflib.unified_diff(
            original_text.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile="a/kernel.cu",
            tofile="b/kernel.cu",
            n=3,
        )
    )
    if not diff:
        raise ValueError("replacement did not produce a source diff")
    return diff
