"""Bounded, controller-owned correction details for public AST tools."""

from __future__ import annotations

import json
from typing import Any


class SyntaxRejection(ValueError):
    """A stable rejection code plus public, actionable fields.

    Never pass through a raw compiler diagnostic, host exception or private
    verifier data. Details are built from the authorized source and typed action.
    """

    def __init__(self, code: str, **detail: Any) -> None:
        self.code = code
        self.detail = detail
        node = detail.get("node_id")
        prefix = f"{code}: {node} " if node is not None else f"{code}: "
        super().__init__(prefix + json.dumps(detail, ensure_ascii=False, sort_keys=True))
