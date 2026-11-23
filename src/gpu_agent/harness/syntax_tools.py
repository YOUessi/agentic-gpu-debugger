"""Controller-owned CUDA syntax-node inspection and atomic scoped rewriting.

tree-sitter locates exact byte intervals in *raw* source, not model-quoted
lines. It does NOT prove semantic correctness: only the separate CUDA oracle
and independent verifier can do that. Refuse unparseable or stale sources.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import tree_sitter_cuda
from tree_sitter import Language, Node, Parser

_NODE_TYPES = frozenset({"expression_statement", "return_statement", "declaration"})
_MAX_NODES = 90
_MAX_SOURCE = 1024 * 1024


@dataclass(frozen=True)
class SyntaxNode:
    node_id: str
    function: str
    kind: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    code: str


def _parser() -> Parser:
    return Parser(Language(tree_sitter_cuda.language()))


def _source_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _functions(root: Node, raw: bytes) -> list[tuple[str, Node]]:
    found: list[tuple[str, Node]] = []

    def walk(node: Node) -> None:
        if node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            if declarator is None:
                return
            text = raw[declarator.start_byte : declarator.end_byte].decode("utf-8")
            match = re.search(r"([A-Za-z_]\w*)\s*\(", text)
            if match:
                found.append((match.group(1), node))
            return
        for child in node.named_children:
            walk(child)

    walk(root)
    return found


def _entries(raw: bytes) -> tuple[str, list[SyntaxNode]]:
    if not raw or len(raw) > _MAX_SOURCE or b"\x00" in raw:
        raise ValueError("invalid bounded CUDA syntax source")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("CUDA source is not valid UTF-8") from None
    tree = _parser().parse(raw)
    if tree.root_node.has_error:
        raise ValueError("CUDA_SYNTAX_SOURCE_HAS_ERRORS")
    source_sha256 = _source_digest(raw)
    entries: list[SyntaxNode] = []
    for function, function_node in _functions(tree.root_node, raw):
        def visit(node: Node) -> None:
            if node.type in _NODE_TYPES:
                chunk = raw[node.start_byte : node.end_byte]
                node_id = hashlib.sha256(
                    f"{source_sha256}:{node.type}:{node.start_byte}:{node.end_byte}".encode()
                    + chunk
                ).hexdigest()[:24]
                entries.append(
                    SyntaxNode(
                        node_id=node_id,
                        function=function,
                        kind=node.type,
                        start_line=node.start_point.row + 1,
                        end_line=node.end_point.row + 1,
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        code=chunk.decode("utf-8"),
                    )
                )
                return
            for child in node.named_children:
                visit(child)
        visit(function_node)
    if len(entries) > _MAX_NODES:
        raise ValueError("CUDA syntax node listing exceeds controller bound")
    return source_sha256, entries


def syntax_nodes(raw: bytes, *, symbol: str = "") -> dict[str, Any]:
    """Expose only ordinary public source structure; never a clean reference."""
    if symbol and not re.fullmatch(r"[A-Za-z_]\w{0,79}", symbol):
        raise ValueError("invalid CUDA function symbol")
    digest, entries = _entries(raw)
    if symbol:
        entries = [item for item in entries if item.function == symbol]
    if not entries:
        raise ValueError("CUDA_SYNTAX_SYMBOL_NOT_FOUND")
    return {
        "schema_version": "gpu-cuda-syntax-map-v1",
        "source_sha256": digest,
        "symbol_filter": symbol,
        "nodes": [
            {
                "id": item.node_id,
                "function": item.function,
                "kind": item.kind,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "code": item.code[:300],
            }
            for item in entries
        ],
    }


def rewrite_syntax_nodes(
    raw: bytes,
    *,
    expected_source_sha256: str,
    edits: list[tuple[str, str]],
) -> bytes:
    """Resolve immutable syntax IDs and atomically assemble a new CUDA source.

    All targets bind to ONE input version. No fallbacks to fuzzy line matching,
    no mutation of any source file, and no silent syntactic repair.
    """
    digest, entries = _entries(raw)
    if digest != expected_source_sha256:
        raise ValueError("CUDA_SOURCE_SHA_STALE")
    if not 1 <= len(edits) <= 4:
        raise ValueError("CUDA_SYNTAX_EDIT_COUNT_OUT_OF_BOUNDS")
    by_id = {item.node_id: item for item in entries}
    spans: list[tuple[int, int, bytes]] = []
    chosen: set[str] = set()
    for node_id, code in edits:
        if node_id in chosen:
            raise ValueError("CUDA_SYNTAX_DUPLICATE_NODE")
        chosen.add(node_id)
        if node_id not in by_id:
            raise ValueError("CUDA_SYNTAX_NODE_NOT_FOUND_OR_STALE")
        if not code or len(code.encode("utf-8")) > 1200 or "\x00" in code:
            raise ValueError("CUDA_SYNTAX_REPLACEMENT_INVALID")
        node = by_id[node_id]
        proposed = code.encode("utf-8")
        if proposed == raw[node.start_byte : node.end_byte]:
            raise ValueError("CUDA_SYNTAX_NOOP")
        spans.append((node.start_byte, node.end_byte, proposed))
    spans.sort(key=lambda x: x[0])
    cursor = 0
    chunks: list[bytes] = []
    for start, end, proposed in spans:
        if start < cursor:
            raise ValueError("CUDA_SYNTAX_OVERLAPPING_NODES")
        chunks.extend((raw[cursor:start], proposed))
        cursor = end
    chunks.append(raw[cursor:])
    rewritten = b"".join(chunks)
    if len(rewritten) > _MAX_SOURCE:
        raise ValueError("CUDA_SYNTAX_RESULT_TOO_LARGE")
    after = _parser().parse(rewritten)
    if after.root_node.has_error:
        raise ValueError("CUDA_SYNTAX_REPLACEMENT_PARSE_ERROR")
    return rewritten
