"""Persistent, scoped LangGraph Store for evidence-derived repair episodes.

This local Store intentionally stores no previous patch, kernel source, goal,
raw model response, credential, hidden input or evaluator artifact. Lessons
are controller-generated descriptions of observable tool/failure categories.
A public pass is NOT an independent strict verification.

The default search is BM25 plus domain tags. Optional local embeddings are
loaded only when an operator explicitly provides a local model directory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from langgraph.store.base import BaseStore, GetOp, Item, ListNamespacesOp, PutOp, SearchItem, SearchOp

from gpu_agent.harness.graph_plan import TOOL_GATES
from gpu_agent.store import reject_symlinks

MEMORY_VERSION = "gpu-agent-repair-memory-v1"
STRATEGIES = {
    "COMPILE_REPAIR": "Use compiler evidence and inspect the exact current source revision.",
    "NUMERIC_REPAIR": "Diagnose public numeric or sanitizer feedback and check the revised candidate.",
    "AST_REBIND": "Use the current source hash and exact observed AST node binding.",
    "AVOID_DUPLICATE": "Reuse existing evidence and avoid repeating an identical rejected action.",
    "REPLAN": "Revise the next task using the observed tool failure.",
    "HALT_INFRASTRUCTURE": "Preserve infrastructure failure evidence without retrying uncertain effects.",
}
_TAGS = {
    "COMPILE_REPAIR": "compile build compiler nvcc syntax diagnostic",
    "NUMERIC_REPAIR": "numeric precision overflow underflow mismatch softmax reduction sanitizer correctness",
    "AST_REBIND": "ast syntax tree sitter binding source revision node hash edit",
    "AVOID_DUPLICATE": "duplicate repeated redundant read rejected",
    "REPLAN": "task plan dependency repair failure",
    "HALT_INFRASTRUCTURE": "infrastructure unavailable timeout failure",
}
_STOP = {"a", "an", "the", "to", "and", "of", "for", "in", "on", "with", "use", "this", "fix"}
MAX_ROWS = 4096
MAX_VALUE_BYTES = 16_384


class Embedder(Protocol):
    identity: str

    def embed(self, text: str) -> Sequence[float]: ...


class LocalSentenceEmbedder:
    """Optional CPU-only, offline local embedding backend."""

    def __init__(self, model_dir: Path):
        reject_symlinks(model_dir)
        root = model_dir.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("memory embedding model must be a local directory")
        parts = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("symlinks are not allowed in embedding model directory")
            if path.is_file():
                h = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(block)
                parts.append((path.relative_to(root).as_posix(), h.hexdigest()))
        if not parts:
            raise ValueError("empty local embedding model")
        self.identity = hashlib.sha256(json.dumps(parts).encode()).hexdigest()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError("install the optional memory-embeddings dependency") from exc
        self.model = SentenceTransformer(
            str(root), device="cpu", local_files_only=True, trust_remote_code=False,
        )

    def embed(self, text: str) -> Sequence[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def rank_texts(texts: list[str], query: str) -> list[float]:
    """Deterministic BM25 ranking; no embedding API or invented semantic score."""
    q = set(tokens(query))
    if not q or not texts:
        return [0.0] * len(texts)
    terms = [Counter(tokens(text)) for text in texts]
    average = max(1.0, sum(sum(row.values()) for row in terms) / len(terms))
    df = {term: sum(term in row for row in terms) for term in q}
    scores = []
    for row in terms:
        norm = 1.2 * (0.25 + 0.75 * sum(row.values()) / average)
        scores.append(sum(
            math.log(1 + (len(terms) - df[term] + 0.5) / (df[term] + 0.5))
            * row[term] * 2.2 / (row[term] + norm)
            for term in q if row[term]
        ))
    return scores


def _vector(values: Sequence[float]) -> list[float]:
    result = [float(x) for x in values]
    if not result or len(result) > 4096 or not all(math.isfinite(x) for x in result):
        raise ValueError("invalid local memory embedding")
    length = math.sqrt(sum(x * x for x in result))
    if length <= 0:
        raise ValueError("zero memory embedding")
    return [x / length for x in result]


def _namespace(namespace: tuple[str, ...]) -> str:
    if not namespace or not all(re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", x) for x in namespace):
        raise ValueError("invalid memory namespace")
    return json.dumps(namespace, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteRepairStore(BaseStore):
    """Local persistent implementation of LangGraph BaseStore's batch contract.

    Exact namespaces, equality filters, lexical retrieval and optional local
    cosine retrieval. The store has a bounded corpus per scope. No raw user
    documents are accepted by the runtime's memory-writing node.
    """

    def __init__(self, database: Path, namespace: tuple[str, ...], *,
                 embedder: Embedder | None = None):
        self.database = database.absolute()
        self.namespace = namespace
        self.ns = _namespace(namespace)
        self.embedder = embedder
        reject_symlinks(self.database)
        self.database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.database.exists():
            try:
                fd = os.open(self.database, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
            except FileExistsError:
                pass
        info = self.database.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
            raise ValueError("memory database must be an operator-owned private file")
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS lessons ("
                         "ns TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
                         "created TEXT NOT NULL, updated TEXT NOT NULL, "
                         "embedding TEXT, embedding_id TEXT, PRIMARY KEY(ns,key))")

    def _connect(self):
        conn = sqlite3.connect(self.database, timeout=3.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=3000")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _require_scope(self, namespace: tuple[str, ...]) -> None:
        if namespace != self.namespace:
            raise ValueError("memory namespace is outside the operator-bound repository scope")

    @staticmethod
    def _item(row, namespace, *, score=None, search=False):
        value = json.loads(row["value"])
        args = dict(namespace=namespace, key=row["key"], value=value,
                    created_at=datetime.fromisoformat(row["created"]),
                    updated_at=datetime.fromisoformat(row["updated"]))
        return SearchItem(**args, score=score) if search else Item(**args)

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        results: list[Any] = []
        conn = self._connect()
        try:
            with conn:
                for op in ops:
                    if isinstance(op, GetOp):
                        self._require_scope(op.namespace)
                        row = conn.execute("SELECT * FROM lessons WHERE ns=? AND key=?",
                                           (self.ns, op.key)).fetchone()
                        results.append(self._item(row, self.namespace) if row else None)
                    elif isinstance(op, PutOp):
                        self._require_scope(op.namespace)
                        if op.ttl is not None:
                            raise ValueError("local repair store does not support TTL")
                        if op.value is None:
                            conn.execute("DELETE FROM lessons WHERE ns=? AND key=?",
                                         (self.ns, op.key))
                            results.append(None)
                            continue
                        raw = json.dumps(dict(op.value), ensure_ascii=False, sort_keys=True)
                        if len(raw.encode()) > MAX_VALUE_BYTES:
                            raise ValueError("memory value exceeds bound")
                        existing = conn.execute("SELECT value FROM lessons WHERE ns=? AND key=?",
                                                (self.ns, op.key)).fetchone()
                        if existing is None and conn.execute(
                            "SELECT count(*) FROM lessons WHERE ns=?", (self.ns,)
                        ).fetchone()[0] >= MAX_ROWS:
                            raise ValueError("memory namespace full; operator maintenance required")
                        if existing is not None and existing["value"] == raw:
                            results.append(None)
                            continue
                        vector = (
                            _vector(self.embedder.embed(str(op.value.get("query_text", ""))))
                            if self.embedder is not None and op.index is not False else None
                        )
                        now = _now()
                        conn.execute(
                            "INSERT INTO lessons VALUES (?,?,?,?,?,?,?) "
                            "ON CONFLICT(ns,key) DO UPDATE SET value=excluded.value,"
                            "updated=excluded.updated,embedding=excluded.embedding,"
                            "embedding_id=excluded.embedding_id",
                            (self.ns, op.key, raw, now, now,
                             json.dumps(vector) if vector else None,
                             self.embedder.identity if vector and self.embedder else None),
                        )
                        results.append(None)
                    elif isinstance(op, SearchOp):
                        self._require_scope(op.namespace_prefix)
                        if not 1 <= op.limit <= 256 or op.offset < 0:
                            raise ValueError("memory query bounds exceeded")
                        rows = list(conn.execute(
                            "SELECT * FROM lessons WHERE ns=? ORDER BY updated DESC,key ASC",
                            (self.ns,),
                        ))
                        values = [json.loads(row["value"]) for row in rows]
                        if op.filter:
                            pairs = [(row, value) for row, value in zip(rows, values, strict=True)
                                     if all(value.get(k) == v for k, v in op.filter.items())]
                            rows, values = [x[0] for x in pairs], [x[1] for x in pairs]
                        lexical = rank_texts([str(v.get("query_text", "")) for v in values],
                                             op.query or "")
                        best = max(lexical, default=0.0)
                        scores = [x / best if best > 0 else 0.0 for x in lexical]
                        if op.query and self.embedder:
                            q = _vector(self.embedder.embed(op.query))
                            for i, row in enumerate(rows):
                                if row["embedding_id"] != self.embedder.identity or not row["embedding"]:
                                    continue
                                vec = _vector(json.loads(row["embedding"]))
                                if len(vec) != len(q):
                                    raise ValueError("memory embedding dimension mismatch")
                                cosine = sum(a * b for a, b in zip(vec, q, strict=True))
                                scores[i] = 0.65 * max(0.0, cosine) + 0.35 * scores[i]
                        order = sorted(range(len(rows)), key=lambda i: (-scores[i], i))
                        if op.query:
                            order = [i for i in order if scores[i] > 0]
                        results.append([
                            self._item(rows[i], self.namespace, score=scores[i] if op.query else None,
                                       search=True)
                            for i in order[op.offset:op.offset + op.limit]
                        ])
                    elif isinstance(op, ListNamespacesOp):
                        def matches(condition):
                            path = condition.path
                            if len(path) > len(self.namespace):
                                return False
                            values = (self.namespace[:len(path)] if condition.match_type == "prefix"
                                      else self.namespace[-len(path):] if path else ())
                            return all(a == "*" or a == b for a, b in zip(path, values, strict=True))
                        exists = conn.execute("SELECT 1 FROM lessons WHERE ns=? LIMIT 1",
                                              (self.ns,)).fetchone()
                        names = [self.namespace[:op.max_depth]] if (
                            exists and all(matches(c) for c in op.match_conditions or ())
                        ) else []
                        results.append(names[op.offset:op.offset + op.limit])
                    else:
                        raise TypeError("unsupported LangGraph Store operation")
        finally:
            conn.close()
        return results

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        return await asyncio.to_thread(self.batch, list(ops))


def memory_configuration(
    repository: Path, output: Path, database: Path | None, scope: str | None,
    mode: str, embedding_model: Path | None,
) -> tuple[dict[str, Any], SQLiteRepairStore | None]:
    if mode == "off":
        if any(x is not None for x in (database, scope, embedding_model)):
            raise ValueError("enable an explicit memory mode before providing memory configuration")
        return {"mode": "off"}, None
    if mode not in {"read", "learn"} or database is None or not scope:
        raise ValueError("memory requires explicit mode, database and repository scope")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", scope):
        raise ValueError("invalid operator memory scope")
    reject_symlinks(database)
    db = database.resolve()
    origin = repository.resolve(strict=True)
    out = output.resolve()
    if db.is_relative_to(origin) or db.is_relative_to(out):
        raise ValueError("memory database must be outside the repository and session")
    namespace = (MEMORY_VERSION, scope, hashlib.sha256(str(origin).encode()).hexdigest())
    embedder = LocalSentenceEmbedder(embedding_model) if embedding_model else None
    config = {
        "mode": mode, "namespace": list(namespace),
        "database_sha256": hashlib.sha256(str(db).encode()).hexdigest(),
        "embedding_identity": embedder.identity if embedder else None,
    }
    return config, SQLiteRepairStore(db, namespace, embedder=embedder)


def make_lesson(state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Derive a typed, source-free episode from actual graph evidence."""
    evidence = state.get("evidence", [])
    if not evidence or state.get("pause_requested"):
        return None
    reasons = [
        e.get("recovery", {}).get("strategy")
        for e in state.get("events", []) if e.get("kind") == "failure_routed"
    ]
    strategies = list(dict.fromkeys(s for s in reasons if s in STRATEGIES))
    tools = [e["tool"] for e in evidence if e.get("tool") in TOOL_GATES]
    snap = state.get("snapshot", {})
    passed = bool(
        snap.get("public_checks_passed") and snap.get("current_candidate_sha256")
        and any(e.get("gate") == "public_check" and e.get("status") == "OK"
                and e.get("candidate") == snap["current_candidate_sha256"] for e in evidence)
    )
    outcome = (
        "PUBLIC_PASS" if passed and state.get("finish_accepted") and state.get("reason") == "WORKSPACE_DIFF_STAGED_UNVERIFIED"
        else "STAGED_UNVERIFIED" if state.get("finish_accepted") and state.get("reason") == "WORKSPACE_DIFF_STAGED_UNVERIFIED"
        else "UNRESOLVED"
    )
    tags = " ".join(_TAGS[s] for s in strategies)
    query_text = f"cuda coding repair {outcome} {tags} {' '.join(tools[-12:])}"
    return {
        "schema": MEMORY_VERSION, "run_id": state["identity"]["run_id"],
        "effect_head_sha256": state["effects_head_sha256"],
        "outcome": outcome, "strategies": strategies,
        "tool_sequence": tools[-12:], "turns": state["turns"],
        "failure_count": sum(e.get("status") != "OK" for e in evidence),
        "query_text": query_text[:1800],
    }


def sanitize_lesson(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """Never trust arbitrary Store content or include it verbatim in a prompt."""
    if value.get("schema") != MEMORY_VERSION:
        return None
    run_id = value.get("run_id", "")
    outcome = value.get("outcome")
    if not re.fullmatch(r"[a-f0-9]{32}", str(run_id)) or outcome not in {
        "PUBLIC_PASS", "STAGED_UNVERIFIED", "UNRESOLVED"
    }:
        return None
    if not isinstance(value.get("strategies"), list) or not isinstance(value.get("tool_sequence"), list):
        return None
    strategies = [s for s in value["strategies"] if isinstance(s, str) and s in STRATEGIES][:6]
    tools = [t for t in value["tool_sequence"] if isinstance(t, str) and t in TOOL_GATES][-12:]
    return {
        "run_id": run_id, "outcome": outcome,
        "strategies": strategies, "tool_sequence": tools,
        "query_text": " ".join(["cuda coding repair", outcome, *(_TAGS[s] for s in strategies), *tools]),
    }


def select_lessons(pool: list[dict[str, Any]], query: str, *, limit: int = 3) -> list[dict[str, Any]]:
    scores = rank_texts([x["query_text"] for x in pool], query)
    order = sorted(range(len(pool)), key=lambda i: (
        -scores[i], -float(pool[i].get("initial_score") or 0), i
    ))
    return [pool[i] for i in order[:limit] if scores[i] > 0 or pool[i].get("initial_score", 0) > 0]
