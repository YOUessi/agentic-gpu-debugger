"""Restricted caller-mask contract plus advisory arrival analysis, not full C++ proof.

The CUDA programming guide states, for __syncwarp(mask) and the *_sync(mask, ...) warp
intrinsics, that each calling thread must have its own bit set in mask and that all
non-exited threads named in mask must execute the same intrinsic with the same mask; and
that __syncthreads must be reached by all non-exited threads of the block. This module
evaluates that rule on concrete thread indices, per warp, for the launch configurations
written in the same file. It uses nothing but the patched source text: no test data, no
reference fix, no execution.

The analysis is deliberately small and conservative. A call site is judged only when the
kernel has integer-literal (or integer-constant) 1-D launch configurations in the file,
the site is not inside a loop/switch, and every condition, early return and mask that
reaches it is a constant expression of threadIdx/blockDim/warpSize, const locals built
from them, and __ballot_sync results. Anything else is UNANALYZABLE and never rejected,
so unknown sites do not yield a finding. The simplified parser is not a complete C++
semantic model. Only a modeled caller missing its own mask bit is a source-contract
violation. Arrival/liveness uncertainty cannot reject a patch. GPU verification remains
independent, and compiler-eliminated calls can differ from the source contract.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SiteStatus = Literal["CONSISTENT", "MISMATCH", "UNANALYZABLE"]

WARP_INTRINSICS = frozenset(
    {
        "__syncwarp",
        "__ballot_sync",
        "__all_sync",
        "__any_sync",
        "__uni_sync",
        "__shfl_sync",
        "__shfl_up_sync",
        "__shfl_down_sync",
        "__shfl_xor_sync",
        "__match_any_sync",
        "__match_all_sync",
        "__reduce_add_sync",
        "__reduce_min_sync",
        "__reduce_max_sync",
        "__reduce_and_sync",
        "__reduce_or_sync",
        "__reduce_xor_sync",
    }
)
BLOCK_BARRIERS = frozenset(
    {"__syncthreads", "__syncthreads_count", "__syncthreads_and", "__syncthreads_or"}
)
_OPAQUE_CONTROL = frozenset({"for", "while", "do", "switch"})
_FULL = 0xFFFFFFFF
_MAX_BLOCK = 1024


class SyncCounterexample(BaseModel):
    """Controller-computed numeric witness; no repair, label or arbitrary text."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    line: int = Field(ge=1)
    intrinsic: str = Field(max_length=32)
    block_size: int = Field(ge=1, le=1024)
    thread_index: int = Field(ge=0, lt=1024)
    lane_id: int = Field(ge=0, lt=32)
    mask: int = Field(ge=0, le=0xFFFFFFFF)

    @model_validator(mode="after")
    def valid_counterexample(self) -> "SyncCounterexample":
        if (
            self.intrinsic not in WARP_INTRINSICS
            or self.thread_index >= self.block_size
            or self.lane_id != self.thread_index % 32
            or self.mask & (1 << self.lane_id)
        ):
            raise ValueError("not a caller-mask counterexample")
        return self


@dataclass(frozen=True)
class SiteResult:
    function: str
    line: int
    intrinsic: str
    status: SiteStatus
    reason: str = ""
    counterexample: SyncCounterexample | None = None


class _Unknown(Exception):
    """The value is not a constant function of the thread index."""


@dataclass(frozen=True)
class _Tok:
    text: str
    line: int


# --------------------------------------------------------------------------- lexing


def _strip(text: str) -> str:
    """Blank comments, literals and preprocessor lines, keeping offsets and newlines."""
    out = list(text)
    i, n = 0, len(text)

    def blank(start: int, end: int) -> None:
        for k in range(start, min(end, n)):
            if out[k] != "\n":
                out[k] = " "

    at_line_start = True
    while i < n:
        c = text[i]
        if at_line_start and c == "#":
            end = i
            while end < n and text[end] != "\n":
                end += 2 if text[end] == "\\" and end + 1 < n else 1
            blank(i, end)
            i = end
            continue
        if c == "\n":
            at_line_start = True
            i += 1
            continue
        if not c.isspace():
            at_line_start = False
        if text.startswith("//", i):
            end = text.find("\n", i)
            end = n if end < 0 else end
            blank(i, end)
            i = end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            blank(i, end)
            i = end
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
        else:
            i += 1
    return "".join(out)


_TOKEN = re.compile(
    r"0[xX][0-9a-fA-F]+[uUlL]*|\d+[uUlL]*|[A-Za-z_]\w*|<<<|>>>|<<=|>>=|::|->|\+\+|--"
    r"|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^]=|\S"
)


def _tokens(text: str) -> list[_Tok]:
    stripped = _strip(text)
    toks: list[_Tok] = []
    line, previous = 1, 0
    for match in _TOKEN.finditer(stripped):
        line += stripped.count("\n", previous, match.start())
        previous = match.start()
        toks.append(_Tok(match.group(), line))
    return toks


def _match(toks: list[_Tok], i: int) -> int:
    """Index of the bracket closing toks[i]."""
    pairs = {"(": ")", "{": "}", "[": "]"}
    opener, closer = toks[i].text, pairs[toks[i].text]
    depth = 0
    for k in range(i, len(toks)):
        if toks[k].text == opener:
            depth += 1
        elif toks[k].text == closer:
            depth -= 1
            if depth == 0:
                return k
    raise _Unknown


def _split_args(toks: list[_Tok]) -> list[list[_Tok]]:
    args: list[list[_Tok]] = [[]]
    depth = 0
    for tok in toks:
        if tok.text in "([{":
            depth += 1
        elif tok.text in ")]}":
            depth -= 1
        if tok.text == "," and depth == 0:
            args.append([])
        else:
            args[-1].append(tok)
    return args


# ----------------------------------------------------------------------- expressions


def _int(text: str) -> int:
    if "l" in text.lower():
        raise _Unknown  # 64-bit promotions are outside this expression model.
    body = text.rstrip("uUlL")
    base = 16 if body[:2] in {"0x", "0X"} else (8 if len(body) > 1 and body[0] == "0" else 10)
    value = int(body, base)
    if not 0 <= value <= _FULL:
        raise _Unknown
    return value


class _Expr:
    """Recursive-descent evaluator for C integer expressions (unsigned 32-bit)."""

    _BINARY = [
        ["||"],
        ["&&"],
        ["|"],
        ["^"],
        ["&"],
        ["==", "!="],
        ["<", "<=", ">", ">="],
        ["<<", ">>"],
        ["+", "-"],
        ["*", "/", "%"],
    ]

    def __init__(self, toks: list[_Tok], env: "_Env", thread: int) -> None:
        self.toks = [t.text for t in toks]
        self.env = env
        self.thread = thread
        self.pos = 0

    def value(self) -> int:
        if not self.toks:
            raise _Unknown
        result = self._binary(0)
        if self.pos != len(self.toks):
            raise _Unknown
        return result

    def _peek(self) -> str | None:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def _take(self, expected: str | None = None) -> str:
        tok = self._peek()
        if tok is None or (expected is not None and tok != expected):
            raise _Unknown
        self.pos += 1
        return tok

    def _binary(self, level: int) -> int:
        if level == len(self._BINARY):
            return self._unary()
        left = self._binary(level + 1)
        while self._peek() in self._BINARY[level]:
            op = self._take()
            right = self._binary(level + 1)
            left = _apply(op, left, right)
        return left

    def _unary(self) -> int:
        tok = self._peek()
        if tok == "!":
            self._take()
            return int(not self._unary())
        if tok == "~":
            raise _Unknown  # Integral promotions require a typed expression model.
        if tok == "-":
            raise _Unknown  # signed arithmetic is outside the unsigned model
        if tok == "+":
            self._take()
            return self._unary()
        return self._primary()

    def _primary(self) -> int:
        tok = self._take()
        if tok == "(":
            result = self._binary(0)
            self._take(")")
            return result
        if tok[0].isdigit():
            return _int(tok) & _FULL
        if tok in {"true", "false"}:
            return int(tok == "true")
        if tok == "static_cast":
            raise _Unknown  # C++ conversion widths/signs are not modeled.
        if tok in {"threadIdx", "blockDim"}:
            self._take(".")
            axis = self._take()
            if axis not in {"x", "y", "z"}:
                raise _Unknown
            if tok == "threadIdx":
                return self.thread if axis == "x" else 0
            return self.env.block if axis == "x" else 1
        if tok == "warpSize":
            return 32
        if tok.isidentifier():
            if self._peek() == "(":
                raise _Unknown  # calls other than a bound __ballot_sync are opaque
            return self.env.lookup(tok, self.thread)
        raise _Unknown


def _apply(op: str, a: int, b: int) -> int:
    if op == "||":
        return int(bool(a) or bool(b))
    if op == "&&":
        return int(bool(a) and bool(b))
    if op in {"/", "%"} and b == 0:
        raise _Unknown
    if op in {"<<", ">>"} and b >= 32:
        raise _Unknown
    if op == "-" and a < b:
        raise _Unknown  # a negative intermediate would need signed/unsigned rules
    # Do not eagerly evaluate shifts for unrelated operators: a mask used as
    # the RHS of '&' may be 0xffffffff, causing a huge allocation otherwise.
    if op == "<<":
        return (a << b) & _FULL
    if op == ">>":
        return a >> b
    table = {
        "|": a | b,
        "^": a ^ b,
        "&": a & b,
        "==": int(a == b),
        "!=": int(a != b),
        "<": int(a < b),
        "<=": int(a <= b),
        ">": int(a > b),
        ">=": int(a >= b),
        "+": a + b,
        "-": a - b,
        "*": a * b,
        "/": a // b if b else 0,
        "%": a % b if b else 0,
    }
    return table[op] & _FULL


# ------------------------------------------------------------------ function model

_Guard = tuple[tuple[_Tok, ...], bool] | None  # None: an opaque (loop/switch) region


@dataclass
class _Binding:
    expr: list[_Tok]
    ctx: tuple[_Guard, ...]
    order: int


@dataclass
class _Site:
    intrinsic: str
    line: int
    mask: list[_Tok] | None  # None for block barriers
    ctx: tuple[_Guard, ...]
    order: int


@dataclass
class _Function:
    name: str
    is_kernel: bool
    bindings: dict[str, _Binding | None] = field(default_factory=dict)
    mutated: set[str] = field(default_factory=set)
    returns: list[tuple[tuple[_Guard, ...], int]] = field(default_factory=list)
    sites: list[_Site] = field(default_factory=list)
    opaque: bool = False
    order: int = 0

    def next_order(self) -> int:
        self.order += 1
        return self.order


class _Env:
    def __init__(self, fn: _Function, block: int, constants: dict[str, int]) -> None:
        self.fn = fn
        self.block = block
        self.constants = constants
        self._active: set[str] = set()
        self._cache: dict[tuple[str, int], int] = {}

    def lookup(self, name: str, thread: int) -> int:
        key = (name, thread)
        if key not in self._cache:
            self._cache[key] = self._lookup(name, thread)
        return self._cache[key]

    def _lookup(self, name: str, thread: int) -> int:
        if name not in self.fn.bindings:
            raise _Unknown
        binding = self.fn.bindings[name]
        if binding is None or name in self.fn.mutated or name in self._active:
            raise _Unknown
        self._active.add(name)
        try:
            expr = binding.expr
            if expr and expr[0].text == "__ballot_sync":
                return self._ballot(binding, thread)
            return _Expr(expr, self, thread).value()
        finally:
            self._active.discard(name)

    def _ballot(self, binding: _Binding, thread: int) -> int:
        close = len(binding.expr) - 1
        if len(binding.expr) < 4 or binding.expr[1].text != "(" or binding.expr[close].text != ")":
            raise _Unknown
        args = _split_args(binding.expr[2:close])
        if len(args) != 2:
            raise _Unknown
        if not self.executes(binding.ctx, binding.order, thread):
            raise _Unknown  # the variable is read by a thread that never computed it
        result = 0
        caller_mask = _Expr(args[0], self, thread).value()
        for other in _warp_of(thread, self.block):
            if not caller_mask >> (other % 32) & 1:
                continue
            if not self.executes(binding.ctx, binding.order, other):
                if self.alive(binding.order, other):
                    raise _Unknown
                continue
            member = _Expr(args[0], self, other).value()
            if member != caller_mask:
                raise _Unknown
            if member >> (other % 32) & 1 and _Expr(args[1], self, other).value():
                result |= 1 << (other % 32)
        return result

    def guards(self, ctx: tuple[_Guard, ...], thread: int) -> bool:
        for guard in ctx:
            if guard is None:
                raise _Unknown
            cond, polarity = guard
            if bool(_Expr(list(cond), self, thread).value()) != polarity:
                return False
        return True

    def alive(self, order: int, thread: int) -> bool:
        return not any(self.guards(ctx, thread) for ctx, at in self.fn.returns if at < order)

    def executes(self, ctx: tuple[_Guard, ...], order: int, thread: int) -> bool:
        return self.alive(order, thread) and self.guards(ctx, thread)


def _warp_of(thread: int, block: int) -> range:
    start = thread - thread % 32
    return range(start, min(start + 32, block))


_TYPE_WORDS = frozenset(
    "const constexpr static volatile unsigned signed int long short char bool auto float "
    "double size_t uint32_t int32_t uint64_t int64_t std".split()
)


class _Parser:
    def __init__(self, toks: list[_Tok], fn: _Function) -> None:
        self.toks = toks
        self.fn = fn

    def block(self, start: int, end: int, ctx: tuple[_Guard, ...]) -> None:
        i = start
        while i < end:
            i = self.statement(i, end, ctx)

    def statement(self, i: int, end: int, ctx: tuple[_Guard, ...]) -> int:
        tok = self.toks[i].text
        if tok == "{":
            close = _match(self.toks, i)
            self.block(i + 1, close, ctx)
            return close + 1
        if tok == ";":
            return i + 1
        if tok == "if":
            j = i + 1
            if self.toks[j].text == "constexpr":
                j += 1
            if self.toks[j].text != "(":
                raise _Unknown
            close = _match(self.toks, j)
            cond = tuple(self.toks[j + 1 : close])
            after = self.statement(close + 1, end, (*ctx, (cond, True)))
            if after < end and self.toks[after].text == "else":
                after = self.statement(after + 1, end, (*ctx, (cond, False)))
            return after
        if tok in _OPAQUE_CONTROL:
            self.fn.opaque = True  # May prevent reaching even a later lexical site.
            j = i + 1
            if tok == "do":
                after = self.statement(j, end, (*ctx, None))
                if self.toks[after].text != "while":
                    raise _Unknown
                close = _match(self.toks, after + 1)
                return close + 2
            if self.toks[j].text != "(":
                raise _Unknown
            close = _match(self.toks, j)
            self.expression(list(self.toks[j + 1 : close]), (*ctx, None))
            return self.statement(close + 1, end, (*ctx, None))
        if tok == "goto":
            self.fn.opaque = True
        # expression, declaration, return, break, continue
        j = i
        depth = 0
        while j < end:
            t = self.toks[j].text
            if t in "([{":
                depth += 1
            elif t in ")]}":
                depth -= 1
            elif t == ";" and depth == 0:
                break
            j += 1
        body = self.toks[i:j]
        if tok == "return":
            self.expression(body[1:], ctx)
            self.fn.returns.append((ctx, self.fn.next_order()))
        else:
            self.declaration_or_expression(body, ctx)
        return j + 1

    def declaration_or_expression(self, body: list[_Tok], ctx: tuple[_Guard, ...]) -> None:
        texts = [t.text for t in body]
        for k, t in enumerate(texts):
            if t in {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}:
                lhs = texts[:k]
                if (
                    t == "="
                    and len(lhs) >= 2
                    and lhs[-1].isidentifier()
                    and all(w in _TYPE_WORDS or w == "::" for w in lhs[:-1])
                ):
                    name = lhs[-1]
                    rhs = body[k + 1 :]
                    declared = [w for w in lhs[:-1] if w not in {"const", "constexpr"}]
                    supported = any(w in {"const", "constexpr"} for w in lhs[:-1]) and declared in (
                        ["unsigned"],
                        ["unsigned", "int"],
                        ["uint32_t"],
                    )
                    if not supported or len(_split_args(rhs)) > 1 or name in self.fn.bindings:
                        self.fn.bindings[name] = None
                    else:
                        self.fn.bindings[name] = _Binding(list(rhs), ctx, self.fn.next_order())
                    self.expression(rhs, ctx)
                    return
                # Only the assigned object is mutated (`a[i] = v` changes a, not i).
                base = next((w for w in lhs if w.isidentifier() and w not in _TYPE_WORDS), None)
                if base is not None:
                    self.fn.mutated.add(base)
                break
        for k, t in enumerate(texts):
            if t in {"++", "--"}:
                for near in (k - 1, k + 1):
                    if 0 <= near < len(texts) and texts[near].isidentifier():
                        self.fn.mutated.add(texts[near])
            if t == "&" and k + 1 < len(texts) and texts[k + 1].isidentifier():
                if k == 0 or texts[k - 1] in {"(", ",", "=", "return"}:
                    self.fn.mutated.add(texts[k + 1])
        self.expression(body, ctx)

    def expression(self, body: list[_Tok], ctx: tuple[_Guard, ...]) -> None:
        if any(tok.text in {"&&", "||", "?"} for tok in body):
            # Operand-level control flow is not represented by the statement
            # parser; do not treat conditional calls as unconditional calls.
            ctx = (*ctx, None)
        k = 0
        while k < len(body):
            t = body[k].text
            if (
                t.isidentifier()
                and k + 1 < len(body)
                and body[k + 1].text == "("
                and t not in WARP_INTRINSICS | BLOCK_BARRIERS
            ):
                self.fn.opaque = True  # Unknown calls may mutate or alter control flow.
            if t == "{":  # lambda or initializer list: anything inside is opaque
                close = _match(body, k)
                self.expression(body[k + 1 : close], (*ctx, None))
                k = close + 1
                continue
            if t in "?":
                ctx = (*ctx, None)  # sync calls in a conditional operand are opaque
            if (
                (t in WARP_INTRINSICS or t in BLOCK_BARRIERS)
                and k + 1 < len(body)
                and body[k + 1].text == "("
            ):
                close = _match(body, k + 1)
                args = _split_args(body[k + 2 : close])
                mask: list[_Tok] | None = None
                if t in WARP_INTRINSICS:
                    mask = args[0] if args[0] else [_Tok("0xffffffffU", body[k].line)]
                    for arg in args:
                        self.expression(arg, ctx)
                self.fn.sites.append(_Site(t, body[k].line, mask, ctx, self.fn.next_order()))
                k = close + 1
                continue
            k += 1


# --------------------------------------------------------------------- whole file


def _functions(toks: list[_Tok]) -> list[tuple[_Function, int, int]]:
    found: list[tuple[_Function, int, int]] = []
    i = 0
    while i < len(toks):
        if toks[i].text in {"__global__", "__device__"}:
            is_kernel = toks[i].text == "__global__"
            j = i + 1
            while j < len(toks) and toks[j].text not in {";", "{", "("}:
                j += 1
            if j < len(toks) and toks[j].text == "(" and toks[j - 1].text.isidentifier():
                name = toks[j - 1].text
                close = _match(toks, j)
                k = close + 1
                while k < len(toks) and toks[k].text not in {"{", ";"}:
                    k += 1
                if k < len(toks) and toks[k].text == "{":
                    body_end = _match(toks, k)
                    fn = _Function(name, is_kernel)
                    if any(
                        t.text in {"threadIdx", "blockDim", "warpSize"} for t in toks[j + 1 : close]
                    ):
                        fn.opaque = True
                    found.append((fn, k + 1, body_end))
                    i = body_end + 1
                    continue
        i += 1
    return found


def _constants(toks: list[_Tok]) -> dict[str, int]:
    """Integer names declared exactly once as `const[expr] <int type> NAME = <literal>;`."""
    seen: dict[str, int | None] = {}
    for k in range(len(toks) - 3):
        if toks[k].text not in {"const", "constexpr"}:
            continue
        j = k + 1
        while j < len(toks) and toks[j].text in _TYPE_WORDS | {"::"}:
            j += 1
        if j + 3 < len(toks) and toks[j + 1].text == "=" and toks[j + 3].text == ";":
            name, value = toks[j].text, toks[j + 2].text
            if name.isidentifier() and value[:1].isdigit():
                seen[name] = None if name in seen else _int(value)
    return {name: value for name, value in seen.items() if value is not None}


def _launch_blocks(toks: list[_Tok], constants: dict[str, int]) -> dict[str, list[int | None]]:
    launches: dict[str, list[int | None]] = {}
    for k, tok in enumerate(toks):
        if tok.text != "<<<" or k == 0 or not toks[k - 1].text.isidentifier():
            continue
        end = k + 1
        while end < len(toks) and toks[end].text != ">>>":
            end += 1
        args = _split_args(toks[k + 1 : end])
        block: int | None = None
        if len(args) >= 2:
            texts = [t.text for t in args[1]]
            if len(texts) == 1 and texts[0][:1].isdigit():
                block = _int(texts[0])
        if block is not None and not 1 <= block <= _MAX_BLOCK:
            block = None
        launches.setdefault(toks[k - 1].text, []).append(block)
    return launches


def _judge(
    fn: _Function,
    site: _Site,
    launches: list[int | None],
    consts: dict[str, int],
    source_hash: str,
    witnesses: list[SyncCounterexample],
) -> tuple[SiteStatus, str]:
    blocks = [block for block in launches if block is not None]
    if fn.opaque or not fn.is_kernel or not blocks or len(blocks) != len(launches):
        return "UNANALYZABLE", "unsupported_launch_or_control"
    uncertain = False
    try:
        for block in blocks:
            env = _Env(fn, block, consts)
            executing = [t for t in range(block) if env.executes(site.ctx, site.order, t)]
            if site.mask is None:
                alive = [t for t in range(block) if env.alive(site.order, t)]
                if executing != alive:
                    # A thread absent at this lexical site can reach another
                    # dynamic barrier instance or exit. We do not model either.
                    uncertain = True
                continue
            for t in executing:
                mask = _Expr(site.mask, env, t).value()
                if not mask >> (t % 32) & 1:
                    witnesses.append(
                        SyncCounterexample(
                            source_sha256=source_hash,
                            line=site.line,
                            intrinsic=site.intrinsic,
                            block_size=block,
                            thread_index=t,
                            lane_id=t % 32,
                            mask=mask,
                        )
                    )
                    return "MISMATCH", "caller_not_in_mask"
                for other in _warp_of(t, block):
                    if not mask >> (other % 32) & 1 or not env.alive(site.order, other):
                        continue
                    if not env.guards(site.ctx, other):
                        uncertain = True
                        continue
                    if _Expr(site.mask, env, other).value() != mask:
                        uncertain = True
    except (_Unknown, RecursionError, IndexError, ValueError):
        return "UNANALYZABLE", "unsupported_expression"
    if uncertain:
        return "UNANALYZABLE", "dynamic_arrival_or_exit_not_modeled"
    return "CONSISTENT", "supported_site_consistent"


def analyze(source: str) -> list[SiteResult]:
    """Judge every synchronizing call site in `source`; never raises on odd input."""
    if len(source) > 1024 * 1024:
        return []
    try:
        toks = _tokens(source)
        functions = _functions(toks)
        constants = _constants(toks)
        launches = _launch_blocks(toks, constants)
    except (_Unknown, IndexError, ValueError):
        return []
    results: list[SiteResult] = []
    for fn, start, end in functions:
        if any(
            t.text
            in {"asm", "__asm__", "struct", "class", "using", "typedef", "try", "throw", "template"}
            for t in toks[start:end]
        ):
            fn.opaque = True
        if sum(other.name == fn.name for other, _, _ in functions) != 1:
            fn.opaque = True
        if re.search(r"^\s*#\s*(?:define|undef|if|elif|else|endif)\b", source, re.MULTILINE):
            fn.opaque = True
        try:
            _Parser(toks, fn).block(start, end, ())
        except (_Unknown, IndexError, RecursionError):
            fn.opaque = True
        for site in fn.sites:
            witnesses: list[SyncCounterexample] = []
            status, reason = _judge(
                fn,
                site,
                launches.get(fn.name, []),
                constants,
                hashlib.sha256(source.encode()).hexdigest(),
                witnesses,
            )
            results.append(
                SiteResult(
                    fn.name,
                    site.line,
                    site.intrinsic,
                    status,
                    reason,
                    witnesses[0] if witnesses else None,
                )
            )
    return results


def has_mismatch(source: str) -> bool:
    return any(result.status == "MISMATCH" for result in analyze(source))


def caller_mask_violation(source: str) -> bool:
    """Narrow source contract: an executing lane must name itself in its mask.

    Only the modeled immutable uint32/literal-launch subset is admitted. Missing
    arrivals are deliberately excluded: lexical position does not establish
    dynamic liveness. This is independent of whether nvcc eliminates a call.
    """
    return any(r.reason == "caller_not_in_mask" for r in analyze(source))


def caller_mask_counterexample(source: str) -> SyncCounterexample | None:
    return next((r.counterexample for r in analyze(source) if r.counterexample is not None), None)
