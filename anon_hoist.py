"""Hoist anonymous `function() ... end` to a named local at file scope.

Every run of `foo(function() ... end)` builds a new closure. We lift the
body to `local foo_anon_1 = function(...)` once, then use that name.

Only when we can prove the rewrite does the same thing. Anything else stays
as-is and is reported RED.

When we can pass extra args (`pcall` only):
- read-only body: hoist, pass used names as extra args
- one `x = expr` to an outer name: anon func returns expr; assign x only if pcall
  succeeded (a failed pcall must not store the error string in x)

Everyone else (`table.sort`, `x or function()`, callbacks): same params, no
extra names. Captures or outer writes -> skip. Lua 5.1 `xpcall` is the same.

Lua 5.1 allows 200 locals per function. The chunk is a function. If hoist
would go over that, emit `name = function` (module/global) instead of `local`.

Own `...` (this function's param) stays. Outer `...` is passed on pcall only
(`pcall(fn, ...)`). Callers that cannot take extra args still skip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Lua 5.1 LUAI_MAXVARS is 200. Leave a few slots for later `--fix` chunk locals.
_CHUNK_LOCAL_CAP = 190

from luaparser.astnodes import (
    AnonymousFunction,
    Assign,
    BinaryOp,
    Block,
    Break,
    Call,
    Chunk,
    Comment,
    Do,
    Dots,
    ElseIf,
    FalseExpr,
    Field,
    Forin,
    Fornum,
    Function,
    Goto,
    If,
    Index,
    Invoke,
    Label,
    LocalAssign,
    LocalFunction,
    Method,
    Name,
    Nil,
    Number,
    Repeat,
    Return,
    SemiColon,
    String,
    Table,
    TrueExpr,
    UnaryOp,
    Varargs,
    While,
)


_TOKEN_START = re.compile(r"\[@\d+,(\d+):\d+='")
_TOKEN_END = re.compile(r"\[@\d+,\d+:(\d+)='")
_FUNC_HEADER = re.compile(r"^function\s*\([^)]*\)", re.DOTALL)
# Skip leading `(`, spaces, and comments. Group 1 is `function(...)`.
_FUNC_LEAD = re.compile(
    r"^[ \t\n]*\(?[ \t\n]*(?:--(?:\[\[.*?\]\]|[^\n]*(?:\n[ \t\n]*|$))[ \t\n]*)*(function\s*\([^)]*\))",
    re.DOTALL,
)
_END_TOKEN = re.compile(r"(?<![A-Za-z0-9_])end(?![A-Za-z0-9_])")
# Parser sometimes glues `-- comment` onto the function span. Not part of it.
_TRAIL_COMMENT = re.compile(r"(?:[ \t]*(?:--[^\n]*)?\n?)*\Z")
# Last line already ends with `end` (`end`, `) end`, `return x end`).
_TRAIL_END = re.compile(r"^(.*?)(?<![A-Za-z0-9_])end\s*$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LUA_KEYWORDS = frozenset({
    "and", "break", "do", "else", "elseif", "end", "false", "for",
    "function", "goto", "if", "in", "local", "nil", "not", "or",
    "repeat", "return", "then", "true", "until", "while",
})
_CHILD_ATTRS = (
    "body", "test", "orelse", "targets", "values", "iter",
    "func", "args", "value", "idx", "key", "left", "right",
    "operand", "fields", "keys", "source", "step", "start", "stop",
)


def token_start(token) -> Optional[int]:
    if token is None or str(token) == "None":
        return None
    m = _TOKEN_START.match(str(token))
    return int(m.group(1)) if m else None


def token_end(token) -> Optional[int]:
    if token is None or str(token) == "None":
        return None
    m = _TOKEN_END.match(str(token))
    return int(m.group(1)) + 1 if m else None


def node_span(node) -> Tuple[Optional[int], Optional[int]]:
    """Character span from first/last tokens. None if tokens are missing."""
    if node is None:
        return None, None
    start = token_start(getattr(node, "first_token", None))
    end = token_end(getattr(node, "last_token", None))
    return start, end


def _anon_span(source: str, anon) -> Tuple[Optional[int], Optional[int]]:
    """True `function ... end` slice. Parser often includes `( ... )` and a trailing comment."""
    start, end = node_span(anon)
    if start is None or end is None:
        return None, None
    lead = _FUNC_LEAD.match(source[start:end])
    if lead:
        start = start + lead.start(1)
    body = source[start:end]
    # Extra `)` from `( function() ... end )`.
    while body.endswith(")") and body.count(")") > body.count("("):
        end -= 1
        while end > start and source[end - 1] in " \t\n":
            end -= 1
        body = source[start:end]
    last = None
    for m in _END_TOKEN.finditer(body):
        last = m
    if last and _TRAIL_COMMENT.match(body[last.end():]):
        end = start + last.end()
    return start, end


def _callee_call_span(source: str, call: Call) -> Tuple[Optional[int], Optional[int]]:
    """Span of the whole `name(...)`, callee included.

    Parser often starts the call at `(`. Replacing from there writes
    the name again and you get `name name(...)`.
    """
    start, end = node_span(call)
    if end is None:
        return None, None

    func_start, _ = node_span(getattr(call, "func", None))
    probe = func_start if func_start is not None else start
    start = _walk_back_to_callee(source, probe) if probe is not None else None
    if start is None:
        return None, None

    # last_token is sometimes `end` of the anon; include the call's `)`.
    i = end
    while i < len(source) and source[i] in " \t\n":
        i += 1
    if i < len(source) and source[i] == ")":
        end = i + 1
    elif end > 0 and source[end - 1] != ")":
        return None, None
    return start, end


def _walk_back_to_callee(source: str, pos: int) -> Optional[int]:
    """From `(`, walk back to the name. Only pcall / xpcall get a full-call rewrite."""
    i = pos
    if 0 <= i < len(source) and source[i] == "(":
        i -= 1
        while i >= 0 and source[i] in " \t\n":
            i -= 1
    while i >= 0 and (source[i].isalnum() or source[i] == "_"):
        i -= 1
    start = i + 1
    j = start
    while j < len(source) and (source[j].isalnum() or source[j] == "_"):
        j += 1
    # Everyone else replaces the function only, not the whole call.
    if source[start:j] in ("pcall", "xpcall"):
        return start
    return None


def _iter_children(node):
    if node is None:
        return
    if isinstance(node, list):
        for item in node:
            yield item
        return
    for attr in _CHILD_ATTRS:
        child = getattr(node, attr, None)
        if child is None:
            continue
        if isinstance(child, list):
            for item in child:
                yield item
        else:
            yield child


def _real_stmts(block) -> List[Any]:
    if not isinstance(block, Block) or not getattr(block, "body", None):
        return []
    return [s for s in block.body if not isinstance(s, (SemiColon, Comment))]


def _name_id(node) -> Optional[str]:
    if isinstance(node, Name) and getattr(node, "id", None):
        return node.id
    return None


def _is_bracket_idx(idx) -> bool:
    """True for `t[foo]`. Dot `t.foo` has no token on the key."""
    tok = getattr(idx, "first_token", None) if idx is not None else None
    return tok is not None and str(tok) != "None"


def _index_path(node) -> Optional[str]:
    """Name or `a.b.c` flattened to `a_b_c`."""
    if isinstance(node, Name):
        return node.id
    if not isinstance(node, Index):
        return None
    parts = []
    cur = node
    while isinstance(cur, Index):
        idx = _name_id(cur.idx)
        if idx:
            parts.append(idx)
        cur = cur.value
    base = _name_id(cur)
    if base:
        parts.append(base)
    parts.reverse()
    return "_".join(parts) if parts else None


def sanitize_ident(raw: str) -> str:
    """Safe Lua name for the caller prefix (`foo.bar` -> `foo_bar`)."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", raw or "")
    s = re.sub(r"_+", "_", s).strip("_")
    if not s or s[0].isdigit() or s in _LUA_KEYWORDS:
        s = f"_{s}" if s else "_chunk"
    if s in _LUA_KEYWORDS:
        s = f"_{s}"
    return s or "_chunk"


def _func_display_name(node) -> Optional[str]:
    if isinstance(node, LocalFunction):
        return _name_id(node.name)
    if isinstance(node, Function):
        return _index_path(node.name)
    if isinstance(node, Method):
        src = _name_id(getattr(node, "source", None)) or "obj"
        meth = _name_id(getattr(node, "name", None)) or "method"
        return f"{src}_{meth}"
    return None


def _assigned_func_name(func_node, parents) -> Optional[str]:
    """Name of `foo = function()` / `local foo = function()` wrapping func_node."""
    if not parents or func_node is None:
        return None
    p = parents.get(id(func_node))
    if not isinstance(p, (Assign, LocalAssign)):
        return None
    targets = getattr(p, "targets", None) or []
    return _index_path(targets[0]) if targets else None


def _caller_name(scope, parents) -> str:
    """Innermost function: `foo = function()` wins over an outer `local function`."""
    s = scope
    while s is not None:
        if getattr(s, "scope_type", None) == "function":
            node = getattr(s, "node", None)
            assigned = _assigned_func_name(node, parents)
            if assigned:
                return assigned
            display = _func_display_name(node) if node is not None else None
            name = display or getattr(s, "name", "") or ""
            if name and not str(name).startswith("<"):
                return name
        s = getattr(s, "parent", None)
    return "_chunk"


def _chunk_stmt(node, parents):
    """File-scope statement that owns `node` (`local function`, `foo =`, `do`).

    Anons go above this, not inside the hot function.
    """
    if not parents or node is None:
        return None
    prev = node
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, Block):
            gp = parents.get(id(cur))
            if isinstance(gp, Chunk):
                return prev
        prev = cur
        cur = parents.get(id(cur))
    return None


def _all_taken_names(scopes) -> Set[str]:
    """Names we must not reuse for a hoisted func or a temp."""
    taken: Set[str] = set(_LUA_KEYWORDS)
    if not scopes:
        return taken
    for s in scopes:
        taken.update(getattr(s, "locals", ()) or ())
        name = getattr(s, "name", None)
        if name and not str(name).startswith("<"):
            taken.add(sanitize_ident(str(name)))
    return taken


def _anon_seq(name: str) -> int:
    m = re.search(r"_anon_(\d+)$", name or "")
    return int(m.group(1)) if m else 0


def pick_anon_name(caller: str, taken: Set[str]) -> Optional[str]:
    """First free `foo_anon_1`, `foo_anon_2`, ... Marks it taken."""
    base = sanitize_ident(caller)
    for n in range(1, 10000):
        name = f"{base}_anon_{n}"
        if name not in taken:
            taken.add(name)
            return name
    return None


def pick_temp(base: str, taken: Set[str]) -> str:
    """`_ok` / `_x` temps that must not clash with locals already in the file."""
    if base not in taken:
        taken.add(base)
        return base
    i = 2
    while True:
        cand = f"{base}{i}"
        if cand not in taken:
            taken.add(cand)
            return cand
        i += 1


@dataclass
class AnonUse:
    """What the body does to names outside itself."""
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)
    uses_dots: bool = False
    has_goto: bool = False
    unknown: bool = False


# Parser leftover, not Lua: `for i = 1, #t` has no step, so step is a raw 1.
_LEAF = (int, float, str, bool, bytes)
_WALK_OK = (
    Block, Call, Return, Table, BinaryOp, UnaryOp, Chunk,
    Number, String, Nil, TrueExpr, FalseExpr, Break,
    Comment, SemiColon, Label,
)


def _walk_anon(anon: AnonymousFunction, on_free=None, on_dots=None, on_goto=None, on_unknown=None):
    """Walk the body. Track names born here. Callers listen for free names.

    Bind vs write:
    - `local x = rhs`: new x inside this function. Walk rhs, then bind x.
    - `x = rhs`: write of an outer x. Then walk rhs.

    Free name = used here, not bound in this function or a nested one.
    """
    stack: List[Set[str]] = [set()]

    def add_inner(name: Optional[str]):
        if name:
            stack[-1].add(name)

    def is_inner(name: str) -> bool:
        return any(name in layer for layer in stack)

    def dots():
        if on_dots:
            on_dots()

    def walk(node, as_write: bool = False):
        if node is None or isinstance(node, _LEAF):
            return
        if isinstance(node, (Dots, Varargs)):
            # Own `...` is a param on the stack. Else it is an outer function's.
            if not is_inner("..."):
                dots()
            return
        if isinstance(node, Goto):
            if on_goto:
                on_goto()
            return
        if isinstance(node, Name):
            name = node.id
            if name == "...":
                if not is_inner("..."):
                    dots()
                return
            if not name or is_inner(name):
                return
            if on_free:
                on_free(name, as_write)
            return
        if isinstance(node, Index):
            # t.foo reads t. t[foo] also reads foo.
            walk(node.value)
            if _is_bracket_idx(node.idx):
                walk(node.idx)
            return
        if isinstance(node, Field):
            walk(getattr(node, "value", None))
            # { foo = 1 } key is a field name, not a variable. { [foo] = 1 } is.
            tok = getattr(node, "first_token", None)
            if tok is not None and str(tok) != "None" and str(tok).find("'['") >= 0:
                walk(getattr(node, "key", None))
            return
        if isinstance(node, Invoke):
            walk(getattr(node, "source", None))
            for a in getattr(node, "args", None) or []:
                walk(a)
            return
        if isinstance(node, LocalAssign):
            # Must be before Assign: local-assign is a kind of assign in the parser.
            for v in getattr(node, "values", None) or []:
                walk(v)
            for t in getattr(node, "targets", None) or []:
                add_inner(_name_id(t))
            return
        if isinstance(node, Assign):
            for t in getattr(node, "targets", None) or []:
                if isinstance(t, Name):
                    walk(t, as_write=True)
                else:
                    walk(t)
            for v in getattr(node, "values", None) or []:
                walk(v)
            return
        if isinstance(node, (Function, LocalFunction, Method, AnonymousFunction)):
            # Nested function: its locals are not free names of the outer.
            if isinstance(node, LocalFunction):
                add_inner(_name_id(node.name))
            stack.append(set())
            if isinstance(node, Method):
                add_inner("self")
            for arg in getattr(node, "args", None) or []:
                if isinstance(arg, (Dots, Varargs)):
                    add_inner("...")
                else:
                    add_inner(_name_id(arg))
            walk(getattr(node, "body", None))
            stack.pop()
            return
        if isinstance(node, Fornum):
            # `for i =` : i is local to the loop.
            walk(getattr(node, "start", None))
            walk(getattr(node, "stop", None))
            walk(getattr(node, "step", None))
            stack.append(set())
            add_inner(_name_id(getattr(node, "target", None)))
            walk(getattr(node, "body", None))
            stack.pop()
            return
        if isinstance(node, Forin):
            # `for k, v in` : k and v are local to the loop.
            for it in getattr(node, "iter", None) or []:
                walk(it)
            stack.append(set())
            for t in getattr(node, "targets", None) or []:
                add_inner(_name_id(t))
            walk(getattr(node, "body", None))
            stack.pop()
            return
        if isinstance(node, (While, Repeat, Do)):
            if isinstance(node, While):
                walk(getattr(node, "test", None))
            stack.append(set())
            walk(getattr(node, "body", None))
            if isinstance(node, Repeat):
                walk(getattr(node, "test", None))
            stack.pop()
            return
        if isinstance(node, If):
            # Each then / elseif / else has its own locals.
            walk(getattr(node, "test", None))
            stack.append(set())
            walk(getattr(node, "body", None))
            stack.pop()
            orelse = getattr(node, "orelse", None)
            while orelse is not None:
                if isinstance(orelse, ElseIf):
                    walk(getattr(orelse, "test", None))
                    stack.append(set())
                    walk(getattr(orelse, "body", None))
                    stack.pop()
                    orelse = getattr(orelse, "orelse", None)
                else:
                    stack.append(set())
                    walk(orelse)
                    stack.pop()
                    orelse = None
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if isinstance(node, _WALK_OK):
            for child in _iter_children(node):
                walk(child)
            return
        if on_unknown:
            # Node type we do not handle. Fail closed.
            on_unknown()

    for arg in getattr(anon, "args", None) or []:
        if isinstance(arg, (Dots, Varargs)):
            add_inner("...")
        else:
            add_inner(_name_id(arg))
    walk(getattr(anon, "body", None))


def _analyze_anon(anon: AnonymousFunction) -> AnonUse:
    """One walk: unknown / outer `...` / goto, plus outer reads and writes."""
    use = AnonUse()

    def on_free(name, is_write):
        if is_write:
            use.writes.add(name)
        else:
            use.reads.add(name)

    def mark_dots():
        use.uses_dots = True

    def mark_goto():
        use.has_goto = True

    def mark_unknown():
        use.unknown = True

    _walk_anon(anon, on_free=on_free, on_dots=mark_dots, on_goto=mark_goto, on_unknown=mark_unknown)
    return use


def _anon_params(anon: AnonymousFunction) -> Optional[List[str]]:
    """Declared args. `...` is allowed and must be last."""
    params: List[str] = []
    for arg in getattr(anon, "args", None) or []:
        if isinstance(arg, (Dots, Varargs)):
            params.append("...")
            continue
        name = _name_id(arg)
        if not name:
            return None
        params.append(name)
    if "..." in params and params[-1] != "...":
        return None
    return params


def _params_with_dots(orig: List[str], extra: List[str], outer_dots: bool) -> List[str]:
    """Named args, then extra captures, then `...` last if we keep or pass it."""
    named = [p for p in orig if p != "..."]
    named.extend(extra)
    if outer_dots or "..." in orig:
        named.append("...")
    return named


def _single_capture_assign(anon: AnonymousFunction, writes: Set[str]) -> Optional[Tuple[str, Any]]:
    """Exactly one `x = expr` that writes an outer x. Not `local x =`."""
    stmts = _real_stmts(getattr(anon, "body", None))
    if len(stmts) != 1 or type(stmts[0]) is not Assign:
        return None
    assign = stmts[0]
    targets = getattr(assign, "targets", None) or []
    values = getattr(assign, "values", None) or []
    if len(targets) != 1 or len(values) != 1:
        return None
    name = _name_id(targets[0])
    if not name or name not in writes:
        return None
    return name, values[0]


def _call_kind(call: Call) -> Optional[str]:
    func = getattr(call, "func", None)
    if not isinstance(func, Name):
        return None
    if func.id in ("pcall", "xpcall"):
        return func.id
    return None


def _collect_anons_and_parents(root) -> Tuple[List[AnonymousFunction], Dict[int, Any]]:
    """One walk: every unnamed function, plus the parent of every node."""
    parents: Dict[int, Any] = {}
    anons: List[AnonymousFunction] = []

    def walk(node, parent=None):
        if node is None or isinstance(node, (str, int, float, bool, bytes)):
            return
        if isinstance(node, list):
            for item in node:
                walk(item, parent)
            return
        parents[id(node)] = parent
        if isinstance(node, AnonymousFunction):
            anons.append(node)
        for child in _iter_children(node):
            walk(child, node)

    walk(root)
    return anons, parents


def _assign_context(call: Call, parent) -> Tuple[str, int, Optional[str], Optional[Any]]:
    """How the pcall sits in the parent.

    stmt: `pcall(...)` alone.
    assign: `ok = pcall(...)` or `local ok = pcall(...)`.
    expr: inside a bigger expression. We cannot rewrite that.
    """
    if isinstance(parent, Block):
        return "stmt", 0, None, None
    if isinstance(parent, (Assign, LocalAssign)):
        values = getattr(parent, "values", None) or []
        if len(values) != 1 or values[0] is not call:
            return "expr", 0, None, None
        targets = getattr(parent, "targets", None) or []
        ok_name = _name_id(targets[0]) if targets else None
        return "assign", len(targets), ok_name, parent
    return "expr", 0, None, None


def _line_start(source: str, pos: int) -> int:
    return source.rfind("\n", 0, pos) + 1


def _line_indent(source: str, pos: int) -> str:
    start = _line_start(source, pos)
    i = start
    while i < len(source) and source[i] in " \t":
        i += 1
    return source[start:i]


def _extra_args_src(source: str, args: List[Any]) -> Optional[str]:
    if not args:
        return ""
    start, _ = node_span(args[0])
    _, end = node_span(args[-1])
    if start is None or end is None:
        return None
    return source[start:end]


def _rhs_src(source: str, value_node) -> Optional[str]:
    start, end = node_span(value_node)
    if start is None or end is None:
        return None
    return source[start:end]


def _unwrap_func_src(func_src: str) -> Optional[str]:
    """Drop a wrapping `(` the parser sometimes includes. Need `function` first."""
    m = _FUNC_LEAD.match(func_src)
    if not m:
        return None
    return func_src[m.start(1):]


def _replace_func_header(func_src: str, anon_name: str, params: List[str]) -> Optional[str]:
    """Prefix: `function(...)` becomes `local name = function(...)`. Rest stays."""
    body = _unwrap_func_src(func_src)
    if body is None or not _FUNC_HEADER.match(body):
        return None
    header = f"local {anon_name} = function({', '.join(params)})"
    return _FUNC_HEADER.sub(header, body, count=1)


def _anon_role(anon, parent) -> str:
    """pcall / xpcall if this function is the first arg. Else expr."""
    if isinstance(parent, Call):
        kind = _call_kind(parent)
        args = list(getattr(parent, "args", None) or [])
        if kind == "pcall" and args and args[0] is anon:
            return "pcall"
        if kind == "xpcall" and args and args[0] is anon:
            return "xpcall"
    return "expr"


def _is_chunk_func_def(anon, parent, parents) -> bool:
    """`foo = function()` / `local foo = function()` at file scope. Already named."""
    if not isinstance(parent, (Assign, LocalAssign)):
        return False
    values = getattr(parent, "values", None) or []
    if len(values) != 1 or values[0] is not anon:
        return False
    return _chunk_stmt(parent, parents) is parent


def _chunk_local_decl_pos(tree) -> Dict[str, int]:
    """First `local x` / `local function x` at file scope, by name."""
    pos: Dict[str, int] = {}
    block = getattr(tree, "body", None)
    stmts = getattr(block, "body", None) if block is not None else None
    if not stmts:
        return pos
    for s in stmts:
        names = []
        if isinstance(s, LocalAssign):
            names = [_name_id(t) for t in (getattr(s, "targets", None) or [])]
        elif isinstance(s, LocalFunction):
            names = [_name_id(s.name)]
        st, _ = node_span(s)
        if st is None:
            continue
        for name in names:
            if name and name not in pos:
                pos[name] = st
    return pos


def _enclosing_bound_names(scope, anon) -> Set[str]:
    """Locals from scopes around this function, not the function itself.

    Globals and file-scope locals are not in here. Those stay visible after
    we hoist to file scope. A name that is here must be passed as an extra
    arg (pcall only) or we skip.
    """
    names: Set[str] = set()
    s = scope
    while s is not None:
        if getattr(s, "node", None) is anon:
            s = getattr(s, "parent", None)
            continue
        st = getattr(s, "scope_type", None)
        if st and st != "global":
            names.update(getattr(s, "locals", ()) or ())
        s = getattr(s, "parent", None)
    return names


def _scope_for(node, analyzer, parents):
    """Nearest analyzer scope that owns this node."""
    by_id = {}
    for s in getattr(analyzer, "scopes", None) or []:
        n = getattr(s, "node", None)
        if n is not None:
            by_id[id(n)] = s
    cur = node
    while cur is not None:
        s = by_id.get(id(cur))
        if s is not None:
            return s
        cur = parents.get(id(cur)) if parents else None
    return getattr(analyzer, "global_scope", None)


def _node_line(source: str, node) -> int:
    start, _ = node_span(node)
    if start is None:
        return 0
    return source.count("\n", 0, start) + 1


def classify_anon(
    anon: AnonymousFunction,
    source: str,
    scope,
    parent,
    taken: Set[str],
    parents=None,
    chunk_locals=None,
) -> Dict[str, Any]:
    """Decide hoist vs skip. `safe` True means we will rewrite.

    Skip if: outer `...` where the caller cannot take extra args, goto,
    unknown syntax, outer writes that are not a single `x = expr` on pcall,
    enclosing-function locals where the caller cannot take extra args
    (everything except pcall; xpcall never can), or we cannot slice tokens.
    Own `...` is fine. Globals and earlier file-scope locals are fine.
    """
    details: Dict[str, Any] = {"safe": False}
    # pcall / xpcall first arg, or just a function sitting in an expression.
    role = _anon_role(anon, parent)
    details["role"] = role
    details["callee"] = role if role in ("pcall", "xpcall") else "function"
    details["full_match"] = f"{details['callee']}(function" if role != "expr" else "function("

    fn_start, fn_end = _anon_span(source, anon)
    if None in (fn_start, fn_end):
        details["skip_reason"] = "missing first_token/last_token"
        return details

    call = parent if role in ("pcall", "xpcall") else None
    call_start = call_end = None
    extra_src = ""
    if call is not None:
        call_start, call_end = _callee_call_span(source, call)
        if None in (call_start, call_end):
            details["skip_reason"] = "missing first_token/last_token"
            return details
        args = list(getattr(call, "args", None) or [])
        # Lua 5.1 xpcall is only (fn, err). No extra args, no other shapes.
        if role == "xpcall" and len(args) != 2:
            details["skip_reason"] = "xpcall must be xpcall(fn, err)"
            return details
        extra_src = _extra_args_src(source, args[1:])
        if extra_src is None:
            details["skip_reason"] = "cannot slice extra pcall args"
            return details

    # Body must be something we can read. Outer `...` is not a skip here;
    # pcall can pass it. Goto or unknown syntax -> skip.
    use = _analyze_anon(anon)
    if use.unknown:
        details["skip_reason"] = "anonymous function has unknown syntax"
        return details
    if use.has_goto:
        details["skip_reason"] = "anonymous function uses goto"
        return details

    orig_params = _anon_params(anon)
    if orig_params is None:
        details["skip_reason"] = "anonymous function has unusable params"
        return details

    # Named after the function we sit in. Inserted above that file-scope statement.
    caller = _caller_name(scope, parents)
    insert_node = _chunk_stmt(anon, parents)
    if insert_node is None:
        details["skip_reason"] = "cannot find chunk-level insert point"
        return details
    ins_start, _ = node_span(insert_node)
    if ins_start is None:
        details["skip_reason"] = "cannot find chunk-level insert point"
        return details
    insert_char = _line_start(source, ins_start)

    writes = set(use.writes)
    reads = set(use.reads) - writes
    # Only safe write: body is exactly `x = expr` on pcall. Else skip.
    assign_info = _single_capture_assign(anon, writes) if writes else None

    if writes and (role != "pcall" or assign_info is None):
        details["skip_reason"] = "writes captured locals (not a single x = expr)"
        details["writes"] = sorted(writes)
        return details

    if writes and assign_info is not None:
        call_parent = parents.get(id(call)) if parents and call is not None else None
        ctx_kind, n_targets, ok_name, assign_node = _assign_context(call, call_parent)
        if ctx_kind == "expr":
            details["skip_reason"] = "assign rewrite not safe in expression context"
            return details
        if ctx_kind == "assign" and n_targets >= 2:
            details["skip_reason"] = "pcall already unpacks 2+ results"
            return details
        assigned, rhs_node = assign_info
        rhs_src = _rhs_src(source, rhs_node)
        if rhs_src is None:
            details["skip_reason"] = "cannot slice assignment rhs"
            return details
        leftover = sorted(reads)
        a_start = a_end = None
        if ctx_kind != "stmt":
            a_start, a_end = node_span(assign_node)
            if a_start is None or a_end is None:
                details["skip_reason"] = "cannot slice parent assign"
                return details

        anon_name = pick_anon_name(caller, taken)
        if anon_name is None:
            details["skip_reason"] = "cannot allocate anon name"
            return details

        # Assign rewrite is pcall. Outer `...` is just another extra arg.
        anon_params = _params_with_dots(orig_params, leftover, use.uses_dots)
        indent = _line_indent(source, insert_char)
        inner = indent + _indent_unit(source)

        anon_text = (
            f"{indent}local {anon_name} = function({', '.join(anon_params)})\n"
            f"{inner}return {rhs_src}\n"
            f"{indent}end\n\n"
        )

        call_args = [anon_name]
        if extra_src:
            call_args.append(extra_src)
        call_args.extend(leftover)
        if use.uses_dots:
            call_args.append("...")
        new_call = f"pcall({', '.join(call_args)})"

        stmt_indent = _line_indent(source, call_start)
        val_name = pick_temp(f"_{assigned}", taken)
        # Return the expr. Copy into x only when pcall succeeded.
        # A failed pcall's second result is the error string; that must not land in x.
        if ctx_kind == "stmt":
            ok_temp = pick_temp("_ok", taken)
            replace_start, replace_end = call_start, call_end
            replace_text = (
                f"local {ok_temp}, {val_name} = {new_call}\n"
                f"{stmt_indent}if {ok_temp} then {assigned} = {val_name} end"
            )
        else:
            ok_temp = ok_name or pick_temp("_ok", taken)
            local_kw = "local " if isinstance(assign_node, LocalAssign) else ""
            replace_start, replace_end = a_start, a_end
            replace_text = (
                f"{local_kw}{ok_temp}, {val_name} = {new_call}\n"
                f"{stmt_indent}if {ok_temp} then {assigned} = {val_name} end"
            )

        details.update({
            "safe": True,
            "rewrite_kind": "assign",
            "anon_name": anon_name,
            "anon_params": anon_params,
            "captures": leftover + (["..."] if use.uses_dots else []),
            "anon_text": anon_text,
            "fn_start": fn_start,
            "fn_end": fn_end,
            "insert_char": insert_char,
            "insert_seq": _anon_seq(anon_name),
            "replace_start": replace_start,
            "replace_end": replace_end,
            "replace_text": replace_text,
            "caller": caller,
        })
        return details

    captures = sorted(reads)
    enclosing = _enclosing_bound_names(scope, anon)
    must_forward = [c for c in captures if c in enclosing]
    # pcall can take extra args. table.sort / xpcall / `x or function()` cannot.
    can_forward = role == "pcall"
    # Outer `...` needs the same extra-arg slot as a captured local.
    if use.uses_dots and not can_forward:
        details["skip_reason"] = (
            "xpcall cannot take extra args (Lua 5.1)"
            if role == "xpcall"
            else "caller cannot take extra args"
        )
        details["captures"] = ["..."]
        return details
    if must_forward and not can_forward:
        details["skip_reason"] = (
            "xpcall cannot take extra args (Lua 5.1)"
            if role == "xpcall"
            else "caller cannot take extra args"
        )
        details["captures"] = must_forward
        return details
    # File-scope `local later` after the insert only matters if this function
    # was compiled after that local (it captured it). If the function sits
    # above the local, it already sees a global. Same after hoist.
    late = []
    if not can_forward and chunk_locals:
        late = [
            c for c in captures
            if c not in enclosing
            and c in chunk_locals
            and insert_char <= chunk_locals[c] <= fn_start
        ]
        if late:
            details["skip_reason"] = "uses local declared after anon"
            details["captures"] = late
            return details

    anon_name = pick_anon_name(caller, taken)
    if anon_name is None:
        details["skip_reason"] = "cannot allocate anon name"
        return details

    # Same body, new name. pcall gets extra args for captures and outer `...`.
    # Everyone else: name only. Own `...` stays on the param list.
    func_src = source[fn_start:fn_end]
    extra = captures if can_forward else []
    anon_params = _params_with_dots(orig_params, extra, use.uses_dots and can_forward)
    anon_body = _replace_func_header(func_src, anon_name, anon_params)
    if anon_body is None:
        details["skip_reason"] = "cannot rewrite function header"
        return details

    indent = _line_indent(source, insert_char)
    anon_text = _reindent_anon(anon_body, indent, source) + "\n\n"

    if role in ("pcall", "xpcall"):
        # `pcall(function() ... end, extra)` -> `pcall(name, extra, captures...)`
        call_args = [anon_name]
        if extra_src:
            call_args.append(extra_src)
        if can_forward:
            call_args.extend(captures)
        if use.uses_dots and can_forward:
            call_args.append("...")
        replace_text = f"{role}({', '.join(call_args)})"
        replace_start, replace_end = call_start, call_end
    else:
        # `table.sort(t, function() ... end)` / `x or function()` -> just the name.
        replace_text = anon_name
        replace_start, replace_end = fn_start, fn_end

    details.update({
        "safe": True,
        "rewrite_kind": "hoist",
        "anon_name": anon_name,
        "anon_params": anon_params,
        "captures": (captures + (["..."] if use.uses_dots else [])) if can_forward else [],
        "anon_text": anon_text,
        "fn_start": fn_start,
        "fn_end": fn_end,
        "insert_char": insert_char,
        "insert_seq": _anon_seq(anon_name),
        "replace_start": replace_start,
        "replace_end": replace_end,
        "replace_text": replace_text,
        "caller": caller,
    })
    return details


def _indent_unit(source: str) -> str:
    """One indent step: tab if the file uses tabs, else 4 spaces."""
    for line in source.split("\n"):
        if line.startswith("\t"):
            return "\t"
        stripped = line.lstrip(" ")
        if stripped != line and line.startswith("    "):
            return "    "
    return "\t"


def _reindent_anon(text: str, indent: str, source: str) -> str:
    """`local name = function` / `end` at insert indent; body one step in."""
    lines = text.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return text
    unit = _indent_unit(source)
    if len(lines) == 1:
        return indent + lines[0].lstrip(" \t")
    first = indent + lines[0].lstrip(" \t")
    last = indent + "end"
    # Keep anything before the last `end` (`) end`, `return x end`). Always
    # write one `end` of our own. Packer `return foo(\n)\n) end` plus a second
    # `end` does not parse.
    trail = _TRAIL_END.match(lines[-1].rstrip())
    if trail:
        middle = list(lines[1:-1])
        rest = trail.group(1).rstrip()
        if rest:
            middle.append(rest)
    else:
        middle = lines[1:]
    out = [first]
    # Strip the old common indent, then indent + one step.
    nonempty = [m for m in middle if m.strip()]
    common = None
    for m in nonempty:
        lead = m[: len(m) - len(m.lstrip(" \t"))]
        if common is None or (lead and m.startswith(common) and len(lead) < len(common)):
            common = lead
        elif common and not m.startswith(common):
            common = ""
            break
    if common is None:
        common = ""
    for m in middle:
        if not m.strip():
            out.append("")
            continue
        body = m[len(common):] if common and m.startswith(common) else m.lstrip(" \t")
        out.append(indent + unit + body)
    out.append(last)
    return "\n".join(out)


def _drop_local_if_chunk_full(drafts, chunk_locals):
    """Lua 5.1: 200 locals per function. Chunk locals plus `local foo_anon_N` count.

    Over the cap, drop `local` so the name is a module/global assign.
    """
    n_new = sum(1 for _, d in drafts if d.get("safe"))
    if len(chunk_locals) + n_new <= _CHUNK_LOCAL_CAP:
        return
    for _, d in drafts:
        name = d.get("anon_name")
        text = d.get("anon_text")
        if d.get("safe") and name and text:
            d["anon_text"] = text.replace(
                f"local {name} = function", f"{name} = function", 1
            )


def analyze_tree(analyzer) -> None:
    """Collect, classify once (smallest first), splice inners into outers, report."""
    tree = getattr(analyzer, "_ast_tree", None)
    if tree is None:
        return
    source = analyzer.source
    anons, parents = _collect_anons_and_parents(tree)
    taken = _all_taken_names(getattr(analyzer, "scopes", None))
    chunk_locals = _chunk_local_decl_pos(tree)
    from models import Finding

    candidates = []
    for anon in anons:
        parent = parents.get(id(anon))
        # File-scope `foo = function()` is already named. Leave it.
        if _is_chunk_func_def(anon, parent, parents):
            continue
        scope = _scope_for(anon, analyzer, parents)
        fn_start, fn_end = _anon_span(source, anon)
        if fn_start is None or fn_end is None:
            size, start = 10**9, 10**9
        else:
            size, start = fn_end - fn_start, fn_start
        candidates.append((size, start, anon, parent, scope))
    # Smallest first: inner `_anon_1` before the outer that holds it.
    candidates.sort()

    drafts = []
    for _size, _start, anon, parent, scope in candidates:
        details = classify_anon(
            anon,
            source,
            scope,
            parent,
            taken,
            parents=parents,
            chunk_locals=chunk_locals,
        )
        drafts.append((anon, details))

    _splice_nested(drafts, source)
    _drop_local_if_chunk_full(drafts, chunk_locals)

    for anon, details in drafts:
        line = _node_line(source, anon)
        src_line = ""
        if hasattr(analyzer, "_get_source_line"):
            src_line = analyzer._get_source_line(line) or ""
        callee = details.get("callee") or "function"
        anon_name = details.get("anon_name") or "?"
        caps = details.get("captures") or []
        if details.get("safe"):
            rewrite = details.get("rewrite_kind")
            if rewrite == "assign":
                msg = f"function() x = ... end -> pcall({anon_name}) then x = result"
                suggestion = f"Hoist to {anon_name}; assign only on success"
            elif details.get("role") in ("pcall", "xpcall"):
                call_args = ", ".join([anon_name] + list(caps))
                msg = f"{callee}(function() ... end) -> {callee}({call_args})"
                suggestion = f"Hoist to {anon_name}"
            else:
                msg = f"function() ... end -> {anon_name}"
                suggestion = f"Hoist to {anon_name}"
            details["suggestion"] = suggestion
            analyzer.findings.append(Finding(
                pattern_name="anon_hoist",
                severity="GREEN",
                line_num=line,
                message=msg,
                details=details,
                source_line=src_line,
            ))
        else:
            reason = details.get("skip_reason") or "unsafe"
            details["suggestion"] = f"Skipped: {reason}"
            analyzer.findings.append(Finding(
                pattern_name="anon_skip",
                severity="RED",
                line_num=line,
                message=f"function() ... end skipped: {reason}",
                details=details,
                source_line=src_line,
            ))


def _span_contains(outer, inner) -> bool:
    """Inner function sits inside outer. Use function span, not the callsite."""
    os_, oe = outer.get("fn_start"), outer.get("fn_end")
    ins, ie = inner.get("fn_start"), inner.get("fn_end")
    if None in (os_, oe, ins, ie):
        return False
    return os_ < ins and oe > ie


def _rebuild_hoist_anon(details, source, func_src) -> bool:
    """Re-header and reindent after a splice into this body."""
    anon_body = _replace_func_header(
        func_src,
        details.get("anon_name"),
        details.get("anon_params") or [],
    )
    if anon_body is None:
        return False
    indent = _line_indent(source, details["insert_char"])
    details["anon_text"] = _reindent_anon(anon_body, indent, source) + "\n\n"
    return True


def _splice_inner(outer, inner, source) -> bool:
    """Rewrite inner's original function text inside the outer anon func.

    Keep a raw (pre-reindent) copy so a later sibling splice does not rebuild
    from the original source and wipe this one.
    """
    fs, fe = inner.get("fn_start"), inner.get("fn_end")
    if fs is None or fe is None:
        return False
    old = source[fs:fe]
    new = inner.get("replace_text") or ""
    if not old or not new:
        return False
    raw = outer.get("raw_func")
    if raw is None:
        ofs, ofe = outer.get("fn_start"), outer.get("fn_end")
        if ofs is None or ofe is None:
            return False
        raw = source[ofs:ofe]
    if old not in raw:
        return False
    raw = raw.replace(old, new, 1)
    outer["raw_func"] = raw
    return _rebuild_hoist_anon(outer, source, raw)


def _splice_nested(drafts, source):
    """Paste each child's rewrite into the nearest outer. Inner site stays as-is in the file."""
    # drafts is already small-to-large.
    greens = [d for *_, d in drafts if d.get("safe")]
    done = []
    for details in greens:
        contained = [c for c in done if c.get("safe") and _span_contains(details, c)]
        for child in contained:
            if any(other is not child and _span_contains(other, child) for other in contained):
                continue  # grandchild: already pasted into the child
            if not _splice_inner(details, child, source):
                details["safe"] = False
                details["skip_reason"] = "contains nested function"
                break
        done.append(details)
    # Child body is already the name inside the outer. Do not also replace it in the file.
    for details in greens:
        if not details.get("safe"):
            continue
        if any(
            other is not details and other.get("safe") and _span_contains(other, details)
            for other in greens
        ):
            details["absorb_callsite"] = True
