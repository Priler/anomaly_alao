"""Classify and rewrite pcall/xpcall(function() ... end) closures.

Hoists the anonymous function to a chunk-level local so the callsite
does not allocate a closure every run. Fail closed: anything we cannot
prove equivalent is reported RED and left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from luaparser.astnodes import (
    AnonymousFunction,
    Assign,
    Block,
    Call,
    Comment,
    Do,
    Dots,
    ElseIf,
    Forin,
    Fornum,
    Function,
    Goto,
    If,
    Index,
    Invoke,
    LocalAssign,
    LocalFunction,
    Method,
    Name,
    Repeat,
    SemiColon,
    Varargs,
    While,
)


_TOKEN_START = re.compile(r"\[@\d+,(\d+):\d+='")
_TOKEN_END = re.compile(r"\[@\d+,\d+:(\d+)='")
_FUNC_HEADER = re.compile(r"^function\s*\([^)]*\)", re.DOTALL)
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
    tok = getattr(idx, "first_token", None) if idx is not None else None
    return tok is not None and str(tok) != "None"


def sanitize_ident(raw: str) -> str:
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
        name = node.name
        if isinstance(name, Name):
            return name.id
        if isinstance(name, Index):
            parts = []
            cur = name
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
        return None
    if isinstance(node, Method):
        src = _name_id(getattr(node, "source", None)) or "obj"
        meth = _name_id(getattr(node, "name", None)) or "method"
        return f"{src}_{meth}"
    return None


def _enclosing_locals(scope) -> Set[str]:
    names: Set[str] = set()
    while scope is not None and getattr(scope, "scope_type", None) != "global":
        names.update(getattr(scope, "locals", ()) or ())
        scope = getattr(scope, "parent", None)
    return names


def _named_func_scopes(scope) -> List[Any]:
    """Innermost-first function scopes that have a usable name + node."""
    out = []
    while scope is not None:
        if getattr(scope, "scope_type", None) == "function":
            node = getattr(scope, "node", None)
            name = getattr(scope, "name", "") or ""
            if node is not None and (
                not name.startswith("<") or isinstance(node, Method)
            ):
                out.append(scope)
        scope = getattr(scope, "parent", None)
    return out


def _all_taken_names(scopes) -> Set[str]:
    taken: Set[str] = set(_LUA_KEYWORDS)
    if not scopes:
        return taken
    for s in scopes:
        taken.update(getattr(s, "locals", ()) or ())
        name = getattr(s, "name", None)
        if name and not str(name).startswith("<"):
            taken.add(sanitize_ident(str(name)))
    return taken


def pick_helper_name(caller: str, taken: Set[str]) -> Optional[str]:
    base = sanitize_ident(caller)
    for n in range(1, 10000):
        name = f"{base}_pcall_{n}"
        if name not in taken:
            taken.add(name)
            return name
    return None


def pick_temp(base: str, taken: Set[str]) -> str:
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
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)
    uses_dots: bool = False
    has_goto: bool = False


def _analyze_anon(anon: AnonymousFunction, enclosing: Set[str]) -> AnonUse:
    use = AnonUse()
    stack: List[Set[str]] = [set()]

    def add_inner(name: Optional[str]):
        if name:
            stack[-1].add(name)

    def is_inner(name: str) -> bool:
        return any(name in layer for layer in stack)

    def push():
        stack.append(set())

    def pop():
        stack.pop()

    for arg in getattr(anon, "args", None) or []:
        if isinstance(arg, (Dots, Varargs)):
            use.uses_dots = True
        add_inner(_name_id(arg))

    def walk(node, as_assign_target: bool = False):
        if node is None:
            return
        if isinstance(node, (Dots, Varargs)):
            use.uses_dots = True
            return
        if isinstance(node, Goto):
            use.has_goto = True
            return
        if isinstance(node, Name):
            name = node.id
            if not name or name == "...":
                if name == "...":
                    use.uses_dots = True
                return
            if is_inner(name):
                return
            if name in enclosing:
                if as_assign_target:
                    use.writes.add(name)
                else:
                    use.reads.add(name)
            return
        if isinstance(node, Index):
            walk(node.value)
            if _is_bracket_idx(node.idx):
                walk(node.idx)
            return
        if isinstance(node, Invoke):
            walk(getattr(node, "source", None))
            for a in getattr(node, "args", None) or []:
                walk(a)
            return
        if isinstance(node, Assign):
            for t in getattr(node, "targets", None) or []:
                if isinstance(t, Name):
                    walk(t, as_assign_target=True)
                else:
                    walk(t)
            for v in getattr(node, "values", None) or []:
                walk(v)
            return
        if isinstance(node, LocalAssign):
            for v in getattr(node, "values", None) or []:
                walk(v)
            for t in getattr(node, "targets", None) or []:
                add_inner(_name_id(t))
            return
        if isinstance(node, (Function, LocalFunction, Method, AnonymousFunction)):
            if isinstance(node, LocalFunction):
                add_inner(_name_id(node.name))
            push()
            if isinstance(node, Method):
                add_inner("self")
            for arg in getattr(node, "args", None) or []:
                if isinstance(arg, (Dots, Varargs)):
                    use.uses_dots = True
                add_inner(_name_id(arg))
            walk(getattr(node, "body", None))
            pop()
            return
        if isinstance(node, Fornum):
            walk(getattr(node, "start", None))
            walk(getattr(node, "stop", None))
            walk(getattr(node, "step", None))
            push()
            add_inner(_name_id(getattr(node, "target", None)))
            walk(getattr(node, "body", None))
            pop()
            return
        if isinstance(node, Forin):
            for it in getattr(node, "iter", None) or []:
                walk(it)
            push()
            for t in getattr(node, "targets", None) or []:
                add_inner(_name_id(t))
            walk(getattr(node, "body", None))
            pop()
            return
        if isinstance(node, (While, Repeat, Do)):
            if isinstance(node, While):
                walk(getattr(node, "test", None))
            push()
            walk(getattr(node, "body", None))
            if isinstance(node, Repeat):
                walk(getattr(node, "test", None))
            pop()
            return
        if isinstance(node, If):
            walk(getattr(node, "test", None))
            push()
            walk(getattr(node, "body", None))
            pop()
            orelse = getattr(node, "orelse", None)
            while orelse is not None:
                if isinstance(orelse, ElseIf):
                    walk(getattr(orelse, "test", None))
                    push()
                    walk(getattr(orelse, "body", None))
                    pop()
                    orelse = getattr(orelse, "orelse", None)
                else:
                    push()
                    walk(orelse)
                    pop()
                    orelse = None
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        for child in _iter_children(node):
            walk(child)

    walk(getattr(anon, "body", None))
    return use


def _anon_params(anon: AnonymousFunction) -> Optional[List[str]]:
    params: List[str] = []
    for arg in getattr(anon, "args", None) or []:
        if isinstance(arg, (Dots, Varargs)):
            return None
        name = _name_id(arg)
        if not name:
            return None
        params.append(name)
    return params


def _single_capture_assign(anon: AnonymousFunction, writes: Set[str]) -> Optional[Tuple[str, Any]]:
    """Body is exactly one `name = expr` whose name is a captured write."""
    stmts = _real_stmts(getattr(anon, "body", None))
    if len(stmts) != 1 or not isinstance(stmts[0], Assign):
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


def _parent_map(root) -> Dict[int, Any]:
    parents: Dict[int, Any] = {}

    def walk(node, parent=None):
        if node is None or isinstance(node, (str, int, float, bool, bytes)):
            return
        if isinstance(node, list):
            for item in node:
                walk(item, parent)
            return
        parents[id(node)] = parent
        for child in _iter_children(node):
            walk(child, node)

    walk(root)
    return parents


def _assign_context(call: Call, parent) -> Tuple[str, int, Optional[str], Optional[Any]]:
    """How the pcall sits in the parent.

    Returns (kind, n_targets, ok_name, assign_node).
    kind: 'stmt' | 'assign' | 'expr'
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


def _replace_func_header(func_src: str, helper_name: str, params: List[str]) -> Optional[str]:
    if not _FUNC_HEADER.match(func_src):
        return None
    header = f"local function {helper_name}({', '.join(params)})"
    return _FUNC_HEADER.sub(header, func_src, count=1)


def classify_pcall(
    call: Call,
    source: str,
    scope,
    parent,
    taken: Set[str],
    scopes=None,
) -> Dict[str, Any]:
    """Return a details dict for a Finding. `safe` True means GREEN hoist."""
    line = 0
    details: Dict[str, Any] = {"safe": False}

    kind = _call_kind(call)
    if kind is None:
        details["skip_reason"] = "callee is not bare pcall/xpcall"
        return details

    args = list(getattr(call, "args", None) or [])
    if not args or not isinstance(args[0], AnonymousFunction):
        details["skip_reason"] = "first arg is not an anonymous function"
        return details

    if kind == "xpcall" and len(args) != 2:
        details["skip_reason"] = "xpcall must be xpcall(fn, err)"
        return details

    anon = args[0]
    fn_start, fn_end = node_span(anon)
    call_start, call_end = node_span(call)
    if None in (fn_start, fn_end, call_start, call_end):
        details["skip_reason"] = "missing first_token/last_token"
        return details

    # Call first_token is often just '('. Walk back to the callee name.
    func_start, _ = node_span(call.func)
    if func_start is not None:
        call_start = func_start

    enclosing = _enclosing_locals(scope)
    use = _analyze_anon(anon, enclosing)
    if use.uses_dots:
        details["skip_reason"] = "anonymous function uses ..."
        return details
    if use.has_goto:
        details["skip_reason"] = "anonymous function uses goto"
        return details

    orig_params = _anon_params(anon)
    if orig_params is None:
        details["skip_reason"] = "anonymous function has unusable params"
        return details

    named = _named_func_scopes(scope)
    caller = "_chunk"
    if named:
        display = _func_display_name(named[0].node) or named[0].name
        caller = display or "_chunk"

    insert_node = named[-1].node if named else None
    if insert_node is not None:
        ins_start, _ = node_span(insert_node)
        if ins_start is None:
            details["skip_reason"] = "cannot find chunk-level insert point"
            return details
        insert_char = _line_start(source, ins_start)
    else:
        insert_char = _line_start(source, call_start)

    helper_name = pick_helper_name(caller, taken)
    if helper_name is None:
        details["skip_reason"] = "cannot allocate helper name"
        return details

    ctx_kind, n_targets, ok_name, assign_node = _assign_context(call, parent)
    extra = args[1:]
    extra_src = _extra_args_src(source, extra)
    if extra_src is None:
        details["skip_reason"] = "cannot slice extra pcall args"
        return details

    writes = set(use.writes)
    reads = set(use.reads) - writes
    assign_info = _single_capture_assign(anon, writes) if writes else None

    if writes and assign_info is None:
        details["skip_reason"] = "writes captured locals (not a single x = expr)"
        details["writes"] = sorted(writes)
        return details

    if writes and assign_info is not None:
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
        if kind == "xpcall" and leftover:
            details["skip_reason"] = "xpcall cannot take extra args (Lua 5.1)"
            return details

        helper_params = list(orig_params) + leftover
        indent = _line_indent(source, insert_char)
        body_indent = _line_indent(source, fn_start)
        if not body_indent:
            # one step in from the helper indent; default tab if helper is flush
            inner = indent + "\t" if indent else "\t"
        else:
            # keep the original body's indent (already inside the anon)
            inner = body_indent
            # if the original was a one-liner `function() x = e end`, body
            # indent equals the helper line indent; step in one tab
            if inner == indent or inner == _line_indent(source, call_start):
                inner = indent + "\t" if indent else "\t"

        helper_text = (
            f"{indent}local function {helper_name}({', '.join(helper_params)})\n"
            f"{inner}return {rhs_src}\n"
            f"{indent}end\n"
        )

        call_args = [helper_name]
        if extra_src:
            call_args.append(extra_src)
        call_args.extend(leftover)
        new_call = f"{kind}({', '.join(call_args)})"

        stmt_indent = _line_indent(source, call_start)
        val_name = pick_temp(f"_{assigned}", taken)
        if ctx_kind == "stmt":
            ok_temp = pick_temp("_ok", taken)
            replace_start, replace_end = call_start, call_end
            replace_text = (
                f"local {ok_temp}, {val_name} = {new_call}\n"
                f"{stmt_indent}if {ok_temp} then {assigned} = {val_name} end"
            )
        else:
            # keep existing ok target; widen to two results
            ok_temp = ok_name or pick_temp("_ok", taken)
            a_start, a_end = node_span(assign_node)
            if a_start is None or a_end is None:
                details["skip_reason"] = "cannot slice parent assign"
                return details
            local_kw = "local " if isinstance(assign_node, LocalAssign) else ""
            replace_start, replace_end = a_start, a_end
            replace_text = (
                f"{local_kw}{ok_temp}, {val_name} = {new_call}\n"
                f"{stmt_indent}if {ok_temp} then {assigned} = {val_name} end"
            )

        details.update({
            "safe": True,
            "rewrite_kind": "assign",
            "helper_name": helper_name,
            "captures": leftover,
            "helper_text": helper_text,
            "insert_char": insert_char,
            "replace_start": replace_start,
            "replace_end": replace_end,
            "replace_text": replace_text,
            "caller": caller,
        })
        return details

    # read-only hoist
    captures = sorted(reads)
    if kind == "xpcall" and captures:
        details["skip_reason"] = "xpcall cannot take extra args (Lua 5.1)"
        details["captures"] = captures
        return details

    func_src = source[fn_start:fn_end]
    helper_params = list(orig_params) + captures
    helper_body = _replace_func_header(func_src, helper_name, helper_params)
    if helper_body is None:
        details["skip_reason"] = "cannot rewrite function header"
        return details

    indent = _line_indent(source, insert_char)
    # Header replace keeps the original `function` indent (often deeper than
    # chunk). Re-indent the whole helper to the insert line's indent.
    helper_text = _reindent_helper(helper_body, indent) + "\n"

    call_args = [helper_name]
    if extra_src:
        call_args.append(extra_src)
    call_args.extend(captures)
    replace_text = f"{kind}({', '.join(call_args)})"

    details.update({
        "safe": True,
        "rewrite_kind": "hoist",
        "helper_name": helper_name,
        "captures": captures,
        "helper_text": helper_text,
        "insert_char": insert_char,
        "replace_start": call_start,
        "replace_end": call_end,
        "replace_text": replace_text,
        "caller": caller,
    })
    return details


def _reindent_helper(helper: str, indent: str) -> str:
    """Force helper to `indent` at the `local function` / `end` lines.

    Inner lines keep relative indent vs the original `function` line.
    """
    lines = helper.split("\n")
    if not lines:
        return helper
    first = lines[0]
    orig_indent = first[: len(first) - len(first.lstrip(" \t"))]
    out = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue
        if orig_indent and line.startswith(orig_indent):
            line = line[len(orig_indent):]
        out.append(indent + line)
    return "\n".join(out)


def analyze_tree(analyzer) -> None:
    """Emit pcall_anon_hoist / pcall_anon_skip findings onto analyzer."""
    tree = getattr(analyzer, "_ast_tree", None)
    if tree is None:
        return
    source = analyzer.source
    parents = _parent_map(tree)
    taken = _all_taken_names(getattr(analyzer, "scopes", None))
    from models import Finding

    for call_info in getattr(analyzer, "calls", []) or []:
        call = getattr(call_info, "node", None)
        if call is None or _call_kind(call) is None:
            continue
        args = getattr(call, "args", None) or []
        if not args or not isinstance(args[0], AnonymousFunction):
            continue

        parent = parents.get(id(call))
        details = classify_pcall(
            call,
            source,
            getattr(call_info, "scope", None),
            parent,
            taken,
        )
        line = getattr(call_info, "line", 0) or 0
        src_line = ""
        if hasattr(analyzer, "_get_source_line"):
            src_line = analyzer._get_source_line(line) or ""

        if details.get("safe"):
            kind = details.get("rewrite_kind")
            helper = details.get("helper_name")
            caps = details.get("captures") or []
            cap_s = f" captures=[{', '.join(caps)}]" if caps else ""
            msg = f"pcall anon -> {helper}({', '.join(caps)}) [{kind}]{cap_s}"
            analyzer.findings.append(Finding(
                pattern_name="pcall_anon_hoist",
                severity="GREEN",
                line_num=line,
                message=msg,
                details=details,
                source_line=src_line,
            ))
        else:
            reason = details.get("skip_reason") or "unsafe"
            analyzer.findings.append(Finding(
                pattern_name="pcall_anon_skip",
                severity="RED",
                line_num=line,
                message=f"pcall anon not hoisted: {reason}",
                details=details,
                source_line=src_line,
            ))
