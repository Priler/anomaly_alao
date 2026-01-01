"""
AST-based Lua code analyzer implemented with https://pypi.org/project/luaparser/
"""

from luaparser import ast
from luaparser.astnodes import (
    Node, Chunk, Block,
    Function, LocalFunction, Method,
    Assign, LocalAssign,
    While, Repeat, Fornum, Forin,
    If, ElseIf,
    Call, Invoke,
    Index, Name, String, Number, Nil, TrueExpr, FalseExpr,
    Table, Field,
    Concat, AddOp, SubOp, MultOp, FloatDivOp, ModOp, ExpoOp,
    Return, Break,
    UMinusOp, UBNotOp, ULNotOp, ULengthOP,
    AndLoOp, OrLoOp,
    LessThanOp, GreaterThanOp, LessOrEqThanOp, GreaterOrEqThanOp, EqToOp, NotEqToOp,
    SemiColon, Comment,
)
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

from models import Finding


# Hot callbacks that run frequently
HOT_CALLBACKS = frozenset({
    'actor_on_update', 'actor_on_first_update',
    'npc_on_update', 'monster_on_update',
    'on_key_press', 'on_key_release', 'on_key_hold',
    'actor_on_weapon_fired', 'actor_on_hud_animation_end',
    'on_before_hit', 'on_hit',
    'physic_object_on_hit_callback',
    'npc_on_before_hit', 'monster_on_before_hit',
    'npc_on_hit_callback', 'monster_on_hit_callback',
    'actor_on_feel_touch',
    'actor_on_item_take', 'actor_on_item_drop',
    'actor_on_item_use',
})

# Bare globals that benefit from caching
CACHEABLE_BARE_GLOBALS = frozenset({
    'pairs', 'ipairs', 'next', 'type', 'tostring', 'tonumber',
    'unpack', 'select', 'rawget', 'rawset',
})

# these are less beneficial to cache (error handling, output)
BARE_GLOBALS_UNSAFE_TO_CACHE = frozenset({
    'pcall', 'xpcall', 'error', 'assert', 'print',
})

# Module functions that benefit from caching
CACHEABLE_MODULE_FUNCS = {
    'math': frozenset({
        'floor', 'ceil', 'abs', 'min', 'max', 'sqrt', 'sin', 'cos', 'tan',
        'random', 'pow', 'log', 'exp', 'atan2', 'atan', 'asin', 'acos',
        'deg', 'rad', 'fmod', 'modf', 'huge',
    }),
    'string': frozenset({
        'find', 'sub', 'gsub', 'match', 'gmatch', 'format',
        'lower', 'upper', 'len', 'rep', 'byte', 'char', 'reverse',
    }),
    'table': frozenset({
        'insert', 'remove', 'concat', 'sort', 'getn', 'unpack',
    }),
    'bit': frozenset({
        'band', 'bor', 'bxor', 'bnot', 'lshift', 'rshift', 'arshift', 'rol', 'ror',
    }),
}

# Debug/logging function patterns
DEBUG_FUNCTIONS = frozenset({
    'print', 'printf', 'printe', 'printd', 'log',
    'log1', 'log2', 'log3',
    'DebugLog', 'debug_log', 'trace', 'dump',
})

# functions that have direct replacement patterns (not cached)
DIRECT_REPLACEMENT_FUNCS = frozenset({
    'table.insert', 'table.getn', 'string.len',
})


@dataclass
class Scope:
    """Represents a variable scope (function, loop, block)."""
    name: str
    start_line: int
    end_line: int = -1
    parent: Optional['Scope'] = None
    scope_type: str = 'block'  # 'function', 'loop', 'block'
    is_hot_callback: bool = False

    # variables declared in this scope
    locals: Set[str] = field(default_factory=set)

    # cached globals in this scope
    cached_globals: Set[str] = field(default_factory=set)

    def __hash__(self):
        # use object's actual id for hashing
        return id(self)

    def __eq__(self, other):
        if isinstance(other, Scope):
            return self is other
        return False


@dataclass
class CallInfo:
    """Information about a function call."""
    full_name: str          # "table.insert", "db.actor", "pairs"
    module: Optional[str]   # "table", "db", None
    func: str               # "insert", "actor", "pairs"
    args: List[Any]         # AST nodes of arguments
    line: int
    node: Node
    scope: Scope
    in_loop: bool = False
    loop_depth: int = 0


@dataclass
class AssignInfo:
    """Information about an assignment."""
    target: str             # variable name
    value_type: str         # 'call', 'index', 'concat', 'literal', 'other'
    value_repr: str         # string representation
    line: int
    node: Node
    scope: Scope
    is_local: bool = False
    in_loop: bool = False


@dataclass
class ConcatInfo:
    """Information about string concatenation."""
    target: Optional[str]   # variable being assigned to
    left_var: Optional[str]  # left operand if it's a variable
    line: int
    scope: Scope
    in_loop: bool = False
    loop_depth: int = 0


class ASTAnalyzer:
    """AST-based Lua code analyzer."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset analyzer state."""
        self.findings: List[Finding] = []
        self.scopes: List[Scope] = []
        self.current_scope: Optional[Scope] = None
        self.global_scope: Optional[Scope] = None

        self.calls: List[CallInfo] = []
        self.assigns: List[AssignInfo] = []
        self.concats: List[ConcatInfo] = []
        self.global_writes: List[Tuple[str, int]] = []

        self.source_lines: List[str] = []
        self.source: str = ""
        self.file_path: Optional[Path] = None

        self.loop_depth: int = 0
        self.function_depth: int = 0

    def analyze_file(self, file_path: Path) -> List[Finding]:
        """Analyze a Lua file and return findings."""
        self.reset()
        self.file_path = file_path

        try:
            self.source = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return []

        self.source_lines = self.source.splitlines()

        try:
            tree = ast.parse(self.source)
        except Exception:
            # parse error, skip
            return []

        # create global scope
        self.global_scope = Scope(
            name='<global>',
            start_line=1,
            end_line=len(self.source_lines),
            scope_type='global',
        )
        self.current_scope = self.global_scope
        self.scopes.append(self.global_scope)

        # walk AST
        self._visit(tree)

        # analyze collected data
        self._analyze_patterns()

        return self.findings

    def _get_line(self, node: Node) -> int:
        """Extract line number from node."""
        ft = getattr(node, 'first_token', None)
        if ft:
            s = str(ft)
            if ',' in s:
                parts = s.rsplit(',', 1)
                if len(parts) == 2:
                    line_col = parts[1].rstrip(']')
                    if ':' in line_col:
                        try:
                            return int(line_col.split(':')[0])
                        except ValueError:
                            pass
        return 0

    def _get_node_source(self, node: Node) -> str:
        """Get source text for a node (approximate)."""
        line = self._get_line(node)
        if 0 < line <= len(self.source_lines):
            return self.source_lines[line - 1].strip()
        return ""

    def _node_to_string(self, node: Node) -> str:
        """Convert an AST node to its string representation."""
        if isinstance(node, Name):
            return node.id
        elif isinstance(node, Number):
            return str(node.n)
        elif isinstance(node, String):
            s = node.s
            if isinstance(s, bytes):
                s = s.decode('utf-8', errors='replace')
            return f'"{s}"'
        elif isinstance(node, (TrueExpr,)):
            return "true"
        elif isinstance(node, (FalseExpr,)):
            return "false"
        elif isinstance(node, (Nil,)):
            return "nil"
        elif isinstance(node, Index):
            value = self._node_to_string(node.value)
            idx = self._node_to_string(node.idx)
            # determine bracket vs dot notation:
            # - dot notation (t.field): idx.first_token is None
            # - bracket notation (t[key]): idx.first_token has a value
            idx_token = getattr(node.idx, 'first_token', None)
            if idx_token is not None and str(idx_token) != 'None':
                # bracket notation: t[key]
                return f"{value}[{idx}]"
            else:
                # dot notation: t.field
                return f"{value}.{idx}"
        elif isinstance(node, Call):
            func = self._node_to_string(node.func)
            args = ", ".join(self._node_to_string(a) for a in node.args)
            return f"{func}({args})"
        elif isinstance(node, Invoke):
            source = self._node_to_string(node.source)
            func = self._node_to_string(node.func)
            args = ", ".join(self._node_to_string(a) for a in node.args)
            return f"{source}:{func}({args})"
        elif isinstance(node, ULengthOP):
            return f"#{self._node_to_string(node.operand)}"
        elif isinstance(node, Concat):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} .. {right}"
        elif isinstance(node, Table):
            return "{...}"
        else:
            return f"<{type(node).__name__}>"

    def _get_call_name(self, node: Call) -> Tuple[Optional[str], str, str]:
        """Get module, function, and full name from a Call node."""
        func = node.func

        if isinstance(func, Name):
            # bare function: pairs(), time_global()
            return None, func.id, func.id
        elif isinstance(func, Index):
            # module.func: table.insert(), db.actor
            if isinstance(func.value, Name) and isinstance(func.idx, Name):
                module = func.value.id
                fn = func.idx.id
                return module, fn, f"{module}.{fn}"

        return None, "", ""

    def _enter_scope(self, name: str, line: int, scope_type: str = 'block', is_hot: bool = False):
        """Enter a new scope."""
        new_scope = Scope(
            name=name,
            start_line=line,
            parent=self.current_scope,
            scope_type=scope_type,
            is_hot_callback=is_hot or (self.current_scope and self.current_scope.is_hot_callback),
        )

        # inherit cached globals from parent
        if self.current_scope:
            new_scope.cached_globals = set(self.current_scope.cached_globals)

        self.scopes.append(new_scope)
        self.current_scope = new_scope
        return new_scope

    def _exit_scope(self, end_line: int):
        """Exit current scope."""
        if self.current_scope:
            self.current_scope.end_line = end_line
            self.current_scope = self.current_scope.parent

    def _is_cached(self, name: str) -> bool:
        """Check if a global is cached in current scope chain."""
        scope = self.current_scope
        while scope:
            if name in scope.cached_globals or name in scope.locals:
                return True
            scope = scope.parent
        return False

    def _visit(self, node: Node):
        """Visit a node and dispatch to specific handler."""
        if node is None:
            return

        handler = getattr(self, f'_visit_{type(node).__name__}', None)
        if handler:
            handler(node)
        else:
            self._visit_children(node)

    def _visit_children(self, node: Node):
        """Visit all children of a node."""
        try:
            node_dict = vars(node) if hasattr(node, '__dict__') else {}
        except TypeError:
            node_dict = {}

        for key, value in node_dict.items():
            if key.startswith('_'):
                continue
            if isinstance(value, Node):
                self._visit(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Node):
                        self._visit(item)

    def _visit_Chunk(self, node: Chunk):
        self._visit(node.body)

    def _visit_Block(self, node: Block):
        for stmt in node.body:
            self._visit(stmt)

    def _visit_Function(self, node: Function):
        """Handle global function definition."""
        line = self._get_line(node)
        func_name = self._node_to_string(node.name) if node.name else '<anon>'

        is_hot = func_name in HOT_CALLBACKS

        self.function_depth += 1
        self._enter_scope(func_name, line, 'function', is_hot)

        # register parameters as locals
        if hasattr(node, 'args') and node.args:
            for arg in node.args:
                if isinstance(arg, Name):
                    self.current_scope.locals.add(arg.id)

        self._visit(node.body)

        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.function_depth -= 1

    def _visit_LocalFunction(self, node: LocalFunction):
        """Handle local function definition."""
        line = self._get_line(node)
        func_name = node.name.id if isinstance(node.name, Name) else '<anon>'

        # register function name in parent scope
        if self.current_scope:
            self.current_scope.locals.add(func_name)

        is_hot = func_name in HOT_CALLBACKS

        self.function_depth += 1
        self._enter_scope(func_name, line, 'function', is_hot)

        if hasattr(node, 'args') and node.args:
            for arg in node.args:
                if isinstance(arg, Name):
                    self.current_scope.locals.add(arg.id)

        self._visit(node.body)

        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.function_depth -= 1

    def _visit_Method(self, node: Method):
        """Handle method definition."""
        line = self._get_line(node)

        # get method name
        if isinstance(node.name, Index):
            func_name = self._node_to_string(node.name)
        else:
            func_name = self._node_to_string(node.name) if node.name else '<method>'

        is_hot = func_name in HOT_CALLBACKS

        self.function_depth += 1
        self._enter_scope(func_name, line, 'function', is_hot)

        # 'self' is implicit first param
        self.current_scope.locals.add('self')

        if hasattr(node, 'args') and node.args:
            for arg in node.args:
                if isinstance(arg, Name):
                    self.current_scope.locals.add(arg.id)

        self._visit(node.body)

        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.function_depth -= 1

    def _get_end_line(self, node: Node) -> int:
        """Try to get end line of a node."""
        lt = getattr(node, 'last_token', None)
        if lt:
            s = str(lt)
            if ',' in s:
                parts = s.rsplit(',', 1)
                if len(parts) == 2:
                    line_col = parts[1].rstrip(']')
                    if ':' in line_col:
                        try:
                            return int(line_col.split(':')[0])
                        except ValueError:
                            pass
        return self._get_line(node)

    def _visit_Forin(self, node: Forin):
        """Handle for-in loop."""
        line = self._get_line(node)

        # visit iterator expression first (outside loop scope)
        for iter_expr in node.iter:
            self._visit(iter_expr)

        self.loop_depth += 1
        self._enter_scope('<forin>', line, 'loop')

        # loop variables are local to loop
        for target in node.targets:
            if isinstance(target, Name):
                self.current_scope.locals.add(target.id)

        self._visit(node.body)

        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.loop_depth -= 1

    def _visit_Fornum(self, node: Fornum):
        """Handle numeric for loop."""
        line = self._get_line(node)

        # visit range expressions first
        self._visit(node.start)
        self._visit(node.stop)
        if node.step:
            self._visit(node.step)

        self.loop_depth += 1
        self._enter_scope('<fornum>', line, 'loop')

        if isinstance(node.target, Name):
            self.current_scope.locals.add(node.target.id)

        self._visit(node.body)

        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.loop_depth -= 1

    def _visit_While(self, node: While):
        """Handle while loop."""
        line = self._get_line(node)

        self._visit(node.test)

        self.loop_depth += 1
        self._enter_scope('<while>', line, 'loop')
        self._visit(node.body)
        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.loop_depth -= 1

    def _visit_Repeat(self, node: Repeat):
        """Handle repeat-until loop."""
        line = self._get_line(node)

        self.loop_depth += 1
        self._enter_scope('<repeat>', line, 'loop')
        self._visit(node.body)
        self._visit(node.test)
        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.loop_depth -= 1

    def _visit_If(self, node: If):
        """Handle if statement."""
        self._visit(node.test)
        self._visit(node.body)
        if node.orelse:
            self._visit(node.orelse)

    def _visit_ElseIf(self, node: ElseIf):
        """Handle elseif clause."""
        self._visit(node.test)
        self._visit(node.body)
        if node.orelse:
            self._visit(node.orelse)

    def _visit_LocalAssign(self, node: LocalAssign):
        """Handle local assignment."""
        line = self._get_line(node)

        # register targets as locals
        for target in node.targets:
            if isinstance(target, Name):
                self.current_scope.locals.add(target.id)

        # check for caching pattern: local xyz = module.func
        if len(node.targets) == 1 and len(node.values) == 1:
            target = node.targets[0]
            value = node.values[0]

            if isinstance(target, Name):
                target_name = target.id

                # check if caching a module.func
                if isinstance(value, Index):
                    if isinstance(value.value, Name) and isinstance(value.idx, Name):
                        module = value.value.id
                        func = value.idx.id
                        full_name = f"{module}.{func}"

                        if module in CACHEABLE_MODULE_FUNCS:
                            self.current_scope.cached_globals.add(full_name)

                # check if caching a bare global
                elif isinstance(value, Name):
                    if value.id in CACHEABLE_BARE_GLOBALS:
                        self.current_scope.cached_globals.add(value.id)

                # record assignment info
                self._record_assignment(target_name, value, line, is_local=True)

        # visit values
        for value in node.values:
            self._visit(value)

    def _visit_Assign(self, node: Assign):
        """Handle assignment."""
        line = self._get_line(node)

        # check for global writes
        for target in node.targets:
            if isinstance(target, Name):
                target_name = target.id
                # it's a global write if not in any scope's locals
                if not self._is_in_locals(target_name):
                    self.global_writes.append((target_name, line))

                if len(node.values) == 1:
                    self._record_assignment(target_name, node.values[0], line, is_local=False)

        # visit values
        for value in node.values:
            self._visit(value)

    def _is_in_locals(self, name: str) -> bool:
        """Check if name is in any scope's locals."""
        scope = self.current_scope
        while scope:
            if name in scope.locals:
                return True
            scope = scope.parent
        return False

    def _record_assignment(self, target: str, value: Node, line: int, is_local: bool):
        """Record an assignment for analysis."""
        if isinstance(value, Call):
            value_type = 'call'
            value_repr = self._node_to_string(value)
        elif isinstance(value, Index):
            value_type = 'index'
            value_repr = self._node_to_string(value)
        elif isinstance(value, Concat):
            value_type = 'concat'
            value_repr = self._node_to_string(value)

            # record concat info
            left_var = None
            if isinstance(value.left, Name):
                left_var = value.left.id

            self.concats.append(ConcatInfo(
                target=target,
                left_var=left_var,
                line=line,
                scope=self.current_scope,
                in_loop=self.loop_depth > 0,
                loop_depth=self.loop_depth,
            ))
        elif isinstance(value, (Number, String, TrueExpr, FalseExpr, Nil)):
            value_type = 'literal'
            value_repr = self._node_to_string(value)
        else:
            value_type = 'other'
            value_repr = self._node_to_string(value)

        self.assigns.append(AssignInfo(
            target=target,
            value_type=value_type,
            value_repr=value_repr,
            line=line,
            node=value,
            scope=self.current_scope,
            is_local=is_local,
            in_loop=self.loop_depth > 0,
        ))

    def _visit_Call(self, node: Call):
        """Handle function call."""
        line = self._get_line(node)
        module, func, full_name = self._get_call_name(node)

        if full_name:
            self.calls.append(CallInfo(
                full_name=full_name,
                module=module,
                func=func,
                args=node.args,
                line=line,
                node=node,
                scope=self.current_scope,
                in_loop=self.loop_depth > 0,
                loop_depth=self.loop_depth,
            ))

        # visit children
        self._visit(node.func)
        for arg in node.args:
            self._visit(arg)

    def _visit_Invoke(self, node: Invoke):
        """Handle method call (obj:method())."""
        line = self._get_line(node)

        # record as call
        source = self._node_to_string(node.source)
        func = node.func.id if isinstance(node.func, Name) else self._node_to_string(node.func)
        full_name = f"{source}:{func}"

        self.calls.append(CallInfo(
            full_name=full_name,
            module=source,
            func=func,
            args=node.args,
            line=line,
            node=node,
            scope=self.current_scope,
            in_loop=self.loop_depth > 0,
            loop_depth=self.loop_depth,
        ))

        self._visit(node.source)
        for arg in node.args:
            self._visit(arg)

    def _visit_Concat(self, node: Concat):
        """Handle concatenation operator."""
        line = self._get_line(node)

        left_var = None
        if isinstance(node.left, Name):
            left_var = node.left.id

        # only interesting if we're in a loop
        if self.loop_depth > 0:
            self.concats.append(ConcatInfo(
                target=None,  # no assignment context here
                left_var=left_var,
                line=line,
                scope=self.current_scope,
                in_loop=True,
                loop_depth=self.loop_depth,
            ))

        self._visit(node.left)
        self._visit(node.right)

    # visitor pass-through for other nodes
    def _visit_Index(self, node: Index):
        self._visit(node.value)
        self._visit(node.idx)

    def _visit_Table(self, node: Table):
        for field in node.fields:
            self._visit(field)

    def _visit_Field(self, node: Field):
        if node.key:
            self._visit(node.key)
        self._visit(node.value)

    def _visit_Return(self, node: Return):
        for val in node.values:
            self._visit(val)

    # binary ops
    def _visit_AddOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_SubOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_MultOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_FloatDivOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_ModOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_ExpoOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_AndLoOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_OrLoOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_LessThanOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_GreaterThanOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_LessOrEqThanOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_GreaterOrEqThanOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_EqToOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_NotEqToOp(self, node): self._visit(node.left); self._visit(node.right)

    # unary ops
    def _visit_UMinusOp(self, node): self._visit(node.operand)
    def _visit_UBNotOp(self, node): self._visit(node.operand)
    def _visit_ULNotOp(self, node): self._visit(node.operand)
    def _visit_ULengthOP(self, node): self._visit(node.operand)

    # terminal nodes - no children
    def _visit_Name(self, node): pass
    def _visit_Number(self, node): pass
    def _visit_String(self, node): pass
    def _visit_Nil(self, node): pass
    def _visit_TrueExpr(self, node): pass
    def _visit_FalseExpr(self, node): pass
    def _visit_SemiColon(self, node): pass
    def _visit_Comment(self, node): pass
    def _visit_Break(self, node): pass


    # PATTERN ANALYSIS

    def _analyze_patterns(self):
        """Analyze collected data and generate findings."""
        self._analyze_table_insert()
        self._analyze_deprecated_funcs()
        self._analyze_math_pow()
        self._analyze_uncached_globals()
        self._analyze_repeated_calls_in_scope()
        self._analyze_string_concat_in_loop()
        self._analyze_debug_statements()
        self._analyze_global_writes()

    def _analyze_table_insert(self):
        """Find table.insert(t, v) that can be t[#t+1] = v."""
        for call in self.calls:
            if call.full_name == 'table.insert' and len(call.args) == 2:
                # 2-arg form: table.insert(t, v)
                table_name = self._node_to_string(call.args[0])
                value = self._node_to_string(call.args[1])

                self.findings.append(Finding(
                    pattern_name='table_insert_append',
                    severity='GREEN',
                    line_num=call.line,
                    message=f'table.insert({table_name}, v) -> {table_name}[#{table_name}+1] = v',
                    details={
                        'table': table_name,
                        'value': value,
                        'full_match': f'table.insert({table_name}, {value})',
                        'node': call.node,
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _analyze_deprecated_funcs(self):
        """Find deprecated functions: table.getn, string.len."""
        for call in self.calls:
            if call.full_name == 'table.getn' and len(call.args) == 1:
                arg = self._node_to_string(call.args[0])
                self.findings.append(Finding(
                    pattern_name='table_getn',
                    severity='GREEN',
                    line_num=call.line,
                    message=f'table.getn({arg}) -> #{arg}',
                    details={
                        'table': arg,
                        'full_match': f'table.getn({arg})',
                        'node': call.node,
                    },
                    source_line=self._get_source_line(call.line),
                ))

            elif call.full_name == 'string.len' and len(call.args) == 1:
                arg = self._node_to_string(call.args[0])
                self.findings.append(Finding(
                    pattern_name='string_len',
                    severity='GREEN',
                    line_num=call.line,
                    message=f'string.len({arg}) -> #{arg}',
                    details={
                        'string': arg,
                        'full_match': f'string.len({arg})',
                        'node': call.node,
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _analyze_math_pow(self):
        """Find math.pow that can be simplified."""
        for call in self.calls:
            if call.full_name == 'math.pow' and len(call.args) == 2:
                base = self._node_to_string(call.args[0])
                exp_node = call.args[1]

                # check for simple cases
                if isinstance(exp_node, Number):
                    exp = exp_node.n
                    full_match = f'math.pow({base}, {exp})'

                    if exp == 0.5:
                        self.findings.append(Finding(
                            pattern_name='math_pow_simple',
                            severity='GREEN',
                            line_num=call.line,
                            message=f'{full_match} -> math.sqrt({base})',
                            details={
                                'base': base,
                                'exponent': exp,
                                'type': 'sqrt',
                                'is_simple': True,
                                'full_match': full_match,
                                'node': call.node,
                            },
                            source_line=self._get_source_line(call.line),
                        ))
                    elif exp in (2, 3, 4) and self._is_simple_expr(call.args[0]):
                        replacement = '*'.join([base] * int(exp))
                        self.findings.append(Finding(
                            pattern_name='math_pow_simple',
                            severity='GREEN',
                            line_num=call.line,
                            message=f'{full_match} -> {replacement}',
                            details={
                                'base': base,
                                'exponent': int(exp),
                                'type': 'power',
                                'is_simple': True,
                                'full_match': full_match,
                                'node': call.node,
                            },
                            source_line=self._get_source_line(call.line),
                        ))

    def _is_simple_expr(self, node: Node) -> bool:
        """Check if node is a simple expression (safe to repeat)."""
        return isinstance(node, (Name, Number))

    def _analyze_uncached_globals(self):
        """Find frequently used globals that should be cached."""
        # count calls by full_name, grouped by function scope
        scope_calls: Dict[Scope, Dict[str, List[CallInfo]]] = defaultdict(lambda: defaultdict(list))

        for call in self.calls:
            # find enclosing function scope
            func_scope = self._find_function_scope(call.scope)
            if func_scope:
                scope_calls[func_scope][call.full_name].append(call)

        # check each function
        for func_scope, calls_by_name in scope_calls.items():
            globals_to_cache = {}

            for name, calls in calls_by_name.items():
                # skip if already cached or has direct replacement
                if name in DIRECT_REPLACEMENT_FUNCS:
                    continue
                if name in func_scope.cached_globals:
                    continue

                # check if it's a cacheable global
                is_bare = name in CACHEABLE_BARE_GLOBALS
                is_module_func = False

                if '.' in name:
                    module, func = name.split('.', 1)
                    if module in CACHEABLE_MODULE_FUNCS and func in CACHEABLE_MODULE_FUNCS[module]:
                        is_module_func = True

                if not is_bare and not is_module_func:
                    continue

                # threshold: 3+ uses, or 2+ if in hot callback
                threshold = 2 if func_scope.is_hot_callback else 3
                if len(calls) >= threshold:
                    globals_to_cache[name] = calls

            if globals_to_cache:
                # skip global scope - only cache inside actual functions
                if func_scope.name == '<global>' or func_scope.scope_type == 'global':
                    continue

                # create summary finding for this function
                example_lines = []
                for name, calls in list(globals_to_cache.items())[:5]:
                    for c in calls[:2]:
                        example_lines.append(f"L{c.line}: {name}")

                self.findings.append(Finding(
                    pattern_name='uncached_globals_summary',
                    severity='GREEN',
                    line_num=func_scope.start_line,
                    message=f'Cache {len(globals_to_cache)} globals in {func_scope.name}',
                    details={
                        'globals': {n: len(c) for n, c in globals_to_cache.items()},
                        'globals_info': globals_to_cache,  # name -> list of CallInfo with nodes
                        'function': func_scope.name,
                        'is_hot': func_scope.is_hot_callback,
                        'scope': func_scope,
                    },
                    source_line='\n'.join(example_lines),
                ))

    def _find_function_scope(self, scope: Scope) -> Optional[Scope]:
        """Find the enclosing function scope."""
        while scope:
            if scope.scope_type == 'function':
                return scope
            scope = scope.parent
        return self.global_scope

    def _analyze_repeated_calls_in_scope(self):
        """Find repeated expensive calls within function scope."""
        # expensive calls to track
        expensive_calls = {'db.actor', 'time_global', 'alife', 'system_ini', 'level.object_by_id'}

        # group by function scope
        scope_calls: Dict[Scope, Dict[str, List[CallInfo]]] = defaultdict(lambda: defaultdict(list))

        for call in self.calls:
            if call.full_name in expensive_calls:
                func_scope = self._find_function_scope(call.scope)
                if func_scope:
                    scope_calls[func_scope][call.full_name].append(call)

            # special: track :section() calls on objects
            if call.func == 'section' and ':' in call.full_name:
                func_scope = self._find_function_scope(call.scope)
                if func_scope:
                    key = f"{call.full_name}()"
                    scope_calls[func_scope][key].append(call)

        for func_scope, calls_by_name in scope_calls.items():
            for name, calls in calls_by_name.items():
                threshold = 2 if func_scope.is_hot_callback else 3

                if len(calls) >= threshold:
                    # suggest caching
                    severity = 'GREEN'

                    if name == 'db.actor':
                        suggestion = 'local actor = db.actor'
                    elif name == 'time_global':
                        suggestion = 'local tg = time_global()'
                    elif name == 'alife':
                        suggestion = 'local sim = alife()'
                    elif name == 'system_ini':
                        suggestion = 'local ini = system_ini()'
                    elif name == 'level.object_by_id':
                        # check if all calls use the same argument
                        args_strs = []
                        for c in calls:
                            if c.args:
                                args_strs.append(self._node_to_string(c.args[0]))
                            else:
                                args_strs.append('')

                        all_same = len(set(args_strs)) == 1 and args_strs[0]

                        if all_same:
                            arg_str = args_strs[0]
                            suggestion = f'local obj = level.object_by_id({arg_str})'
                            severity = 'GREEN'
                        else:
                            suggestion = 'local obj = level.object_by_id(id)'
                            severity = 'YELLOW'  # different args, needs review
                    else:
                        suggestion = f'Cache {name} result'

                    self.findings.append(Finding(
                        pattern_name=f'repeated_{name.replace(".", "_").replace(":", "_")}',
                        severity=severity,
                        line_num=calls[0].line,
                        message=f'{name} called {len(calls)}x in {func_scope.name}',
                        details={
                            'count': len(calls),
                            'function': func_scope.name,
                            'is_hot': func_scope.is_hot_callback,
                            'suggestion': suggestion,
                            'lines': [c.line for c in calls],
                            'calls': calls,  # list of CallInfo with nodes
                            'scope': func_scope,
                        },
                        source_line=suggestion,
                    ))

    def _analyze_string_concat_in_loop(self):
        """Find string concatenation patterns in loops."""
        # find self-concatenation: s = s .. x
        loop_concats: Dict[Tuple[Scope, str], List[ConcatInfo]] = defaultdict(list)

        for concat in self.concats:
            if concat.in_loop and concat.target and concat.left_var:
                if concat.target == concat.left_var:
                    # self concat: s = s .. x
                    key = (concat.scope, concat.target)
                    loop_concats[key].append(concat)

        for (scope, var), concats in loop_concats.items():
            if len(concats) >= 1:
                self.findings.append(Finding(
                    pattern_name='string_concat_in_loop',
                    severity='YELLOW',
                    line_num=concats[0].line,
                    message=f'String concat in loop: {var} = {var} .. x',
                    details={
                        'variable': var,
                        'count': len(concats),
                        'loop_depth': concats[0].loop_depth,
                        'suggestion': 'Use table.insert() + table.concat()',
                    },
                    source_line=self._get_source_line(concats[0].line),
                ))

    def _analyze_debug_statements(self):
        """Find debug/logging statements."""
        for call in self.calls:
            func_name = call.func
            # exclude math.log - it's mathematical logarithm, not logging
            if call.full_name and call.full_name.startswith('math.'):
                continue
            if func_name in DEBUG_FUNCTIONS:
                self.findings.append(Finding(
                    pattern_name='debug_statement',
                    severity='DEBUG',
                    line_num=call.line,
                    message=f'Debug call: {func_name}()',
                    details={
                        'function': func_name,
                        'node': call.node,
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _analyze_global_writes(self):
        """Track global variable writes."""
        for name, line in self.global_writes:
            # skip common patterns that are intentional
            if name.startswith('_') or name.isupper():
                continue

            self.findings.append(Finding(
                pattern_name='global_write',
                severity='RED',
                line_num=line,
                message=f'Global write: {name}',
                details={
                    'variable': name,
                },
                source_line=self._get_source_line(line),
            ))

    def _get_source_line(self, line_num: int) -> str:
        """Get source line by number."""
        if 0 < line_num <= len(self.source_lines):
            return self.source_lines[line_num - 1].rstrip()
        return ""


def analyze_file(file_path: Path) -> List[Finding]:
    """Convenience function to analyze a file."""
    analyzer = ASTAnalyzer()
    return analyzer.analyze_file(file_path)
