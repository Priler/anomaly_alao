"""
AST-based Lua code analyzer implemented with https://pypi.org/project/luaparser/
"""

from luaparser import ast
from luaparser.astnodes import (
    Node, Chunk, Block,
    Function, LocalFunction, Method, AnonymousFunction,
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
import sys
import io
import re

from models import Finding, detect_file_encoding


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

# Per-frame callbacks detection
PER_FRAME_CALLBACKS = frozenset({
    'actor_on_update',
    'npc_on_update',
    'monster_on_update',
    'physic_object_on_update',
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

# Index-style accesses that are worth caching when repeated (property reads,
# not function calls). `db.actor` and `db.storage` are property accesses that
# resolve through metatable lookups in the engine - caching the reference in
# a local once is a real win in hot code. The repeated-call analysis treats
# entries here equivalently to function calls in expensive_calls
EXPENSIVE_INDEXES = frozenset({'db.actor'})

# Functions/properties that can return nil - calling methods on these without
# nil checks can cause CTD (crash to desktop)
# Format: full_name -> description of when it returns nil
NIL_RETURNING_FUNCTIONS = {
    # level functions
    'level.object_by_id': 'object is offline or does not exist',
    'level.get_target_obj': 'nothing under crosshair',
    'level.get_target_element': 'nothing under crosshair',
    'level.get_target_pos': 'nothing under crosshair',
    'level.vertex_position': 'invalid vertex ID',
    'level.get_view_entity': 'no view entity set',
    
    # alife functions
    'alife': 'called from main menu or during loading',
    'alife().object': 'object does not exist in simulation',
    'alife():object': 'object does not exist in simulation',
    'alife().story_object': 'no object with that story_id',
    'alife():story_object': 'no object with that story_id',
    'alife().actor': 'actor not spawned yet',
    'alife():actor': 'actor not spawned yet',
    
    # game object methods that can return nil
    ':parent': 'object has no parent (not in inventory)',
    ':best_enemy': 'no enemy detected',
    ':best_item': 'no item of interest found',
    ':best_danger': 'no danger detected',
    ':best_weapon': 'no weapon available',
    ':best_cover': 'no cover available',
    ':active_item': 'no weapon/item currently equipped',
    ':active_detector': 'no detector currently active',
    ':object': 'item not in inventory or index out of bounds',
    ':get_enemy': 'no current enemy target',
    ':get_corpse': 'no corpse being investigated',
    ':get_current_outfit': 'no outfit equipped',
    ':item_in_slot': 'slot is empty',
    ':get_helicopter': 'not a helicopter or no helicopter',
    ':get_car': 'not in a vehicle',
    ':get_campfire': 'not a campfire zone',
    ':get_artefact': 'not an artefact',
    ':get_physics_shell': 'object has no physics shell',
    ':spawn_ini': 'no spawn ini defined',
    ':motivation_action_manager': 'not an NPC with action manager',
    ':get_current_holder': 'not in a vehicle/turret',
    ':get_old_holder': 'was not in a vehicle/turret',
    ':memory_position': 'object never seen',
    ':get_dest_enemy': 'no destination enemy',
    ':bone_id': 'bone name does not exist',
    ':bone_position': 'bone does not exist',
    ':bone_direction': 'bone does not exist',
    
    # common patterns
    'db.actor': 'called from main menu or during loading',
    'db.storage': 'storage not initialized',  # db.storage[id] can be nil
    
    # story object helpers
    'get_story_object': 'no object with that story_id',
    'get_object_by_name': 'object not found or offline',
}

# Method patterns that indicate the variable is being nil-checked
# These patterns mean the variable is safe to use after the check
NIL_CHECK_PATTERNS = {
    'if {var} then',
    'if {var} and',
    'if not {var} then return',
    'if not {var} then return end',
    'if {var} == nil then return',
    'if {var} == nil then return end', 
    'if {var} ~= nil then',
    '{var} and {var}:',
    '{var} and {var}.',
}

# Callback parameters that are guaranteed non-nil by the engine
# Format: (callback_name, param_index) - 0-indexed
SAFE_CALLBACK_PARAMS = {
    'actor_on_item_take': {0},      # item
    'actor_on_item_drop': {0},      # item
    'actor_on_item_use': {0},       # item
    'actor_on_trade': {0, 1},       # item, sell_buy  
    'npc_on_death_callback': {0, 1}, # npc, killer
    'monster_on_death_callback': {0, 1}, # monster, killer
    'npc_on_hit_callback': {0},     # npc
    'monster_on_hit_callback': {0}, # monster
    'on_before_hit': {0, 1, 2},     # obj, shit, bone_id
    'physic_object_on_hit_callback': {0}, # obj
    'actor_on_before_death': {0, 1}, # who, flags
    'save_state': {0},              # m_data
    'load_state': {0},              # m_data
}


@dataclass
class Scope:
    """Represents a variable scope (function, loop, block)."""
    name: str
    start_line: int
    end_line: int = -1
    parent: Optional['Scope'] = None
    scope_type: str = 'block'  # 'function', 'loop', 'block'
    is_hot_callback: bool = False

    # for function scopes: the AST node (Function/LocalFunction/Method/AnonymousFunction).
    # used by ast_transformer to locate the function body's actual start position
    # when inserting cache declarations, instead of fragile paren-counting on the
    # declaration line (which breaks for anonymous functions passed as call args, as an examply).
    node: Optional[Any] = None

    # variables declared in this scope
    locals: Set[str] = field(default_factory=set)

    # cached globals in this scope
    cached_globals: Set[str] = field(default_factory=set)

    # function aliases: local_name -> canonical_name (e.g., "tinsert" -> "table.insert")
    func_aliases: Dict[str, str] = field(default_factory=dict)

    def __hash__(self):
        # use object's actual id for hashing
        return id(self)

    def __eq__(self, other):
        if isinstance(other, Scope):
            return self is other
        return False


@dataclass
class CallInfo:
    """Information about a function call.

    `if_chain_path` is the sequence of (if_id, branch_index) tuples - one per
    enclosing if-chain - that must all be entered to reach this call. It
    powers nested branch-aware counting. An empty tuple means the call runs
    unconditionally (within whatever function/loop scope it lives in).
    """
    full_name: str          # "table.insert", "db.actor", "pairs"
    module: Optional[str]   # "table", "db", None
    func: str               # "insert", "actor", "pairs"
    args: List[Any]         # AST nodes of arguments
    line: int
    node: Node
    scope: Scope
    in_loop: bool = False
    loop_depth: int = 0
    if_chain_path: Tuple[Tuple[int, int], ...] = ()


@dataclass
class IndexInfo:
    """Information about a property access (e.g. `db.actor`).

    Mirrors CallInfo enough that _analyze_repeated_calls_in_scope and
    _edit_repeated_calls can iterate either kind through the same code paths.
    Only single-level dot-access is recorded (Index(Name, Name)) - that's the
    only shape we know how to safely cache as `local x = base.field`.
    """
    full_name: str          # "db.actor"
    module: str             # "db"
    func: str               # "actor"  (named `func` for duck-compat with CallInfo)
    line: int
    node: Node
    scope: Scope
    in_loop: bool = False
    loop_depth: int = 0
    if_chain_path: Tuple[Tuple[int, int], ...] = ()


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
    loop_scope: Optional[Scope] = None  # the innermost loop scope
    right_expr: Optional[str] = None    # string repr of right side of concat


@dataclass
class NilSourceInfo:
    """Information about a variable assigned from a nil-returning function."""
    var_name: str           # variable name
    source_call: str        # the call that might return nil (e.g. "level.object_by_id(id)")
    source_func: str        # just the function name (e.g. "level.object_by_id")
    assign_line: int        # line where assignment happened
    scope: Scope            # scope of the variable
    is_local: bool          # whether it's a local variable
    is_guarded: bool = False  # whether a nil check was found after assignment


@dataclass 
class NilAccessInfo:
    """Information about accessing a potentially nil variable."""
    var_name: str           # the variable being accessed
    access_type: str        # 'method' or 'index'
    access_call: str        # full call (e.g. "obj:section()")
    access_line: int
    nil_source: NilSourceInfo  # the nil source info
    is_safe_to_fix: bool = False  # whether this can be auto-fixed


@dataclass
class DeadCodeInfo:
    """Information about dead/unreachable code."""
    dead_type: str          # 'after_return', 'after_break', 'if_false', 'while_false', 'unused_local_var', 'unused_local_func'
    start_line: int
    end_line: int
    scope_name: str
    description: str
    is_safe_to_remove: bool = False  # True only for 100% safe cases
    code_preview: str = ""
    node: Optional[Node] = None


@dataclass
class LocalVarInfo:
    """Information about a local variable for dead code analysis."""
    name: str
    assign_line: int
    scope: Scope
    is_read: bool = False       # has the variable been read?
    is_function: bool = False   # is it a local function?
    read_lines: List[int] = field(default_factory=list)
    is_loop_var: bool = False   # is it a for loop variable?
    is_param: bool = False      # is it a function parameter?


@dataclass
class PerFrameCallbackInfo:
    """Information about a per-frame callback function for performance analysis."""
    name: str
    start_line: int
    end_line: int
    scope: Scope
    # Collected during analysis pass
    expensive_calls: List[CallInfo] = field(default_factory=list)
    loop_count: int = 0
    uncached_globals: List[str] = field(default_factory=list)


@dataclass
class DistanceComparisonInfo:
    """Information about distance_to() used in comparison (can be optimized to distance_to_sqr())."""
    line: int
    source_obj: str          # the object calling distance_to (e.g., "pos", "actor:position()")
    target_obj: str          # the argument to distance_to (e.g., "target_pos")
    comparison_op: str       # '<', '<=', '>', '>='
    threshold_value: float   # the numeric threshold (e.g., 10)
    threshold_node: Node     # the AST node for the threshold (for replacement)
    full_node: Node          # the full comparison node
    invoke_node: Node        # the distance_to invoke node


@dataclass
class VectorAllocationInfo:
    """Information about vector() allocation in a loop."""
    line: int
    call_node: Node
    loop_depth: int
    scope: Scope
    in_per_frame_callback: bool = False


class ASTAnalyzer:
    """AST-based Lua code analyzer."""

    def __init__(self, cache_threshold: int = 4, experimental: bool = False):
        self.cache_threshold = cache_threshold
        self.experimental = experimental
        self.reset()

    def reset(self):
        """Reset analyzer state for reuse."""
        self.findings: List[Finding] = []
        self.scopes: List[Scope] = []
        self.current_scope: Optional[Scope] = None
        self.global_scope: Optional[Scope] = None
        self.calls: List[CallInfo] = []
        self.indexes: List[IndexInfo] = []
        # Index nodes that are the callee of a Call/Invoke - they should NOT
        # be recorded as standalone property reads (the corresponding Call /
        # Invoke already accounts for them).
        self._suppress_indexes: Set[int] = set()
        self.assigns: List[AssignInfo] = []
        self.concats: List[ConcatInfo] = []
        self.global_writes: List[Tuple[str, int]] = []
        
        self.nil_sources: Dict[Tuple[int, str], NilSourceInfo] = {}
        self.nil_accesses: List[NilAccessInfo] = []
        self.nil_guards: Set[Tuple[str, int]] = set()
        
        self.dead_code: List[DeadCodeInfo] = []
        self.local_vars: Dict[Tuple[int, str], LocalVarInfo] = {}
        self.local_funcs: Dict[Tuple[int, str], LocalVarInfo] = {}
        self.callback_registrations: Set[str] = set()
        self.per_frame_callbacks: List[PerFrameCallbackInfo] = []
        self.distance_comparisons: List[DistanceComparisonInfo] = []
        self.vector_allocations: List[VectorAllocationInfo] = []
        self.assignment_target_ids: Set[int] = set()

        self.source_lines: List[str] = []
        self.source: str = ""
        self.file_path: Optional[Path] = None

        self.loop_depth: int = 0
        self.function_depth: int = 0

        # Stack of (if_id, branch_index) ancestors. Pushed when we enter a
        # branch body, popped when we leave. Calls/indexes record a snapshot
        # of this stack so _count_calls_branch_aware can walk the nested
        # if-chain tree and compute the true max-coexisting-call count.
        self.if_chain_stack: List[Tuple[int, int]] = []

        # cleared per-run so a re-used analyzer can't see a stale tree
        self._ast_tree: Optional[Node] = None

    def analyze_file(self, file_path: Path, source: Optional[str] = None) -> List[Finding]:
        """Analyze a Lua file and return findings.

        `source` skips the disk read (hoist-then-fix re-analyze).
        """
        self.reset()
        self.file_path = file_path

        if source is not None:
            self.source = source
        else:
            try:
                encoding = detect_file_encoding(file_path)
                self.source = file_path.read_text(encoding=encoding)
                self._file_encoding = encoding
            except Exception:
                return []

        self.source_lines = self.source.splitlines()

        try:
            # suppress ANTLR lexer error output during parse
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                tree = ast.parse(self.source)
            finally:
                sys.stderr = old_stderr
        except Exception:
            # parse error, skip
            return []

        # store AST tree for dead code analysis
        self._ast_tree = tree

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
            # Escape special characters for Lua string literal
            # Must re-escape because luaparser stores decoded values
            escaped = s.replace('\\', '\\\\')  # backslash first!
            escaped = escaped.replace('\a', '\\a')  # bell
            escaped = escaped.replace('\b', '\\b')  # backspace
            escaped = escaped.replace('\f', '\\f')  # form feed
            escaped = escaped.replace('\n', '\\n')  # newline
            escaped = escaped.replace('\r', '\\r')  # carriage return
            escaped = escaped.replace('\t', '\\t')  # tab
            escaped = escaped.replace('\v', '\\v')  # vertical tab
            escaped = escaped.replace('\0', '\\0')  # null
            # Choose quote style and escape the chosen quote
            if '"' in escaped and "'" not in escaped:
                escaped = escaped.replace("'", "\\'")
                return f"'{escaped}'"
            else:
                escaped = escaped.replace('"', '\\"')
                return f'"{escaped}"'
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
        elif isinstance(node, UMinusOp):
            return f"-{self._node_to_string(node.operand)}"
        elif isinstance(node, ULNotOp):
            return f"not {self._node_to_string(node.operand)}"
        elif isinstance(node, UBNotOp):
            return f"~{self._node_to_string(node.operand)}"
        elif isinstance(node, Concat):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} .. {right}"
        elif isinstance(node, OrLoOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"({left} or {right})"
        elif isinstance(node, AndLoOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"({left} and {right})"
        elif isinstance(node, AddOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} + {right}"
        elif isinstance(node, SubOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} - {right}"
        elif isinstance(node, MultOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} * {right}"
        elif isinstance(node, FloatDivOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} / {right}"
        elif isinstance(node, ModOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} % {right}"
        elif isinstance(node, ExpoOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} ^ {right}"
        elif isinstance(node, EqToOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} == {right}"
        elif isinstance(node, NotEqToOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} ~= {right}"
        elif isinstance(node, LessThanOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} < {right}"
        elif isinstance(node, GreaterThanOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} > {right}"
        elif isinstance(node, LessOrEqThanOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} <= {right}"
        elif isinstance(node, GreaterOrEqThanOp):
            left = self._node_to_string(node.left)
            right = self._node_to_string(node.right)
            return f"{left} >= {right}"
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

    def _enter_scope(self, name: str, line: int, scope_type: str = 'block', is_hot: bool = False, node: Optional[Any] = None):
        """Enter a new scope."""
        new_scope = Scope(
            name=name,
            start_line=line,
            parent=self.current_scope,
            scope_type=scope_type,
            is_hot_callback=is_hot or (self.current_scope and self.current_scope.is_hot_callback),
            node=node,
        )

        # inherit cached globals from parent
        if self.current_scope:
            new_scope.cached_globals = set(self.current_scope.cached_globals)
            # inherit function aliases from parent scope
            new_scope.func_aliases = dict(self.current_scope.func_aliases)

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

    def _resolve_alias(self, name: str) -> Optional[str]:
        """
        Resolve a function alias to its canonical name.
        
        If 'name' is an alias for a stdlib function (e.g., 'tinsert' -> 'table.insert'),
        returns the canonical name. Otherwise returns None.
        """
        scope = self.current_scope
        while scope:
            if name in scope.func_aliases:
                return scope.func_aliases[name]
            scope = scope.parent
        return None

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
        is_per_frame = func_name in PER_FRAME_CALLBACKS

        self.function_depth += 1
        self._enter_scope(func_name, line, 'function', is_hot, node=node)

        # Track per-frame callback for performance analysis
        if is_per_frame:
            self.per_frame_callbacks.append(PerFrameCallbackInfo(
                name=func_name,
                start_line=line,
                end_line=-1,  # Will be set on scope exit
                scope=self.current_scope,
            ))

        # register parameters as locals
        if hasattr(node, 'args') and node.args:
            for arg in node.args:
                if isinstance(arg, Name):
                    self.current_scope.locals.add(arg.id)

        self._visit(node.body)

        end_line = self._get_end_line(node)
        
        # Update end_line for per-frame callback
        if is_per_frame and self.per_frame_callbacks:
            self.per_frame_callbacks[-1].end_line = end_line
        
        self._exit_scope(end_line)
        self.function_depth -= 1

    def _visit_LocalFunction(self, node: LocalFunction):
        """Handle local function definition."""
        line = self._get_line(node)
        func_name = node.name.id if isinstance(node.name, Name) else '<anon>'

        # register function name in parent scope (before entering function scope)
        if self.current_scope:
            self.current_scope.locals.add(func_name)
            
            # Mark function name as assignment target
            if isinstance(node.name, Name):
                self.assignment_target_ids.add(id(node.name))
                
                # Track for unused function detection (skip _ prefixed)
                if not func_name.startswith('_'):
                    key = (id(self.current_scope), func_name)
                    self.local_funcs[key] = LocalVarInfo(
                        name=func_name,
                        assign_line=line,
                        scope=self.current_scope,
                        is_read=False,
                        is_function=True,
                    )

        is_hot = func_name in HOT_CALLBACKS

        self.function_depth += 1
        self._enter_scope(func_name, line, 'function', is_hot, node=node)

        if hasattr(node, 'args') and node.args:
            for arg in node.args:
                if isinstance(arg, Name):
                    self.current_scope.locals.add(arg.id)

        self._visit(node.body)

        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.function_depth -= 1

    def _visit_AnonymousFunction(self, node: AnonymousFunction):
        """Handle anonymous function expression (e.g. `tbl.x = function(...) ... end`).

        Without this, scope tracking never enters such bodies, so references
        inside them (db.actor, etc.) are attributed to the enclosing module
        scope. _edit_repeated_calls would then hoist a cache to file top,
        where db.actor is nil at script-load time.
        """
        line = self._get_line(node)

        self.function_depth += 1
        self._enter_scope('<anon>', line, 'function', is_hot=False, node=node)

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
        self._enter_scope(func_name, line, 'function', is_hot, node=node)

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

    def _iter_children(self, node):
        """Iterate over all child nodes of an AST node."""
        if node is None:
            return
        
        # Handle lists
        if isinstance(node, list):
            for item in node:
                yield item
            return
        
        # For AST nodes, iterate over known child attributes
        child_attrs = [
            'body', 'test', 'orelse', 'targets', 'values', 'iter',
            'func', 'args', 'value', 'idx', 'key', 'left', 'right',
            'operand', 'fields', 'keys', 'source', 'step', 'start', 'stop',
        ]
        
        for attr in child_attrs:
            child = getattr(node, attr, None)
            if child is not None:
                if isinstance(child, list):
                    for item in child:
                        yield item
                else:
                    yield child

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
                var_name = target.id
                self.current_scope.locals.add(var_name)
                
                # Mark as assignment target (not a read)
                self.assignment_target_ids.add(id(target))
                
                # Track for unused variable detection (skip _ prefixed)
                if not var_name.startswith('_'):
                    key = (id(self.current_scope), var_name)
                    self.local_vars[key] = LocalVarInfo(
                        name=var_name,
                        assign_line=line,
                        scope=self.current_scope,
                        is_read=False,
                        is_function=False,
                        is_loop_var=True,
                    )

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
            var_name = node.target.id
            self.current_scope.locals.add(var_name)
            
            # Mark as assignment target (not a read)
            self.assignment_target_ids.add(id(node.target))
            
            # Track for unused variable detection (skip _ prefixed)
            if not var_name.startswith('_'):
                key = (id(self.current_scope), var_name)
                self.local_vars[key] = LocalVarInfo(
                    name=var_name,
                    assign_line=line,
                    scope=self.current_scope,
                    is_read=False,
                    is_function=False,
                    is_loop_var=True,
                )

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
        """Handle if statement.

        In Lua every if/elseif/else body is its own scope, so we open a fresh
        block scope around each. Without this, a `local x = ...` inside an
        if-body bleeds into the enclosing function's locals tracking and can
        clobber nil-source tracking when the same name is reassigned in
        another branch - see C2 in the audit notes.
        """
        if_id = id(node)

        # test runs in the parent scope (it can read outer locals but
        # `local`s declared inside it would still belong to the parent).
        # The test is also OUTSIDE this if-chain's branches in the
        # if_chain_stack sense - so we don't push for the test.
        self._visit(node.test)

        # main body - branch 0, fresh block scope, push onto chain stack
        body_line = self._get_line(node.body) if node.body is not None else self._get_line(node)
        body_end = self._get_end_line(node.body) if node.body is not None else self._get_end_line(node)
        self.if_chain_stack.append((if_id, 0))
        self._enter_scope('<if-body>', body_line, 'block')
        self._visit(node.body)
        self._exit_scope(body_end)
        self.if_chain_stack.pop()

        # elseif/else chain - keeps the same if_id, increments branch index
        if node.orelse:
            self._visit_orelse(node.orelse, if_id, 1)

    def _visit_orelse(self, node, if_id, branch_idx):
        """Helper to visit elseif/else with branch + scope tracking."""
        if isinstance(node, ElseIf):
            # elseif: test and body share a scope (elseif test only runs after
            # all prior branches have failed, so it's effectively in the same
            # control-flow position as the body that follows it).
            line = self._get_line(node)
            end = self._get_end_line(node)
            self.if_chain_stack.append((if_id, branch_idx))
            self._enter_scope('<elseif-body>', line, 'block')
            self._visit(node.test)
            self._visit(node.body)
            self._exit_scope(end)
            self.if_chain_stack.pop()

            if node.orelse:
                self._visit_orelse(node.orelse, if_id, branch_idx + 1)
        elif isinstance(node, Block):
            # else block - branch index -1
            line = self._get_line(node)
            end = self._get_end_line(node)
            self.if_chain_stack.append((if_id, -1))
            self._enter_scope('<else-body>', line, 'block')
            self._visit(node)
            self._exit_scope(end)
            self.if_chain_stack.pop()
        else:
            # defensive fallback
            line = self._get_line(node)
            end = self._get_end_line(node)
            self.if_chain_stack.append((if_id, -1))
            self._enter_scope('<else-body>', line, 'block')
            self._visit(node)
            self._exit_scope(end)
            self.if_chain_stack.pop()

    def _visit_ElseIf(self, node: ElseIf):
        """Handle elseif clause - this is called from _visit_orelse."""
        # already handled by _visit_orelse
        pass

    def _visit_Do(self, node):
        """Handle `do ... end` block - its body is a fresh local scope."""
        line = self._get_line(node)
        end_line = self._get_end_line(node)
        self._enter_scope('<do>', line, 'block')
        body = getattr(node, 'body', None)
        if body is not None:
            self._visit(body)
        self._exit_scope(end_line)

    def _visit_LocalAssign(self, node: LocalAssign):
        """Handle local assignment."""
        line = self._get_line(node)

        # register targets as locals and track for unused variable detection
        for target in node.targets:
            if isinstance(target, Name):
                var_name = target.id
                self.current_scope.locals.add(var_name)
                
                # Mark this Name node as an assignment target (not a read)
                self.assignment_target_ids.add(id(target))
                
                # Track for unused variable detection (skip _ prefixed)
                if not var_name.startswith('_'):
                    key = (id(self.current_scope), var_name)
                    self.local_vars[key] = LocalVarInfo(
                        name=var_name,
                        assign_line=line,
                        scope=self.current_scope,
                        is_read=False,
                        is_function=False,
                    )

        # check for caching pattern: local xyz = module.func
        is_new_alias = False
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
                            # Record alias mapping: target_name -> canonical function name
                            self.current_scope.func_aliases[target_name] = full_name
                            is_new_alias = True

                # check if caching a bare global
                elif isinstance(value, Name):
                    if value.id in CACHEABLE_BARE_GLOBALS:
                        self.current_scope.cached_globals.add(value.id)
                        # Record alias mapping for bare globals too
                        self.current_scope.func_aliases[target_name] = value.id
                        is_new_alias = True
                
                # If NOT creating a new alias, invalidate any existing alias with this name
                # (e.g., local f = table.insert; local f = other_func)
                if not is_new_alias:
                    if target_name in self.current_scope.func_aliases:
                        del self.current_scope.func_aliases[target_name]

                # record assignment info
                self._record_assignment(target_name, value, line, is_local=True)

        # visit values
        for value in node.values:
            self._visit(value)

    def _visit_Assign(self, node: Assign):
        """Handle assignment."""
        line = self._get_line(node)

        # Mark Name targets as assignment targets (not reads)
        for target in node.targets:
            if isinstance(target, Name):
                self.assignment_target_ids.add(id(target))
                
                # Invalidate alias if this variable was an alias
                # (reassigning breaks the alias relationship)
                target_name = target.id
                self._invalidate_alias(target_name)

        # check for global writes
        for target in node.targets:
            if isinstance(target, Name):
                target_name = target.id
                # it's a global write if not in any scope's locals
                if not self._is_in_locals(target_name):
                    self.global_writes.append((target_name, line))

                if len(node.values) == 1:
                    self._record_assignment(target_name, node.values[0], line, is_local=False)

        # visit targets (for calls inside index expressions like db.storage[npc:id()])
        for target in node.targets:
            self._visit(target)

        # visit values
        for value in node.values:
            self._visit(value)
    
    def _invalidate_alias(self, var_name: str):
        """Remove alias mapping when a variable is reassigned."""
        scope = self.current_scope
        while scope:
            if var_name in scope.func_aliases:
                del scope.func_aliases[var_name]
                return  # Only remove from innermost scope where it exists
            scope = scope.parent

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
            
            # get the right side expression
            right_expr = self._node_to_string(value.right)
            
            # find innermost loop scope
            loop_scope = None
            if self.loop_depth > 0:
                # walk up scopes to find loop
                s = self.current_scope
                while s:
                    if s.scope_type == 'loop':
                        loop_scope = s
                        break
                    s = s.parent

            self.concats.append(ConcatInfo(
                target=target,
                left_var=left_var,
                line=line,
                scope=self.current_scope,
                in_loop=self.loop_depth > 0,
                loop_depth=self.loop_depth,
                loop_scope=loop_scope,
                right_expr=right_expr,
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
        
        # Track nil-returning function assignments
        self._track_nil_source(target, value, value_repr, line, is_local)

    def _find_defining_scope(self, var_name: str) -> Optional[Scope]:
        """Find the scope that owns this variable (walks up the scope chain).

        Returns None if the variable is not declared anywhere in the current
        chain (i.e. it's a global). This matters for nil-tracking: when a
        function parameter or outer local is reassigned inside a nested block,
        the nil-source must be keyed under the *defining* scope, otherwise it
        evaporates when the inner block exits and later access sites don't
        find it.
        """
        scope = self.current_scope
        while scope:
            if var_name in scope.locals:
                return scope
            scope = scope.parent
        return None

    def _track_nil_source(self, target: str, value: Node, value_repr: str, line: int, is_local: bool):
        """Track if a variable is assigned from a nil-returning function."""
        source_func = None

        # Check if it's a call to a nil-returning function
        if isinstance(value, Call):
            # get the function name
            _, _, full_name = self._get_call_name(value)
            if full_name and full_name in NIL_RETURNING_FUNCTIONS:
                source_func = full_name

        # Check for method call (Invoke) - e.g., obj:parent()
        elif isinstance(value, Invoke):
            method_name = value.func.id if isinstance(value.func, Name) else ''
            method_pattern = f':{method_name}'
            if method_pattern in NIL_RETURNING_FUNCTIONS:
                source_func = method_pattern

        # Check for index access - e.g., db.actor, alife():object(id)
        elif isinstance(value, Index):
            full_name = self._node_to_string(value)
            # check direct matches like db.actor
            if full_name in NIL_RETURNING_FUNCTIONS:
                source_func = full_name

        # Determine the scope under which to register/clear the nil source.
        #  - LocalAssign (`local x = ...`) introduces a brand-new local in the
        #    current scope, so we use current_scope.
        #  - Plain Assign (`x = ...`) targets an already-declared variable:
        #    walk up to find which scope owns it. If it's a global (no scope),
        #    we skip nil-tracking entirely - the analyzer doesn't reason about
        #    cross-call global state.
        if is_local:
            owning_scope = self.current_scope
        else:
            owning_scope = self._find_defining_scope(target)
            if owning_scope is None:
                return  # global write - out of scope for this analysis

        key = (id(owning_scope), target)

        if source_func:
            self.nil_sources[key] = NilSourceInfo(
                var_name=target,
                source_call=value_repr,
                source_func=source_func,
                assign_line=line,
                scope=owning_scope,
                is_local=is_local,
                is_guarded=False,
            )
        else:
            # variable reassigned from a non-nil source - drop the tracking entry
            if key in self.nil_sources:
                del self.nil_sources[key]

    def _check_nil_access(self, source_node: Node, source_str: str, full_call: str, line: int, access_type: str):
        """Check if we're accessing a potentially nil variable."""
        # only check simple variable names for now
        if not isinstance(source_node, Name):
            return
        
        var_name = source_node.id
        
        # check if this variable is from a nil-returning function in an enclosing scope
        nil_source = self._find_nil_source(var_name)
        if not nil_source:
            return
        
        # check if there's a nil guard before this access
        if self._has_nil_guard(var_name, nil_source.assign_line, line):
            nil_source.is_guarded = True
            return
        
        # determine if this is safe to auto-fix
        # Safe if: assignment is on previous line, this is the only usage before any branch
        is_safe = self._is_safe_nil_fix(nil_source, line)
        
        self.nil_accesses.append(NilAccessInfo(
            var_name=var_name,
            access_type=access_type,
            access_call=full_call,
            access_line=line,
            nil_source=nil_source,
            is_safe_to_fix=is_safe,
        ))

    def _find_nil_source(self, var_name: str) -> Optional[NilSourceInfo]:
        """Find nil source for a variable in current or enclosing scopes."""
        # key is (scope_id, var_name)
        scope = self.current_scope
        while scope:
            key = (id(scope), var_name)
            if key in self.nil_sources:
                return self.nil_sources[key]
            scope = scope.parent
        return None

    @staticmethod
    def _strip_line_comments_and_strings(text: str) -> str:
        """Strip comment and string content from a line so regex only matches real code."""
        result = []
        i = 0
        while i < len(text):
            c = text[i]
            # line comment
            if c == '-' and i + 1 < len(text) and text[i + 1] == '-':
                break
            # string literal
            elif c in ('"', "'"):
                quote = c
                result.append(c)
                i += 1
                while i < len(text):
                    sc = text[i]
                    if sc == '\\':
                        result.append(' ')
                        i += 1
                        if i < len(text):
                            result.append(' ')
                            i += 1
                        continue
                    if sc == quote:
                        result.append(sc)
                        i += 1
                        break
                    result.append(' ')
                    i += 1
                continue
            # long string [[...]]
            elif c == '[' and i + 1 < len(text) and text[i + 1] in ('[', '='):
                eq_count = 0
                j = i + 1
                while j < len(text) and text[j] == '=':
                    eq_count += 1
                    j += 1
                if j < len(text) and text[j] == '[':
                    close = ']' + '=' * eq_count + ']'
                    end = text.find(close, j + 1)
                    if end != -1:
                        result.append(' ' * (end + len(close) - i))
                        i = end + len(close)
                        continue
                result.append(c)
                i += 1
                continue
            else:
                result.append(c)
                i += 1
        return ''.join(result)

    def _has_nil_guard(self, var_name: str, assign_line: int, access_line: int) -> bool:
        """Check if there's a nil guard between assignment and access."""
        if assign_line >= access_line:
            return False
        
        var_escaped = re.escape(var_name)
        
        guard_patterns = [
            rf'\bif\s+{var_escaped}\s+then\b',
            rf'\bif\s+{var_escaped}\s+and\b',
            rf'\bif\s+not\s+{var_escaped}\s+then\b',
            rf'\bif\s+{var_escaped}\s*~=\s*nil\b',
            rf'\bif\s+{var_escaped}\s*==\s*nil\s+then\s+return\b',
            rf'\b{var_escaped}\s+and\s+{var_escaped}[:\.]',
            rf'\bif\s*\(\s*{var_escaped}\s*\)\s*then\b',
        ]
        
        combined_pattern = re.compile('|'.join(guard_patterns), re.IGNORECASE)
        
        for line_num in range(assign_line, access_line):
            if line_num <= 0 or line_num > len(self.source_lines):
                continue
            line_text = self.source_lines[line_num - 1]
            
            # only match against actual code, not comments or string contents
            cleaned = self._strip_line_comments_and_strings(line_text)
            if combined_pattern.search(cleaned):
                return True
        
        return False

    def _is_safe_nil_fix(self, nil_source: NilSourceInfo, access_line: int) -> bool:
        """
        Determine if a nil access is safe to auto-fix.
        
        Safe conditions:
        1. Access is on the line immediately after assignment
        2. It's a local variable (not global)
        3. Assignment and access are in the same scope
        4. No complex control flow between them
        5. Access line is NOT a local declaration (would break scope if wrapped)
        6. Access line is NOT a control flow statement (if/for/while - too complex)
        """
        # must be immediately after (next line)
        if access_line != nil_source.assign_line + 1:
            return False
        
        # must be local
        if not nil_source.is_local:
            return False
        
        # must be in same scope
        if self.current_scope != nil_source.scope:
            return False
        
        # check that the line between is not a control flow statement
        if nil_source.assign_line <= 0 or nil_source.assign_line > len(self.source_lines):
            return False
        
        # check access line content
        if access_line > 0 and access_line <= len(self.source_lines):
            access_text = self.source_lines[access_line - 1].strip()
            
            # CRITICAL: access line must NOT be a local declaration
            if access_text.startswith('local '):
                return False
            
            # CRITICAL: access line must NOT be control flow (too complex to wrap)
            control_keywords = ('if ', 'if(', 'for ', 'while ', 'repeat', 'function ', 'function(')
            if any(access_text.startswith(kw) for kw in control_keywords):
                return False
            
        return True

    def _visit_Call(self, node: Call):
        """Handle function call."""
        line = self._get_line(node)
        module, func, full_name = self._get_call_name(node)

        # Check if this is an aliased stdlib call
        # e.g., if 'local tinsert = table.insert' was declared,
        # then 'tinsert(t, v)' should be recognized as 'table.insert(t, v)'
        if full_name and module is None:
            # This is a bare function call - check if it's an alias
            canonical = self._resolve_alias(full_name)
            if canonical:
                full_name = canonical
                # Also update module/func if it's a module.func pattern
                if '.' in canonical:
                    module, func = canonical.split('.', 1)

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
                if_chain_path=tuple(self.if_chain_stack),
            ))
            
            # Track RegisterScriptCallback for unused variable/function detection
            if full_name == 'RegisterScriptCallback' and len(node.args) >= 2:
                callback_func = self._node_to_string(node.args[1])
                if callback_func:
                    self.callback_registrations.add(callback_func)
            
            # Track vector() allocations in loops
            if full_name == 'vector' and self.loop_depth > 0:
                # check if we're inside a per-frame callback
                in_per_frame = False
                scope = self.current_scope
                while scope:
                    if scope.is_hot_callback:
                        in_per_frame = True
                        break
                    scope = scope.parent
                
                self.vector_allocations.append(VectorAllocationInfo(
                    line=line,
                    call_node=node,
                    loop_depth=self.loop_depth,
                    scope=self.current_scope,
                    in_per_frame_callback=in_per_frame,
                ))

        # the Index that names the function being called is *not* a standalone
        # property read - record it under the Call we just emitted, not again
        # via _visit_Index. This affects only the outermost Index attached to
        # node.func; nested Indexes (the `a.b` in `a.b.c()`) are still real
        # property reads and must be recorded normally.
        if isinstance(node.func, Index):
            self._suppress_indexes.add(id(node.func))

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
            if_chain_path=tuple(self.if_chain_stack),
        ))

        # Check for potential nil access
        self._check_nil_access(node.source, source, full_name, line, 'method')

        # source of an invoke (e.g. `db.actor` in `db.actor:method()`) is the
        # receiver - the Invoke itself records the full `db.actor:method`
        # signature, so don't double-count `db.actor` as a standalone read.
        if isinstance(node.source, Index):
            self._suppress_indexes.add(id(node.source))

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
        # Distinguish bracket vs. dot notation. For dot access (t.field), the
        # idx is a synthetic Name representing a field literal - visiting it
        # would falsely register the field name as a variable read and hide
        # genuinely-unused locals that happen to share the field name.
        # For bracket access (t[expr]), idx is a real expression that must be
        # visited (it can read locals: t[my_var], t[foo() + 1], etc.).
        idx_token = getattr(node.idx, 'first_token', None)
        is_bracket = idx_token is not None and str(idx_token) != 'None'
        if is_bracket:
            self._visit(node.idx)

        # Record property reads for repeated-access caching (e.g. `db.actor`).
        # Only single-level dot accesses where both sides are plain Names -
        # those are the only shape `_edit_repeated_calls` knows how to cache
        # as `local x = base.field`.
        if id(node) in self._suppress_indexes:
            return
        if is_bracket:
            return
        if not isinstance(node.value, Name) or not isinstance(node.idx, Name):
            return

        full_name = f"{node.value.id}.{node.idx.id}"
        if full_name not in EXPENSIVE_INDEXES:
            return

        self.indexes.append(IndexInfo(
            full_name=full_name,
            module=node.value.id,
            func=node.idx.id,
            line=self._get_line(node),
            node=node,
            scope=self.current_scope,
            in_loop=self.loop_depth > 0,
            loop_depth=self.loop_depth,
            if_chain_path=tuple(self.if_chain_stack),
        ))

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
    def _visit_EqToOp(self, node): self._visit(node.left); self._visit(node.right)
    def _visit_NotEqToOp(self, node): self._visit(node.left); self._visit(node.right)
    
    def _visit_LessThanOp(self, node):
        self._check_distance_comparison(node, '<')
        self._visit(node.left)
        self._visit(node.right)
    
    def _visit_GreaterThanOp(self, node):
        self._check_distance_comparison(node, '>')
        self._visit(node.left)
        self._visit(node.right)
    
    def _visit_LessOrEqThanOp(self, node):
        self._check_distance_comparison(node, '<=')
        self._visit(node.left)
        self._visit(node.right)
    
    def _visit_GreaterOrEqThanOp(self, node):
        self._check_distance_comparison(node, '>=')
        self._visit(node.left)
        self._visit(node.right)
    
    def _check_distance_comparison(self, node, op: str):
        """Check if this comparison involves distance_to() that could use distance_to_sqr()."""
        # Pattern: obj:distance_to(target) < N  or  N > obj:distance_to(target)
        invoke_node = None
        threshold_node = None
        
        # check left side for distance_to invoke
        if isinstance(node.left, Invoke):
            func_name = node.left.func.id if isinstance(node.left.func, Name) else None
            if func_name == 'distance_to':
                invoke_node = node.left
                threshold_node = node.right
        
        # check right side for distance_to invoke (reversed comparison)
        if invoke_node is None and isinstance(node.right, Invoke):
            func_name = node.right.func.id if isinstance(node.right.func, Name) else None
            if func_name == 'distance_to':
                invoke_node = node.right
                threshold_node = node.left
                # Reverse the operator for analysis
                op = {'<': '>', '>': '<', '<=': '>=', '>=': '<='}[op]
        
        if invoke_node is None:
            return
        
        # check if threshold is a numeric literal
        if not isinstance(threshold_node, Number):
            return
        
        threshold_value = threshold_node.n
        
        # extract source and target
        source_obj = self._node_to_string(invoke_node.source)
        target_obj = self._node_to_string(invoke_node.args[0]) if invoke_node.args else ""
        
        self.distance_comparisons.append(DistanceComparisonInfo(
            line=self._get_line(node),
            source_obj=source_obj,
            target_obj=target_obj,
            comparison_op=op,
            threshold_value=threshold_value,
            threshold_node=threshold_node,
            full_node=node,
            invoke_node=invoke_node,
        ))

    # unary ops
    def _visit_UMinusOp(self, node): self._visit(node.operand)
    def _visit_UBNotOp(self, node): self._visit(node.operand)
    def _visit_ULNotOp(self, node): self._visit(node.operand)
    def _visit_ULengthOP(self, node): self._visit(node.operand)

    # terminal nodes - no children
    def _visit_Name(self, node):
        """Handle Name node - track variable reads for unused detection."""
        # Only count as read if NOT an assignment target
        if id(node) not in self.assignment_target_ids:
            var_name = node.id
            # Find which scope this variable belongs to (walk up scope chain)
            scope = self.current_scope
            while scope:
                key = (id(scope), var_name)
                if key in self.local_vars:
                    self.local_vars[key].is_read = True
                    break
                if key in self.local_funcs:
                    self.local_funcs[key].is_read = True
                    break
                # Check if it's in this scope's locals (even if not tracked)
                if var_name in scope.locals:
                    break  # Found the scope, but might not be tracked (e.g., _ prefixed)
                scope = scope.parent
    
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
        self._analyze_pow_operator()
        self._analyze_string_literal_concat()
        self._analyze_string_find_plain()
        self._analyze_redundant_not_eq()
        self._analyze_uncached_globals()
        self._analyze_repeated_calls_in_scope()
        self._analyze_string_concat_in_loop()
        self._analyze_debug_statements()
        self._analyze_global_writes()
        self._analyze_nil_access()
        self._analyze_dead_code()
        self._analyze_per_frame_callbacks()
        self._analyze_distance_to_comparisons()
        self._analyze_vector_allocations_in_loops()
        self._analyze_anon_hoist()

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
                            message=f'{full_match} -> {base}^0.5',
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

    def _analyze_string_literal_concat(self):
        """Find `"a" .. "b" .. "c"` chains where every leaf is a String literal.

        These can be combined at compile time into a single string literal,
        eliminating the runtime concat allocation. Walk the AST top-down: when
        we hit a Concat whose entire subtree is all-Strings, emit a finding
        for it and *don't* recurse into the children (their Concats are
        subsumed by the outer one we're rewriting).
        """
        if not getattr(self, '_ast_tree', None):
            return

        def all_string_leaves(node, out):
            """Return True iff every leaf of this Concat tree is a String."""
            if isinstance(node, Concat):
                return all_string_leaves(node.left, out) and all_string_leaves(node.right, out)
            if isinstance(node, String):
                out.append(node)
                return True
            return False

        def walk(node):
            if isinstance(node, Concat):
                leaves = []
                if all_string_leaves(node, leaves) and len(leaves) >= 2:
                    self._emit_string_literal_concat_finding(node, leaves)
                    return  # children subsumed - don't recurse
            for child in self._iter_children(node):
                walk(child)

        walk(self._ast_tree)

    def _emit_string_literal_concat_finding(self, concat_node, leaves):
        """Emit a finding for an all-literal Concat tree.

        Stores a list of (start, end) spans for each String leaf along with
        the decoded text. The transformer reuses the source text of the
        first leaf as the new literal, splicing in the combined contents.
        """
        # decode each String into the concatenated value. luaparser keeps
        # the decoded form on `.s` (without quotes/escape processing of the
        # source - bytes/str depending on encoding).
        parts = []
        for leaf in leaves:
            s = leaf.s
            if isinstance(s, bytes):
                s = s.decode('utf-8', errors='replace')
            parts.append(s)
        combined_value = ''.join(parts)

        line = self._get_line(concat_node)
        # produce a Lua literal for the combined value. Use _node_to_string
        # which already escapes properly.
        # Build a synthetic String AST node would be overkill; just call
        # the same escape routine indirectly by constructing a fake node.
        # Easiest: do the escape inline (mirrors _node_to_string).
        escaped = combined_value.replace('\\', '\\\\')
        escaped = escaped.replace('\a', '\\a')
        escaped = escaped.replace('\b', '\\b')
        escaped = escaped.replace('\f', '\\f')
        escaped = escaped.replace('\n', '\\n')
        escaped = escaped.replace('\r', '\\r')
        escaped = escaped.replace('\t', '\\t')
        escaped = escaped.replace('\v', '\\v')
        escaped = escaped.replace('\0', '\\0')
        if '"' in escaped and "'" not in escaped:
            escaped_lit = "'" + escaped.replace("'", "\\'") + "'"
        else:
            escaped_lit = '"' + escaped.replace('"', '\\"') + '"'

        self.findings.append(Finding(
            pattern_name='string_literal_concat',
            severity='GREEN',
            line_num=line,
            message=f'fold {len(leaves)} string literals into one',
            details={
                'parts_count': len(leaves),
                'combined': escaped_lit,
                'node': concat_node,
            },
            source_line=self._get_source_line(line),
        ))

    # Lua-pattern metacharacters. If a needle contains none of these, it's a
    # plain text search and `string.find(s, needle, 1, true)` is much faster
    # than the default pattern interpretation.
    _LUA_PATTERN_META = frozenset('^$().%[]*+-?')

    def _analyze_string_find_plain(self):
        """Find `string.find(s, "literal", ...)` calls where the needle is a
        plain string (no pattern metacharacters). Adding `, 1, true` (or
        `, true` if `init` is already given) skips pattern compilation. Only
        rewrite the 2-arg and 3-arg forms - with 4+ args the user has
        already supplied a `plain` flag explicitly.
        """
        for call in self.calls:
            if call.full_name != 'string.find':
                continue
            n_args = len(call.args)
            if n_args < 2 or n_args > 3:
                continue
            needle = call.args[1]
            if not isinstance(needle, String):
                continue
            s = needle.s
            if isinstance(s, bytes):
                s = s.decode('utf-8', errors='replace')
            if any(ch in self._LUA_PATTERN_META for ch in s):
                continue

            self.findings.append(Finding(
                pattern_name='string_find_plain',
                severity='GREEN',
                line_num=call.line,
                message=f'string.find with plain literal "{s[:30]}" - add plain=true',
                details={
                    'n_args': n_args,
                    'node': call.node,
                },
                source_line=self._get_source_line(call.line),
            ))

    def _analyze_redundant_not_eq(self):
        """Find `not (a == b)` / `not (a ~= b)` and rewrite to `a ~= b` / `a == b`.

        Pure boolean identity - works for any operand types Lua supports
        (no NaN trap like comparison operators have, since `==` already
        handles NaN consistently).

        Skips patterns where the operand is anything other than EqToOp /
        NotEqToOp (e.g. `not (a < b)` is *not* equivalent to `a >= b` for
        floats with NaN, so we leave inequalities alone).
        """
        if not getattr(self, '_ast_tree', None):
            return
        for node in ast.walk(self._ast_tree):
            if not isinstance(node, ULNotOp):
                continue
            op = node.operand
            if isinstance(op, EqToOp):
                new_op = '~='
                shape = '== '
            elif isinstance(op, NotEqToOp):
                new_op = '=='
                shape = '~= '
            else:
                continue

            self.findings.append(Finding(
                pattern_name='redundant_not_eq',
                severity='GREEN',
                line_num=self._get_line(node),
                message=f'not (a {shape.strip()} b) -> a {new_op} b',
                details={
                    'new_op': new_op,
                    'left_node': op.left,
                    'right_node': op.right,
                    'outer_node': node,
                },
                source_line=self._get_source_line(self._get_line(node)),
            ))

    def _analyze_pow_operator(self):
        """Find `x ^ 2` / `x ^ 3` patterns rewriteable to `x*x` / `x*x*x`.

        In LuaJIT 2.0 the `^` operator with an integer exponent dispatches
        through a generic pow VM helper, while `x*x` compiles to a single
        MUL bytecode. Same shape as the math.pow optimization, but the
        operator form was previously not detected.
        """
        if not getattr(self, '_ast_tree', None):
            return
        for node in ast.walk(self._ast_tree):
            if not isinstance(node, ExpoOp):
                continue
            right = node.right
            if not isinstance(right, Number):
                continue
            exp = right.n
            if exp not in (2, 3) or not self._is_simple_expr(node.left):
                continue
            base = self._node_to_string(node.left)
            replacement = '*'.join([base] * int(exp))
            full_match = f'{base}^{int(exp)}'
            self.findings.append(Finding(
                pattern_name='pow_op_simple',
                severity='GREEN',
                line_num=self._get_line(node),
                message=f'{full_match} -> {replacement}',
                details={
                    'base': base,
                    'exponent': int(exp),
                    'replacement': replacement,
                    'full_match': full_match,
                    'node': node,
                },
                source_line=self._get_source_line(self._get_line(node)),
            ))
    
    def _count_calls_branch_aware(self, calls: List) -> int:
        """Maximum number of calls executable on any single control-flow path.

        Each item carries its full ancestor `if_chain_path` (sequence of
        (if_id, branch_index) tuples). The algorithm walks that tree depth by
        depth: at each level, items whose path ends here count directly, and
        items whose path reaches deeper are partitioned by next-step
        (if_id, branch). For each if-chain encountered we sum (across the
        chain) the *max* across its branches - only one branch executes. The
        recursion handles arbitrarily nested if-chains correctly.
        """

        def count_at(items, depth: int) -> int:
            if not items:
                return 0

            direct = 0
            # if_id -> branch_idx -> list of items still descending
            nested: Dict[int, Dict[int, list]] = {}

            for item in items:
                path = item.if_chain_path
                if len(path) <= depth:
                    direct += 1
                else:
                    if_id, branch_idx = path[depth]
                    chain = nested.setdefault(if_id, {})
                    chain.setdefault(branch_idx, []).append(item)

            total = direct
            for if_id, branches in nested.items():
                # exactly one branch of this if-chain runs - take the worst case
                max_in_chain = 0
                for branch_idx, branch_items in branches.items():
                    branch_count = count_at(branch_items, depth + 1)
                    if branch_count > max_in_chain:
                        max_in_chain = branch_count
                total += max_in_chain
            return total

        return count_at(calls, 0)

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

                # threshold: configurable (default 4), hot callbacks use threshold-1
                # with --experimental, use branch-aware counting
                threshold = self.cache_threshold - 1 if func_scope.is_hot_callback else self.cache_threshold
                call_count = self._count_calls_branch_aware(calls)
                if call_count >= threshold:
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
        """Find the enclosing function scope, or None if at module level.

        Must NOT fall back to global_scope: callers use the returned scope as
        the insertion target for cache decls, and inserting at module scope
        captures runtime-only values (db.actor, alife()) at script-load time
        when they are still nil.
        """
        while scope:
            if scope.scope_type == 'function':
                return scope
            scope = scope.parent
        return None

    def _analyze_repeated_calls_in_scope(self):
        """Find repeated expensive calls within function scope."""
        # expensive function calls (need parens) to track
        # NOTE: time_global() is NOT included because it returns different values
        # each call (current time) - caching it breaks elapsed time calculations
        # NOTE: level.object_by_id() is NOT auto-fixed because different IDs give
        # different objects, and even same IDs can change if object is destroyed
        # NOTE: `db.actor` is a *property*, not a function - it's tracked via
        # self.indexes (see EXPENSIVE_INDEXES) and folded into the same buckets
        # below.
        expensive_calls = {'alife', 'system_ini', 'game_ini', 'getFS',
                           'device', 'get_console', 'get_hud', 'level.name'}

        # method calls that are safe to cache (immutable object properties)
        # based on X-Ray engine source analysis:
        # - :section() returns stored NameSection member (xr_object.h:155)
        # - :id() returns stored Props.net_ID member (xr_object.h:98)
        # - :clsid() returns stored m_script_clsid member (GameObject.h:257)
        # - :story_id() returns m_story_id set once from config (xrServer_Objects_ALife.cpp:375)
        cacheable_methods = {'section', 'id', 'clsid', 'story_id'}

        # group by function scope. Both CallInfo and IndexInfo carry the same
        # set of fields the downstream code depends on (line, node, scope,
        # parent_if_node, branch_index), so we can mix them in one bucket.
        scope_calls: Dict[Scope, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

        for call in self.calls:
            if call.full_name in expensive_calls:
                func_scope = self._find_function_scope(call.scope)
                if func_scope:
                    scope_calls[func_scope][call.full_name].append(call)

            # track cacheable method calls on objects (:section(), :id(), :clsid())
            if call.func in cacheable_methods and ':' in call.full_name:
                func_scope = self._find_function_scope(call.scope)
                if func_scope:
                    key = f"{call.full_name}()"
                    scope_calls[func_scope][key].append(call)

        # Property-style accesses (db.actor, etc.). These flow through the
        # same bucket as calls so the threshold check, branch-aware count,
        # and finding emission below all work uniformly.
        for idx in self.indexes:
            if idx.full_name in EXPENSIVE_INDEXES:
                func_scope = self._find_function_scope(idx.scope)
                if func_scope:
                    scope_calls[func_scope][idx.full_name].append(idx)

        for func_scope, calls_by_name in scope_calls.items():
            for name, calls in calls_by_name.items():
                threshold = self.cache_threshold - 1 if func_scope.is_hot_callback else self.cache_threshold
                call_count = self._count_calls_branch_aware(calls)

                if call_count >= threshold:
                    # suggest caching
                    severity = 'GREEN'

                    if name == 'db.actor':
                        suggestion = 'local actor = db.actor'
                    elif name == 'alife':
                        suggestion = 'local sim = alife()'
                    elif name == 'system_ini':
                        suggestion = 'local ini = system_ini()'
                    elif name == 'device':
                        suggestion = 'local dev = device()'
                    elif name == 'get_console':
                        suggestion = 'local console = get_console()'
                    elif name == 'get_hud':
                        suggestion = 'local hud = get_hud()'
                    elif name == 'level.name':
                        suggestion = 'local level_name = level.name()'
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
                            'original_call': name,  # preserve original like "self.object:id()"
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
                concat_info = concats[0]
                loop_scope = concat_info.loop_scope
                
                # check if variable is initialized to empty string before loop
                init_line = None
                is_safe = False
                
                # SAFETY: don't auto-fix nested loops (loop_depth > 1) because we can't
                # reliably determine which loop's end to place table.concat after
                if loop_scope and concat_info.loop_depth == 1:
                    # look for var = "" or var = '' IMMEDIATELY before the loop
                    # must be: within 3 lines, NOT inside any loop, and must be local declaration
                    empty_strings = ('""', "''", '[[]]')
                    for assign in self.assigns:
                        if (assign.target == var and
                            assign.value_type == 'literal' and
                            assign.value_repr in empty_strings and
                            assign.line < loop_scope.start_line and
                            assign.line >= loop_scope.start_line - 3 and
                            not assign.in_loop and
                            assign.is_local and
                            # SAFETY: the loop's enclosing scope must match the
                            # init's scope. If the loop is nested deeper (e.g.
                            # inside an `if` between init and loop), inserting
                            # `local var = table.concat(...)` after the loop's
                            # `end` lands in the wrong scope and breaks code
                            # that uses the variable past that scope.
                            loop_scope.parent is assign.scope):
                            init_line = assign.line
                            is_safe = True
                            break
                
                self.findings.append(Finding(
                    pattern_name='string_concat_in_loop',
                    severity='YELLOW',
                    line_num=concat_info.line,
                    message=f'String concat in loop: {var} = {var} .. x',
                    details={
                        'variable': var,
                        'count': len(concats),
                        'loop_depth': concat_info.loop_depth,
                        'suggestion': 'Use table.insert() + table.concat()',
                        'right_expr': concat_info.right_expr,
                        'loop_start': loop_scope.start_line if loop_scope else None,
                        'loop_end': loop_scope.end_line if loop_scope else None,
                        'init_line': init_line,
                        'is_safe': is_safe,
                        'concat_lines': [c.line for c in concats],
                    },
                    source_line=self._get_source_line(concat_info.line),
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

    def _analyze_nil_access(self):
        """Generate findings for potential nil access patterns."""
        for access in self.nil_accesses:
            nil_source = access.nil_source
            reason = NIL_RETURNING_FUNCTIONS.get(nil_source.source_func, 'may return nil')
            
            # determine severity based on whether it's safe to fix
            if access.is_safe_to_fix:
                severity = 'YELLOW'  # can be auto-fixed with --fix-nil
                message = (f"Potential nil access: '{access.var_name}' from {nil_source.source_func}() "
                          f"used without nil check (auto-fixable)")
            else:
                severity = 'YELLOW'  # warning only, needs manual review
                message = (f"Potential nil access: '{access.var_name}' from {nil_source.source_func}() "
                          f"used without nil check")
            
            self.findings.append(Finding(
                pattern_name='potential_nil_access',
                severity=severity,
                line_num=access.access_line,
                message=message,
                details={
                    'var_name': access.var_name,
                    'source_func': nil_source.source_func,
                    'source_call': nil_source.source_call,
                    'assign_line': nil_source.assign_line,
                    'access_call': access.access_call,
                    'access_type': access.access_type,
                    'is_safe_to_fix': access.is_safe_to_fix,
                    'is_local': nil_source.is_local,
                    'reason': reason,
                },
                source_line=self._get_source_line(access.access_line),
            ))

    def _analyze_dead_code(self):
        """Analyze for dead/unreachable code patterns."""
        if not hasattr(self, '_ast_tree') or self._ast_tree is None:
            return
        
        # Phase 1: 100% safe patterns (auto-fixable)
        self._detect_code_after_return()
        self._detect_code_after_break()
        self._detect_if_false_blocks()
        self._detect_while_false_loops()
        self._detect_unnecessary_else()
        self._detect_constant_conditions()
        
        # Phase 2: Warning patterns (not auto-fixable)
        self._detect_unused_local_vars()
        self._detect_unused_local_funcs()

    def _detect_code_after_return(self):
        """Detect unreachable code after unconditional return statements."""
        self._walk_for_dead_after_terminator(Return, 'return')

    def _detect_code_after_break(self):
        """Detect unreachable code after break statements in loops."""
        self._walk_for_dead_after_terminator(Break, 'break')

    def _walk_for_dead_after_terminator(self, terminator_type, terminator_name: str):
        """Walk AST to find dead code after terminators (return/break)."""
        
        def check_block(block_body: List[Node], scope_name: str, in_loop: bool = False):
            """Check a block for dead code after terminators."""
            if not block_body:
                return
            
            for i, stmt in enumerate(block_body):
                # check if this is a terminator
                is_terminator = isinstance(stmt, terminator_type)
                
                # for break, only count as terminator if we're in a loop
                if isinstance(stmt, Break) and not in_loop:
                    continue
                
                if is_terminator and i < len(block_body) - 1:
                    # there are statements after the terminator
                    dead_start = i + 1
                    dead_stmts = block_body[dead_start:]
                    
                    # filter out comments and semicolons
                    real_dead = [s for s in dead_stmts 
                                if not isinstance(s, (Comment, SemiColon))]
                    
                    if real_dead:
                        first_dead = real_dead[0]
                        last_dead = real_dead[-1]
                        start_line = self._get_line(first_dead)
                        end_line = self._get_end_line(last_dead) or start_line
                        
                        # get code preview
                        preview_lines = []
                        for ln in range(start_line, min(start_line + 3, end_line + 1)):
                            if 0 < ln <= len(self.source_lines):
                                preview_lines.append(self.source_lines[ln - 1].rstrip())
                        code_preview = '\n'.join(preview_lines)
                        if end_line > start_line + 2:
                            code_preview += '\n...'
                        
                        self.dead_code.append(DeadCodeInfo(
                            dead_type=f'after_{terminator_name}',
                            start_line=start_line,
                            end_line=end_line,
                            scope_name=scope_name,
                            description=f'Unreachable code after {terminator_name}',
                            is_safe_to_remove=True,
                            code_preview=code_preview,
                            node=first_dead,
                        ))
                        
                        self.findings.append(Finding(
                            pattern_name=f'dead_code_after_{terminator_name}',
                            severity='GREEN',  # safe to auto-fix
                            line_num=start_line,
                            message=f'Unreachable code after {terminator_name} statement (lines {start_line}-{end_line})',
                            details={
                                'dead_type': f'after_{terminator_name}',
                                'start_line': start_line,
                                'end_line': end_line,
                                'scope_name': scope_name,
                                'is_safe_to_remove': True,
                                'dead_stmt_count': len(real_dead),
                            },
                            source_line=self._get_source_line(start_line),
                        ))
                
                # recurse into nested structures
                if isinstance(stmt, (Function, LocalFunction, Method)):
                    if hasattr(stmt, 'body') and stmt.body:
                        body = stmt.body.body if isinstance(stmt.body, Block) else [stmt.body]
                        func_name = self._get_func_name(stmt)
                        check_block(body, func_name, False)
                
                elif isinstance(stmt, If):
                    if hasattr(stmt, 'body') and stmt.body:
                        body = stmt.body.body if isinstance(stmt.body, Block) else [stmt.body]
                        check_block(body, scope_name, in_loop)
                    if hasattr(stmt, 'orelse') and stmt.orelse:
                        if isinstance(stmt.orelse, Block):
                            check_block(stmt.orelse.body, scope_name, in_loop)
                        elif isinstance(stmt.orelse, (If, ElseIf)):
                            check_block([stmt.orelse], scope_name, in_loop)
                
                elif isinstance(stmt, ElseIf):
                    if hasattr(stmt, 'body') and stmt.body:
                        body = stmt.body.body if isinstance(stmt.body, Block) else [stmt.body]
                        check_block(body, scope_name, in_loop)
                    if hasattr(stmt, 'orelse') and stmt.orelse:
                        if isinstance(stmt.orelse, Block):
                            check_block(stmt.orelse.body, scope_name, in_loop)
                        elif isinstance(stmt.orelse, (If, ElseIf)):
                            check_block([stmt.orelse], scope_name, in_loop)
                
                elif isinstance(stmt, (While, Repeat)):
                    if hasattr(stmt, 'body') and stmt.body:
                        body = stmt.body.body if isinstance(stmt.body, Block) else [stmt.body]
                        check_block(body, scope_name, True)  # now in a loop
                
                elif isinstance(stmt, (Fornum, Forin)):
                    if hasattr(stmt, 'body') and stmt.body:
                        body = stmt.body.body if isinstance(stmt.body, Block) else [stmt.body]
                        check_block(body, scope_name, True)  # now in a loop
        
        # start from the root
        if hasattr(self._ast_tree, 'body') and self._ast_tree.body:
            body = self._ast_tree.body.body if isinstance(self._ast_tree.body, Block) else [self._ast_tree.body]
            check_block(body, '<global>', False)

    def _get_func_name(self, node: Node) -> str:
        """Get function name from function node."""
        if isinstance(node, Function):
            return self._node_to_string(node.name) if node.name else '<anon>'
        elif isinstance(node, LocalFunction):
            return node.name.id if isinstance(node.name, Name) else '<anon>'
        elif isinstance(node, Method):
            source = self._node_to_string(node.source)
            method = node.name.id if isinstance(node.name, Name) else ""
            return f"{source}:{method}"
        return '<unknown>'

    def _detect_if_false_blocks(self):
        """Detect 'if false then ... end' blocks."""
        self._walk_for_false_conditions(If, 'if_false')

    def _detect_while_false_loops(self):
        """Detect 'while false do ... end' loops."""
        self._walk_for_false_conditions(While, 'while_false')

    def _walk_for_false_conditions(self, node_type, dead_type: str):
        """Walk AST to find if/while with literal false conditions."""

        def is_literal_false(node: Node) -> bool:
            """Check if node is literal false or nil."""
            return isinstance(node, (FalseExpr, Nil))

        # single flat walk - O(n) instead of O(n²)
        for node in ast.walk(self._ast_tree):
            if isinstance(node, node_type):
                if hasattr(node, 'test') and is_literal_false(node.test):
                    # CRITICAL: only auto-remove `if false then ... end` when
                    # there are no elseif/else branches. With branches, the
                    # if-chain still has reachable code (`if false then ...
                    # elseif cond then BAR end` - BAR runs when cond is true)
                    # and removing the entire chain drops live behaviour.
                    if node_type is If and getattr(node, 'orelse', None) is not None:
                        continue

                    start_line = self._get_line(node)
                    end_line = self._get_end_line(node) or start_line
                    
                    # get code preview
                    preview_lines = []
                    for ln in range(start_line, min(start_line + 3, end_line + 1)):
                        if 0 < ln <= len(self.source_lines):
                            preview_lines.append(self.source_lines[ln - 1].rstrip())
                    code_preview = '\n'.join(preview_lines)
                    if end_line > start_line + 2:
                        code_preview += '\n...'
                    
                    type_name = 'if' if node_type == If else 'while'
                    
                    self.dead_code.append(DeadCodeInfo(
                        dead_type=dead_type,
                        start_line=start_line,
                        end_line=end_line,
                        scope_name='<unknown>',
                        description=f'{type_name} false block (never executes)',
                        is_safe_to_remove=True,
                        code_preview=code_preview,
                        node=node,
                    ))
                    
                    self.findings.append(Finding(
                        pattern_name=f'dead_code_{dead_type}',
                        severity='GREEN',  # safe to auto-fix
                        line_num=start_line,
                        message=f'Dead code: {type_name} false (lines {start_line}-{end_line})',
                        details={
                            'dead_type': dead_type,
                            'start_line': start_line,
                            'end_line': end_line,
                            'scope_name': '<unknown>',
                            'is_safe_to_remove': True,
                        },
                        source_line=self._get_source_line(start_line),
                    ))

    def _detect_unnecessary_else(self):
        """Detect unnecessary else blocks after if blocks that always return/break.
        
        Pattern:
            if condition then
                return x
            else            -- This else is unnecessary
                return y
            end
        
        Can be simplified to:
            if condition then
                return x
            end
            return y
        """
        from luaparser.astnodes import If, ElseIf, Block, Return, Break
        
        def block_always_terminates(body) -> bool:
            """Check if a block always ends with return or break."""
            if not body:
                return False
            
            # Get the body list
            if isinstance(body, Block):
                stmts = body.body if hasattr(body, 'body') else []
            elif isinstance(body, list):
                stmts = body
            else:
                return False
            
            if not stmts:
                return False
            
            # Check if last statement is return or break
            last_stmt = stmts[-1] if stmts else None
            return isinstance(last_stmt, (Return, Break))
        
        def check_if_node(node: If):
            """Check an if node for unnecessary else."""
            if not node.orelse:
                return  # No else clause
            
            # Check if the if body always terminates
            if not block_always_terminates(node.body):
                return  # If body doesn't always terminate, else is needed
            
            # Get else clause info
            orelse = node.orelse
            
            # Handle elseif chain - check if ALL branches terminate
            if isinstance(orelse, ElseIf):
                # For elseif, we need more complex analysis
                # Skip for now - just handle simple if/else
                return
            
            # Simple else block
            if isinstance(orelse, Block):
                else_start = self._get_line(orelse)
                if_start = self._get_line(node)
                
                # Find the 'else' keyword line (should be just before the else block content)
                # The else block starts at the 'else' keyword
                else_keyword_line = else_start
                
                # Look backwards from else block to find 'else' keyword
                for search_line in range(else_start, if_start, -1):
                    line_text = self._get_source_line(search_line)
                    if line_text and line_text.strip().lower() == 'else':
                        else_keyword_line = search_line
                        break
                
                self.findings.append(Finding(
                    pattern_name='unnecessary_else',
                    severity='YELLOW',  # style suggestion, not auto-fix for now
                    line_num=else_keyword_line,
                    message=f'Unnecessary else after return/break - code can be simplified',
                    details={
                        'if_line': if_start,
                        'else_line': else_keyword_line,
                        'suggestion': 'Remove else and dedent the else body',
                    },
                    source_line=self._get_source_line(else_keyword_line),
                ))
        
        def walk(node):
            """Walk AST looking for if statements."""
            if isinstance(node, If):
                check_if_node(node)
            
            # Recurse into children
            for child in self._iter_children(node):
                walk(child)
        
        walk(self._ast_tree)

    def _detect_constant_conditions(self):
        """Detect if/while statements with constant conditions.
        
        Patterns detected:
        - if 1 then (always true)
        - if 0 then (always false in Lua? No - 0 is truthy!)
        - if nil then (always false)
        - if "string" then (always true)
        - while true do (intentional infinite loop - skip)
        - while 1 do (always true)
        
        Note: In Lua, only nil and false are falsy. 0, "", etc are truthy.
        """
        from luaparser.astnodes import If, While, TrueExpr, FalseExpr, Nil, Number, String
        
        def is_constant_truthy(node) -> tuple:
            """Check if a node is a constant truthy/falsy value.
            Returns (is_constant, is_truthy, description)
            """
            # nil is always falsy
            if isinstance(node, Nil):
                return (True, False, 'nil')
            
            # false is always falsy
            if isinstance(node, FalseExpr):
                return (True, False, 'false')
            
            # true is always truthy (but usually intentional)
            if isinstance(node, TrueExpr):
                return (True, True, 'true')
            
            # Numbers are always truthy (including 0!)
            if isinstance(node, Number):
                val = node.n if hasattr(node, 'n') else '?'
                return (True, True, f'number {val}')
            
            # Strings are always truthy (including "")
            if isinstance(node, String):
                return (True, True, 'string literal')
            
            return (False, None, None)
        
        def check_condition(node, node_type: str, line: int):
            """Check a condition node."""
            is_const, is_truthy, desc = is_constant_truthy(node)

            if not is_const:
                return

            # Skip "while true do" - this is intentional infinite loop pattern
            if node_type == 'while' and isinstance(node, TrueExpr):
                return

            # Literal nil/false are already handled by _detect_if_false_blocks /
            # _detect_while_false_loops (which emit dead_code_* patterns). Skipping
            # them here avoids duplicate findings AND ensures the auto-fix path
            # (gated on pattern.startswith('dead_code_')) actually runs for them.
            if isinstance(node, (FalseExpr, Nil)):
                return

            # All remaining cases are "always truthy" - the body always runs and the
            # else (if any) is dead. We do not currently auto-remove this case
            # because removing only the else branch is a more involved AST rewrite.
            if node_type == 'if':
                msg = f'Constant condition: if {desc} (always true, else branch is dead code)'
            else:
                msg = f'Constant condition: while {desc} (infinite loop)'
            severity = 'YELLOW'

            self.findings.append(Finding(
                pattern_name='constant_condition',
                severity=severity,
                line_num=line,
                message=msg,
                details={
                    'node_type': node_type,
                    'condition_desc': desc,
                    'is_truthy': True,
                },
                source_line=self._get_source_line(line),
            ))
        
        def walk(node):
            """Walk AST looking for if/while statements."""
            if isinstance(node, If):
                line = self._get_line(node)
                check_condition(node.test, 'if', line)
            elif isinstance(node, While):
                line = self._get_line(node)
                check_condition(node.test, 'while', line)
            
            # Recurse into children
            for child in self._iter_children(node):
                walk(child)
        
        walk(self._ast_tree)

    def _detect_unused_local_vars(self):
        """Detect local variables that are assigned but never read (Phase 2 - warning only).
        
        Uses scope-aware tracking from the visitor pass - variables are keyed by
        (scope_id, var_name) to correctly handle shadowing.
        """
        # Report unused locals from scope-aware tracking
        for (scope_id, name), info in self.local_vars.items():
            # Skip functions (handled separately) and loop vars
            if info.is_function or info.is_loop_var:
                continue
                
            if not info.is_read:
                # Check if it's used as callback (RegisterScriptCallback)
                if name in self.callback_registrations:
                    continue
                
                # Get scope name for better error message
                scope_name = info.scope.name if info.scope else '<unknown>'
                
                self.findings.append(Finding(
                    pattern_name='unused_local_variable',
                    severity='YELLOW',  # warning only, don't auto-fix
                    line_num=info.assign_line,
                    message=f"Local variable '{name}' is assigned but never used in {scope_name}",
                    details={
                        'var_name': name,
                        'assign_line': info.assign_line,
                        'scope_name': scope_name,
                        'is_safe_to_remove': False,  # not safe - might be intentional
                    },
                    source_line=self._get_source_line(info.assign_line),
                ))

    def _detect_unused_local_funcs(self):
        """Detect local functions that are never called (Phase 2 - warning only).
        
        Uses scope-aware tracking from the visitor pass - functions are keyed by
        (scope_id, func_name) to correctly handle shadowing.
        """
        # Report unused local functions from scope-aware tracking
        for (scope_id, name), info in self.local_funcs.items():
            if not info.is_read:
                # Check if it's used as callback (RegisterScriptCallback)
                if name in self.callback_registrations:
                    continue
                
                # Check if it's a known callback name
                if name in HOT_CALLBACKS or name in SAFE_CALLBACK_PARAMS:
                    continue
                
                # Get scope name for better error message
                scope_name = info.scope.name if info.scope else '<unknown>'
                
                self.findings.append(Finding(
                    pattern_name='unused_local_function',
                    severity='YELLOW',  # warning only
                    line_num=info.assign_line,
                    message=f"Local function '{name}' appears to be unused in {scope_name}",
                    details={
                        'func_name': name,
                        'assign_line': info.assign_line,
                        'scope_name': scope_name,
                        'is_safe_to_remove': False,  # not safe - might be callback
                    },
                    source_line=self._get_source_line(info.assign_line),
                ))

    def _analyze_per_frame_callbacks(self):
        """Analyze per-frame callbacks and flag them with performance info.
        
        These callbacks run every frame and deserve extra attention:
        - actor_on_update
        - npc_on_update
        - monster_on_update
        - physic_object_on_update
        - etc
        """
        # Expensive calls that should be avoided or cached in per-frame callbacks
        EXPENSIVE_CALLS = frozenset({
            'pairs', 'ipairs', 'string.find', 'string.match', 'string.gmatch',
            'string.gsub', 'string.format', 'table.sort', 'table.concat',
            'io.open', 'io.read', 'io.write', 'os.execute',
            'alife', 'alife_object', 'level.object_by_id', 'simulation_objects',
            'get_story_object', 'get_object_by_name',
        })
        
        for callback_info in self.per_frame_callbacks:
            scope = callback_info.scope
            scope_id = id(scope)

            # `_is_descendant_of(s)` returns True iff s == scope or any ancestor
            # of s is scope. The previous implementation only checked the call's
            # immediate scope and its direct parent, so anything inside a nested
            # if/loop/block (which is exactly where the cost lives) was invisible.
            def _is_in_callback(s):
                cur = s
                while cur is not None:
                    if id(cur) == scope_id:
                        return True
                    # stop at the next enclosing FUNCTION boundary - calls
                    # inside nested closures belong to those closures, not us
                    if cur is not s and cur.scope_type == 'function':
                        return False
                    cur = cur.parent
                return False

            # gather statistics about the callback
            calls_in_scope = [c for c in self.calls if c.scope and _is_in_callback(c.scope)]

            # count loops anywhere inside the callback (not just direct children)
            loop_count = sum(
                1 for s in self.scopes
                if s.scope_type == 'loop' and s is not scope and _is_in_callback(s)
            )
            
            # find expensive calls
            expensive_calls = []
            for call in calls_in_scope:
                if call.full_name in EXPENSIVE_CALLS or call.func in EXPENSIVE_CALLS:
                    expensive_calls.append(f"{call.full_name} (line {call.line})")
            
            # find uncached globals used multiple times
            global_usage = defaultdict(list)
            for call in calls_in_scope:
                if call.full_name in CACHEABLE_BARE_GLOBALS or \
                   (call.module and call.module in CACHEABLE_MODULE_FUNCS):
                    global_usage[call.full_name].append(call.line)
            
            uncached_globals = []
            for name, lines in global_usage.items():
                if len(lines) >= 2 and name not in scope.cached_globals:
                    uncached_globals.append(f"{name} ({len(lines)}x)")
            
            # build message
            issues = []
            if loop_count > 0:
                issues.append(f"{loop_count} loop(s)")
            if expensive_calls:
                issues.append(f"{len(expensive_calls)} expensive call(s)")
            if uncached_globals:
                issues.append(f"{len(uncached_globals)} uncached global(s)")
            
            # determine severity based on issues found
            if expensive_calls or loop_count > 0:
                severity = 'RED'
            elif uncached_globals:
                severity = 'YELLOW'
            else:
                severity = 'DEBUG'
            
            # always report per-frame callbacks
            message = f"Per-frame callback: {callback_info.name} (lines {callback_info.start_line}-{callback_info.end_line})"
            if issues:
                message += f" - {', '.join(issues)}"
            
            self.findings.append(Finding(
                pattern_name='per_frame_callback',
                severity=severity,
                line_num=callback_info.start_line,
                message=message,
                details={
                    'callback_name': callback_info.name,
                    'start_line': callback_info.start_line,
                    'end_line': callback_info.end_line,
                    'loop_count': loop_count,
                    'expensive_calls': expensive_calls,
                    'uncached_globals': uncached_globals,
                    'total_calls': len(calls_in_scope),
                },
                source_line=self._get_source_line(callback_info.start_line),
            ))

    # @TODO: Check distance_to() implementation in xray source code more thoroughly
    def _analyze_distance_to_comparisons(self):
        """
        Find distance_to() calls in comparisons that can use distance_to_sqr() instead.
        Should replace compared value with its square too.
        
        Pattern: pos:distance_to(target) < 10
        Optimized: pos:distance_to_sqr(target) < 100  -- (10^2, avoids sqrt)
        
        This is auto-fixable and provides performance improvement
        since distance_to() requires a square root operation.
        """
        for comp in self.distance_comparisons:
            squared_threshold = comp.threshold_value ** 2
            
            # format the squared value nicely
            if squared_threshold == int(squared_threshold):
                squared_str = str(int(squared_threshold))
            else:
                squared_str = f"{squared_threshold:.6g}"
            
            original = f"{comp.source_obj}:distance_to({comp.target_obj}) {comp.comparison_op} {comp.threshold_value}"
            optimized = f"{comp.source_obj}:distance_to_sqr({comp.target_obj}) {comp.comparison_op} {squared_str}"
            
            self.findings.append(Finding(
                pattern_name='distance_to_comparison',
                severity='GREEN',  # Auto-fixable
                line_num=comp.line,
                message=f'Use distance_to_sqr() to avoid sqrt: {original} -> {optimized}',
                details={
                    'source_obj': comp.source_obj,
                    'target_obj': comp.target_obj,
                    'comparison_op': comp.comparison_op,
                    'original_threshold': comp.threshold_value,
                    'squared_threshold': squared_threshold,
                    'squared_threshold_str': squared_str,
                    'invoke_node': comp.invoke_node,
                    'threshold_node': comp.threshold_node,
                    'full_node': comp.full_node,
                },
                source_line=self._get_source_line(comp.line),
            ))

    def _analyze_vector_allocations_in_loops(self):
        """
        Find vector() allocations inside loops.
        
        Each vector() call allocates memory that must be garbage collected.
        In loops, especially in per-frame callbacks, this can cause significant
        GC pressure and frame drops.
        
        Solution: Pre-allocate vectors at module level and reuse with :set()
        
        Example:
            -- Bad: allocates every iteration
            for i = 1, 100 do
                local pos = vector():set(x, y, z)
            end
            
            -- Good: reuse pre-allocated vector
            local temp_vec = vector()  -- at module level
            for i = 1, 100 do
                temp_vec:set(x, y, z)
            end
        
        NOT auto-fixable because it requires:
        1. Moving allocation to module/function level
        2. Understanding which vectors can be safely reused
        3. Ensuring no aliasing issues
        """
        for alloc in self.vector_allocations:
            severity = 'RED'
            
            context = ""
            if alloc.in_per_frame_callback:
                context = " in per-frame callback"
            if alloc.loop_depth > 1:
                context += f" (nested {alloc.loop_depth} loops deep)"
            
            message = f"vector() allocation in loop{context} - pre-allocate and reuse with :set()"
            
            self.findings.append(Finding(
                pattern_name='vector_alloc_in_loop',
                severity=severity,
                line_num=alloc.line,
                message=message,
                details={
                    'loop_depth': alloc.loop_depth,
                    'in_per_frame_callback': alloc.in_per_frame_callback,
                    'node': alloc.call_node,
                },
                source_line=self._get_source_line(alloc.line),
            ))

    def _analyze_anon_hoist(self):
        """Opt-in: lift function() ... end to a named local at file scope."""
        from anon_hoist import analyze_tree
        analyze_tree(self)

    def _get_source_line(self, line_num: int) -> str:
        """Get source line by number."""
        if 0 < line_num <= len(self.source_lines):
            return self.source_lines[line_num - 1].rstrip()
        return ""


def analyze_file(file_path: Path, cache_threshold: int = 4, experimental: bool = False) -> List[Finding]:
    """Convenience function to analyze a file."""
    analyzer = ASTAnalyzer(cache_threshold=cache_threshold, experimental=experimental)
    return analyzer.analyze_file(file_path)
