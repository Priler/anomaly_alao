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
import sys
import io

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

# Functions/properties that can return nil - calling methods on these without
# nil checks can cause CTD (crash to desktop)
# Format: full_name -> description of when it returns nil
NIL_RETURNING_FUNCTIONS = {
    # level functions
    'level.object_by_id': 'object is offline or does not exist',
    'level.get_target_obj': 'nothing under crosshair',
    'level.get_target_element': 'nothing under crosshair',
    
    # alife functions
    'alife': 'called from main menu or during loading',
    'alife().object': 'object does not exist in simulation',
    'alife():object': 'object does not exist in simulation',
    
    # game object methods that can return nil
    ':parent': 'object has no parent (not in inventory)',
    ':best_enemy': 'no enemy detected',
    ':best_item': 'no item of interest found',
    ':best_danger': 'no danger detected',
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
    
    # common patterns
    'db.actor': 'called from main menu or during loading',
    'db.storage': 'storage not initialized',  # db.storage[id] can be nil
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
        
        # nil access tracking
        self.nil_sources: Dict[Tuple[int, str], NilSourceInfo] = {}  # (scope_id, var_name) -> nil source info
        self.nil_accesses: List[NilAccessInfo] = []      # potential nil accesses
        self.nil_guards: Set[Tuple[str, int]] = set()    # (var_name, line) pairs where nil check exists
        
        # dead code tracking
        self.dead_code: List[DeadCodeInfo] = []
        self.local_vars: Dict[Tuple[int, str], LocalVarInfo] = {}  # (scope_id, name) -> info
        self.local_funcs: Dict[Tuple[int, str], LocalVarInfo] = {}  # (scope_id, name) -> info
        self.callback_registrations: Set[str] = set()  # names registered as callbacks

        self.source_lines: List[str] = []
        self.source: str = ""
        self.file_path: Optional[Path] = None

        self.loop_depth: int = 0
        self.function_depth: int = 0

    def analyze_file(self, file_path: Path) -> List[Finding]:
        """Analyze a Lua file and return findings."""
        self.reset()
        self.file_path = file_path
        self._ast_tree = None

        try:
            # use latin-1 encoding which maps bytes 0-255 directly to unicode 0-255
            # this preserves non-UTF-8 characters (like Windows-1252 bullet points)
            # and allows the Lua parser to work correctly
            self.source = file_path.read_text(encoding='latin-1')
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
            # use single quotes if string contains double quotes
            if '"' in s and "'" not in s:
                return f"'{s}'"
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

        # visit targets (for calls inside index expressions like db.storage[npc:id()])
        for target in node.targets:
            self._visit(target)

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
        
        if source_func:
            key = (id(self.current_scope), target)
            self.nil_sources[key] = NilSourceInfo(
                var_name=target,
                source_call=value_repr,
                source_func=source_func,
                assign_line=line,
                scope=self.current_scope,
                is_local=is_local,
                is_guarded=False,
            )
        else:
            # if variable is reassigned from non-nil source, remove from tracking
            key = (id(self.current_scope), target)
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

    def _has_nil_guard(self, var_name: str, assign_line: int, access_line: int) -> bool:
        """Check if there's a nil guard between assignment and access."""
        if assign_line >= access_line:
            return False
        
        # check lines between assignment and access for nil guard patterns
        for line_num in range(assign_line, access_line):
            if line_num <= 0 or line_num > len(self.source_lines):
                continue
            line_text = self.source_lines[line_num - 1]
            
            # check for common nil guard patterns
            # if var then / if var and / if not var then return
            patterns = [
                f'if {var_name} then',
                f'if {var_name} and',
                f'if not {var_name} then',
                f'if {var_name} ~= nil',
                f'if {var_name} == nil then return',
                f'{var_name} and {var_name}:',
                f'{var_name} and {var_name}.',
            ]
            for pattern in patterns:
                if pattern in line_text:
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
        
        # Check for potential nil access
        self._check_nil_access(node.source, source, full_name, line, 'method')

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
        self._analyze_nil_access()
        self._analyze_dead_code()

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
        # NOTE: time_global() is NOT included because it returns different values
        # each call (current time) - caching it breaks elapsed time calculations
        # NOTE: level.object_by_id() is NOT auto-fixed because different IDs give
        # different objects, and even same IDs can change if object is destroyed
        expensive_calls = {'db.actor', 'alife', 'system_ini', 'game_ini', 'getFS',
                           'device', 'get_console', 'get_hud', 'level.name'}

        # method calls that are safe to cache (immutable object properties)
        # based on X-Ray engine source analysis:
        # - :section() returns stored NameSection member (xr_object.h:155)
        # - :id() returns stored Props.net_ID member (xr_object.h:98)
        # - :clsid() returns stored m_script_clsid member (GameObject.h:257)
        # - :story_id() returns m_story_id set once from config (xrServer_Objects_ALife.cpp:375)
        cacheable_methods = {'section', 'id', 'clsid', 'story_id'}

        # group by function scope
        scope_calls: Dict[Scope, Dict[str, List[CallInfo]]] = defaultdict(lambda: defaultdict(list))

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

        for func_scope, calls_by_name in scope_calls.items():
            for name, calls in calls_by_name.items():
                threshold = 2 if func_scope.is_hot_callback else 3

                if len(calls) >= threshold:
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
                
                if loop_scope:
                    # look for var = "" or var = '' IMMEDIATELY before the loop
                    # must be: within 3 lines, NOT inside any loop, and must be local declaration
                    for assign in self.assigns:
                        if (assign.target == var and 
                            assign.value_type == 'literal' and
                            assign.value_repr in ('""', "''") and
                            assign.line < loop_scope.start_line and
                            assign.line >= loop_scope.start_line - 3 and  # must be within 3 lines
                            not assign.in_loop and  # init must NOT be inside any loop
                            assign.is_local):  # must be a local declaration
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

    def _detect_unused_local_vars(self):
        """Detect local variables that are assigned but never read (Phase 2 - warning only)."""
        
        local_vars: Dict[str, LocalVarInfo] = {}
        assignment_targets: Set[int] = set()  # ids of Name nodes that are assignment targets
        
        # First pass: collect all local variable assignments - O(n)
        for node in ast.walk(self._ast_tree):
            if isinstance(node, LocalAssign):
                line = self._get_line(node)
                for target in node.targets:
                    if isinstance(target, Name):
                        assignment_targets.add(id(target))
                        var_name = target.id
                        # don't track if it starts with _ (intentionally unused)
                        if not var_name.startswith('_'):
                            local_vars[var_name] = LocalVarInfo(
                                name=var_name,
                                assign_line=line,
                                scope=self.current_scope,
                                is_read=False,
                                is_function=False,
                            )
            
            elif isinstance(node, LocalFunction):
                line = self._get_line(node)
                if isinstance(node.name, Name):
                    assignment_targets.add(id(node.name))
                    func_name = node.name.id
                    if not func_name.startswith('_'):
                        local_vars[func_name] = LocalVarInfo(
                            name=func_name,
                            assign_line=line,
                            scope=self.current_scope,
                            is_read=False,
                            is_function=True,
                        )
            
            # Also track for loop variables as assigned
            elif isinstance(node, Fornum):
                if isinstance(node.target, Name):
                    assignment_targets.add(id(node.target))
                    var_name = node.target.id
                    if not var_name.startswith('_'):
                        local_vars[var_name] = LocalVarInfo(
                            name=var_name,
                            assign_line=self._get_line(node),
                            scope=self.current_scope,
                            is_read=False,
                            is_function=False,
                            is_loop_var=True,
                        )
            
            elif isinstance(node, Forin):
                if hasattr(node, 'targets'):
                    for target in node.targets:
                        if isinstance(target, Name):
                            assignment_targets.add(id(target))
                            var_name = target.id
                            if not var_name.startswith('_'):
                                local_vars[var_name] = LocalVarInfo(
                                    name=var_name,
                                    assign_line=self._get_line(node),
                                    scope=self.current_scope,
                                    is_read=False,
                                    is_function=False,
                                    is_loop_var=True,
                                )
        
        # Second pass: find all reads - O(n)
        for node in ast.walk(self._ast_tree):
            if isinstance(node, Name):
                # only count as read if NOT an assignment target
                if id(node) not in assignment_targets:
                    var_name = node.id
                    if var_name in local_vars:
                        local_vars[var_name].is_read = True
            
            # Check RegisterScriptCallback for callback registration
            elif isinstance(node, Call):
                func_name = self._node_to_string(node.func)
                if func_name == 'RegisterScriptCallback' and len(node.args) >= 2:
                    callback_func = self._node_to_string(node.args[1])
                    if callback_func:
                        self.callback_registrations.add(callback_func)
        
        # report unused locals
        for name, info in local_vars.items():
            if not info.is_read and not info.is_function and not info.is_loop_var:
                # check if it's used as callback (RegisterScriptCallback)
                if name in self.callback_registrations:
                    continue
                
                self.findings.append(Finding(
                    pattern_name='unused_local_variable',
                    severity='YELLOW',  # warning only, don't auto-fix
                    line_num=info.assign_line,
                    message=f"Local variable '{name}' is assigned but never used",
                    details={
                        'var_name': name,
                        'assign_line': info.assign_line,
                        'is_safe_to_remove': False,  # not safe - might be intentional
                    },
                    source_line=self._get_source_line(info.assign_line),
                ))

    def _detect_unused_local_funcs(self):
        """Detect local functions that are never called (Phase 2 - warning only)."""
        # Track local function definitions and calls
        local_funcs: Dict[str, LocalVarInfo] = {}
        called_funcs: Set[str] = set()
        
        # Single pass through AST - O(n)
        for node in ast.walk(self._ast_tree):
            if isinstance(node, LocalFunction):
                line = self._get_line(node)
                if isinstance(node.name, Name):
                    func_name = node.name.id
                    if not func_name.startswith('_'):
                        local_funcs[func_name] = LocalVarInfo(
                            name=func_name,
                            assign_line=line,
                            scope=self.current_scope,
                            is_read=False,
                            is_function=True,
                        )
            
            elif isinstance(node, Call):
                func_name = self._node_to_string(node.func)
                if func_name:
                    called_funcs.add(func_name)
                
                # check for RegisterScriptCallback
                if func_name == 'RegisterScriptCallback' and len(node.args) >= 2:
                    callback_func = self._node_to_string(node.args[1])
                    if callback_func:
                        self.callback_registrations.add(callback_func)
                        called_funcs.add(callback_func)
            
            elif isinstance(node, Name):
                # function reference (not call)
                called_funcs.add(node.id)
        
        # report unused local functions
        for name, info in local_funcs.items():
            if name not in called_funcs and name not in self.callback_registrations:
                # check if it's a known callback name
                if name in HOT_CALLBACKS or name in SAFE_CALLBACK_PARAMS:
                    continue
                
                self.findings.append(Finding(
                    pattern_name='unused_local_function',
                    severity='YELLOW',  # warning only
                    line_num=info.assign_line,
                    message=f"Local function '{name}' appears to be unused",
                    details={
                        'func_name': name,
                        'assign_line': info.assign_line,
                        'is_safe_to_remove': False,  # not safe - might be callback
                    },
                    source_line=self._get_source_line(info.assign_line),
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
