"""
AST-based Lua source transformer
This fixes Lua source code based on AST analysis findings
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import shutil

from ast_analyzer import analyze_file, ASTAnalyzer, Scope
from models import Finding


# Lua reserved keywords. Used to distinguish a grouping paren after a keyword
# (e.g. `if (foo())`) from a real function-call name preceding a paren
_LUA_KEYWORDS = frozenset({
    'and', 'break', 'do', 'else', 'elseif', 'end', 'false', 'for',
    'function', 'goto', 'if', 'in', 'local', 'nil', 'not', 'or',
    'repeat', 'return', 'then', 'true', 'until', 'while',
})


@dataclass
class SourceEdit:
    """A source code edit with character positions."""
    start_char: int      # start character offset in source
    end_char: int        # end character offset (exclusive)
    replacement: str     # replacement text
    priority: int = 0    # higher priority edits applied first
    # Optional grouping: links an insertion (e.g. `local tostr = tostring`) to
    # the replacement edits it enables (e.g. tostring -> tostr at call sites).
    # If all linked replacements get rejected (because they overlap a higher-
    # priority edit such as a debug-statement comment-out), the insertion is
    # dropped too -- otherwise we'd leave a dead `local X = ...` declaration
    # with no callers.
    group_id: Optional[int] = None
    # When set on an insertion, marks it as an "enabler" that should be dropped
    # if its group has no surviving replacements. Replacements in the same
    # group leave this empty.
    is_enabler: bool = False
    # Same-position insertions apply high-seq first so they land later in the
    # file (end-to-start apply). Used to keep foo_anon_1 above foo_anon_2.
    seq: int = 0
    # Anon insert: (fn_start, fn_end). Other GREEN edits inside that span
    # get folded into the anon func.
    anon_meta: Optional[tuple] = None


class ASTTransformer:
    """Transform Lua source using AST-based analysis."""

    def __init__(self):
        self.source: str = ""
        self.edits: List[SourceEdit] = []
        self.file_path: Optional[Path] = None
        self._next_group_id: int = 1
        self.analyzer: Optional[ASTAnalyzer] = None
        self._line_offsets: List[int] = []  # cached line start offsets

    def _compute_line_offsets(self):
        """Compute and cache line start offsets for efficient lookups."""
        self._line_offsets = [0]
        for i, char in enumerate(self.source):
            if char == '\n':
                self._line_offsets.append(i + 1)

    def transform_file(self, file_path: Path, backup: bool = True, dry_run: bool = False,
                       fix_debug: bool = False, fix_yellow: bool = False,
                       experimental: bool = False, fix_nil: bool = False,
                       remove_dead_code: bool = False,
                       cache_threshold: int = 4,
                       hoist_anon_funcs: bool = False) -> Tuple[bool, str, int]:
        """
        Transform a file based on findings.
        Returns (was_modified, new_content, edit_count).
        
        Args:
            fix_nil: If True, auto-fix safe nil access patterns
            remove_dead_code: If True, remove 100% safe dead code (after return, if false, etc.)
            cache_threshold: Minimum call count to trigger caching suggestions (default: 4)
        """
        self.file_path = file_path
        self.edits = []
        self._next_group_id = 1
        self.experimental = experimental
        self.fix_nil = fix_nil
        self.remove_dead_code = remove_dead_code
        self.hoist_anon_funcs = bool(hoist_anon_funcs)

        # run analyzer with user-specified cache_threshold
        self.analyzer = ASTAnalyzer(cache_threshold=cache_threshold, experimental=experimental)
        findings = self.analyzer.analyze_file(file_path)

        # get source from analyzer and compute line offsets
        self.source = self.analyzer.source
        self._compute_line_offsets()

        # filter to fixable severities
        allowed_severities = {'GREEN'}
        if fix_yellow:
            allowed_severities.add('YELLOW')
        if fix_debug:
            allowed_severities.add('DEBUG')

        # Hoist is opt-in. Other GREEN still runs.
        fixable = [f for f in findings if f.severity in allowed_severities
                   and f.pattern_name != 'anon_hoist']

        if self.hoist_anon_funcs:
            fixable.extend(f for f in findings if f.pattern_name == 'anon_hoist')
        
        # add experimental fixes (string_concat_in_loop) if enabled
        # only add if not already included via fix_yellow
        if experimental and not fix_yellow:
            experimental_fixes = [f for f in findings 
                                  if f.pattern_name == 'string_concat_in_loop' 
                                  and f.severity == 'YELLOW']
            fixable.extend(experimental_fixes)
        
        # add safe nil fixes if enabled
        if fix_nil:
            nil_fixes = [f for f in findings 
                        if f.pattern_name == 'potential_nil_access'
                        and f.details.get('is_safe_to_fix', False)]
            # only add if not already in fixable
            existing_lines = {f.line_num for f in fixable}
            for nf in nil_fixes:
                if nf.line_num not in existing_lines:
                    fixable.append(nf)
        
        # add safe dead code removal if enabled
        if remove_dead_code:
            dead_code_fixes = [f for f in findings
                              if f.pattern_name.startswith('dead_code_')
                              and f.details.get('is_safe_to_remove', False)]
            existing_lines = {f.line_num for f in fixable}
            for df in dead_code_fixes:
                if df.line_num not in existing_lines:
                    fixable.append(df)

        if not fixable:
            return False, self.source, 0

        # generate edits for each finding
        for finding in fixable:
            self._generate_edits(finding)

        if not self.edits:
            return False, self.source, 0

        self._fold_inner_into_anon()

        edit_count = len(self.edits)

        # apply edits
        new_content = self._apply_edits()

        if new_content == self.source:
            return False, self.source, 0

        if not dry_run:
            if backup:
                # use .alao-bak extension to distinguish from mod author backups
                backup_path = file_path.with_suffix(file_path.suffix + '.alao-bak')
                if not backup_path.exists():
                    shutil.copy2(file_path, backup_path)

            file_path.write_text(new_content, encoding=getattr(self.analyzer, '_file_encoding', 'latin-1'))

        return True, new_content, edit_count

    def _generate_edits(self, finding: Finding):
        """Generate source edits for a finding."""
        pattern = finding.pattern_name

        if pattern == 'table_insert_append':
            self._edit_table_insert(finding)
        elif pattern == 'table_getn':
            self._edit_table_getn(finding)
        elif pattern == 'string_len':
            self._edit_string_len(finding)
        elif pattern == 'math_pow_simple':
            self._edit_math_pow(finding)
        elif pattern == 'pow_op_simple':
            self._edit_pow_op_simple(finding)
        elif pattern == 'string_literal_concat':
            self._edit_string_literal_concat(finding)
        elif pattern == 'string_find_plain':
            self._edit_string_find_plain(finding)
        elif pattern == 'redundant_not_eq':
            self._edit_redundant_not_eq(finding)
        elif pattern == 'debug_statement':
            self._edit_debug_statement(finding)
        elif pattern == 'uncached_globals_summary':
            self._edit_uncached_globals(finding)
        elif pattern == 'string_concat_in_loop':
            if getattr(self, 'experimental', False):
                self._edit_string_concat_in_loop(finding)
        elif pattern == 'potential_nil_access':
            if getattr(self, 'fix_nil', False):
                self._edit_nil_access(finding)
        elif pattern.startswith('dead_code_'):
            if getattr(self, 'remove_dead_code', False):
                self._edit_dead_code(finding)
        elif pattern.startswith('repeated_'):
            self._edit_repeated_calls(finding)
        elif pattern == 'distance_to_comparison':
            self._edit_distance_to_comparison(finding)
        elif pattern == 'anon_hoist':
            if getattr(self, 'hoist_anon_funcs', False):
                self._edit_anon(finding)


    # Edit methods using AST positions

    def _edit_anon(self, finding: Finding):
        """Insert the named func at file scope. Replace `function() ... end` with the name."""
        d = finding.details or {}
        if not d.get('safe'):
            return
        anon_text = d.get('anon_text')
        insert_char = d.get('insert_char')
        replace_start = d.get('replace_start')
        replace_end = d.get('replace_end')
        replace_text = d.get('replace_text')
        if not anon_text or insert_char is None:
            return
        gid = self._next_group_id
        self._next_group_id += 1
        absorbed = bool(d.get('absorb_callsite'))
        fn_start = d.get('fn_start')
        fn_end = d.get('fn_end')
        meta = (fn_start, fn_end) if fn_start is not None and fn_end is not None else None
        if absorbed:
            # Already pasted into an outer. Only insert this named func.
            self.edits.append(SourceEdit(
                start_char=insert_char,
                end_char=insert_char,
                replacement=anon_text,
                priority=100,
                seq=int(d.get('insert_seq') or 0),
                anon_meta=meta,
            ))
            return
        if replace_start is None or replace_end is None or replace_text is None:
            return
        self.edits.append(SourceEdit(
            start_char=insert_char,
            end_char=insert_char,
            replacement=anon_text,
            priority=100,
            group_id=gid,
            is_enabler=True,
            seq=int(d.get('insert_seq') or 0),
            anon_meta=meta,
        ))
        self.edits.append(SourceEdit(
            start_char=replace_start,
            end_char=replace_end,
            replacement=replace_text,
            priority=50,
            group_id=gid,
        ))

    def _edit_table_insert(self, finding: Finding):
        """Convert table.insert(t, v) to t[#t+1] = v."""
        node = finding.details.get('node')
        if not node:
            return

        table_name = finding.details.get('table', '')
        if not table_name:
            return

        # get position from node tokens
        start, end = self._get_node_span(node)
        if start is None:
            return

        # extract value from source
        call_text = self.source[start:end]
        value = self._extract_table_insert_value(call_text, table_name)
        if not value:
            return

        replacement = f'{table_name}[#{table_name}+1] = {value}'

        self.edits.append(SourceEdit(
            start_char=start,
            end_char=end,
            replacement=replacement,
        ))

    def _extract_table_insert_value(self, call_text: str, table_name: str) -> Optional[str]:
        """Extract the value argument from table.insert(t, v) call text.
        
        Handles:
        - Regular strings: "..." and '...'
        - Long strings: [[...]] and [=[...]=] (with any number of =)
        - Nested parentheses, braces, and brackets
        """
        # find opening paren
        paren_start = call_text.find('(')
        if paren_start == -1:
            return None

        # find comma after table name
        comma_pos = call_text.find(',', paren_start)
        if comma_pos == -1:
            return None

        value_start = comma_pos + 1

        # find matching closing paren with proper tracking
        depth = 1
        brace_depth = 0
        in_string = False
        string_char = None
        in_long_string = False
        long_string_level = 0  # number of = signs in long string delimiter
        i = paren_start + 1

        while i < len(call_text) and depth > 0:
            c = call_text[i]

            if in_long_string:
                # Look for closing long string delimiter: ]=*]
                if c == ']':
                    # Check if this is the closing delimiter
                    # Need to match the same number of = signs
                    if i + 1 + long_string_level < len(call_text):
                        expected_close = ']' + '=' * long_string_level + ']'
                        if call_text[i:i + len(expected_close)] == expected_close:
                            in_long_string = False
                            i += len(expected_close)
                            continue
                i += 1
                continue
            
            if in_string:
                # Check for escaped quote or end of string
                if c == string_char:
                    # Check if escaped (count preceding backslashes)
                    num_backslashes = 0
                    j = i - 1
                    while j >= 0 and call_text[j] == '\\':
                        num_backslashes += 1
                        j -= 1
                    # If even number of backslashes, quote is not escaped
                    if num_backslashes % 2 == 0:
                        in_string = False
                i += 1
                continue

            # Not in any string - check what we have
            if c in ('"', "'"):
                in_string = True
                string_char = c
            elif c == '[':
                # Check for long string start: [=*[
                # Count = signs
                eq_count = 0
                j = i + 1
                while j < len(call_text) and call_text[j] == '=':
                    eq_count += 1
                    j += 1
                # Check if followed by [
                if j < len(call_text) and call_text[j] == '[':
                    # This is a long string
                    in_long_string = True
                    long_string_level = eq_count
                    i = j + 1  # skip past the opening [[
                    continue
                # Otherwise it's a regular bracket (for indexing)
                # Don't track bracket depth - it's handled by context
            elif c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == '{':
                brace_depth += 1
            elif c == '}':
                brace_depth -= 1

            i += 1

        if depth != 0:
            return None

        value = call_text[value_start:i - 1].strip()
        return value

    def _edit_table_getn(self, finding: Finding):
        """Convert table.getn(t) to #t."""
        node = finding.details.get('node')
        if not node:
            return

        table_name = finding.details.get('table', '')
        if not table_name:
            return

        start, end = self._get_node_span(node)
        if start is None:
            return

        self.edits.append(SourceEdit(
            start_char=start,
            end_char=end,
            replacement=f'#{table_name}',
        ))

    def _edit_string_len(self, finding: Finding):
        """Convert string.len(s) to #s."""
        node = finding.details.get('node')
        if not node:
            return

        str_name = finding.details.get('string', '')
        if not str_name:
            return

        start, end = self._get_node_span(node)
        if start is None:
            return

        self.edits.append(SourceEdit(
            start_char=start,
            end_char=end,
            replacement=f'#{str_name}',
        ))

    def _edit_math_pow(self, finding: Finding):
        """Convert math.pow(x, n) to x^n or x*x*..."""
        node = finding.details.get('node')
        if not node:
            return

        base = finding.details.get('base', '')
        exp = finding.details.get('exponent')
        pow_type = finding.details.get('type')

        if not base:
            return

        start, end = self._get_node_span(node)
        if start is None:
            return

        if pow_type == 'sqrt':
            # `^` already binds tighter than every surrounding operator we
            # would replace into, so no parens needed.
            replacement = f'{base}^0.5'
        elif pow_type == 'power' and isinstance(exp, int):
            # Wrap multi-MUL replacement in parens: the original `math.pow(x,2)`
            # is a single primary expression, but `x*x` introduces a binary
            # operator. Without parens, surroundings like `1/math.pow(x,2)`
            # would silently rewrite to `1/x*x` ((1/x)*x - wrong) instead of
            # `1/(x*x)`.
            replacement = '(' + '*'.join([base] * exp) + ')'
        else:
            return

        self.edits.append(SourceEdit(
            start_char=start,
            end_char=end,
            replacement=replacement,
        ))

    def _edit_string_literal_concat(self, finding: Finding):
        """Replace an all-literal Concat tree with the folded string literal."""
        node = finding.details.get('node')
        combined = finding.details.get('combined')
        if not node or not combined:
            return
        start, end = self._get_node_span(node)
        if start is None:
            return
        self.edits.append(SourceEdit(
            start_char=start,
            end_char=end,
            replacement=combined,
        ))

    def _edit_string_find_plain(self, finding: Finding):
        """Add the plain-search flag to `string.find` with a plain-text needle.

        2-arg form: `string.find(s, "x")`        -> insert `, 1, true`
        3-arg form: `string.find(s, "x", N)`     -> insert `, true`

        The insertion goes right after the last argument's source span.
        We can't rely on the Call's `last_token` being the closing paren -
        when the call expression is wrapped in extra parens (e.g. inside
        an `if (foo(x))` test) luaparser's last_token is the outer `)`,
        and inserting there lands the new args OUTSIDE the call.
        """
        node = finding.details.get('node')
        n_args = finding.details.get('n_args')
        if not node or n_args not in (2, 3):
            return

        # Use the last argument's end position as the insertion point.
        args = getattr(node, 'args', None) or []
        if not args:
            return
        last_arg = args[-1]
        _, arg_end = self._get_node_span(last_arg)
        if arg_end is None or arg_end <= 0 or arg_end > len(self.source):
            return

        addition = ', 1, true' if n_args == 2 else ', true'
        self.edits.append(SourceEdit(
            start_char=arg_end,
            end_char=arg_end,
            replacement=addition,
        ))

    def _edit_redundant_not_eq(self, finding: Finding):
        """Rewrite `not (a == b)` to `a ~= b` (and the dual).

        We replace the entire ULNotOp source span with `<L> <newop> <R>`,
        using the original source text for L and R (so any user-chosen
        parenthesization or whitespace within the operands is preserved).

        Edge case: if the ULNotOp's source span starts with `(`, that means
        the ULNotOp was syntactically wrapped in parens by an enclosing
        context (e.g. inside another `not (...)`). Stripping those parens
        in the rewrite would change precedence - `not (not (a == b))` would
        end up as `not a ~= b`, parsed as `(not a) ~= b`. Detect that case
        and re-wrap the replacement in parens.
        """
        outer = finding.details.get('outer_node')
        left = finding.details.get('left_node')
        right = finding.details.get('right_node')
        new_op = finding.details.get('new_op')
        if outer is None or left is None or right is None or not new_op:
            return

        outer_start, outer_end = self._get_node_span(outer)
        left_start, left_end = self._get_node_span(left)
        right_start, right_end = self._get_node_span(right)
        if (outer_start is None or outer_end is None or
                left_start is None or left_end is None or
                right_start is None or right_end is None):
            return

        left_text = self.source[left_start:left_end]
        right_text = self.source[right_start:right_end]
        # Sanity check: the extracted operand text must look like a valid
        # Lua expression (start with `(`, identifier char, digit, string
        # quote, or `-` for unary minus). If it starts with `[` or `]` the
        # span resolution lost a leading base identifier - bail out rather
        # than emit invalid Lua.
        for txt in (left_text, right_text):
            if not txt:
                return
            c = txt[0]
            if not (c.isalnum() or c in '_("\'-{ '):
                return
        replacement = f'{left_text} {new_op} {right_text}'

        # If the rewritten span begins with `(` rather than `not`, the parens
        # belong to whatever wraps us - keep them in the replacement.
        first_tok = getattr(outer, 'first_token', None)
        if first_tok is not None and "='('" in str(first_tok):
            replacement = f'({replacement})'

        self.edits.append(SourceEdit(
            start_char=outer_start,
            end_char=outer_end,
            replacement=replacement,
        ))

    def _edit_pow_op_simple(self, finding: Finding):
        """Convert `x ^ 2` / `x ^ 3` to `(x*x)` / `(x*x*x)`.

        Same precedence concern as _edit_math_pow: wrap in parens so that
        `1 / x^2` and `-x^2` keep meaning `1/(x*x)` and `-(x*x)` after the
        rewrite. Strictly speaking `-x^2` rewrites to `-(x*x)` which equals
        `-x*x` either way (unary minus has lower precedence than `*`), but
        the universal rule is simpler and never wrong.
        """
        node = finding.details.get('node')
        replacement = finding.details.get('replacement')
        if not node or not replacement:
            return
        start, end = self._get_node_span(node)
        if start is None:
            return
        self.edits.append(SourceEdit(
            start_char=start,
            end_char=end,
            replacement=f'({replacement})',
        ))

    def _edit_distance_to_comparison(self, finding: Finding):
        """
        Convert distance_to() comparison to distance_to_sqr().
        Compared values should be replaced with square.
        
        Example:
            pos:distance_to(target) < 10
        Becomes:
            pos:distance_to_sqr(target) < 100
        
        This avoids the sqrt operation inside distance_to().
        """
        invoke_node = finding.details.get('invoke_node')
        threshold_node = finding.details.get('threshold_node')
        squared_threshold_str = finding.details.get('squared_threshold_str', '')
        
        if not invoke_node or not threshold_node or not squared_threshold_str:
            return
        
        # edit 1: change distance_to to distance_to_sqr in the method name
        # get the span of the invoke node and find "distance_to" within it
        invoke_start, invoke_end = self._get_node_span(invoke_node)
        if invoke_start is not None:
            invoke_text = self.source[invoke_start:invoke_end]
            # find ":distance_to(" pattern
            method_idx = invoke_text.find(':distance_to(')
            if method_idx != -1:
                # position of "distance_to" (after the colon)
                method_name_start = invoke_start + method_idx + 1  # +1 to skip ':'
                method_name_end = method_name_start + len('distance_to')
                
                self.edits.append(SourceEdit(
                    start_char=method_name_start,
                    end_char=method_name_end,
                    replacement='distance_to_sqr',
                    priority=1,  # apply method name change first
                ))
        
        # edit 2: change the threshold value to its squared version
        threshold_start, threshold_end = self._get_node_span(threshold_node)
        if threshold_start is not None:
            self.edits.append(SourceEdit(
                start_char=threshold_start,
                end_char=threshold_end,
                replacement=squared_threshold_str,
                priority=0,
            ))

    @staticmethod
    def _strip_strings_and_comments(text: str) -> str:
        """Strip string contents and comments from a line of Lua code.
        
        Replaces string bodies with spaces and removes comments,
        so keyword checks don't match inside string literals.
        """
        result = []
        i = 0
        while i < len(text):
            c = text[i]
            # line comment
            if c == '-' and i + 1 < len(text) and text[i + 1] == '-':
                # check for long comment --[[
                if i + 2 < len(text) and text[i + 2] == '[':
                    eq_count = 0
                    j = i + 3
                    while j < len(text) and text[j] == '=':
                        eq_count += 1
                        j += 1
                    if j < len(text) and text[j] == '[':
                        # long comment, skip to closing ]=*]
                        close = ']' + '=' * eq_count + ']'
                        end = text.find(close, j + 1)
                        if end != -1:
                            result.append(' ' * (end + len(close) - i))
                            i = end + len(close)
                            continue
                # regular line comment, rest of line is gone
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
            elif c == '[':
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

    def _has_control_flow_keyword(self, text: str) -> bool:
        """Check if a line of code contains control flow keywords outside of strings/comments."""
        control_flow_keywords = ['if ', 'then ', 'else', 'elseif ', 'end', 'for ', 'while ',
                                 'do ', 'repeat', 'until ', 'function ', 'return ']
        cleaned = self._strip_strings_and_comments(text).lower()
        for kw in control_flow_keywords:
            if kw in cleaned:
                return True
        return False

    def _is_inside_multiline_comment(self, pos: int) -> bool:
        """Check if a position in source is inside a multi-line comment.
        
        Handles --[[...]], --[=[...]=], --[==[...]==], etc.
        Scans forward through all long comments to see if pos falls within any.
        """
        text = self.source
        i = 0
        while i < pos:
            c = text[i]
            # skip string literals so we don't match --[[ inside strings
            if c in ('"', "'"):
                quote = c
                i += 1
                while i < len(text) and text[i] != quote:
                    if text[i] == '\\':
                        i += 1
                    i += 1
                i += 1
                continue
            # check for long string [[ or [=[ (not a comment, but skip it)
            if c == '[':
                eq = 0
                j = i + 1
                while j < len(text) and text[j] == '=':
                    eq += 1
                    j += 1
                if j < len(text) and text[j] == '[':
                    close = ']' + '=' * eq + ']'
                    end = text.find(close, j + 1)
                    if end != -1:
                        i = end + len(close)
                        continue
            # check for long comment --[[ or --[=[ etc
            if c == '-' and i + 1 < len(text) and text[i + 1] == '-':
                if i + 2 < len(text) and text[i + 2] == '[':
                    eq = 0
                    j = i + 3
                    while j < len(text) and text[j] == '=':
                        eq += 1
                        j += 1
                    if j < len(text) and text[j] == '[':
                        close = ']' + '=' * eq + ']'
                        end = text.find(close, j + 1)
                        if end == -1:
                            # unclosed long comment, everything after is "inside"
                            return True
                        comment_end = end + len(close)
                        if pos < comment_end:
                            return True
                        i = comment_end
                        continue
                # regular line comment, skip to EOL
                eol = text.find('\n', i)
                if eol == -1:
                    break
                i = eol + 1
                continue
            i += 1
        return False

    def _edit_debug_statement(self, finding: Finding):
        """Comment out debug statement (handles multi-line calls)."""
        node = finding.details.get('node')

        if node:
            # use AST node to get full span of call
            start_char, end_char = self._get_node_span(node)
            if start_char is not None:
                # find all lines this call spans
                start_line = self.source[:start_char].count('\n') + 1
                end_line = self.source[:end_char].count('\n') + 1

                # Dont comment a statement that BINDS the debug call's result to
                # a variable (e.g. `local tc = self:log(...)`). Commenting the
                # line removes the declaration while later uses of the variable
                # survive -> "attempt to use global 'tc' (a nil value)"
                stmt_line_start, _ = self._get_line_span(start_line)
                if stmt_line_start is not None:
                    prefix = self.source[stmt_line_start:start_char]
                    if re.search(r'\blocal\b', prefix) or \
                            re.search(r'(?<![=~<>!])=(?!=)', prefix):
                        return

                # expression continuations - if prev line ends with these, call is part of expr
                expr_continuations = [' and', ' or', '(', ',', '=', '{', '[']

                if start_line > 1:
                    prev_line_start, prev_line_end = self._get_line_span(start_line - 1)
                    if prev_line_start is not None:
                        prev_line = self.source[prev_line_start:prev_line_end].rstrip()
                        for cont in expr_continuations:
                            if prev_line.endswith(cont):
                                return  # skip - this is part of an expression

                # collect all lines and check them ALL for control flow
                lines_to_comment = []
                has_control_flow = False

                for line_num in range(start_line, end_line + 1):
                    line_start, line_end = self._get_line_span(line_num)
                    if line_start is None:
                        continue

                    line = self.source[line_start:line_end]
                    stripped = line.lstrip()

                    # skip if already commented
                    if stripped.startswith('--'):
                        continue

                    # check for control flow outside strings/comments
                    if self._has_control_flow_keyword(stripped):
                        has_control_flow = True

                    lines_to_comment.append((line_num, line_start, line_end, line, stripped))

                # if ANY line has control flow, skip the ENTIRE statement
                if has_control_flow:
                    return

                # comment out all lines
                for line_num, line_start, line_end, line, stripped in lines_to_comment:
                    indent = line[:len(line) - len(stripped)]
                    new_line = f'{indent}-- {stripped}'

                    self.edits.append(SourceEdit(
                        start_char=line_start,
                        end_char=line_end,
                        replacement=new_line,
                        priority=200,  # high priority to override variable replacements inside debug calls
                    ))
                return

        # fallback to single line if no node
        line_num = finding.line_num
        start, end = self._get_line_span(line_num)
        if start is None:
            return

        # check for expression continuation on prev line
        expr_continuations = [' and', ' or', '(', ',', '=', '{', '[']
        if line_num > 1:
            prev_line_start, prev_line_end = self._get_line_span(line_num - 1)
            if prev_line_start is not None:
                prev_line = self.source[prev_line_start:prev_line_end].rstrip()
                for cont in expr_continuations:
                    if prev_line.endswith(cont):
                        return  # skip - part of expression

        line = self.source[start:end]
        stripped = line.lstrip()

        if stripped.startswith('--'):
            return

        # skip lines with control flow outside strings/comments
        if self._has_control_flow_keyword(stripped):
            return

        # skip assignment statements - commenting them drops a binding that
        # later code may still reference (see node-path note above)
        if re.match(r'(local\s+)?[\w.\[\]\'"]+\s*=(?!=)', stripped):
            return

        indent = line[:len(line) - len(stripped)]
        new_line = f'{indent}-- {stripped}'

        self.edits.append(SourceEdit(
            start_char=start,
            end_char=end,
            replacement=new_line,
            priority=200,
        ))

    def _edit_nil_access(self, finding: Finding):
        """
        Wrap unsafe nil access with if-then guard.
        
        Before:
            local obj = level.object_by_id(id)
            obj:set_visual("stalker")
            
        After:
            local obj = level.object_by_id(id)
            if obj then
                obj:set_visual("stalker")
            end
            
        Only applies to safe-to-fix cases (immediately after assignment).
        Only wraps SINGLE lines to avoid corrupting nested control structures.
        
        IMPORTANT: Does NOT wrap lines containing 'local' declarations,
        as that would change variable scope and break code that uses
        the variable outside the if block.
        """
        details = finding.details
        
        # only fix safe cases
        if not details.get('is_safe_to_fix', False):
            return
        
        var_name = details.get('var_name')
        assign_line = details.get('assign_line')
        access_line = finding.line_num
        
        if not var_name or not assign_line:
            return
        
        # get the access line
        access_line_start, access_line_end = self._get_line_span(access_line)
        if access_line_start is None:
            return
        
        access_line_text = self.source[access_line_start:access_line_end]
        stripped_access = access_line_text.strip()
        
        # SAFETY CHECK: don't wrap lines with local declarations
        if stripped_access.startswith('local '):
            return
        
        # SAFETY CHECK: don't wrap control flow statements (if, for, while, etc.)
        # these have complex nested structures that can get corrupted
        control_keywords = ('if ', 'if(', 'for ', 'while ', 'repeat', 'function ', 'function(')
        if any(stripped_access.startswith(kw) for kw in control_keywords):
            return
        
        # SAFETY CHECK: don't wrap incomplete statements (multi-line function calls, etc.)
        # check for unbalanced parentheses - if line has more '(' than ')', it continues on next line
        open_parens = stripped_access.count('(')
        close_parens = stripped_access.count(')')
        if open_parens > close_parens:
            return
        
        # SAFETY CHECK: don't wrap lines ending with opening constructs
        rstripped = stripped_access.rstrip()
        if rstripped.endswith('(') or rstripped.endswith(',') or rstripped.endswith('..'):
            return
        
        # SAFETY CHECK: don't wrap if next line also uses the same variable directly
        # this would result in partial protection (first line guarded, second line crashes)
        next_line_start, next_line_end = self._get_line_span(access_line + 1)
        if next_line_start is not None:
            next_line_text = self.source[next_line_start:next_line_end].strip()
            # check if next line starts with VAR: (direct method call on same variable)
            if next_line_text.startswith(f'{var_name}:'):
                return
        
        # determine indent from the access line
        indent = ''
        for ch in access_line_text:
            if ch in ' \t':
                indent += ch
            else:
                break
        
        # ONLY wrap this single line - don't try to wrap multiple lines
        # Multi-line wrapping is error-prone with nested structures
        wrapped_content = stripped_access
        
        # build the replacement
        new_content = f'{indent}if {var_name} then\n'
        new_content += f'{indent}    {wrapped_content}\n'
        new_content += f'{indent}end'
        
        # preserve trailing newline if original had one
        if access_line_text.endswith('\n'):
            new_content += '\n'
        
        self.edits.append(SourceEdit(
            start_char=access_line_start,
            end_char=access_line_end,
            replacement=new_content,
            priority=50,
        ))

    def _edit_dead_code(self, finding: Finding):
        """
        Remove dead code that is 100% safe to remove.
        
        Handles:
        - Code after unconditional return
        - Code after break in loops
        - if false then ... end blocks
        - while false do ... end loops
        """
        details = finding.details
        
        # only remove if marked as safe
        if not details.get('is_safe_to_remove', False):
            return
        
        dead_type = details.get('dead_type', '')
        start_line = details.get('start_line', 0)
        end_line = details.get('end_line', 0)
        
        if not start_line or not end_line:
            return
        
        # get the character positions for the lines to remove
        start_pos, _ = self._get_line_span(start_line)
        _, end_pos = self._get_line_span(end_line)
        
        if start_pos is None or end_pos is None:
            return
        
        # determine what to replace with
        if dead_type in ('after_return', 'after_break'):
            # remove the dead statements entirely
            # but preserve any trailing newline to keep formatting
            replacement = ''
        elif dead_type in ('if_false', 'while_false'):
            # remove the entire if/while block
            # check if there's only whitespace before on the same line
            line_content = self.source[start_pos:end_pos]
            
            # preserve indentation context - just remove the block
            replacement = ''
        else:
            return
        
        self.edits.append(SourceEdit(
            start_char=start_pos,
            end_char=end_pos,
            replacement=replacement,
            priority=10,  # high priority - remove dead code first
        ))

    def _edit_string_concat_in_loop(self, finding: Finding):
        """
        Transform string concatenation in loops to table.concat pattern.
        
        Before:
            local result = ""
            for i = 1, 10 do
                result = result .. get_part(i)
            end
            
        After:
            local _result_parts = {}
            for i = 1, 10 do
                _result_parts[#_result_parts+1] = get_part(i)
            end
            local result = table.concat(_result_parts)
        """
        details = finding.details
        var = details.get('variable')
        init_line = details.get('init_line')
        loop_start = details.get('loop_start')
        loop_end = details.get('loop_end')
        concat_lines = details.get('concat_lines', [])
        is_safe = details.get('is_safe', False)
        
        if not var or not init_line or not loop_end or not concat_lines:
            return
        
        if not is_safe:
            return  # only transform safe patterns
        
        # skip one-liner loops (loop start == loop end)
        if loop_start == loop_end:
            return  # one-liner loop, too complex to transform safely
        
        # skip if any concat is on same line as loop start or loop end (embedded/one-liner)
        for concat_line in concat_lines:
            if concat_line == loop_start or concat_line == loop_end:
                return  # concat embedded in loop header/footer, skip
        
        # Bug #32 fix: Additional check - verify concat lines don't contain loop keywords
        # This catches one-liners that the AST might report with different start/end lines
        for concat_line in concat_lines:
            line_start, line_end = self._get_line_span(concat_line)
            if line_start is not None:
                line_text = self.source[line_start:line_end]
                # if the concat line contains 'for ' and ' end', it's a one-liner loop
                if re.search(r'\bfor\b.*\bend\b', line_text):
                    return  # one-liner loop detected via text analysis
                # if line contains 'for ' at all, it's embedded in loop header
                if re.search(r'\bfor\s+', line_text):
                    return  # embedded in for header
        
        parts_var = f'_{var}_parts'
        
        # SAFETY: check if the variable is referenced on any lines between init and loop_end
        # that aren't the concat_lines we're converting. If so, the optimization would
        # break those references (since local var = "" becomes local _var_parts = {})
        concat_set = set(concat_lines)
        var_pattern = re.compile(rf'\b{re.escape(var)}\b')
        for check_ln in range(init_line + 1, loop_end + 1):
            if check_ln in concat_set:
                continue
            ls, le = self._get_line_span(check_ln)
            if ls is None:
                continue
            line_text = self.source[ls:le]
            # strip comments
            comment_pos = line_text.find('--')
            if comment_pos >= 0:
                line_text = line_text[:comment_pos]
            if var_pattern.search(line_text):
                return  # variable used outside concat lines, not safe to transform
        
        # VALIDATION PHASE: check all lines can be transformed before making any edits
        
        # validate init line
        init_start, init_end = self._get_line_span(init_line)
        if init_start is None:
            return
        
        init_text = self.source[init_start:init_end]
        indent = self._get_indent_at_line(init_line)
        
        # validate all concat lines match the expected pattern
        concat_replacements = []
        for concat_line in concat_lines:
            line_start, line_end = self._get_line_span(concat_line)
            if line_start is None:
                return  # can't find line, abort
            
            line_text = self.source[line_start:line_end]
            line_indent = self._get_indent_at_line(concat_line)
            
            # pattern: var = var .. expr (must be the whole line content, not embedded)
            concat_pattern = re.compile(
                rf'^(\s*){re.escape(var)}\s*=\s*{re.escape(var)}\s*\.\.\s*(.+)$',
                re.DOTALL
            )
            match = concat_pattern.match(line_text.rstrip('\n\r'))
            if not match:
                return  # pattern not on its own line, abort entire transformation
            
            expr = match.group(2).rstrip()
            
            # Bug #31 fix: check if expr references the variable itself
            # e.g. (var == "" and "" or ", ") - this would break after transformation
            # because 'var' won't exist until after table.concat
            if re.search(rf'\b{re.escape(var)}\b', expr):
                # try to replace common patterns like (var == "" and X or Y) with (#parts == 0 and X or Y)
                empty_check = re.compile(
                    rf'\(\s*{re.escape(var)}\s*==\s*""\s+and\s+',
                    re.IGNORECASE
                )
                if empty_check.search(expr):
                    # replace var == "" with #parts_var == 0
                    expr = re.sub(
                        rf'\b{re.escape(var)}\s*==\s*""',
                        f'#{parts_var} == 0',
                        expr
                    )
                else:
                    # can't safely transform, abort
                    return
            
            new_line = f'{line_indent}{parts_var}[#{parts_var}+1] = {expr}\n'
            concat_replacements.append((line_start, line_end, new_line))
        
        # validate loop end line
        end_line_end = self._get_line_end(loop_end)
        if end_line_end is None:
            return
        
        # EDIT PHASE: all validations passed, now add edits
        
        # step 1: replace initialization line
        stripped = init_text.strip()
        if stripped.startswith('local '):
            new_init = f'{indent}local {parts_var} = {{}}\n'
        else:
            new_init = f'{indent}{parts_var} = {{}}\n'
        
        self.edits.append(SourceEdit(
            start_char=init_start,
            end_char=init_end,
            replacement=new_init,
            priority=50,
        ))
        
        # step 2: replace each concat line
        for line_start, line_end, new_line in concat_replacements:
            self.edits.append(SourceEdit(
                start_char=line_start,
                end_char=line_end,
                replacement=new_line,
                priority=50,
            ))
        
        # step 3: add table.concat after loop ends
        concat_decl = f'\n{indent}local {var} = table.concat({parts_var})'
        
        self.edits.append(SourceEdit(
            start_char=end_line_end,
            end_char=end_line_end,
            replacement=concat_decl,
            priority=50,
        ))

    # More idiomatic cache names for common globals (kinda)
    IDIOMATIC_CACHE_NAMES = {
        # math module - use m prefix
        'math.floor': 'mfloor',
        'math.ceil': 'mceil',
        'math.abs': 'mabs',
        'math.min': 'mmin',
        'math.max': 'mmax',
        'math.sqrt': 'msqrt',
        'math.sin': 'msin',
        'math.cos': 'mcos',
        'math.random': 'mrandom',
        'math.pow': 'mpow',
        'math.huge': 'mhuge',

        # table module - use t prefix
        'table.insert': 'tinsert',
        'table.remove': 'tremove',
        'table.concat': 'tconcat',
        'table.sort': 'tsort',

        # string module - use s prefix
        'string.find': 'sfind',
        'string.sub': 'ssub',
        'string.len': 'slen',
        'string.format': 'sformat',
        'string.gsub': 'sgsub',
        'string.match': 'smatch',
        'string.gmatch': 'sgmatch',
        'string.lower': 'slower',
        'string.upper': 'supper',

        # bare globals - use descriptive short names
        'pairs': 'pairs_',  # trailing underscore to avoid shadowing
        'ipairs': 'ipairs_',
        'type': 'type_',
        'tostring': 'tostr',
        'tonumber': 'tonum',
        'print': 'pr',
        'assert': 'assert_',
        'error': 'err',
        'pcall': 'pcall_',
        'xpcall': 'xpcall_',
        'next': 'next_',
        'select': 'sel',
        'unpack': 'unpack_',
        'rawget': 'rawget_',
        'rawset': 'rawset_',
        'setmetatable': 'setmt',
        'getmetatable': 'getmt',
    }

    def _edit_uncached_globals(self, finding: Finding):
        """Add local caching for globals at function start."""
        details = finding.details
        globals_info = details.get('globals_info', {})
        scope = details.get('scope')

        if not globals_info or not scope:
            return

        if getattr(scope, 'scope_type', None) != 'function':
            return

        # collect existing locals so we don't shadow a user-defined name
        # (e.g. user already has `local mfloor = ...` somewhere in this function)
        existing_locals = self._collect_function_locals(scope)

        # build cache declarations and track replacements
        cache_lines = []
        replacements: Dict[str, str] = {}

        for name in sorted(globals_info.keys()):
            # use idiomatic name if available, otherwise generate one
            if name in self.IDIOMATIC_CACHE_NAMES:
                base_cache_name = self.IDIOMATIC_CACHE_NAMES[name]
            elif '.' in name:
                module, func = name.split('.', 1)
                # use first letter of module + func name
                base_cache_name = f'{module[0]}{func}'
            else:
                base_cache_name = f'g_{name}'

            cache_name = self._resolve_cache_name(base_cache_name, existing_locals)
            existing_locals.add(cache_name)  # avoid colliding with our other inserts

            cache_lines.append(f'local {cache_name} = {name}')
            replacements[name] = cache_name

        if not cache_lines:
            return

        # Preferred path: use the function literal's AST node to locate the body's
        # actual start. Required for anonymous functions passed as call arguments
        # (e.g. `CreateTimeEvent(..., function() ... end)`) where naive paren-
        # counting on the declaration line confuses the surrounding call's open
        # paren with the function's own params, and would otherwise insert the
        # cache decl AFTER the closure -- outside the closure's scope, so the
        # cache name is unresolved when the closure runs.
        insert_pos: Optional[int] = None
        indent: Optional[str] = None

        scope_node = getattr(scope, 'node', None)
        if scope_node is not None:
            body = getattr(scope_node, 'body', None)
            first_token = getattr(body, 'first_token', None) if body is not None else None
            if first_token is not None:
                body_start = self._parse_token_start(str(first_token))
                if body_start is not None:
                    line_start = self.source.rfind('\n', 0, body_start) + 1
                    leading = self.source[line_start:body_start]
                    if leading.strip() == '':
                        # body's first token is at the start of its own line
                        # (after indentation). Insert cache lines BEFORE this line.
                        insert_pos = line_start
                        indent = leading

        # Group the cache insertion with its replacement edits so the post-
        # filtering pass can drop the insertion if every replacement gets
        # rejected (e.g. all call sites are inside a debug_log that --fix-debug
        # comments out; without this, we'd leave dead `local X = ...` lines).
        gid = self._next_group_id
        self._next_group_id += 1

        if insert_pos is not None:
            # build cache block as full lines, terminated by newline so the
            # next line (body's original first statement) stays on its own line.
            cache_block = ''.join(f'{indent}{line}\n' for line in cache_lines)

            self.edits.append(SourceEdit(
                start_char=insert_pos,
                end_char=insert_pos,
                replacement=cache_block,
                priority=100,
                group_id=gid,
                is_enabler=True,
            ))
        else:
            # Fallback: legacy paren-counting on the declaration line. Used when
            # the AST node isn't available or the body starts mid-line (e.g.
            # single-line `function(...) body end`). Skipping is safer than
            # inserting at the wrong scope.
            func_body_start_line = scope.start_line

            func_decl_start, func_decl_end = self._get_line_span(scope.start_line)
            if func_decl_start is not None:
                func_decl_text = self.source[func_decl_start:func_decl_end]
                # check if function definition has unclosed paren (multi-line params)
                open_parens = func_decl_text.count('(')
                close_parens = func_decl_text.count(')')

                if open_parens > close_parens:
                    # multi-line function definition - find closing paren
                    paren_depth = open_parens - close_parens
                    for search_line in range(scope.start_line + 1, scope.start_line + 30):  # reasonable limit
                        search_start, search_end = self._get_line_span(search_line)
                        if search_start is None:
                            break
                        search_text = self.source[search_start:search_end]
                        paren_depth += search_text.count('(') - search_text.count(')')
                        if paren_depth <= 0:
                            # found the closing paren - insert after this line
                            func_body_start_line = search_line
                            break

            fallback_pos = self._get_line_end(func_body_start_line)
            if fallback_pos is None:
                return

            fallback_indent = self._get_indent_at_line(func_body_start_line + 1)
            if not fallback_indent:
                fallback_indent = self._detect_indent_unit()

            cache_block = '\n' + '\n'.join(f'{fallback_indent}{line}' for line in cache_lines)

            self.edits.append(SourceEdit(
                start_char=fallback_pos,
                end_char=fallback_pos,
                replacement=cache_block,
                priority=100,
                group_id=gid,
                is_enabler=True,
            ))

        # replace usages using AST node positions
        for name, calls in globals_info.items():
            new_name = replacements.get(name)
            if not new_name:
                continue

            for call in calls:
                node = call.node
                if not node:
                    continue

                # for calls like pairs(), ipairs() - replace the function name part
                if '.' not in name:
                    # bare global - find and replace just the name
                    start, end = self._get_call_func_span(node, name)
                else:
                    # module.func - replace the whole func reference
                    start, end = self._get_call_func_span(node, name)

                if start is None:
                    continue

                self.edits.append(SourceEdit(
                    start_char=start,
                    end_char=end,
                    replacement=new_name,
                    group_id=gid,
                ))

    def _edit_repeated_calls(self, finding: Finding):
        """Add caching for repeated expensive calls."""
        details = finding.details
        calls = details.get('calls', [])
        scope = details.get('scope')
        suggestion = details.get('suggestion', '')

        if not calls or not scope:
            return

        if getattr(scope, 'scope_type', None) != 'function':
            return

        pattern = finding.pattern_name

        # determine cache variable name and cache line
        if pattern == 'repeated_db_actor':
            cache_line = 'local actor = db.actor'
            new_name = 'actor'
            call_pattern = 'db.actor'
            call_pattern_re = re.compile(r'\bdb\.actor\b')
        elif pattern == 'repeated_time_global':
            cache_line = 'local tg = time_global()'
            new_name = 'tg'
            call_pattern = 'time_global()'
            call_pattern_re = re.compile(r'\btime_global\s*\(')
        elif pattern == 'repeated_alife':
            cache_line = 'local sim = alife()'
            new_name = 'sim'
            call_pattern = 'alife()'
            call_pattern_re = re.compile(r'\balife\s*\(')
        elif pattern == 'repeated_system_ini':
            cache_line = 'local ini = system_ini()'
            new_name = 'ini'
            call_pattern = 'system_ini()'
            call_pattern_re = re.compile(r'\bsystem_ini\s*\(')
        elif pattern == 'repeated_device':
            cache_line = 'local dev = device()'
            new_name = 'dev'
            call_pattern = 'device()'
            call_pattern_re = re.compile(r'\bdevice\s*\(')
        elif pattern == 'repeated_get_console':
            cache_line = 'local console = get_console()'
            new_name = 'console'
            call_pattern = 'get_console()'
            call_pattern_re = re.compile(r'\bget_console\s*\(')
        elif pattern == 'repeated_get_hud':
            cache_line = 'local hud = get_hud()'
            new_name = 'hud'
            call_pattern = 'get_hud()'
            call_pattern_re = re.compile(r'\bget_hud\s*\(')
        elif pattern == 'repeated_game_ini':
            cache_line = 'local g_ini = game_ini()'
            new_name = 'g_ini'
            call_pattern = 'game_ini()'
            call_pattern_re = re.compile(r'\bgame_ini\s*\(')
        elif pattern == 'repeated_getFS':
            cache_line = 'local fs = getFS()'
            new_name = 'fs'
            call_pattern = 'getFS()'
            call_pattern_re = re.compile(r'\bgetFS\s*\(')
        elif pattern == 'repeated_level_name':
            cache_line = 'local level_name = level.name()'
            new_name = 'level_name'
            call_pattern = 'level.name()'
            call_pattern_re = re.compile(r'\blevel\.name\s*\(')
        elif pattern.endswith('_story_id()') or pattern.endswith('_section()') or pattern.endswith('_id()') or pattern.endswith('_clsid()'):
            # dynamic method caching: repeated_obj_section(), repeated_item_id(), etc
            # extract object name and method from pattern: repeated_obj_section() -> obj, section
            # pattern format: repeated_{objname}_{method}()
            # NOTE: story_id must be checked before id since _id() is suffix of _story_id()
            # NOTE: Use non-greedy (.+?) to avoid capturing part of method name
            match = re.match(r'repeated_(.+?)_(story_id|section|clsid|id)\(\)$', pattern)
            if not match:
                return
            sanitized_obj_name = match.group(1)
            method_name = match.group(2)
            
            # get original object name from details if available (e.g. "self.object:id()")
            original_call = details.get('original_call', '') if details else ''
            if original_call and ':' in original_call:
                # extract original object name: "self.object:id()" -> "self.object"
                real_obj_name = original_call.split(':')[0]
            else:
                # fallback: try to restore dots from underscores for common patterns
                if sanitized_obj_name.startswith('self_'):
                    real_obj_name = 'self.' + sanitized_obj_name[5:]
                else:
                    real_obj_name = sanitized_obj_name
            
            # SAFETY CHECK: skip if object is an indexed expression (e.g., t[a], arr[i])
            # These can't be converted to valid Lua variable names
            if '[' in real_obj_name or ']' in real_obj_name:
                return

            # SAFETY CHECK: skip if the object expression itself contains a call
            # (a method chain, e.g. obj:active_item():section()). The sanitized
            # name would carry the '()' -> invalid identifier, and split(':')[0]
            # mis-derives the cached value. Caching these is unsafe.
            if '(' in sanitized_obj_name or ')' in sanitized_obj_name \
                    or '(' in real_obj_name or ')' in real_obj_name:
                return
            
            # generate cache variable name (always use sanitized for variable)
            if method_name == 'section':
                new_name = f'{sanitized_obj_name}_sec'
            elif method_name == 'id':
                new_name = f'{sanitized_obj_name}_id'
            elif method_name == 'clsid':
                new_name = f'{sanitized_obj_name}_cls'
            elif method_name == 'story_id':
                new_name = f'{sanitized_obj_name}_sid'
            else:
                new_name = f'{sanitized_obj_name}_{method_name}'
            
            cache_line = f'local {new_name} = {real_obj_name}:{method_name}()'
            call_pattern = f'{real_obj_name}:{method_name}()'
            call_pattern_re = re.compile(re.escape(real_obj_name) + r'\s*:\s*' + re.escape(method_name) + r'\s*\(')
        else:
            return

        # check if first call is already a cache declaration for THIS pattern
        # i.e., "local obj = level.object_by_id(...)" where obj is the cache var
        # NB: this check uses the ORIGINAL new_name, so a pre-existing cache
        # is recognised even if a separate local would have forced a rename.
        original_cache_decl_pattern = rf'\blocal\s+{re.escape(new_name)}\s*='
        first_call = calls[0]
        first_line_start, first_line_end = self._get_line_span(first_call.line)
        is_already_cached = False
        cache_indent = None

        if first_line_start is not None:
            first_line = self.source[first_line_start:first_line_end]
            if re.search(original_cache_decl_pattern, first_line) and call_pattern_re.search(first_line):
                is_already_cached = True
                cache_indent = self._get_indent_at_line(first_call.line)

        # Collision check: if we're going to INSERT a new local but the chosen
        # name already exists somewhere in the function (param, sibling local,
        # nested local), pick an alternative so we don't shadow user code.
        if not is_already_cached:
            existing_locals = self._collect_function_locals(scope)
            resolved = self._resolve_cache_name(new_name, existing_locals)
            if resolved != new_name:
                # rebuild cache_line with the renamed identifier; the call_pattern
                # itself describes the source expression (db.actor / obj:id()),
                # which is unaffected by the rename.
                cache_line = cache_line.replace(
                    f'local {new_name} ',
                    f'local {resolved} ',
                    1,
                )
                new_name = resolved

        # cache_decl_pattern is rebuilt with the possibly-renamed identifier;
        # downstream code uses it only to skip-over the cache decl line itself
        # when applying replacements.
        cache_decl_pattern = rf'\blocal\s+{re.escape(new_name)}\s*='

        # insert cache if not already present
        if not is_already_cached:
            insert_pos = self._get_line_start(first_call.line)
            if insert_pos is None:
                return

            indent = self._get_indent_at_line(first_call.line)
            
            # check if insertion point is inside a multi-line comment
            if self._is_inside_multiline_comment(insert_pos):
                return
            
            # Check if first call is inside a multi-line if/elseif/while condition
            # Pattern: "if\n  (expr with call)" - we're between keyword and then/do
            # In this case, we can't insert a local declaration at the call's line
            if scope and hasattr(scope, 'start_line'):
                control_keyword_line = None
                in_condition = False
                
                # scan backwards from first_call.line to find unmatched if/elseif/while
                for check_line in range(first_call.line, scope.start_line - 1, -1):
                    ls, le = self._get_line_span(check_line)
                    if ls is not None:
                        line_text = self.source[ls:le]
                        # remove comments
                        if '--' in line_text:
                            line_text = line_text[:line_text.find('--')]
                        stripped = line_text.strip().lower()
                        
                        # check for then/do - if found before if/while, we're NOT in a condition
                        if stripped.endswith('then') or stripped == 'then' or ' then' in stripped:
                            break
                        if stripped.endswith('do') or stripped == 'do' or ' do' in stripped:
                            break
                        
                        # check for if/elseif/while at the START of a line (not inside string)
                        # these keywords without then/do on same line indicate multi-line condition
                        if stripped.startswith(('if ', 'if(', 'elseif ', 'elseif(', 'while ', 'while(')):
                            # check if 'then' or 'do' is on the same line
                            if 'then' not in stripped and 'do' not in stripped:
                                control_keyword_line = check_line
                                in_condition = True
                                break
                        # bare 'if' on its own line
                        if stripped == 'if' or stripped == 'elseif' or stripped == 'while':
                            control_keyword_line = check_line
                            in_condition = True
                            break
                
                if in_condition and control_keyword_line is not None:
                    # we're inside a multi-line condition - insert BEFORE the control statement
                    insert_pos = self._get_line_start(control_keyword_line)
                    if insert_pos is None:
                        return
                    indent = self._get_indent_at_line(control_keyword_line)
            
            # is this a method cache (obj:method()) vs global cache (func())
            is_method_cache = ':' in call_pattern

            # check if first call is inside a table constructor or function call arguments
            # look for unbalanced { or ( in lines before this one within scope
            if scope and hasattr(scope, 'start_line'):
                brace_depth = 0
                paren_depth = 0
                has_loop_before_first_call = False
                has_branch_between_calls = False

                for check_line in range(scope.start_line, first_call.line + 1):
                    ls, le = self._get_line_span(check_line)
                    if ls is not None:
                        line_text = self.source[ls:le]
                        # skip comments
                        if '--' in line_text:
                            line_text = line_text[:line_text.find('--')]
                        # skip strings
                        in_string = False
                        clean_line = ""
                        i = 0
                        while i < len(line_text):
                            c = line_text[i]
                            if c in ('"', "'") and not in_string:
                                in_string = c
                            elif c == in_string:
                                # count preceding backslashes
                                num_bs = 0
                                j = i - 1
                                while j >= 0 and line_text[j] == '\\':
                                    num_bs += 1
                                    j -= 1
                                if num_bs % 2 == 0:
                                    in_string = False
                            elif not in_string:
                                clean_line += c
                            i += 1
                        
                        brace_depth += clean_line.count('{') - clean_line.count('}')
                        paren_depth += clean_line.count('(') - clean_line.count(')')

                        # check for loop constructs before first call
                        if check_line < first_call.line:
                            stripped = line_text.strip().lower()
                            if stripped.startswith(('for ', 'while ', 'repeat')):
                                has_loop_before_first_call = True

                if brace_depth > 0:
                    return  # inside table constructor, skip optimization
                
                if paren_depth > 0:
                    return  # inside function call arguments, skip optimization

                # SAFETY CHECK: detect nil-guarded method calls
                # Pattern: "obj and obj:method()" or "if obj and ... obj:method()"
                # These rely on short-circuit evaluation for safety - caching breaks this
                if is_method_cache:
                    first_ls, first_le = self._get_line_span(first_call.line)
                    if first_ls is not None:
                        first_line_text = self.source[first_ls:first_le]
                        # extract object name from call_pattern: "obj:method()" -> "obj"
                        obj_name = call_pattern.split(':')[0]
                        
                        # check if this line has pattern: "obj and" before "obj:method"
                        call_pos = first_line_text.find(call_pattern)
                        if call_pos > 0:
                            before_call = first_line_text[:call_pos]
                            nil_guard_pattern = rf'\b{re.escape(obj_name)}\s+and\b'
                            if re.search(nil_guard_pattern, before_call):
                                # object is nil-guarded, skip caching to preserve safety
                                return

                # check if calls span different blocks. Indentation is unreliable
                # (some scripts are mis-indented), so track block nesting by
                # counting Lua block keywords. If the block CONTAINING the first
                # call closes (depth drops below 0) or branches (else/elseif at
                # depth 0) before the last call, later calls are in a different
                # scope -> the decl can't dominate them, so hoist to func top.
                last_call = calls[-1]
                if last_call.line > first_call.line:
                    depth = 0
                    for check_line in range(first_call.line + 1, last_call.line + 1):
                        cls, cle = self._get_line_span(check_line)
                        if cls is None:
                            continue
                        text = self.source[cls:cle]
                        # strip line comment and quoted strings (keywords inside
                        # them must not count toward block nesting)
                        cpos = text.find('--')
                        if cpos != -1:
                            text = text[:cpos]
                        text = re.sub(r'"(?:\\.|[^"\\])*"', '', text)
                        text = re.sub(r"'(?:\\.|[^'\\])*'", '', text)
                        words = re.findall(r'\b[a-z]+\b', text.lower())
                        has_for_while = any(w in ('for', 'while') for w in words)
                        opens = closes = 0
                        branch_at_zero = False
                        for w in words:
                            if w in ('function', 'if', 'for', 'while', 'repeat'):
                                opens += 1
                            elif w == 'do' and not has_for_while:
                                opens += 1
                            elif w in ('end', 'until'):
                                closes += 1
                            elif w in ('else', 'elseif') and depth == 0:
                                branch_at_zero = True
                        if branch_at_zero or depth - closes < 0:
                            has_branch_between_calls = True
                            break
                        depth += opens - closes

                if (has_loop_before_first_call or has_branch_between_calls) and last_call.line > first_call.line:
                    if is_method_cache:
                        # for method caching, skip if branches exist - too risky to hoist
                        return
                    
                    # Insert right after function declaration
                    # but first, check if function definition spans multiple lines
                    # pattern: "function name(" with arguments on following lines until ")"
                    func_body_start_line = scope.start_line + 1
                    
                    func_decl_start, func_decl_end = self._get_line_span(scope.start_line)
                    if func_decl_start is not None:
                        func_decl_text = self.source[func_decl_start:func_decl_end]
                        # check if function definition has unclosed paren (multi-line params)
                        open_parens = func_decl_text.count('(')
                        close_parens = func_decl_text.count(')')
                        
                        if open_parens > close_parens:
                            # multi-line function definition - find closing paren
                            paren_depth = open_parens - close_parens
                            for search_line in range(scope.start_line + 1, scope.start_line + 20):  # reasonable limit
                                search_start, search_end = self._get_line_span(search_line)
                                if search_start is None:
                                    break
                                search_text = self.source[search_start:search_end]
                                paren_depth += search_text.count('(') - search_text.count(')')
                                if paren_depth <= 0:
                                    # found the closing paren - insert after this line
                                    func_body_start_line = search_line + 1
                                    break
                    
                    new_insert_pos = self._get_line_start(func_body_start_line)
                    if new_insert_pos is None:
                        return
                    
                    # check if new insertion point is inside a multi-line comment
                    if self._is_inside_multiline_comment(new_insert_pos):
                        return
                    
                    insert_pos = new_insert_pos
                    # use indent from first call (which is inside the function body)
                    # but reduced by one level since first_call may be inside if/for
                    call_indent = self._get_indent_at_line(first_call.line)
                    if call_indent and len(call_indent) > 0:
                        # detect indent char (tab or spaces)
                        if call_indent[0] == '\t':
                            indent = self._detect_indent_unit()
                        else:
                            # count spaces per indent level (usually 4 or 2)
                            indent = call_indent[:len(call_indent)//2] if len(call_indent) >= 2 else call_indent
                    else:
                        indent = self._detect_indent_unit()

            # Final safety: the chosen insert_pos must be a valid statement
            # boundary. Bail out if it lands before an elseif/else/until (would
            # split an if-chain) or on a line continuing the previous statement
            # (e.g. an expression broken across lines by a trailing `or`/`=`)
            if not self._is_safe_local_insert_pos(insert_pos):
                return

            self.edits.append(SourceEdit(
                start_char=insert_pos,
                end_char=insert_pos,
                replacement=f'{indent}{cache_line}\n',
                priority=100,
            ))

        # replace usages - skip only lines that are the cache declaration itself
        for call in calls:
            line_start, line_end = self._get_line_span(call.line)
            if line_start is not None:
                line = self.source[line_start:line_end]
                # only skip if this is THE cache declaration (local obj = pattern)
                if re.search(cache_decl_pattern, line) and call_pattern_re.search(line):
                    continue

            # if cache already existed (we didn't insert it), check if we're in same scope
            # by looking for else/elseif/end at cache indent level between cache and call
            if is_already_cached and cache_indent is not None and call.line > first_call.line:
                in_sibling_scope = False
                cache_indent_len = len(cache_indent)

                for check_line in range(first_call.line + 1, call.line):
                    cls, cle = self._get_line_span(check_line)
                    if cls is not None:
                        check_text = self.source[cls:cle]
                        check_stripped = check_text.lstrip()
                        check_indent_len = len(check_text) - len(check_stripped)

                        # if we see else/elseif/end at same or shallower indent, scope changed
                        if check_indent_len <= cache_indent_len:
                            first_word = check_stripped.split()[0] if check_stripped.split() else ''
                            if first_word in ('else', 'elseif', 'end'):
                                in_sibling_scope = True
                                break

                if in_sibling_scope:
                    continue  # skip - we're in a sibling scope

            node = call.node
            if not node:
                continue

            # get span for the call expression
            start, end = self._get_node_span(node)
            if start is None:
                continue

            self.edits.append(SourceEdit(
                start_char=start,
                end_char=end,
                replacement=new_name,
            ))


    def _is_safe_local_insert_pos(self, insert_pos: int) -> bool:
        """True if a `local x = ...` line can be inserted at insert_pos without
        breaking syntax. insert_pos must be a line-start offset."""
        # line number of the insertion target (1-based)
        ins_line = self.source.count('\n', 0, insert_pos) + 1

        # the insertion line must not be a branch-continuation keyword: putting
        # a statement before `elseif`/`else`/`until` splits the construct.
        ls, le = self._get_line_span(ins_line)
        if ls is not None:
            m = re.match(r'\s*([A-Za-z]+)', self.source[ls:le])
            if m and m.group(1) in ('elseif', 'else', 'until'):
                return False

        # the previous code line must not leave the statement unterminated
        # (expression broken across lines by a trailing operator / `=` / comma).
        pl = ins_line - 1
        while pl >= 1:
            pls, ple = self._get_line_span(pl)
            if pls is None:
                break
            ptext = self.source[pls:ple]
            cpos = ptext.find('--')
            if cpos != -1:
                ptext = ptext[:cpos]
            pstr = ptext.rstrip()
            if pstr == '':
                pl -= 1
                continue
            # trailing binary keyword (and/or/not) -> continuation
            if re.search(r'(?:^|[^\w])(and|or|not)$', pstr):
                return False
            # trailing operator / open bracket / separator -> continuation
            if pstr[-1] in '+-*/%^#<>=~,({[.':
                return False
            return True
        return True

    # Position helpers using AST tokens

    def _get_node_span(self, node) -> Tuple[Optional[int], Optional[int]]:
        """Get character span (start, end) for an AST node."""
        from luaparser.astnodes import Call, Index, Invoke, Name

        # for Call nodes with Index func (like table.insert),
        # the first_token might not include the base object
        if isinstance(node, Call):
            func = getattr(node, 'func', None)

            # check if first_token looks like it's just '(' - means we need to find the func name
            first = getattr(node, 'first_token', None)
            first_str = str(first) if first else ''

            if "='('" in first_str or "=''" in first_str:
                # the call's first_token is just the paren, we need to find the function name
                # search backwards from paren position to find the identifier
                paren_start = self._parse_token_start(first_str)
                if paren_start is not None:
                    # find the identifier before the paren
                    pos = paren_start - 1
                    # skip whitespace
                    while pos >= 0 and self.source[pos] in ' \t\n':
                        pos -= 1
                    # find end of identifier
                    end_of_name = pos + 1
                    # find start of identifier
                    id_end = pos  # position of last identifier char (or last non-ident)
                    while pos >= 0 and (self.source[pos].isalnum() or self.source[pos] == '_'):
                        pos -= 1
                    found_identifier = pos != id_end
                    start = pos + 1

                    # A Lua keyword before the paren (e.g. `if (foo())`) is NOT a
                    # call name - it's a grouping paren after the keyword. Treat
                    # it as "no identifier" so the wrap branch below returns just
                    # the parenthesised expression, not `if (...)`
                    if found_identifier and self.source[start:end_of_name] in _LUA_KEYWORDS:
                        found_identifier = False

                    # end is the closing paren
                    last = getattr(node, 'last_token', None)
                    end = self._parse_token_end(str(last)) if last else None

                    if found_identifier and start is not None and end is not None:
                        return start, end

                    # No identifier before the paren - this is a wrapping
                    # paren (e.g. `if (foo(x))`). Return the span of the
                    # whole wrap so callers replace it consistently. Using
                    # the Index path's `value.first_token` would start
                    # mid-wrap (`math` of `(math.pow(...))`) but the end
                    # would still be the outer `)`, leaving an unmatched
                    # bracket after replacement.
                    if not found_identifier and end is not None:
                        return paren_start, end

            if isinstance(func, Index):
                # get the start from the base value
                value = getattr(func, 'value', None)
                if value:
                    value_first = getattr(value, 'first_token', None)
                    if value_first and str(value_first) != 'None':
                        start = self._parse_token_start(str(value_first))
                    else:
                        # fallback to finding base before the dot
                        func_start = self._parse_token_start(
                            str(func.first_token)) if func.first_token else None
                        if func_start is not None:
                            # search backwards for the base name
                            pos = func_start - 1
                            while pos >= 0 and self.source[pos] in ' \t':
                                pos -= 1
                            # find start of identifier
                            while pos >= 0 and (
                                    self.source[pos].isalnum() or self.source[pos] == '_'):
                                pos -= 1
                            start = pos + 1
                        else:
                            start = None
                else:
                    start = self._parse_token_start(
                        str(node.first_token)) if node.first_token else None

                # end from the call's last_token
                last = getattr(node, 'last_token', None)
                end = self._parse_token_end(str(last)) if last else None

                return start, end

        # Index nodes (e.g. `db.actor`, `self.args.mode`) - `first_token` is
        # sometimes the base identifier (`db`), sometimes the dot, depending
        # on parsing context. For nested chains like `self.args.mode`, the
        # outer Index's value IS another Index whose first_token may be the
        # `.` between `self` and `args`. We must reach all the way back to
        # the chain's *base* identifier, recursing into nested Indexes.
        if isinstance(node, Index):
            value = getattr(node, 'value', None)
            start = None
            if value is not None:
                if isinstance(value, Index):
                    # recursive: the inner Index has the same start
                    # determination concern. Reuse the helper to walk
                    # all the way down to the base name.
                    v_start, _ = self._get_node_span(value)
                    if v_start is not None:
                        start = v_start
                else:
                    value_first = getattr(value, 'first_token', None)
                    if value_first and str(value_first) != 'None':
                        start = self._parse_token_start(str(value_first))
            if start is None:
                first_tok = getattr(node, 'first_token', None)
                if first_tok and str(first_tok) != 'None':
                    first_str = str(first_tok)
                    first_pos = self._parse_token_start(first_str)
                    # Token-string format: "[@idx,start:end='text',type,line:col]".
                    # Detect a dot/bracket token by its quoted text payload.
                    is_dot_token = "='.'" in first_str
                    is_bracket_token = "='['" in first_str
                    if first_pos is not None:
                        if is_dot_token and first_pos > 0:
                            pos = first_pos - 1
                            while pos >= 0 and self.source[pos] in ' \t':
                                pos -= 1
                            while pos >= 0 and (self.source[pos].isalnum() or self.source[pos] == '_'):
                                pos -= 1
                            start = pos + 1
                        elif is_bracket_token and first_pos > 0:
                            # Bracket-index Index whose own first_token is the
                            # opening `[` rather than the base identifier
                            # (happens when the parser puts the token on the
                            # outer Index rather than the inner Name).
                            # Walk back over whitespace, any chain of nested
                            # `]...[` (multi-level bracket indexing on the
                            # same base, e.g. `m[a][b]`), and a dotted-
                            # identifier chain to find the base. If the base
                            # is a paren-expression like `(a+b)[i]`, give up
                            # and return None so the caller skips the edit.
                            pos = first_pos - 1
                            while pos >= 0 and self.source[pos] in ' \t':
                                pos -= 1
                            while pos >= 0 and self.source[pos] == ']':
                                depth = 1
                                pos -= 1
                                while pos >= 0 and depth > 0:
                                    if self.source[pos] == ']':
                                        depth += 1
                                    elif self.source[pos] == '[':
                                        depth -= 1
                                    pos -= 1
                                while pos >= 0 and self.source[pos] in ' \t':
                                    pos -= 1
                            if pos >= 0 and (self.source[pos].isalnum() or self.source[pos] == '_'):
                                while pos >= 0 and (self.source[pos].isalnum() or self.source[pos] in '_.'):
                                    pos -= 1
                                start = pos + 1
                            else:
                                # base is something we can't safely reconstruct
                                # (paren-expression, function-call result, ...)
                                start = None
                        else:
                            # first_token IS the base name - use it directly
                            start = first_pos

            last = getattr(node, 'last_token', None)
            end = self._parse_token_end(str(last)) if last and str(last) != 'None' else None
            return start, end

        # default: use first/last tokens directly
        first = getattr(node, 'first_token', None)
        last = getattr(node, 'last_token', None)

        if not first or not last or str(first) == 'None' or str(last) == 'None':
            return None, None

        start = self._parse_token_start(str(first))
        end = self._parse_token_end(str(last))

        return start, end

    def _get_call_func_span(self, node, func_name: str) -> Tuple[Optional[int], Optional[int]]:
        """Get span for the function name part of a call node."""
        from luaparser.astnodes import Index, Name

        func_node = getattr(node, 'func', None)

        # for module.func patterns (like bit.band), need special handling
        if '.' in func_name and func_node and isinstance(func_node, Index):
            # get the full span from base name to func name
            # Index.value is the base (e.g., "bit")
            # Index.idx is the function (e.g., "band")
            value = getattr(func_node, 'value', None)
            idx = getattr(func_node, 'idx', None)

            # try to get start from value's token, or search backwards from Index token
            start = None
            if value:
                value_first = getattr(value, 'first_token', None)
                if value_first and str(value_first) != 'None':
                    start = self._parse_token_start(str(value_first))

            if start is None:
                # fallback: search backwards from the dot/bracket to find base name
                func_first = getattr(func_node, 'first_token', None)
                if func_first and str(func_first) != 'None':
                    dot_pos = self._parse_token_start(str(func_first))
                    if dot_pos is not None and dot_pos > 0:
                        # search backwards for identifier start
                        pos = dot_pos - 1
                        while pos >= 0 and self.source[pos] in ' \t':
                            pos -= 1
                        while pos >= 0 and (self.source[pos].isalnum() or self.source[pos] == '_'):
                            pos -= 1
                        start = pos + 1

            # get end from idx's last token or Index's last token
            end = None
            func_last = getattr(func_node, 'last_token', None)
            if func_last and str(func_last) != 'None':
                end = self._parse_token_end(str(func_last))

            if start is not None and end is not None:
                return start, end

        # for bare names, use the func node span directly
        if func_node and not isinstance(func_node, Index):
            start, end = self._get_node_span(func_node)
            if start is not None:
                return start, end

        # fallback: find in source
        node_start, node_end = self._get_node_span(node)
        if node_start is None:
            return None, None

        # find func_name within the node text
        text = self.source[node_start:node_end]

        # for bare names like "pairs", find exact match
        if '.' not in func_name:
            # find the name followed by (
            pos = 0
            while pos < len(text):
                idx = text.find(func_name, pos)
                if idx == -1:
                    break
                # check it's a word boundary
                before_ok = (idx == 0 or not text[idx - 1].isalnum() and text[idx - 1] != '_')
                after_idx = idx + len(func_name)
                after_ok = (after_idx >= len(text)
                            or not text[after_idx].isalnum() and text[after_idx] != '_')
                if before_ok and after_ok:
                    return node_start + idx, node_start + idx + len(func_name)
                pos = idx + 1
        else:
            # for module.func, find the whole thing
            idx = text.find(func_name)
            if idx != -1:
                return node_start + idx, node_start + idx + len(func_name)

        return None, None

    def _parse_token_start(self, token_str: str) -> Optional[int]:
        """Parse start character position from token string."""
        # Format: [@index,start:end='text',<type>,line:col]
        # Positions appear to be 0-indexed
        match = re.match(r"\[@\d+,(\d+):\d+='", token_str)
        if match:
            return int(match.group(1))
        return None

    def _parse_token_end(self, token_str: str) -> Optional[int]:
        """Parse end character position from token string."""
        # End position is inclusive, we want exclusive, so add 1
        match = re.match(r"\[@\d+,\d+:(\d+)='", token_str)
        if match:
            return int(match.group(1)) + 1
        return None

    def _get_line_span(self, line_num: int) -> Tuple[Optional[int], Optional[int]]:
        """Get character span for a line (1-indexed), including newline."""
        if line_num < 1 or line_num > len(self._line_offsets):
            return None, None

        start = self._line_offsets[line_num - 1]
        
        # end is start of next line, or end of source
        if line_num < len(self._line_offsets):
            end = self._line_offsets[line_num]
        else:
            end = len(self.source)

        return start, end

    def _get_line_start(self, line_num: int) -> Optional[int]:
        """Get character position of line start."""
        if line_num < 1 or line_num > len(self._line_offsets):
            return None
        return self._line_offsets[line_num - 1]

    def _get_line_end(self, line_num: int) -> Optional[int]:
        """Get character position of line end (before newline)."""
        if line_num < 1 or line_num > len(self._line_offsets):
            return None

        start = self._line_offsets[line_num - 1]
        
        # find end of content (before newline)
        if line_num < len(self._line_offsets):
            # next line starts after newline, so content ends at offset - 1
            end = self._line_offsets[line_num] - 1
        else:
            end = len(self.source)
        
        return end

    def _get_indent_at_line(self, line_num: int) -> str:
        """Get indentation at a line."""
        start, end = self._get_line_span(line_num)
        if start is None:
            return ''
        
        line = self.source[start:end].rstrip('\n\r')
        stripped = line.lstrip()
        return line[:len(line) - len(stripped)]

    def _collect_function_locals(self, func_scope) -> Set[str]:
        """Return every local name visible anywhere inside `func_scope`'s body.

        Includes the function's own scope and every descendant scope (loop /
        block / nested function bodies). Used to avoid colliding with a cache
        name we are about to introduce - declaring `local mfloor = math.floor`
        on top of a user-written `local mfloor = ...` would silently shadow it.
        """
        if func_scope is None or self.analyzer is None:
            return set()

        target_id = id(func_scope)
        descendant_ids = {target_id}

        # walk every recorded scope; if any ancestor is func_scope, it's a descendant
        for s in self.analyzer.scopes:
            if id(s) == target_id:
                continue
            anc = s.parent
            while anc is not None:
                if id(anc) == target_id:
                    descendant_ids.add(id(s))
                    break
                anc = anc.parent

        names: Set[str] = set()
        for s in self.analyzer.scopes:
            if id(s) in descendant_ids:
                names.update(s.locals)
        return names

    @staticmethod
    def _resolve_cache_name(base_name: str, taken: Set[str]) -> str:
        """Pick an unused identifier based on `base_name`, avoiding `taken`.

        On conflict appends `_alao`, then `_alao2`, `_alao3`... - keeps the
        suggestion legible while guaranteeing uniqueness.
        """
        if base_name not in taken:
            return base_name
        candidate = f"{base_name}_alao"
        if candidate not in taken:
            return candidate
        i = 2
        while True:
            candidate = f"{base_name}_alao{i}"
            if candidate not in taken:
                return candidate
            i += 1

    def _detect_indent_unit(self) -> str:
        """Detect the file's indent style (tab or N spaces). Cached per transform."""
        if hasattr(self, '_cached_indent_unit'):
            return self._cached_indent_unit
        
        tab_lines = 0
        space_widths = []
        
        for line in self.source.split('\n')[:200]:
            if not line or not line[0] in (' ', '\t'):
                continue
            stripped = line.lstrip()
            if not stripped:
                continue
            indent = line[:len(line) - len(stripped)]
            if '\t' in indent:
                tab_lines += 1
            else:
                w = len(indent)
                if w > 0:
                    space_widths.append(w)
        
        if tab_lines > len(space_widths):
            self._cached_indent_unit = '\t'
        elif space_widths:
            # find most common indent width (likely the base unit)
            from collections import Counter
            diffs = []
            sorted_widths = sorted(set(space_widths))
            for i in range(1, len(sorted_widths)):
                d = sorted_widths[i] - sorted_widths[i - 1]
                if d > 0:
                    diffs.append(d)
            if diffs:
                unit = Counter(diffs).most_common(1)[0][0]
            else:
                unit = sorted_widths[0] if sorted_widths else 4
            self._cached_indent_unit = ' ' * unit
        else:
            self._cached_indent_unit = '\t'
        
        return self._cached_indent_unit

    def _fold_inner_into_anon(self):
        """Same-pass `--fix` (table.insert etc.) sits inside the function we hoist.

        Leave those replacements and they overlap the hoist; the hoist is dropped.
        Patch them into the anon text, then drop the inner edits.
        """
        inserts = [e for e in self.edits if e.anon_meta]
        if not inserts:
            return
        drop: Set[int] = set()
        for ins in inserts:
            fn_start, fn_end = ins.anon_meta
            # Skip this hoist's own replace. Only fold other GREEN (table.insert etc.).
            own = {
                id(e) for e in self.edits
                if e.group_id is not None
                and e.group_id == ins.group_id
                and e.start_char != e.end_char
            }
            inner = [
                e for e in self.edits
                if e.start_char != e.end_char
                and e is not ins
                and id(e) not in own
                and fn_start <= e.start_char and e.end_char <= fn_end
            ]
            if not inner:
                continue
            # Do not rebuild from the original function. That wipes nested
            # splices already in the anon text. Patch the text we have.
            ht = ins.replacement
            for e in sorted(inner, key=lambda x: -x.start_char):
                old = self.source[e.start_char:e.end_char]
                if old and old in ht:
                    ht = ht.replace(old, e.replacement, 1)
            ins.replacement = ht
            for e in inner:
                drop.add(id(e))
        if drop:
            self.edits = [e for e in self.edits if id(e) not in drop]

    def _apply_edits(self) -> str:
        """Apply all edits and return new source.

        Two-pass filter:
          (1) admit non-overlapping replacements (priority desc, start desc)
          (2) admit insertions whose position is *outside* every admitted
              replacement's span. Insertions inside a replacement would be
              lost - when the replacement is later applied end-to-start its
              new text overwrites the inserted bytes - so we drop them.

        Then apply admitted edits end-to-start so positions don't shift.
        """
        if not self.edits:
            return self.source

        from bisect import bisect_left, bisect_right

        replacements = [e for e in self.edits if e.start_char != e.end_char]
        insertions = [e for e in self.edits if e.start_char == e.end_char]

        # Pass 1: replacements - same overlap rules as before.
        replacements.sort(key=lambda e: (-e.priority, -e.start_char))
        admitted_repl = []
        covered_starts: List[int] = []
        covered_ends: List[int] = []
        for edit in replacements:
            s, e = edit.start_char, edit.end_char
            overlaps = False
            if covered_starts:
                i = bisect_right(covered_starts, s) - 1
                if i >= 0 and covered_ends[i] > s:
                    overlaps = True
                if not overlaps:
                    j = bisect_left(covered_starts, s)
                    if j < len(covered_starts) and covered_starts[j] < e:
                        overlaps = True
            if not overlaps:
                admitted_repl.append(edit)
                idx = bisect_left(covered_starts, edit.start_char)
                covered_starts.insert(idx, edit.start_char)
                covered_ends.insert(idx, edit.end_char)

        # Pass 2: insertions - dedupe by (pos, text), then drop any whose
        # position is inside an admitted replacement's span. Allow insertion
        # at the boundary (s == replacement.start) because that lands at the
        # leading edge before the replacement text.
        insertions.sort(key=lambda e: (-e.priority, -e.start_char))
        admitted_ins = []
        seen_insertions: set = set()
        for edit in insertions:
            key = (edit.start_char, edit.replacement)
            if key in seen_insertions:
                continue
            seen_insertions.add(key)
            s = edit.start_char
            inside_replacement = False
            if covered_starts:
                i = bisect_right(covered_starts, s) - 1
                if i >= 0 and covered_starts[i] < s < covered_ends[i]:
                    inside_replacement = True
            if not inside_replacement:
                admitted_ins.append(edit)

        # Pass 3: drop "enabler" insertions whose group has no surviving
        # replacement. This is how `local tostr = tostring` style cache decls
        # disappear when --fix-debug comments out every call site that would
        # have referenced them - the cache decl would otherwise sit as a dead
        # local with no callers.
        groups_with_repl: Set[int] = {
            e.group_id for e in admitted_repl if e.group_id is not None
        }
        admitted_ins = [
            e for e in admitted_ins
            if not (e.is_enabler and e.group_id is not None and e.group_id not in groups_with_repl)
        ]

        # Apply end-to-start so earlier positions stay valid.
        admitted = admitted_repl + admitted_ins
        admitted.sort(key=lambda e: (-e.start_char, -e.seq))
        result = self.source
        for edit in admitted:
            result = result[:edit.start_char] + edit.replacement + result[edit.end_char:]
        return result


def transform_file(file_path: Path, backup: bool = True, dry_run: bool = False,
                   fix_debug: bool = False, fix_yellow: bool = False,
                   experimental: bool = False, fix_nil: bool = False,
                   remove_dead_code: bool = False,
                   cache_threshold: int = 4,
                   hoist_anon_funcs: bool = False) -> Tuple[bool, str, int]:
    """Convenience function to transform a file. Returns (modified, content, edit_count)."""
    transformer = ASTTransformer()
    return transformer.transform_file(file_path, backup, dry_run, fix_debug, fix_yellow, 
                                       experimental, fix_nil, remove_dead_code, cache_threshold,
                                       hoist_anon_funcs=hoist_anon_funcs)
