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


@dataclass
class SourceEdit:
    """A source code edit with character positions."""
    start_char: int      # start character offset in source
    end_char: int        # end character offset (exclusive)
    replacement: str     # replacement text
    priority: int = 0    # higher priority edits applied first


class ASTTransformer:
    """Transform Lua source using AST-based analysis."""

    def __init__(self):
        self.source: str = ""
        self.edits: List[SourceEdit] = []
        self.file_path: Optional[Path] = None
        self.analyzer: Optional[ASTAnalyzer] = None

    def transform_file(self, file_path: Path, backup: bool = True, dry_run: bool = False,
                       fix_debug: bool = False, fix_yellow: bool = False) -> Tuple[bool, str, int]:
        """
        Transform a file based on findings.
        Returns (was_modified, new_content, edit_count).
        """
        self.file_path = file_path
        self.edits = []

        # run analyzer
        self.analyzer = ASTAnalyzer()
        findings = self.analyzer.analyze_file(file_path)

        # get source from analyzer
        self.source = self.analyzer.source

        # filter to fixable severities
        allowed_severities = {'GREEN'}
        if fix_yellow:
            allowed_severities.add('YELLOW')
        if fix_debug:
            allowed_severities.add('DEBUG')

        fixable = [f for f in findings if f.severity in allowed_severities]

        if not fixable:
            return False, self.source, 0

        # generate edits for each finding
        for finding in fixable:
            self._generate_edits(finding)

        if not self.edits:
            return False, self.source, 0

        edit_count = len(self.edits)

        # apply edits
        new_content = self._apply_edits()

        if new_content == self.source:
            return False, self.source, 0

        if not dry_run:
            if backup:
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                if not backup_path.exists():
                    shutil.copy2(file_path, backup_path)

            file_path.write_text(new_content, encoding='utf-8')

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
        elif pattern == 'debug_statement':
            self._edit_debug_statement(finding)
        elif pattern == 'uncached_globals_summary':
            self._edit_uncached_globals(finding)
        elif pattern.startswith('repeated_'):
            self._edit_repeated_calls(finding)


    # Edit methods using AST positions

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
        """Extract the value argument from table.insert(t, v) call text."""
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
        bracket_depth = 0
        in_string = False
        string_char = None
        i = paren_start + 1

        while i < len(call_text) and depth > 0:
            c = call_text[i]

            if not in_string:
                if c in ('"', "'"):
                    in_string = True
                    string_char = c
                elif c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif c == '{':
                    brace_depth += 1
                elif c == '}':
                    brace_depth -= 1
                elif c == '[':
                    bracket_depth += 1
                elif c == ']':
                    bracket_depth -= 1
            else:
                if c == string_char and (i == 0 or call_text[i - 1] != '\\'):
                    in_string = False

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
            replacement = f'{base}^0.5'
        elif pow_type == 'power' and isinstance(exp, int):
            replacement = '*'.join([base] * exp)
        else:
            return

        self.edits.append(SourceEdit(
            start_char=start,
            end_char=end,
            replacement=replacement,
        ))

    def _edit_debug_statement(self, finding: Finding):
        """Comment out debug statement (handles multi-line calls)."""
        node = finding.details.get('node')

        # control flow keywords that shouldn't be commented out
        control_flow_keywords = ['if ', 'then ', 'else', 'elseif ', 'end', 'for ', 'while ',
                                 'do ', 'repeat', 'until ', 'function ', 'return ']

        if node:
            # use AST node to get full span of call
            start_char, end_char = self._get_node_span(node)
            if start_char is not None:
                # find all lines this call spans
                start_line = self.source[:start_char].count('\n') + 1
                end_line = self.source[:end_char].count('\n') + 1

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

                    # check for control flow on ANY line of the multi-line statement
                    for kw in control_flow_keywords:
                        if kw in stripped.lower():
                            has_control_flow = True
                            break

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

        # skip lines with control flow
        for kw in control_flow_keywords:
            if kw in stripped.lower():
                return

        indent = line[:len(line) - len(stripped)]
        new_line = f'{indent}-- {stripped}'

        self.edits.append(SourceEdit(
            start_char=start,
            end_char=end,
            replacement=new_line,
            priority=200,
        ))

    def _edit_uncached_globals(self, finding: Finding):
        """Add local caching for globals at function start."""
        details = finding.details
        globals_info = details.get('globals_info', {})
        scope = details.get('scope')

        if not globals_info or not scope:
            return

        # build cache declarations and track replacements
        cache_lines = []
        replacements: Dict[str, str] = {}

        for name in sorted(globals_info.keys()):
            if '.' in name:
                module, func = name.split('.', 1)
                cache_name = f'{module}_{func}'
            else:
                cache_name = f'g_{name}'

            cache_lines.append(f'local {cache_name} = {name}')
            replacements[name] = cache_name

        if not cache_lines:
            return

        # find insertion point - after function definition line
        insert_pos = self._get_line_end(scope.start_line)
        if insert_pos is None:
            return

        # get indentation
        indent = self._get_indent_at_line(scope.start_line + 1)
        if not indent:
            indent = '\t'

        # build cache block
        cache_block = '\n' + '\n'.join(f'{indent}{line}' for line in cache_lines)

        self.edits.append(SourceEdit(
            start_char=insert_pos,
            end_char=insert_pos,
            replacement=cache_block,
            priority=100,
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
                ))

    def _edit_repeated_calls(self, finding: Finding):
        """Add caching for repeated expensive calls."""
        details = finding.details
        calls = details.get('calls', [])
        scope = details.get('scope')
        suggestion = details.get('suggestion', '')

        if not calls or not scope:
            return

        pattern = finding.pattern_name

        # determine cache variable name and cache line
        if pattern == 'repeated_db_actor':
            cache_line = 'local actor = db.actor'
            new_name = 'actor'
            call_pattern = 'db.actor'
        elif pattern == 'repeated_time_global':
            cache_line = 'local tg = time_global()'
            new_name = 'tg'
            call_pattern = 'time_global()'
        elif pattern == 'repeated_alife':
            cache_line = 'local sim = alife()'
            new_name = 'sim'
            call_pattern = 'alife()'
        elif pattern == 'repeated_system_ini':
            cache_line = 'local ini = system_ini()'
            new_name = 'ini'
            call_pattern = 'system_ini()'
        elif pattern == 'repeated_level_object_by_id':
            cache_line = suggestion if suggestion else 'local obj = level.object_by_id(id)'
            new_name = 'obj'
            call_pattern = 'level.object_by_id'
        else:
            return

        # pattern to detect exact cache declaration (local obj =, not local obj1 =)
        cache_decl_pattern = rf'\blocal\s+{re.escape(new_name)}\s*='

        # check if first call is already a cache declaration for THIS pattern
        # i.e., "local obj = level.object_by_id(...)" where obj is the cache var
        first_call = calls[0]
        first_line_start, first_line_end = self._get_line_span(first_call.line)
        is_already_cached = False
        cache_indent = None

        if first_line_start is not None:
            first_line = self.source[first_line_start:first_line_end]
            if re.search(cache_decl_pattern, first_line) and call_pattern in first_line:
                is_already_cached = True
                cache_indent = self._get_indent_at_line(first_call.line)

        # insert cache if not already present
        if not is_already_cached:
            insert_pos = self._get_line_start(first_call.line)
            if insert_pos is None:
                return

            indent = self._get_indent_at_line(first_call.line)

            # check if first call is inside a table constructor
            # look for unbalanced { in lines before this one within scope
            if scope and hasattr(scope, 'start_line'):
                brace_depth = 0
                has_loop_before_first_call = False

                for check_line in range(scope.start_line, first_call.line + 1):
                    ls, le = self._get_line_span(check_line)
                    if ls is not None:
                        line_text = self.source[ls:le]
                        # skip comments
                        if '--' in line_text:
                            line_text = line_text[:line_text.find('--')]
                        brace_depth += line_text.count('{') - line_text.count('}')

                        # check for loop constructs before first call
                        if check_line < first_call.line:
                            stripped = line_text.strip().lower()
                            if stripped.startswith(('for ', 'while ', 'repeat')):
                                has_loop_before_first_call = True

                if brace_depth > 0:
                    return  # inside table constructor, skip optimization

                # if calls span multiple lines and first call is inside a loop,
                # insert at function body start instead to avoid scope issues
                last_call = calls[-1]
                if has_loop_before_first_call and last_call.line > first_call.line:
                    # insert right after function declaration
                    insert_pos = self._get_line_start(scope.start_line + 1)
                    if insert_pos is None:
                        return
                    indent = self._get_indent_at_line(scope.start_line + 1)

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
                if re.search(cache_decl_pattern, line) and call_pattern in line:
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
                    while pos >= 0 and (self.source[pos].isalnum() or self.source[pos] == '_'):
                        pos -= 1
                    start = pos + 1

                    # end is the closing paren
                    last = getattr(node, 'last_token', None)
                    end = self._parse_token_end(str(last)) if last else None

                    if start is not None and end is not None:
                        return start, end

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
        lines = self.source.split('\n')
        if line_num < 1 or line_num > len(lines):
            return None, None

        start = sum(len(l) + 1 for l in lines[:line_num - 1])
        end = start + len(lines[line_num - 1])

        # include newline if not last line
        if line_num < len(lines):
            end += 1

        return start, end

    def _get_line_start(self, line_num: int) -> Optional[int]:
        """Get character position of line start."""
        start, _ = self._get_line_span(line_num)
        return start

    def _get_line_end(self, line_num: int) -> Optional[int]:
        """Get character position of line end (before newline)."""
        lines = self.source.split('\n')
        if line_num < 1 or line_num > len(lines):
            return None

        pos = sum(len(l) + 1 for l in lines[:line_num - 1])
        pos += len(lines[line_num - 1])
        return pos

    def _get_indent_at_line(self, line_num: int) -> str:
        """Get indentation at a line."""
        lines = self.source.split('\n')
        if 0 < line_num <= len(lines):
            line = lines[line_num - 1]
            stripped = line.lstrip()
            return line[:len(line) - len(stripped)]
        return ''

    def _apply_edits(self) -> str:
        """Apply all edits and return new source."""
        if not self.edits:
            return self.source

        # sort by priority descending, then position descending
        self.edits.sort(key=lambda e: (-e.priority, -e.start_char))

        # remove overlapping edits (keep higher priority / earlier in sort)
        filtered = []
        covered_ranges: List[Tuple[int, int]] = []

        for edit in self.edits:
            overlaps = False
            for start, end in covered_ranges:
                # check for overlap
                if edit.start_char < end and edit.end_char > start:
                    overlaps = True
                    break

            if not overlaps:
                filtered.append(edit)
                if edit.start_char != edit.end_char:  # don't track insertions as covered
                    covered_ranges.append((edit.start_char, edit.end_char))

        # apply edits from end to start (so positions don't shift)
        result = self.source
        for edit in sorted(filtered, key=lambda e: -e.start_char):
            result = result[:edit.start_char] + edit.replacement + result[edit.end_char:]

        return result


def transform_file(file_path: Path, backup: bool = True, dry_run: bool = False,
                   fix_debug: bool = False, fix_yellow: bool = False) -> Tuple[bool, str, int]:
    """Convenience function to transform a file. Returns (modified, content, edit_count)."""
    transformer = ASTTransformer()
    return transformer.transform_file(file_path, backup, dry_run, fix_debug, fix_yellow)
