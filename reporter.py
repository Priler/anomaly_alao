"""
This generates HTML analysis reports using Jinja2 templating.
Templates are inside "templates" folder.

bruh wish we had Twig :3
"""

import html
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict
from collections import defaultdict
from datetime import datetime

from models import Finding

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


# performance impact levels for different patterns (based on real-world benchmarks)
PERFORMANCE_IMPACT = {
    # CRITICAL - can destroy frame time
    'expensive_in_hotpath': 'critical',
    'string_concat_in_loop': 'critical',

    # HIGH - moderate to high impact in tight loops
    'table_insert_append': 'high',
    'math_pow_simple': 'high',
    'string_format_in_loop': 'high',

    # MEDIUM - low to moderate impact
    'uncached_globals_summary': 'medium',
    'math_pow_dotted': 'medium',
    'pairs_on_array': 'medium',
    'debug_statement': 'medium',

    # LOW - minor impact
    'table_getn': 'low',
    'string_len': 'low',
    'global_write': 'low',
    'empty_function': 'low',
    'math_pow_complex': 'low',
    'file_too_large': 'low',
    'read_error': 'low',
    'file_too_many_lines': 'low',
}


def get_performance_impact(pattern_name: str) -> str:
    """Get performance impact level for a pattern."""
    return PERFORMANCE_IMPACT.get(pattern_name, 'low')


def highlight_code_match(line_content: str, details: dict, pattern_name: str) -> str:
    """Add HTML highlighting to the part of code that will be changed."""
    if not line_content:
        return ""
    if not details:
        return html.escape(line_content)

    escaped = html.escape(line_content)

    # get the match to highlight based on pattern type
    match_text = None

    if pattern_name in ('table_insert_append', 'table_getn', 'string_len',
                        'math_pow_simple', 'math_pow_dotted', 'math_pow_complex'):
        match_text = details.get('full_match')
    elif pattern_name == 'global_write':
        var = details.get('variable')
        if var:
            match_text = var
    elif pattern_name == 'expensive_in_hotpath':
        ops = details.get('operations', [])
        if ops:
            match_text = ops[0].rstrip('()')
    elif pattern_name == 'string_concat_in_loop':
        var = details.get('variable')
        if var:
            match_text = f"{var} = {var} .."
    elif pattern_name == 'string_format_in_loop':
        match_text = 'string.format'
    elif pattern_name == 'pairs_on_array':
        table = details.get('table')
        if table:
            match_text = f"pairs({table})"
    elif pattern_name == 'debug_statement':
        funcs = details.get('functions', [])
        if funcs:
            match_text = funcs[0].rstrip('()')
    elif pattern_name == 'uncached_globals_summary':
        # for summary, don't highlight (multi-line examples)
        return escaped

    if match_text:
        escaped_match = html.escape(match_text)
        if escaped_match in escaped:
            highlighted = f'<span class="highlight">{escaped_match}</span>'
            escaped = escaped.replace(escaped_match, highlighted, 1)

    return escaped


def get_templates_dir() -> Path:
    """Get path to templates directory."""
    module_dir = Path(__file__).parent
    templates_dir = module_dir / "templates"

    if templates_dir.exists():
        return templates_dir

    cwd_templates = Path.cwd() / "templates"
    if cwd_templates.exists():
        return cwd_templates

    return templates_dir


class Reporter:
    """Collects findings and generates reports."""

    def __init__(self):
        # mod_name -> file_path -> list of findings
        self.findings: Dict[str, Dict[str, List[Finding]]] = defaultdict(lambda: defaultdict(list))
        self.start_time = datetime.now()

        # setup jinja2 if available
        self._jinja_env = None
        if JINJA2_AVAILABLE:
            templates_dir = get_templates_dir()
            if templates_dir.exists():
                self._jinja_env = Environment(
                    loader=FileSystemLoader(str(templates_dir)),
                    autoescape=select_autoescape(['html', 'xml'])
                )
                self._jinja_env.filters['basename'] = lambda p: Path(p).name

    def add_finding(self, mod_name: str, file_path: Path, finding: Finding):
        """Add a finding to the report."""
        self.findings[mod_name][str(file_path)].append(finding)

    def count_by_severity(self, severity: str) -> int:
        """Count total findings of a specific severity."""
        count = 0
        for mod in self.findings.values():
            for file_findings in mod.values():
                count += sum(1 for f in file_findings if f.severity == severity)
        return count

    def total_findings(self) -> int:
        """Total number of findings."""
        return sum(
            len(findings)
            for mod in self.findings.values()
            for findings in mod.values()
        )

    def get_top_issues(self, limit: int = 10) -> List[tuple]:
        """Get top issues by pattern count with severity info."""
        pattern_counts = defaultdict(int)
        pattern_severity = {}
        for mod in self.findings.values():
            for file_findings in mod.values():
                for f in file_findings:
                    pattern_counts[f.pattern_name] += 1
                    if f.pattern_name not in pattern_severity:
                        pattern_severity[f.pattern_name] = f.severity

        result = []
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1])[:limit]:
            severity = pattern_severity.get(pattern, 'RED')
            result.append((pattern, count, severity))
        return result

    def get_mod_severity_breakdown(self, mod_name: str) -> dict:
        """Get severity breakdown for a specific mod."""
        counts = {'GREEN': 0, 'YELLOW': 0, 'RED': 0, 'DEBUG': 0}
        if mod_name in self.findings:
            for file_findings in self.findings[mod_name].values():
                for f in file_findings:
                    if f.severity in counts:
                        counts[f.severity] += 1
        return counts

    def print_summary(self):
        """Print a summary to stdout."""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)

        green = self.count_by_severity("GREEN")
        yellow = self.count_by_severity("YELLOW")
        red = self.count_by_severity("RED")
        debug = self.count_by_severity("DEBUG")

        print(f"\n  GREEN  (auto-fixable):  {green:5d}")
        print(f"  YELLOW (review needed): {yellow:5d}")
        print(f"  RED    (info only):     {red:5d}")
        print(f"  DEBUG  (logging):       {debug:5d}")
        print(f"  {'-' * 28}")
        print(f"  TOTAL:                  {green + yellow + red + debug:5d}")

        top_issues = self.get_top_issues()
        if top_issues:
            print("\nTop issues by type:")
            for pattern, count, severity in top_issues:
                marker = {
                    'GREEN': '[G]',
                    'YELLOW': '[Y]',
                    'RED': '[R]',
                    'DEBUG': '[D]'}.get(
                    severity,
                    '[ ]')
                print(f"  {marker} {pattern}: {count}")

        mod_counts = {
            mod: sum(len(f) for f in files.values())
            for mod, files in self.findings.items()
        }

        if mod_counts:
            print("\nMods with most issues:")
            for mod, count in sorted(mod_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  [{count:4d}] {mod}")

    def print_detailed(self):
        """Print detailed findings to stdout."""
        print("\n" + "=" * 60)
        print("DETAILED FINDINGS")
        print("=" * 60)

        for mod_name, files in sorted(self.findings.items()):
            print(f"\n{'-' * 60}")
            print(f"MOD: {mod_name}")
            print(f"{'-' * 60}")

            for file_path, findings in sorted(files.items()):
                file_name = Path(file_path).name
                print(f"\n  {file_name}:")

                by_severity = defaultdict(list)
                for f in findings:
                    by_severity[f.severity].append(f)

                for severity in ['GREEN', 'YELLOW', 'RED', 'DEBUG']:
                    if severity in by_severity:
                        for f in by_severity[severity]:
                            marker = {
                                'GREEN': '[G]',
                                'YELLOW': '[Y]',
                                'RED': '[R]',
                                'DEBUG': '[D]'}[severity]
                            print(f"    {marker} L{f.line_num}: {f.pattern_name}")
                            if f.details:
                                detail_str = format_details(f.details)
                                if detail_str:
                                    print(f"        {detail_str}")

    def save(self, path: Path, verbose: bool = False):
        """Save report to file (txt, html, or json)."""
        suffix = path.suffix.lower()

        if suffix == '.json':
            self._save_json(path, verbose)
        elif suffix == '.html':
            self._save_html(path, verbose)
        else:
            self._save_txt(path, verbose)

    def _get_template_data(self) -> dict:
        """Prepare data for template rendering."""
        findings_data = {}
        mod_breakdowns = {}
        impact_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        pattern_counts = {}  # pattern_name -> count

        for mod, files in sorted(self.findings.items()):
            findings_data[mod] = {}
            mod_breakdowns[mod] = self.get_mod_severity_breakdown(mod)

            for file_path, file_findings in sorted(files.items()):
                findings_data[mod][file_path] = []
                for f in file_findings:
                    impact = get_performance_impact(f.pattern_name)
                    findings_data[mod][file_path].append({
                        'line_num': f.line_num,
                        'line_content': highlight_code_match(f.line_content, f.details, f.pattern_name),
                        'pattern': f.pattern_name,
                        'severity': f.severity,
                        'description': f.description,
                        'details': f.details,
                        'performance_impact': impact,
                    })
                    # count impacts
                    if impact in impact_counts:
                        impact_counts[impact] += 1
                    # count patterns
                    if f.pattern_name not in pattern_counts:
                        pattern_counts[f.pattern_name] = 0
                    pattern_counts[f.pattern_name] += 1

        # sort patterns by count
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])

        return {
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total': self.total_findings(),
                'green': self.count_by_severity('GREEN'),
                'yellow': self.count_by_severity('YELLOW'),
                'red': self.count_by_severity('RED'),
                'debug': self.count_by_severity('DEBUG'),
            },
            'impact': impact_counts,
            'patterns': sorted_patterns,
            'top_issues': self.get_top_issues(),
            'findings': findings_data,
            'mod_breakdowns': mod_breakdowns,
        }

    def _save_json(self, path: Path, verbose: bool = False):
        """Save as JSON."""
        if verbose:
            print("  Preparing JSON data...", end="", flush=True)

        data = {
            'generated': datetime.now().isoformat(),
            'summary': {
                'total': self.total_findings(),
                'green': self.count_by_severity('GREEN'),
                'yellow': self.count_by_severity('YELLOW'),
                'red': self.count_by_severity('RED'),
                'debug': self.count_by_severity('DEBUG'),
            },
            'findings': {}
        }

        total_mods = len(self.findings)
        for idx, (mod, files) in enumerate(self.findings.items()):
            if verbose and total_mods > 10:
                progress = (idx + 1) / total_mods * 100
                print(f"\r  Processing mods: {progress:.0f}%  ", end="", flush=True)

            data['findings'][mod] = {}
            for file_path, findings in files.items():
                data['findings'][mod][file_path] = [
                    {
                        'line': f.line_num,
                        'pattern': f.pattern_name,
                        'severity': f.severity,
                        'description': f.description,
                        'details': f.details
                    }
                    for f in findings
                ]

        if verbose:
            print("\r  Writing file...              ", end="", flush=True)

        path.write_text(json.dumps(data, indent=2), encoding='utf-8')

        if verbose:
            print("\r  Done.                        ")

    def _save_txt(self, path: Path, verbose: bool = False):
        """Save as plain text."""
        if verbose:
            print("  Generating text report...", end="", flush=True)

        lines = []
        lines.append("Anomaly Lua Script Analysis Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")

        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"GREEN  (auto-fixable):  {self.count_by_severity('GREEN')}")
        lines.append(f"YELLOW (review needed): {self.count_by_severity('YELLOW')}")
        lines.append(f"RED    (info only):     {self.count_by_severity('RED')}")
        lines.append(f"DEBUG  (logging):       {self.count_by_severity('DEBUG')}")
        lines.append(f"TOTAL: {self.total_findings()}")
        lines.append("")

        lines.append("DETAILED FINDINGS")
        lines.append("=" * 60)

        total_mods = len(self.findings)
        for idx, (mod_name, files) in enumerate(sorted(self.findings.items())):
            if verbose and total_mods > 10:
                progress = (idx + 1) / total_mods * 100
                print(f"\r  Processing mods: {progress:.0f}%  ", end="", flush=True)

            lines.append("")
            lines.append(f"MOD: {mod_name}")
            lines.append("-" * 40)

            for file_path, findings in sorted(files.items()):
                file_name = Path(file_path).name
                lines.append(f"  {file_name}:")

                for f in sorted(findings, key=lambda x: x.line_num):
                    lines.append(f"    [{f.severity}] L{f.line_num}: {f.pattern_name}")
                    lines.append(f"           {f.description}")

        if verbose:
            print("\r  Writing file...              ", end="", flush=True)

        path.write_text('\n'.join(lines), encoding='utf-8')

        if verbose:
            print("\r  Done.                        ")

    def _save_html(self, path: Path, verbose: bool = False):
        """Save as HTML report using Jinja2 template if available."""
        if verbose:
            print("  Preparing template data...", end="", flush=True)

        if self._jinja_env:
            try:
                template = self._jinja_env.get_template('report.html')
                data = self._get_template_data()

                if verbose:
                    print("\r  Rendering template...        ", end="", flush=True)

                html_content = template.render(**data)

                if verbose:
                    print("\r  Writing file...              ", end="", flush=True)

                path.write_text(html_content, encoding='utf-8')

                if verbose:
                    print("\r  Done.                        ")
                return
            except Exception as e:
                print(f"\n  Warning: Template rendering failed ({e}), make sure to install Jinja2 and have the templates available.")

def format_details(details: dict) -> str:
    """Format details dict for display."""
    if not details:
        return ""

    parts = []
    for key, value in details.items():
        if key == 'globals' and isinstance(value, dict):
            items = [f"{k}({v}x)" for k, v in value.items()]
            parts.append(f"globals: {', '.join(items)}")
        elif key == 'globals' and isinstance(value, list):
            parts.append(f"globals: {', '.join(value)}")
        elif isinstance(value, list):
            parts.append(f"{key}: {', '.join(str(v) for v in value)}")
        elif isinstance(value, bool):
            if value:
                parts.append(key)
        else:
            parts.append(f"{key}: {value}")

    return "; ".join(parts)
