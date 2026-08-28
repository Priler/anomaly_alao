"""Tests for pcall/xpcall anonymous-function hoisting."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ast_analyzer import ASTAnalyzer
from ast_transformer import transform_file


def _write(src: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".script",
        delete=False,
        newline="\n",
    )
    tmp.write(src)
    tmp.close()
    return Path(tmp.name)


def analyze(src: str):
    path = _write(src)
    try:
        a = ASTAnalyzer()
        return a.analyze_file(path)
    finally:
        path.unlink(missing_ok=True)


def transform(src: str, fix_pcall: bool = True) -> str:
    path = _write(src)
    try:
        _, content, _ = transform_file(
            path,
            backup=False,
            dry_run=True,
            fix_pcall=fix_pcall,
        )
        if fix_pcall:
            assert "pcallpcall" not in content, content
            assert "pcalllocal" not in content, content
            assert "xpcallxpcall" not in content, content
        return content
    finally:
        path.unlink(missing_ok=True)


def _findings(src: str, pattern: str):
    return [f for f in analyze(src) if f.pattern_name == pattern]


class TestPcallHoist(unittest.TestCase):
    def test_no_capture_hoist(self):
        src = (
            "local function foo()\n"
            "\tpcall(function()\n"
            "\t\tlevel.enable_input()\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local function foo_pcall_1()", out)
        self.assertIn("\tpcall(foo_pcall_1)", out)
        self.assertNotIn("pcall(function()", out)
        helpers = out.split("local function foo()")[0]
        self.assertEqual(
            helpers,
            "local function foo_pcall_1()\n"
            "\tlevel.enable_input()\n"
            "end\n\n",
        )

    def test_read_only_capture(self):
        src = (
            "local function apply_description_visibility(info)\n"
            "\tpcall(function()\n"
            "\t\tinfo.desc:AdjustHeightToText()\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local function apply_description_visibility_pcall_1(info)", out)
        self.assertIn("pcall(apply_description_visibility_pcall_1, info)", out)
        self.assertIn("info.desc:AdjustHeightToText()", out)

    def test_field_write_is_capture_read(self):
        src = (
            "local function set_power(info)\n"
            "\tpcall(function() info.dest_gvid = 1 end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("set_power_pcall_1(info)", out)
        self.assertIn("pcall(set_power_pcall_1, info)", out)
        green = _findings(src, "pcall_anon_hoist")
        self.assertEqual(len(green), 1)
        self.assertEqual(green[0].details.get("rewrite_kind"), "hoist")

    def test_single_assign_rewrite(self):
        src = (
            "local function active_slot()\n"
            "\tlocal slot = nil\n"
            "\tpcall(function() slot = db.actor:active_slot() end)\n"
            "\treturn slot\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local function active_slot_pcall_1()", out)
        self.assertIn("return db.actor:active_slot()", out)
        self.assertIn("local _ok, _slot = pcall(active_slot_pcall_1)", out)
        self.assertIn("if _ok then slot = _slot end", out)
        self.assertNotIn("pcall(function()", out)

    def test_local_ok_kept(self):
        src = (
            "local function wrap()\n"
            "\tlocal slot = nil\n"
            "\tlocal ok = pcall(function() slot = db.actor:active_slot() end)\n"
            "\treturn ok, slot\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local ok, _slot = pcall(wrap_pcall_1)", out)
        self.assertIn("if ok then slot = _slot end", out)

    def test_skip_two_unpack(self):
        src = (
            "local function wrap()\n"
            "\tlocal slot = nil\n"
            "\tlocal ok, extra = pcall(function() slot = db.actor:active_slot() end)\n"
            "\treturn ok, extra, slot\n"
            "end\n"
        )
        out = transform(src)
        self.assertEqual(out, src)
        red = _findings(src, "pcall_anon_skip")
        self.assertTrue(any("2+" in (f.details.get("skip_reason") or "") for f in red))

    def test_skip_if_pcall_assign(self):
        src = (
            "local function wrap()\n"
            "\tlocal slot = nil\n"
            "\tif pcall(function() slot = db.actor:active_slot() end) then\n"
            "\t\treturn slot\n"
            "\tend\n"
            "end\n"
        )
        out = transform(src)
        self.assertEqual(out, src)
        red = _findings(src, "pcall_anon_skip")
        self.assertTrue(any("expression" in (f.details.get("skip_reason") or "") for f in red))

    def test_skip_xpcall_with_captures(self):
        src = (
            "local function wrap(pui)\n"
            "\txpcall(function()\n"
            "\t\tif not pui then return end\n"
            "\tend, print)\n"
            "end\n"
        )
        out = transform(src)
        self.assertEqual(out, src)
        red = _findings(src, "pcall_anon_skip")
        self.assertTrue(any("xpcall" in (f.details.get("skip_reason") or "") for f in red))

    def test_skip_dots(self):
        src = (
            "local function wrap(...)\n"
            "\tpcall(function(...)\n"
            "\t\treturn ...\n"
            "\tend, ...)\n"
            "end\n"
        )
        out = transform(src)
        self.assertEqual(out, src)
        red = _findings(src, "pcall_anon_skip")
        self.assertTrue(red)

    def test_skip_multi_stmt_write(self):
        src = (
            "local function wrap()\n"
            "\tlocal slot = nil\n"
            "\tpcall(function()\n"
            "\t\tlevel.enable_input()\n"
            "\t\tslot = db.actor:active_slot()\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertEqual(out, src)
        red = _findings(src, "pcall_anon_skip")
        self.assertTrue(any("writes" in (f.details.get("skip_reason") or "") for f in red))

    def test_helper_above_outermost_named(self):
        src = (
            "local function outer()\n"
            "\tlocal function inner(x)\n"
            "\t\tpcall(function() return x + 1 end)\n"
            "\tend\n"
            "\tinner(1)\n"
            "end\n"
        )
        out = transform(src)
        self.assertTrue(out.startswith("local function inner_pcall_1(x)"), out)
        self.assertIn("pcall(inner_pcall_1, x)", out)
        # helper is not nested inside outer
        after_outer = out.split("local function outer()", 1)[1]
        self.assertNotIn("local function inner_pcall_1", after_outer)

    def test_two_pcalls_numbered(self):
        src = (
            "local function foo(a, b)\n"
            "\tpcall(function() return a end)\n"
            "\tpcall(function() return b end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local function foo_pcall_1(a)", out)
        self.assertIn("local function foo_pcall_2(b)", out)
        self.assertIn("pcall(foo_pcall_1, a)", out)
        self.assertIn("pcall(foo_pcall_2, b)", out)
        self.assertLess(out.find("foo_pcall_1"), out.find("foo_pcall_2"))

    def test_nested_pcall_hoists_both(self):
        src = (
            "local function foo()\n"
            "\tpcall(function()\n"
            "\t\tpcall(function() return 1 end)\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local function foo_pcall_1()", out)
        self.assertIn("local function foo_pcall_2()", out)
        self.assertIn("pcall(foo_pcall_1)", out)
        self.assertIn("pcall(foo_pcall_2)", out)
        self.assertNotIn("pcall(function()", out)
        self.assertLess(out.find("foo_pcall_1"), out.find("foo_pcall_2"))
        green = _findings(src, "pcall_anon_hoist")
        self.assertEqual(len(green), 2)

    def test_skip_does_not_eat_helper_number(self):
        src = (
            "local function foo(a, b)\n"
            "\tlocal x = 0\n"
            "\tpcall(function() return a end)\n"
            "\tpcall(function()\n"
            "\t\tx = 1\n"
            "\t\tx = 2\n"
            "\tend)\n"
            "\tpcall(function() return b end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local function foo_pcall_1(a)", out)
        self.assertIn("local function foo_pcall_2(b)", out)
        self.assertNotIn("foo_pcall_3", out)
        self.assertIn("pcall(function()", out)

    def test_flag_off_does_not_rewrite(self):
        src = (
            "local function foo()\n"
            "\tpcall(function() level.enable_input() end)\n"
            "end\n"
        )
        out = transform(src, fix_pcall=False)
        self.assertEqual(out, src)

    def test_xpcall_no_capture_hoist(self):
        src = (
            "local function foo()\n"
            "\txpcall(function() level.enable_input() end, print)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local function foo_pcall_1()", out)
        self.assertIn("xpcall(foo_pcall_1, print)", out)

    def test_assign_anon_hoists_above_assign(self):
        src = (
            "foo = function(x)\n"
            "\tpcall(function() return x + 1 end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertTrue(out.startswith("local function foo_pcall_1(x)"), out)
        self.assertIn("pcall(foo_pcall_1, x)", out)
        self.assertNotIn("local function", out.split("foo = function", 1)[1])

    def test_do_block_hoists_above_do(self):
        src = (
            "do\n"
            "\tlocal ok, t = pcall(function() return profile_timer() end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertTrue(out.startswith("local function chunk_pcall_1()"), out)
        self.assertIn("pcall(chunk_pcall_1)", out)
        self.assertNotIn("local function", out.split("do\n", 1)[1])

    def test_assign_anon_inside_named_hoists_to_chunk(self):
        src = (
            "local function outer()\n"
            "\tfoo = function(x)\n"
            "\t\tpcall(function() return x end)\n"
            "\tend\n"
            "end\n"
        )
        out = transform(src)
        self.assertTrue(out.startswith("local function foo_pcall_1(x)"), out)
        after_outer = out.split("local function outer()", 1)[1]
        self.assertNotIn("local function foo_pcall_1", after_outer)

    def test_local_func_inside_do_hoists_above_do(self):
        src = (
            "do\n"
            "\tlocal function foo()\n"
            "\t\tpcall(function() level.enable_input() end)\n"
            "\tend\n"
            "end\n"
        )
        out = transform(src)
        self.assertTrue(out.startswith("local function foo_pcall_1()"), out)
        self.assertNotIn("local function foo_pcall_1", out.split("do\n", 1)[1])

    def test_finding_message_matches_alao_style(self):
        src = (
            "local function apply(info)\n"
            "\tpcall(function() info.desc:AdjustHeightToText() end)\n"
            "end\n"
        )
        green = _findings(src, "pcall_anon_hoist")
        self.assertEqual(len(green), 1)
        f = green[0]
        self.assertIn("pcall(function() ... end) -> pcall(apply_pcall_1, info)", f.message)
        self.assertEqual(f.details.get("full_match"), "pcall(function")
        self.assertEqual(f.details.get("suggestion"), "Hoist to apply_pcall_1")
        from reporter import format_details, get_performance_impact
        shown = format_details(f.details)
        self.assertNotIn("helper_text", shown)
        self.assertNotIn("insert_char", shown)
        self.assertNotIn("insert_seq", shown)
        self.assertIn("helper_name", shown)
        self.assertEqual(get_performance_impact("pcall_anon_hoist"), "high")
        self.assertEqual(get_performance_impact("pcall_anon_skip"), "high")


if __name__ == "__main__":
    unittest.main()
