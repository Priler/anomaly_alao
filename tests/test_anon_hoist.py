"""Tests for anonymous-function hoisting."""

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
        a = ASTAnalyzer(hoist_anon_funcs=True)
        return a.analyze_file(path)
    finally:
        path.unlink(missing_ok=True)


def transform(src: str, hoist_anon_funcs: bool = True) -> str:
    path = _write(src)
    try:
        _, content, _ = transform_file(
            path,
            backup=False,
            dry_run=True,
            hoist_anon_funcs=hoist_anon_funcs,
        )
        if hoist_anon_funcs:
            assert "pcallpcall" not in content, content
            assert "pcalllocal" not in content, content
            assert "xpcallxpcall" not in content, content
        return content
    finally:
        path.unlink(missing_ok=True)


def _findings(src: str, pattern: str):
    return [f for f in analyze(src) if f.pattern_name == pattern]


class TestAnonHoist(unittest.TestCase):
    def test_no_capture_hoist(self):
        src = (
            "local function foo()\n"
            "\tpcall(function()\n"
            "\t\tlevel.enable_input()\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local foo_anon_1 = function(level)", out)
        self.assertIn("\tpcall(foo_anon_1, level)", out)
        self.assertNotIn("pcall(function()", out)
        anons = out.split("local function foo()")[0]
        self.assertEqual(
            anons,
            "local foo_anon_1 = function(level)\n"
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
        self.assertIn("local apply_description_visibility_anon_1 = function(info)", out)
        self.assertIn("pcall(apply_description_visibility_anon_1, info)", out)
        self.assertIn("info.desc:AdjustHeightToText()", out)

    def test_table_record_key_is_not_a_capture(self):
        src = (
            "local function foo()\n"
            "\tlocal bar = 1\n"
            "\tpcall(function() return { bar = 1 } end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local foo_anon_1 = function()", out)
        self.assertIn("pcall(foo_anon_1)", out)
        self.assertNotIn("foo_anon_1(bar)", out)

    def test_field_write_is_capture_read(self):
        src = (
            "local function set_power(info)\n"
            "\tpcall(function() info.dest_gvid = 1 end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local set_power_anon_1 = function(info)", out)
        self.assertIn("pcall(set_power_anon_1, info)", out)
        green = _findings(src, "anon_hoist")
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
        self.assertIn("local active_slot_anon_1 = function(db)", out)
        self.assertIn("return db.actor:active_slot()", out)
        self.assertIn("local _ok, _slot = pcall(active_slot_anon_1, db)", out)
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
        self.assertIn("local ok, _slot = pcall(wrap_anon_1, db)", out)
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
        red = _findings(src, "anon_skip")
        self.assertTrue(any("2+" in (f.details.get("skip_reason") or "") for f in red))

    def test_skip_if_anon_assign(self):
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
        red = _findings(src, "anon_skip")
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
        red = _findings(src, "anon_skip")
        self.assertTrue(any("xpcall" in (f.details.get("skip_reason") or "") for f in red))

    def test_own_dots_pcall_hoists(self):
        src = (
            "local function wrap(...)\n"
            "\tpcall(function(...)\n"
            "\t\treturn ...\n"
            "\tend, ...)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local wrap_anon_1 = function(...)", out)
        self.assertIn("pcall(wrap_anon_1, ...)", out)
        self.assertNotIn("pcall(function", out)

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
        red = _findings(src, "anon_skip")
        self.assertTrue(any("writes" in (f.details.get("skip_reason") or "") for f in red))

    def test_local_bind_is_not_a_write(self):
        # `local sm =` is a new name inside the pcall, not a write of outer sm.
        src = (
            "local function wrap()\n"
            "\tpcall(function()\n"
            "\t\tlocal sm = surge_manager.get_surge_manager()\n"
            "\t\tif sm then sm.started = false end\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local wrap_anon_1 = function(surge_manager)", out)
        self.assertIn("pcall(wrap_anon_1, surge_manager)", out)
        self.assertNotIn("pcall(function()", out)
        green = _findings(src, "anon_hoist")
        self.assertEqual(len(green), 1)
        self.assertIn("captures", green[0].details)
        self.assertNotIn("sm", green[0].details["captures"])

    def test_multiline_return_does_not_double_end(self):
        # Packer style: `return foo(\n...\n) end` must not get a second end.
        src = (
            "local function sort_by_dots(parts, sorter, table)\n"
            "\tlocal ok = pcall(function() return table.sort(\n"
            "\t\tparts,\n"
            "\t\tfunction(a, b) return sorter(a, b) end\n"
            "\t) end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("pcall(sort_by_dots_anon_1,", out)
        self.assertNotIn("pcall(function()", out)
        self.assertNotRegex(out, r"\) end\nend")
        from luaparser import ast as lua_ast
        lua_ast.parse(out)

    def test_fornum_without_step_hoists(self):
        # `for i = 1, #t` has no step. Parser leaves a raw 1. That is not unknown syntax.
        src = (
            "local function remove_spots(obj_id)\n"
            "\tpcall(function()\n"
            "\t\tlocal spots = (tft_path and tft_path.dots) or {}\n"
            "\t\tfor i = 1, #spots do\n"
            "\t\t\tlocal spot = spots[i]\n"
            "\t\t\tlevel.map_remove_object_spot(obj_id, spot)\n"
            "\t\tend\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("pcall(remove_spots_anon_1,", out)
        self.assertNotIn("pcall(function()", out)
        red = _findings(src, "anon_skip")
        self.assertEqual(red, [])

    def test_anon_above_outermost_named(self):
        src = (
            "local function outer()\n"
            "\tlocal function inner(x)\n"
            "\t\tpcall(function() return x + 1 end)\n"
            "\tend\n"
            "\tinner(1)\n"
            "end\n"
        )
        out = transform(src)
        self.assertTrue(out.startswith("local inner_anon_1 = function(x)"), out)
        self.assertIn("pcall(inner_anon_1, x)", out)
        # anon is not nested inside outer
        after_outer = out.split("local function outer()", 1)[1]
        self.assertNotIn("local inner_anon_1 = function", after_outer)

    def test_two_pcalls_numbered(self):
        src = (
            "local function foo(a, b)\n"
            "\tpcall(function() return a end)\n"
            "\tpcall(function() return b end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local foo_anon_1 = function(a)", out)
        self.assertIn("local foo_anon_2 = function(b)", out)
        self.assertIn("pcall(foo_anon_1, a)", out)
        self.assertIn("pcall(foo_anon_2, b)", out)
        self.assertLess(out.find("foo_anon_1"), out.find("foo_anon_2"))

    def test_nested_anon_hoists_both(self):
        src = (
            "local function foo()\n"
            "\tpcall(function()\n"
            "\t\tpcall(function() return 1 end)\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local foo_anon_1 = function()", out)
        self.assertIn("local foo_anon_2 = function(pcall)", out)
        self.assertIn("pcall(foo_anon_1)", out)
        self.assertIn("pcall(foo_anon_2, pcall)", out)
        self.assertNotIn("pcall(function()", out)
        self.assertLess(out.find("foo_anon_1"), out.find("foo_anon_2"))
        green = _findings(src, "anon_hoist")
        self.assertEqual(len(green), 2)

    def test_skip_does_not_eat_anon_number(self):
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
        self.assertIn("local foo_anon_1 = function(a)", out)
        self.assertIn("local foo_anon_2 = function(b)", out)
        self.assertNotIn("foo_anon_3", out)
        self.assertIn("pcall(function()", out)

    def test_flag_off_does_not_rewrite(self):
        src = (
            "local function foo()\n"
            "\tpcall(function() level.enable_input() end)\n"
            "end\n"
        )
        out = transform(src, hoist_anon_funcs=False)
        self.assertEqual(out, src)

    def test_xpcall_global_hoists(self):
        src = (
            "local function foo()\n"
            "\txpcall(function() level.enable_input() end, print)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local foo_anon_1 = function()", out)
        self.assertIn("xpcall(foo_anon_1, print)", out)
        self.assertIn("level.enable_input()", out)

    def test_xpcall_no_free_name_hoist(self):
        src = (
            "local function foo()\n"
            "\txpcall(function() return 1 end, print)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local foo_anon_1 = function()", out)
        self.assertIn("xpcall(foo_anon_1, print)", out)

    def test_assign_anon_hoists_above_assign(self):
        src = (
            "foo = function(x)\n"
            "\tpcall(function() return x + 1 end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertTrue(out.startswith("local foo_anon_1 = function(x)"), out)
        self.assertIn("pcall(foo_anon_1, x)", out)
        self.assertNotIn("local function", out.split("foo = function", 1)[1])

    def test_do_block_hoists_above_do(self):
        src = (
            "do\n"
            "\tlocal ok, t = pcall(function() return profile_timer() end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertTrue(out.startswith("local chunk_anon_1 = function(profile_timer)"), out)
        self.assertIn("pcall(chunk_anon_1, profile_timer)", out)
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
        self.assertTrue(out.startswith("local foo_anon_1 = function(x)"), out)
        after_outer = out.split("local function outer()", 1)[1]
        self.assertNotIn("local foo_anon_1 = function", after_outer)

    def test_local_func_inside_do_hoists_above_do(self):
        src = (
            "do\n"
            "\tlocal function foo()\n"
            "\t\tpcall(function() level.enable_input() end)\n"
            "\tend\n"
            "end\n"
        )
        out = transform(src)
        self.assertTrue(out.startswith("local foo_anon_1 = function(level)"), out)
        self.assertNotIn("local foo_anon_1 = function", out.split("do\n", 1)[1])

    def test_finding_message_matches_alao_style(self):
        src = (
            "local function apply(info)\n"
            "\tpcall(function() info.desc:AdjustHeightToText() end)\n"
            "end\n"
        )
        green = _findings(src, "anon_hoist")
        self.assertEqual(len(green), 1)
        f = green[0]
        self.assertIn("pcall(function() ... end) -> pcall(apply_anon_1, info)", f.message)
        self.assertEqual(f.details.get("full_match"), "pcall(function")
        self.assertEqual(f.details.get("suggestion"), "Hoist to apply_anon_1")
        from reporter import format_details, get_performance_impact
        shown = format_details(f.details)
        self.assertNotIn("anon_text", shown)
        self.assertNotIn("insert_char", shown)
        self.assertNotIn("insert_seq", shown)
        self.assertIn("anon_name", shown)
        self.assertEqual(get_performance_impact("anon_hoist"), "high")
        self.assertEqual(get_performance_impact("anon_skip"), "high")

    def test_table_sort_no_capture(self):
        src = (
            "local function sort_rows(data)\n"
            "\ttable.sort(data, function(a, b)\n"
            "\t\treturn (a and a[1] or 0) < (b and b[1] or 0)\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local sort_rows_anon_1 = function(a, b)", out)
        self.assertIn("table.sort(data, sort_rows_anon_1)", out)
        self.assertNotIn("table.sort(data, function", out)

    def test_table_sort_capture_skipped(self):
        src = (
            "local function sort_rows(data, cmp)\n"
            "\ttable.sort(data, function(a, b) return cmp(a, b) end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertEqual(out, src)
        red = _findings(src, "anon_skip")
        self.assertTrue(any("extra args" in (f.details.get("skip_reason") or "") for f in red))

    def test_or_function_no_capture(self):
        src = (
            "local function wrap()\n"
            "\tlocal noop = noop or function() return 1 end\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local wrap_anon_1 = function()", out)
        self.assertIn("local noop = noop or wrap_anon_1", out)

    def test_inner_local_cb_hoists(self):
        src = (
            "local function wrap()\n"
            "\tlocal cb = function() return 1 end\n"
            "\treturn cb\n"
            "end\n"
        )
        out = transform(src)
        self.assertTrue(out.startswith("local cb_anon_1 = function()"), out)
        self.assertIn("local cb = cb_anon_1", out)

    def test_inner_assign_cb_hoists(self):
        src = (
            "local function outer()\n"
            "\tfoo = function(x)\n"
            "\t\treturn x\n"
            "\tend\n"
            "end\n"
        )
        out = transform(src)
        self.assertTrue(out.startswith("local foo_anon_1 = function(x)"), out)
        self.assertIn("foo = foo_anon_1", out)
        self.assertNotIn("foo = function", out)

    def test_chunk_named_assign_untouched(self):
        src = "foo = function(x)\n\treturn x\nend\n"
        out = transform(src)
        self.assertEqual(out, src)
        self.assertEqual(_findings(src, "anon_hoist"), [])
        self.assertEqual(_findings(src, "anon_skip"), [])
        local_src = "local foo = function(x)\n\treturn x\nend\n"
        self.assertEqual(transform(local_src), local_src)
        self.assertEqual(_findings(local_src, "anon_hoist"), [])

    def test_or_function_own_dots_hoists(self):
        src = (
            "local try = try or function(func, ...)\n"
            "\tlocal status, error_or_result = pcall(func, ...)\n"
            "\tif not status then\n"
            "\t\treturn false, status, error_or_result\n"
            "\telse\n"
            "\t\treturn error_or_result, status\n"
            "\tend\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local chunk_anon_1 = function(func, ...)", out)
        self.assertIn("local try = try or chunk_anon_1", out)
        self.assertNotIn("or function", out)

    def test_outer_dots_pcall_hoists(self):
        src = (
            "function foo(...)\n"
            "\tpcall(function()\n"
            "\t\tbar(...)\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local foo_anon_1 = function(bar, ...)", out)
        self.assertIn("pcall(foo_anon_1, bar, ...)", out)
        self.assertNotIn("pcall(function", out)

    def test_outer_dots_sort_skipped(self):
        src = (
            "function foo(...)\n"
            "\ttable.sort(t, function(a, b)\n"
            "\t\treturn cmp(a, b, ...)\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertEqual(out, src)
        red = _findings(src, "anon_skip")
        self.assertTrue(any("extra args" in (f.details.get("skip_reason") or "") for f in red))

    def test_paren_block_comment_then_function(self):
        src = (
            "local t = {\n"
            "\tTutorials = ( --[[\n"
            "\t\thook notes\n"
            "\t]]\n"
            "\t\tfunction(obj)\n"
            "\t\t\treturn obj\n"
            "\t\tend\n"
            "\t),\n"
            "}\n"
        )
        out = transform(src)
        self.assertNotIn("\t\tfunction(obj)", out)
        self.assertIn("chunk_anon_1", out)
        self.assertIn("return obj", out)
        from luaparser import ast as lua_ast
        lua_ast.parse(out)

    def test_paren_comment_then_function(self):
        src = (
            "local t = {\n"
            "\t[\"waypoint\"] = (\n"
            "\t\t-- preset\n"
            "\t\tfunction(args)\n"
            "\t\t\treturn args\n"
            "\t\tend\n"
            "\t),\n"
            "}\n"
        )
        out = transform(src)
        self.assertNotIn("\t\tfunction(args)", out)
        self.assertIn("chunk_anon_1", out)
        self.assertIn("return args", out)
        from luaparser import ast as lua_ast
        lua_ast.parse(out)

    def test_paren_function_hoists(self):
        src = (
            "local function wrap()\n"
            "\tlocal InQuint = (function(x) return x * x * x * x * x end)\n"
            "\treturn InQuint\n"
            "end\n"
        )
        out = transform(src)
        self.assertTrue(out.startswith("local InQuint_anon_1 = function(x)"), out)
        self.assertNotIn("(function(x)", out)
        self.assertIn("return x * x * x * x * x", out)

    def test_or_paren_function_hoists(self):
        src = (
            "local DIK_name = ui_mcm and ui_mcm.display_key or (function() return \"\" end)\n"
        )
        out = transform(src)
        self.assertIn("or (chunk_anon_1)", out)
        self.assertNotIn("or (function", out)
        self.assertNotIn("end)", out)

    def test_generic_global_hoists(self):
        src = (
            "local function foo()\n"
            "\tRegisterScriptCallback(\"x\", function() level.enable_input() end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local foo_anon_1 = function()", out)
        self.assertIn("RegisterScriptCallback(\"x\", foo_anon_1)", out)

    def test_name_declared_below_already_global(self):
        # `later` is a local below `foo`. This function was compiled above it,
        # so `later` is already a global. Hoist does not change that.
        src = (
            "function foo()\n"
            "\ttable.sort(t, function(a, b) return later(a, b) end)\n"
            "end\n"
            "local function later(a, b)\n"
            "\treturn a < b\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local foo_anon_1 = function(a, b)", out)
        self.assertIn("table.sort(t, foo_anon_1)", out)

    def test_try_or_with_printf_below_hoists(self):
        src = (
            "local try = try or function(func, ...)\n"
            "\tprintf(func, ...)\n"
            "end\n"
            "local function printf(...)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local chunk_anon_1 = function(func, ...)", out)
        self.assertIn("local try = try or chunk_anon_1", out)

    def test_comment_after_one_liner_not_in_span(self):
        src = (
            "local t = {\n"
            "\tf = function(obj) return obj end\n"
            "\t-- Equippable bodywear\n"
            "}\n"
        )
        out = transform(src)
        self.assertIn("local chunk_anon_1 = function(obj) return obj end\n", out)
        self.assertNotRegex(out, r"return obj end\n\t-- Equippable bodywear\nend")
        self.assertIn("-- Equippable bodywear", out)
        from luaparser import ast as lua_ast
        lua_ast.parse(out)

    def test_sibling_iifes_both_spliced(self):
        src = (
            "allowed = (function() -- outer\n"
            "\tlocal allow = (function()\n"
            "\t\tlocal t = {}\n"
            "\t\treturn t\n"
            "\tend)()\n"
            "\tlocal ignore = (function()\n"
            "\t\tlocal t = {}\n"
            "\t\tt[\"af_lucifer\"] = true -- PBA\n"
            "\t\tt[\"af_black\"] = true\n"
            "\t\treturn t\n"
            "\tend)()\n"
            "\treturn allow\n"
            "end)()\n"
        )
        out = transform(src)
        self.assertNotIn("(function()", out)
        self.assertIn("(chunk_anon_1)()", out)
        self.assertIn("(chunk_anon_2)()", out)
        from luaparser import ast as lua_ast
        lua_ast.parse(out)

    def test_nested_iife_plus_table_insert(self):
        src = (
            "allowed = (function()\n"
            "\tlocal allow = (function()\n"
            "\t\treturn 1\n"
            "\tend)()\n"
            "\tlocal t = {}\n"
            "\ttable.insert(t, allow)\n"
            "\treturn t\n"
            "end)()\n"
        )
        out = transform(src)
        self.assertNotIn("(function()", out)
        self.assertIn("(chunk_anon_", out)
        self.assertIn("t[#t+1] = allow", out)
        from luaparser import ast as lua_ast
        lua_ast.parse(out)

    def test_nested_iife_splices_inner(self):
        src = (
            "allowed = (function()\n"
            "\tlocal allow = (function()\n"
            "\t\treturn 1\n"
            "\tend)()\n"
            "\treturn allow\n"
            "end)()\n"
        )
        out = transform(src)
        self.assertIn("local chunk_anon_1 = function()", out)
        self.assertIn("local chunk_anon_2 = function()", out)
        self.assertNotIn("(function()", out)
        self.assertIn("(chunk_anon_1)()", out)
        self.assertIn("allowed = (chunk_anon_2)()", out)
        from luaparser import ast as lua_ast
        lua_ast.parse(out)

    def test_fix_inside_hoisted_function(self):
        src = (
            "local function wrap()\n"
            "\ta.add = function(self, x)\n"
            "\t\ttable.insert(self.q, x)\n"
            "\tend\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("self.q[#self.q+1] = x", out)
        self.assertIn("a.add = a_add_anon_1", out)
        self.assertNotIn("a.add = function", out)
        self.assertNotIn("table.insert", out)

    def test_fix_cache_decl_inside_hoisted_function(self):
        # --fix rewrites math.min -> mmin and inserts `local mmin = math.min`
        # at the original function start. That insert sits inside the hoist
        # replace span and used to be dropped; hoisted body then called nil.
        src = (
            "local function wrap()\n"
            "\ta.add = function(self, x)\n"
            "\t\tlocal a = math.min(x, 1)\n"
            "\t\tlocal b = math.min(x, 2)\n"
            "\t\tlocal c = math.min(x, 3)\n"
            "\t\tlocal d = math.min(x, 4)\n"
            "\tend\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local a_add_anon_1 = function(self, x)", out)
        self.assertIn("local mmin = math.min", out)
        self.assertIn("mmin(x, 1)", out)
        self.assertNotIn("math.min(x, 1)", out)
        hoist, _, rest = out.partition("a.add = a_add_anon_1")
        self.assertIn("local mmin = math.min", hoist)
        self.assertNotIn("local mmin = math.min", rest)
        from luaparser import ast as lua_ast
        lua_ast.parse(out)

    def test_long_string_skips_hoist(self):
        # Reindent would change [[...]] data. Skip instead.
        src = (
            "function on_xml_read()\n"
            "\tRegisterScriptCallback(\"on_xml_read\", function(name, xml)\n"
            "\t\tlocal inc =\n"
            "[[\n"
            "#include \"ui\\map_spots_paw.xml\"\n"
            "]]\n"
            "\t\txml:insertFromXMLString(inc)\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertEqual(out, src)
        red = _findings(src, "anon_skip")
        self.assertTrue(any("long string" in (f.details.get("skip_reason") or "") for f in red))

    def test_long_comment_still_hoists(self):
        src = (
            "local function foo()\n"
            "\tpcall(function()\n"
            "\t\t--[[ keep me ]]\n"
            "\t\tlevel.enable_input()\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local foo_anon_1 = function(level)", out)
        self.assertNotIn("pcall(function()", out)

    def test_nested_function_on_local_is_a_capture(self):
        src = (
            "local function outer()\n"
            "\tlocal obj = {}\n"
            "\tpcall(function()\n"
            "\t\tfunction obj.foo() end\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("local outer_anon_1 = function(obj)", out)
        self.assertIn("pcall(outer_anon_1, obj)", out)
        self.assertIn("function obj.foo()", out)

    def test_nested_function_name_write_skips(self):
        src = (
            "local function outer()\n"
            "\tlocal foo\n"
            "\tpcall(function()\n"
            "\t\tfunction foo() end\n"
            "\tend)\n"
            "end\n"
        )
        out = transform(src)
        self.assertEqual(out, src)
        red = _findings(src, "anon_skip")
        self.assertTrue(any("writes" in (f.details.get("skip_reason") or "") for f in red))

    def test_assign_rewrite_skips_nested_function(self):
        src = (
            "local function f()\n"
            "\tlocal x\n"
            "\tpcall(function() x = function() return 1 end end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertIn("pcall(function()", out)
        self.assertNotIn("return function()", out)
        red = _findings(src, "anon_skip")
        self.assertTrue(
            any("nested function" in (f.details.get("skip_reason") or "") for f in red)
        )

    def test_reindent_strips_padded_hash_include(self):
        from anon_hoist import _reindent_anon
        text = (
            "local foo_anon_1 = function()\n"
            "    local inc =\n"
            "    [[\n"
            "    #include \"ui\\map_spots_paw.xml\"\n"
            "    ]]\n"
            "end"
        )
        out = _reindent_anon(text, "", "    x")
        self.assertIn("\n#include \"ui\\map_spots_paw.xml\"\n", out)

    def test_chunk_local_cap_skips(self):
        # Lua 5.1: 200 locals per chunk. Past the cap, do not emit globals.
        locs = "\n".join(f"local v{i} = {i}" for i in range(190))
        src = (
            locs + "\n"
            "local function wrap()\n"
            "\tpcall(function() return 1 end)\n"
            "end\n"
        )
        out = transform(src)
        self.assertEqual(out, src)
        red = _findings(src, "anon_skip")
        self.assertTrue(any("local limit" in (f.details.get("skip_reason") or "") for f in red))


if __name__ == "__main__":
    unittest.main()
