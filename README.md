# ALAO - Anomaly Lua Auto Optimizer

AST-based Lua analyzer and optimizer for Anomaly mods _(a Lua swiss-knife, in some way)_.  
Made for **LuaJIT 2.0.4** _(Lua 5.1)_ in the first place (comes with the latest [Modded Exes](https://github.com/themrdemonized/xray-monolith)).  
Highly experimental _(mostly proof-of-concept)_, but battle tested with huge modpacks (600+ mods).

## How it works?

Lua is a programming language.  
And as all programming languages, it has a syntax based code.  
Thus, it can be parsed into so-called AST _(abstract syntax tree)_.

With AST we can manipulate the code however we want without the high risk of breaking things.  
Lua VM itself parses code into AST, then compiles it to bytecode, then executes the bytecode _(obviously)_.  
ALAO also converts the code to AST.

After that, we search for potential poorly optimized code entities.  
And switch them to a better alternatives _(direct opcodes, caching, reduced allocations, etc)_.  

One of the examples: https://onecompiler.com/lua/449f75hkd  
The original function has a complexity of **O(n²)**.  
The auto-fixed _(by ALAO)_ function has a complexity of **O(n)**.  
For huge data _(say, 100k iterations)_, it works approximately 150x faster.  
It also prevents unnecessary memory allocations, further reducing GC pressure.

Another example ALAO handles is the usage of `math.pow(v, 2)`.  
We can replace the function call with a single MUL bytecode instruction `v*v`.   
The same pattern applies to `math.pow(v, 3)` and `math.pow(v, 0.5)`.


## Quick Start

```bash
python stalker_lua_lint.py [path_to_mods] [options]

# Basic Usage (combinable)
--fix              Fix safe (GREEN) issues automatically
--fix-yellow       Fix unsafe (YELLOW) issues automatically
--fix-debug        Comment out debug statements (log, printf, print, etc.)
--fix-nil          Auto-fix safe nil access patterns (wrap with if-then guard)
--remove-dead-code Remove 100% safe dead code (unreachable code, if false blocks)
--fix-pcall        Hoist pcall/xpcall(function() ... end) to a named local
--cache-threshold  Minimum function call count to trigger caching (default: 4)

--direct           Process scripts directly (no gamedata/scripts structure required)
--exclude "file"   Exclude certain mods from reports/fixes (one mod name per line)

# Experimental
--experimental     Enable experimental fixes (string concat in loops)

# Reports & Restore
--report [file]    Generate comprehensive report (.txt, .html, .json)
--revert           Restore all .alao-bak backup files (undo fixes)

# Performance
--timeout [sec]    Timeout per file (default: 10)
--workers / -j     Parallel workers for fixes (default: CPU count)
--single-thread    Disable multiprocessing (for debugging)

# Output
--verbose / -v     Show detailed output
--quiet / -q       Only show summary

# Backup Management
--backup-all-scripts [path]  Backup ALL scripts to zip before modifications
--backup / --no-backup       Create .alao-bak files (default: True)
--list-backups               List all .alao-bak backup files

# Danger Zone (do not use this)
--clean-backups    Remove all .alao-bak backup files (not recommended)
```

## Benchmarks

The experience differs based on your PC specs and a specific modpack.  
In other words, if it works for me... it doesn't mean it will work for you.  
However, here's a comparison I made recently:

![Benchmark Comparison](benchmark.jpg)

_Notice a decreased frame time and AVG FPS increase. Keep in mind this was tested on warfare mode._

## Detected Patterns

### GREEN (safe to auto-fix with `--fix`)

| Pattern | Replacement | Impact |
|---------|-------------|--------|
| `table.insert(t, v)` | `t[#t+1] = v` | High - avoids function call overhead |
| `table.getn(t)` | `#t` | Low - deprecated function |
| `string.len(s)` | `#s` | Low - unnecessary function call |
| `math.pow(x, 2)` | `x*x` | High - single MUL opcode |
| `math.pow(x, 3)` | `x*x*x` | High - MUL opcodes |
| `math.pow(x, 0.5)` | `x^0.5` | Medium - native operator |
| `pos:distance_to(t) < N` | `pos:distance_to_sqr(t) < N*N` | High - avoids sqrt |
| Uncached globals (4+ calls) | `local mfloor = math.floor` | Medium - reduces lookups |
| Repeated `db.actor` | `local actor = db.actor` | High - cached reference |
| Repeated `alife()` | `local sim = alife()` | High - cached singleton |
| Repeated `device()` | `local dev = device()` | Medium - cached singleton |
| Repeated `system_ini()` | `local ini = system_ini()` | Medium - cached singleton |
| Repeated `get_console()` | `local console = get_console()` | Medium - cached singleton |
| Repeated `get_hud()` | `local hud = get_hud()` | Medium - cached singleton |
| Repeated `:section()` | `local sec = obj:section()` | Medium - immutable property |
| Repeated `:id()` | `local id = obj:id()` | Medium - immutable property |
| `pcall(function() ... end)` | named `local function` + `pcall(name, captures...)` | High - no per-call closure alloc. Opt-in via `--fix-pcall` (not part of `--fix`) |


### YELLOW (may cause CTDs, fix with `--fix-yellow`)

Pay attention some of this fixes requires `--experimental` flag.

| Pattern | Description | Impact |
|---------|-------------|--------|
| `s = s .. x` in loop | String concatenation builds O(n²) garbage | Critical |


### RED (info only (for modders), no auto-fix)

| Pattern | Description |
|---------|-------------|
| Global variable writes | Writing to global scope (potential pollution) |
| Per-frame callback warnings | Performance issues in `actor_on_update`, etc. |
| `vector()` in hot loop | Allocates new vector each iteration | Critical |
| Constant conditions | `if true then` / `if false then` |
| Unnecessary else | `if x then return end else ...` |
| `pcall` anon skipped | Writes that are not a single `x = expr`, `xpcall` with captures, `...`, expression-context assigns |


### DEBUG (comment out with `--fix-debug`)

| Functions |
|-----------|
| `print`, `printf`, `printe`, `printd` |
| `log`, `log1`, `log2`, `log3` |
| `DebugLog`, `debug_log`, `trace`, `dump` |


### NIL Access Detection (fix some with `--fix-nil`)

Detects potential CTD from nil access on these functions/methods:

**Level functions:**
- `level.object_by_id()`, `level.get_target_obj()`, `level.vertex_position()`

**Alife functions:**
- `alife()`, `alife():object()`, `alife():story_object()`, `alife():actor()`

**Game object methods:**
- `:parent()`, `:best_enemy()`, `:active_item()`, `:object()`
- `:item_in_slot()`, `:get_current_outfit()`, `:spawn_ini()`
- `:bone_id()`, `:bone_position()`, and 20+ more

**Common patterns:**
- `db.actor`, `db.storage[id]`, `get_story_object()`, `get_object_by_name()`


### Dead Code Detection (remove some with `--remove-dead-code`)

| Pattern | Description |
|---------|-------------|
| Code after `return` | Unreachable statements after return |
| Code after `break` | Unreachable statements after break |
| `if false then ... end` | Never-executed blocks |
| `while false do ... end` | Never-executed loops |
| Unused local variables | Declared but never read |
| Unused local functions | Declared but never called |


### Cacheable Globals

ALAO uses branch-aware counting for function calls in if/elseif/else chains.  
This prevents over-optimization of mutually exclusive code paths.  
When used 4+ times in a function (3+ in hot callbacks), these get cached:

**bare globals:**
`pairs`, `ipairs`, `next`, `type`, `tostring`, `tonumber`, `unpack`, `select`, `rawget`, `rawset`  
**math module:**
`floor`, `ceil`, `abs`, `min`, `max`, `sqrt`, `sin`, `cos`, `tan`, `random`, `pow`, `log`, `exp`, `atan2`, `atan`, `asin`, `acos`, `deg`, `rad`, `fmod`, `modf`, `huge`  
**string module:**
`find`, `sub`, `gsub`, `match`, `gmatch`, `format`, `lower`, `upper`, `len`, `rep`, `byte`, `char`, `reverse`  
**table module:**
`insert`, `remove`, `concat`, `sort`, `getn`, `unpack`  
**bit module:**
`band`, `bor`, `bxor`, `bnot`, `lshift`, `rshift`, `arshift`, `rol`, `ror`

### Hot Callbacks (lower caching threshold)

These callbacks use `cache_threshold - 1` for more aggressive optimization:

```
actor_on_update, actor_on_first_update
npc_on_update, monster_on_update
on_key_press, on_key_release, on_key_hold
actor_on_weapon_fired, actor_on_hud_animation_end
on_before_hit, on_hit, npc_on_hit_callback, monster_on_hit_callback
actor_on_item_take, actor_on_item_drop, actor_on_item_use
```


## Experimental: String Concat Fix

The `--experimental` flag enables automatic transformation of string concatenation in loops:

**Before:**
```lua
local result = ""
for i = 1, 10 do
    result = result .. get_line(i)
end
```

**After:**
```lua
local _result_parts = {}
for i = 1, 10 do
    _result_parts[#_result_parts+1] = get_line(i)
end
local result = table.concat(_result_parts)
```

This optimization reduces GC pressure from O(n²) to O(n) for string building.

**Safety:** Only applied when:
- Variable is initialized to `""` before the loop
- Pattern is simple `var = var .. expr`

## pcall hoist (`--fix-pcall`)

`pcall(function() ... end)` allocates a new closure every time that line runs.  
`--fix-pcall` hoists the anonymous function to a chunk-level `local function {caller}_pcall_{n}` and passes read-only captured locals as arguments.

**Before:**
```lua
local function apply(info)
    pcall(function()
        info.desc:AdjustHeightToText()
    end)
end
```

**After:**
```lua
local function apply_pcall_1(info)
    info.desc:AdjustHeightToText()
end
local function apply(info)
    pcall(apply_pcall_1, info)
end
```

A single-statement `pcall(function() x = expr end)` becomes `return expr` plus `if ok then x = v end`, so a failed pcall does not store the error string into `x`.

Skipped (RED, no edit): writes that are not that single-assign shape, `xpcall` with leftover captures (Lua 5.1 cannot pass extra args), `...`, and assign-rewrites used as `if pcall(...)` expressions.

Not part of default `--fix`.

## Nil checks performance impact

Honestly, there are little to none performance impact even for thousands of `nil` guard checks.  
In terms of bytecode, it compiles to _(assuming it's a local variable)_:  
- `TEST` - checks if the value is truthy  
- `JMP` - conditional jump

It'll take like ~2-5 CPU nanoseconds per check and zero memory usage.  
So feel free to apply that, as it prevents most of the CTDs caused by evil `nil`.


## Safety Measures

In order to prevent loosing original scripts, make sure to backup them before applying the fixes.  
However, as an additional protection level this tool automatically creates `.alao-bak` files before any changes _(next to modified script files)_.  

You can also use `--backup-all-scripts` flag to make the backup of all your .script files inside your mods _(keeping the folders structure, of course)_.  
In this case there's no need to manually backup the mods folder.  
Because ALAO only touches .script files and all of them will have a full backup now with this option.

```bash
# Make a full backup of all .script files
python stalker_lua_lint.py /path/to/mods --backup-all-scripts

# Restore from backups
python stalker_lua_lint.py /path/to/mods --revert

# List all backups
python stalker_lua_lint.py /path/to/mods --list-backups

# DANGER ZONE - Delete backups (not recommended)
python stalker_lua_lint.py /path/to/mods --clean-backups

# Disable per-file backups (not recommended)
python stalker_lua_lint.py /path/to/mods --fix --no-backup
```


## Requirements

```
Python 3.8+
luaparser
jinja2
```


## User Guide

☢️ I've made a Google Docs guide on how to use ALAO:  
https://docs.google.com/document/d/1isS0Gn9MWrJZ6eSYjh2cFSC8NfPcdWqobB9RdHkAxXI/edit?usp=sharing

Make sure to thoroughly read it all, so you understand how it works.  
It's kinda verbose, but I guess it's a good thing.


## Author

Abraham (Priler)
