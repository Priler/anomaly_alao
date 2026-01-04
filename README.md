# ALAO - Anomaly Lua Auto Optimizer

AST-based Lua analyzer and optimizer for Anomaly mods.
Experimental.

## Quick Start

```bash
python stalker_lua_lint.py [path_to_mods] [options]

# Basic Usage (combinable)
--fix - Fix safe (GREEN) issues automatically
--fix-yellow - Fix unsafe (YELLOW) issues automatically
--fix-debug - Comment out debug statements (log, printf, print, etc.)
--experimental - Enable experimental fixes (string concat in loops)
--direct - Process scripts directly (searches for .script files recursively in the path or you can provide single .script path)
--exclude "alao_exclude.txt" -- Allows to exclude certain mods from reports/fixes (you can specify any other custom .txt list)

# Reports & Restore
--report [file] - Generate comprehensive report (.txt, .html, .json)
--revert - Restore all .bak backup files (undo fixes)

# Performance
--timeout [seconds] - Timeout per file (default: 10)
--workers / -j - Parallel workers for fixes (default: CPU count)

# Output
--verbose / -v - Show detailed output
--quiet / -q - Only show summary

# Backup Management
--backup / --no-backup - Create .bak files before modifying (default: True)
--list-backups - List all .bak backup files

# Danger Zone (do not use this)
--clean-backups - Remove all .bak backup files
```
## Requires

```
- Python 3.8+
- luaparser
- jinja2
```

## Currently Detected Patterns

### GREEN (safe to auto-fix)
- `table.insert(t, v)` → `t[#t+1] = v`
- `table.getn(t)` → `#t`
- `string.len(s)` → `#s`
- `math.pow(x, 2)` → `x*x`
- `math.pow(x, 0.5)` → `math.sqrt(x)`
- Uncached globals used 3+ times
- Repeated `db.actor`, `alife()`, `device()`, `get_console()` calls

### YELLOW (review needed)
- String concatenation in loops (`s = s .. x`) - fixable with `--experimental`
### RED (info only, no auto-fix)
- Global variable writes

### DEBUG (can be auto commented out)
- `print()`, `printf()`, `log()` calls

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
- Pattern is a simple `var = var .. expr`

## Safety measures

To prevent loosing original scripts, make sure to backup them.
However, as an additional protection level this tool creates `.bak` files before any changes.

```bash
# List all backups
python stalker_lua_lint.py /path/to/mods --list-backups

# Restore from backups
python stalker_lua_lint.py /path/to/mods --revert

# Delete backups (DO NOT USE THIS, no point removing og scripts, better keep them)
python stalker_lua_lint.py /path/to/mods --clean-backups

# Disable backups (NOT RECOMMENDED)
python stalker_lua_lint.py /path/to/mods --fix --no-backup
```

## Author

Abraham (Priler)
