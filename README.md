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
And switch them to a better alternatives _(direct opcodes, caching, 
reduced allocations, etc)_.  

One of the examples: https://onecompiler.com/lua/449f75hkd  
The original function has a complexity of **O(n²)**.  
The auto-fixed _(by ALAO)_ function has a complexity of **O(n)**.  
For huge data _(say, 100k iterations)_, it works approximately 150x faster.  
It also prevents unnecessary memory allocations, further reducing GC pressure.

Another example ALAO handles is the usage of `math.pow(v, 2)`.  
We can replace the function call with a single MUL bytecode instruction `v*v`.   
The same pattern applies to  `math.pow(v, 3)` and `math.pow(v, 0.5)`.


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

# Experimental features
--fix-nil - Allows to auto-fix some nil checks (that could cause CTDs in some cases)
--remove-dead-code / --debloat - Allows to remove dead code from the scripts (faster load times)

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

## Nil checks performance impact
Honestly, there are little to none performance impact even for thousands of `nil` guard checks.  
In terms of bytecode, it compiles to _(assuming it's a local variable)_:  
`TEST` - checks if the value is truthy  
`JMP` - conditional jump

It'll take like ~2-5 CPU nanoseconds per check and zero memory usage.  
So feel free to apply that, as it prevents most of the CTDs caused by evil `nil`.

## Safety measures

In order to prevent loosing original scripts, make sure to backup them before applying the fixes.  
However, as an additional protection level this tool automatically creates `.bak` files before any changes _(next to modified script files)_.  

You can also use `--backup-all-scripts` flag to make the backup of all your .script files inside your mods _(keeping the folders structure, of course)_.  
In this case there's no need to manually backup the mods folder.  
Because ALAO only touches .script files and all of them will have a full backup now with this option.

```bash
# Make a full backup of all .script files in a given path
# it will create a .zip archive containing all your current scripts
# archive will be named according to current date (ex. scripts-backup-2026-01-05_09-03-23.zip)
python stalker_lua_lint.py /path/to/mods --backup-all-scripts

# Restore from backups
python stalker_lua_lint.py /path/to/mods --revert

# List all backups
python stalker_lua_lint.py /path/to/mods --list-backups

# DANGER ZONE
# Delete backups (DO NOT USE THIS, no point removing og scripts, better keep them)
python stalker_lua_lint.py /path/to/mods --clean-backups

# Disable backups (NOT RECOMMENDED)
python stalker_lua_lint.py /path/to/mods --fix --no-backup
```

## Author

Abraham (Priler)
