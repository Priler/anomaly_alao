"""
This finds mod folders and their Lua scripts.

Handles the standard MO2 structure:
  <path>/gamedata/scripts/*.lua
  <path>/gamedata/scripts/*.script
"""

from pathlib import Path
from typing import Dict, List


def discover_mods(root_path: Path) -> Dict[str, List[Path]]:
    """
    Discover all mods and their script files.

    Returns dict mapping mod name -> list of script file paths
    """
    mods = {}
    root_path = Path(root_path)

    # check if this IS a gamedata folder directly
    if root_path.name == "gamedata":
        scripts_dir = root_path / "scripts"
        if scripts_dir.exists():
            scripts = find_scripts(scripts_dir)
            if scripts:
                mods["(root)"] = scripts
        return mods

    # check if this folder contains gamedata directly (single mod)
    direct_gamedata = root_path / "gamedata" / "scripts"
    if direct_gamedata.exists():
        scripts = find_scripts(direct_gamedata)
        if scripts:
            mods[root_path.name] = scripts

    # scan subdirectories for mod folders
    for item in root_path.iterdir():
        if not item.is_dir():
            continue

        # skip common non-mod folders
        if item.name.startswith('.') or item.name in ('__pycache__', 'backup', 'backups'):
            continue

        scripts_dir = item / "gamedata" / "scripts"
        if scripts_dir.exists():
            scripts = find_scripts(scripts_dir)
            if scripts:
                mods[item.name] = scripts
        else:
            # maybe nested structure, check one level deeper
            for subitem in item.iterdir():
                if subitem.is_dir():
                    nested_scripts = subitem / "gamedata" / "scripts"
                    if nested_scripts.exists():
                        scripts = find_scripts(nested_scripts)
                        if scripts:
                            mod_name = f"{item.name}/{subitem.name}"
                            mods[mod_name] = scripts

    return mods


def find_scripts(scripts_dir: Path) -> List[Path]:
    """Find all Lua script files in a directory."""
    scripts = []

    # .lua and .script are both used
    for ext in ("*.lua", "*.script"):
        scripts.extend(scripts_dir.glob(ext))

    # also check subdirectories (some mods organize scripts in folders)
    for ext in ("**/*.lua", "**/*.script"):
        for f in scripts_dir.glob(ext):
            if f not in scripts:
                scripts.append(f)

    return sorted(scripts)


def get_mod_info(mod_path: Path) -> dict:
    """
    Try to extract mod info from common metadata files.
    Returns dict with name, version, author if found.
    """
    info = {"name": mod_path.name}

    # check for meta.ini (MO2 style)
    meta_ini = mod_path / "meta.ini"
    if meta_ini.exists():
        try:
            content = meta_ini.read_text(encoding='utf-8', errors='ignore')
            for line in content.splitlines():
                if line.startswith("name="):
                    info["name"] = line.split("=", 1)[1].strip()
                elif line.startswith("version="):
                    info["version"] = line.split("=", 1)[1].strip()
                elif line.startswith("author="):
                    info["author"] = line.split("=", 1)[1].strip()
        except (OSError, IOError):
            pass

    # check for modinfo.txt
    modinfo = mod_path / "modinfo.txt"
    if modinfo.exists():
        try:
            content = modinfo.read_text(encoding='utf-8', errors='ignore')
            lines = content.splitlines()
            if lines:
                info["name"] = lines[0].strip()
        except (OSError, IOError):
            pass

    return info
