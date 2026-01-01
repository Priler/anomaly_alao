"""
Extracts all .script files from a mods folder to an output directory,
preserving the mod folder structure.
Useful for creating script archives for analysis etc.

Usage:
    py script_extractor.py <path-to-mods>
    py script_extractor.py <path-to-mods> <output-path>
"""

import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def find_script_files(mods_path: Path) -> list:
    """Find all .script files in the mods directory."""
    scripts = []

    for mod_dir in mods_path.iterdir():
        if not mod_dir.is_dir():
            continue

        # look for gamedata/scripts
        scripts_dir = mod_dir / "gamedata" / "scripts"
        if not scripts_dir.exists():
            continue

        # find all .script files
        for script_file in scripts_dir.rglob("*.script"):
            rel_path = script_file.relative_to(mods_path)
            scripts.append((mod_dir.name, script_file, rel_path))

    return scripts


def extract_scripts(mods_path: Path, output_path: Path, verbose: bool = False) -> dict:
    """
    Extract all .script files to output directory.

    Returns dict with stats.
    """
    stats = {
        'mods': set(),
        'files': 0,
        'total_size': 0,
        'errors': []
    }

    # find all scripts
    scripts = find_script_files(mods_path)

    if not scripts:
        print("No .script files found!")
        return stats

    # create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # copy files
    for mod_name, src_path, rel_path in scripts:
        dst_path = output_path / rel_path

        try:
            # create parent directories
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # copy file
            shutil.copy2(src_path, dst_path)

            stats['mods'].add(mod_name)
            stats['files'] += 1
            stats['total_size'] += src_path.stat().st_size

            if verbose:
                print(f"  [OK] {rel_path}")

        except Exception as e:
            stats['errors'].append((rel_path, str(e)))
            if verbose:
                print(f"  [ERROR] {rel_path}: {e}")

    return stats


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Extract .script files from Anomaly lua mods"
    )
    parser.add_argument(
        "mods_path",
        help="Path to mods directory"
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Output directory (default: ./extracted_scripts)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show each file being copied"
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a zip archive of extracted scripts"
    )

    args = parser.parse_args()

    mods_path = Path(args.mods_path)
    if not mods_path.exists():
        print(f"Error: Path not found: {mods_path}")
        sys.exit(1)

    # default output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path("./extracted_scripts")

    print(f"Script Extractor")
    print(f"=" * 40)
    print(f"Source: {mods_path}")
    print(f"Output: {output_path}")
    print()

    # extract
    print("Extracting .script files...")
    stats = extract_scripts(mods_path, output_path, verbose=args.verbose)

    # summary
    print()
    print(f"=" * 40)
    print(f"Extraction complete!")
    print(f"  Mods processed: {len(stats['mods'])}")
    print(f"  Files copied:   {stats['files']}")
    print(f"  Total size:     {format_size(stats['total_size'])}")

    if stats['errors']:
        print(f"  Errors:         {len(stats['errors'])}")

    # create zip if requested
    if args.zip and stats['files'] > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"scripts_{timestamp}"
        zip_path = output_path.parent / zip_name

        print()
        print(f"Creating archive: {zip_name}.zip ...")
        shutil.make_archive(str(zip_path), 'zip', output_path)

        zip_file = Path(f"{zip_path}.zip")
        print(f"Archive created: {zip_file} ({format_size(zip_file.stat().st_size)})")

    print()
    print(f"Output directory: {output_path.absolute()}")


if __name__ == "__main__":
    main()
