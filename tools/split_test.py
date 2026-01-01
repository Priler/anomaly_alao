"""
Split extracted mods directory into bunch zip archives for testing.
Each archive will contain up to N mods (default 50).
"""

import argparse
import zipfile
import os
import sys
from pathlib import Path
from math import ceil


def get_mods(mods_path: Path) -> list:
    """Get list of mod directories."""
    mods = []

    for item in sorted(mods_path.iterdir()):
        if item.is_dir():
            # check if it has gamedata/scripts
            scripts_dir = item / "gamedata" / "scripts"
            if scripts_dir.exists():
                script_count = len(list(scripts_dir.glob("*.script")))
                if script_count > 0:
                    mods.append((item, script_count))

    return mods


def create_chunk_zip(mods: list, chunk_num: int, output_dir: Path) -> Path:
    """Create a zip archive for a chunk of mods."""
    zip_path = output_dir / f"mods_chunk_{chunk_num:02d}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for mod_path, _ in mods:
            scripts_dir = mod_path / "gamedata" / "scripts"

            for script_file in scripts_dir.glob("*.script"):
                # archive path: ModName/gamedata/scripts/file.script
                arc_path = f"{mod_path.name}/gamedata/scripts/{script_file.name}"
                zf.write(script_file, arc_path)

            # also include .bak files if present
            for bak_file in scripts_dir.glob("*.bak"):
                arc_path = f"{mod_path.name}/gamedata/scripts/{bak_file.name}"
                zf.write(bak_file, arc_path)

    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Split mods directory into zip archives for testing"
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to mods directory"
    )
    parser.add_argument(
        "-c", "--chunks",
        type=int,
        default=50,
        help="Number of mods per chunk (default: 50)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for zip files (default: current directory)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Just list mods without creating archives"
    )

    args = parser.parse_args()

    # get path
    if args.path:
        mods_path = Path(args.path)
    else:
        user_input = input("Enter path to mods directory: ").strip()
        if not user_input:
            print("No path provided.")
            sys.exit(1)
        mods_path = Path(user_input)

    if not mods_path.exists():
        print(f"Path does not exist: {mods_path}")
        sys.exit(1)

    # get mods
    print(f"Scanning: {mods_path}")
    mods = get_mods(mods_path)

    total_scripts = sum(count for _, count in mods)
    print(f"Found {len(mods)} mods with {total_scripts} script files")

    if not mods:
        print("No mods found.")
        sys.exit(1)

    if args.list:
        print("\nMods:")
        for mod_path, count in mods:
            print(f"  {mod_path.name}: {count} scripts")
        sys.exit(0)

    # calculate chunks
    chunk_size = args.chunks
    num_chunks = ceil(len(mods) / chunk_size)

    print(f"\nSplitting into {num_chunks} chunks of ~{chunk_size} mods each")

    # output directory
    output_dir = Path(args.output) if args.output else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # create chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(mods))
        chunk_mods = mods[start_idx:end_idx]

        chunk_scripts = sum(count for _, count in chunk_mods)
        print(
            f"  Chunk {
                i + 1}/{num_chunks}: {
                len(chunk_mods)} mods, {chunk_scripts} scripts...",
            end=" ",
            flush=True)

        zip_path = create_chunk_zip(chunk_mods, i + 1, output_dir)
        zip_size = zip_path.stat().st_size / (1024 * 1024)

        print(f"-> {zip_path.name} ({zip_size:.1f} MB)")

    print(f"\nDone! Created {num_chunks} archives in {output_dir}")


if __name__ == "__main__":
    main()
