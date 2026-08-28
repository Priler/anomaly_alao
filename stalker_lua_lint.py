"""
ALAO main orchestrator script (entry point).
Written by: Abraham (Priler)

AST bases Lua parser & analyzer for S.T.A.L.K.E.R. Anomaly mods.
Should help to automatically prevent common scripts optimization issues.

Usage:
    python stalker_lua_lint.py [path_to_mods] [options]

Options:
    # BASIC USAGE (options can be combined)
    --fix              Fix safe (GREEN) issues automatically
    --fix-yellow       Fix unsafe (YELLOW) issues automatically
    --fix-debug        Fix (DEBUG) entries automatically (comment out all: log, printf, print, etc.)
    --fix-nil          Fix safe nil access patterns (wrap with if-then guard)
    --remove-dead-code / --debloat
                       Remove 100% safe dead code (unreachable code, if false blocks)
    --fix-pcall        Hoist pcall/xpcall(function() ... end) to a named local
    --cache-threshold N
                       Minimum function call count to trigger caching (default: 4)
                       Hot callbacks use N-1. Lower = more aggressive caching.

    # IMPORTANT
    --backup-all-scripts [path]
                       Backup ALL scripts to a zip archive before modifications
                       (default: scripts-backup-<date>.zip)
    --revert          Restore all .alao-bak backup files (undo ALAO fixes)
    --report [file]   Generate a comprehensive report (supports .txt, .html, .json)

    # SAFETY (auto-backup on first fix run)
    When any --fix* flag is used and no scripts-backup-*.zip exists in the mods folder,
    ALAO will automatically create a backup BEFORE making any changes.
    This ensures you always have a way to restore your original scripts.
    
    --no-first-time-auto-backup
                       Skip automatic backup creation (not recommended)

    # MULTITHREAD processing
    --timeout [seconds]
                       Timeout per file in seconds (default: 10)
    --workers / -j    Number of parallel workers for fixes (default: CPU count)
    --single-thread   Disable multiprocessing (for debugging)
    
    # USELESS things
    --backup / --no-backup
                       Create .alao-bak files before modifying (default: True)
    --verbose / -v    Show detailed output
    --quiet / -q      Only show summary

    # DANGER ZONE
    --list-backups    List all .alao-bak backup files without restoring
    --clean-backups   Remove all .alao-bak backup files
    
    # DEBUG
    --extract-debug [path]
                       Extract all modified files (.script + .alao-bak) to a zip
                       for debugging (default: alao_debug_extract.zip)
    --split N          Split --extract-debug into multiple zips with N mods each
"""

import sys
import os
import argparse
import shutil
import zipfile
import glob
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed, BrokenExecutor

from discovery import discover_mods, discover_direct
from ast_analyzer import analyze_file
from ast_transformer import transform_file
from reporter import Reporter
from models import Finding


def analyze_file_with_timeout(file_path: Path, timeout: float, cache_threshold: int = 4, experimental: bool = False):
    """Analyze a file with a timeout to prevent hanging on problematic files."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(analyze_file, file_path, cache_threshold=cache_threshold, experimental=experimental)
    try:
        result = future.result(timeout=timeout)
        executor.shutdown(wait=False)
        return result
    except FuturesTimeoutError:
        future.cancel()
        executor.shutdown(wait=False)
        raise TimeoutError(f"Analysis timed out for {file_path.name}")


def analyze_file_worker(args_tuple):
    """Worker function for parallel analyze_file calls."""
    mod_name, script_path, timeout, cache_threshold, experimental = args_tuple
    try:
        if timeout and timeout > 0:
            findings = analyze_file_with_timeout(script_path, timeout, cache_threshold, experimental)
        else:
            findings = analyze_file(script_path, cache_threshold=cache_threshold, experimental=experimental)
        return (mod_name, script_path, findings, None)
    except TimeoutError as e:
        return (mod_name, script_path, [], f"TimeoutError: {e}")
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return (mod_name, script_path, [], err_msg)


def transform_file_worker(args_tuple):
    """Worker function for parallel transform_file calls."""
    script_path, backup, fix_debug, fix_yellow, experimental, fix_nil, remove_dead_code, cache_threshold, fix_pcall = args_tuple
    try:
        modified, _, edit_count = transform_file(
            script_path,
            backup=backup,
            fix_debug=fix_debug,
            fix_yellow=fix_yellow,
            experimental=experimental,
            fix_nil=fix_nil,
            remove_dead_code=remove_dead_code,
            cache_threshold=cache_threshold,
            fix_pcall=fix_pcall,
        )
        return (script_path, modified, edit_count, None)
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        return (script_path, False, 0, err_msg)


def backup_all_scripts(all_files, output_path=None, mods_root=None, quiet=False):
    """
    Backup all script files to a zip archive.
    
    Args:
        all_files: List of (mod_name, script_path) tuples
        output_path: Path to output zip file (auto-generated if None or 'auto')
        mods_root: Root mods directory (for relative paths in archive and default output location)
        quiet: Suppress output
    
    Returns:
        Path to created zip file, or None on failure
    """
    if not all_files:
        if not quiet:
            print("No scripts to backup.")
        return None
    
    # generate default filename if needed
    if output_path is None or output_path == 'auto':
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'scripts-backup-{timestamp}.zip'
        # save in mods_root if provided, otherwise current directory
        if mods_root:
            output_path = Path(mods_root) / filename
        else:
            output_path = Path(filename)
    else:
        output_path = Path(output_path)
    
    # ensure .zip extension
    if output_path.suffix.lower() != '.zip':
        output_path = output_path.with_suffix('.zip')
    
    if not quiet:
        print(f"Creating backup: {output_path}")
        print(f"Backing up {len(all_files)} script files...")
    
    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, (mod_name, script_path) in enumerate(all_files):
                # create archive path preserving mod structure
                if mods_root and script_path.is_relative_to(mods_root):
                    archive_path = script_path.relative_to(mods_root)
                else:
                    # fallback: use mod_name/filename
                    archive_path = Path(mod_name) / script_path.name
                
                zf.write(script_path, archive_path)
                
                if not quiet and (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(all_files)} files...")
        
        # get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        
        if not quiet:
            print(f"Backup complete: {output_path} ({size_mb:.2f} MB)")
        
        return output_path
    
    except Exception as e:
        if not quiet:
            print(f"Backup failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Anomaly Lua Auto Optimizer (ALAO)"
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to mods directory"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix safe (GREEN) issues automatically"
    )
    parser.add_argument(
        "--fix-yellow",
        action="store_true",
        help="Fix unsafe (YELLOW) issues automatically"
    )
    parser.add_argument(
        "--fix-debug",
        action="store_true",
        help="Fix (DEBUG) entries automatically (comment out all: log, printf, print, etc.)"
    )
    parser.add_argument(
        "--fix-nil",
        action="store_true",
        help="Fix safe nil access issues (wrap with if-then guard)"
    )
    parser.add_argument(
        "--remove-dead-code", "--debloat",
        action="store_true",
        dest="remove_dead_code",
        help="Remove 100%% safe dead code (unreachable code after return, if false blocks, etc.)"
    )
    parser.add_argument(
        "--fix-pcall",
        action="store_true",
        help="Hoist pcall/xpcall(function() ... end) to a named chunk-level local"
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Enable experimental features: string concat in loops auto-fix"
    )
    parser.add_argument(
        "--cache-threshold",
        type=int,
        default=4,
        metavar="N",
        help="Minimum function call count to trigger caching (default: 4, hot callbacks use N-1, min: 2)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create .alao-bak files before modifying (default: True)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Don't create backup files"
    )
    parser.add_argument(
        "--no-first-time-auto-backup",
        action="store_true",
        help="Skip automatic zip backup on first fix run (not recommended)"
    )
    parser.add_argument(
        "--single-thread",
        action="store_true",
        help="Disable multiprocessing (for debugging)"
    )
    parser.add_argument(
        "--backup-all-scripts",
        type=str,
        nargs='?',
        const='auto',
        default=None,
        metavar="PATH",
        help="Backup all scripts to a zip archive before any modifications (default name: scripts-backup-<date>.zip)"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Generate a comprehensive report (supports .txt, .html, .json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only show summary"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout per file in seconds (default: 10)"
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=None,
        help="Number of parallel workers for fixes (default: CPU count, max 8)"
    )
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Restore all .alao-bak backup files (undo ALAO fixes)"
    )
    parser.add_argument(
        "--list-backups",
        action="store_true",
        help="List all .alao-bak backup files without restoring"
    )
    parser.add_argument(
        "--clean-backups",
        action="store_true",
        help="Remove all .alao-bak backup files"
    )
    parser.add_argument(
        "--extract-debug",
        type=str,
        nargs='?',
        const='alao_debug_extract.zip',
        default=None,
        metavar="PATH",
        help="Extract all modified files (.script + .alao-bak) to a zip for debugging (default: alao_debug_extract.zip)"
    )
    parser.add_argument(
        "--split",
        type=int,
        default=0,
        metavar="N",
        help="Split --extract-debug into multiple zips with N mods each (0 = no split, default)"
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Process path directly (single .script file or folder with scripts, no gamedata/scripts structure)"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Path to file containing mod names to exclude (one per line)"
    )

    args = parser.parse_args()

    # validate cache-threshold
    if args.cache_threshold < 2:
        print(f"Warning: Invalid cache threshold ({args.cache_threshold}), using minimum value of 2")
        args.cache_threshold = 2

    # try get path interactively if not provided
    if args.path:
        # clean path: strip quotes, trailing slashes, and fix common issues
        clean_path = args.path
        # detect if arguments got concatenated due to Windows \" escape issue
        if ' --' in clean_path:
            print("WARNING: Detected malformed input - likely caused by trailing backslash before quote.")
            print("         Windows interprets \\\" as an escaped quote, breaking argument parsing.")
            print("")
            print("Instead of:  \"C:\\\\path\\\\\" --flags")
            print("Use:         \"C:\\\\path\" --flags")
            print("")
            
            # try to recover flags
            parts = clean_path.split(' --')
            clean_path = parts[0]
            for part in parts[1:]:
                flag = part.split()[0] if part.split() else part
                if flag == 'direct':
                    args.direct = True
                    print(f"  Recovered: --direct")
                elif flag == 'fix':
                    args.fix = True
                    print(f"  Recovered: --fix")
                elif flag == 'fix-yellow':
                    args.fix_yellow = True
                    print(f"  Recovered: --fix-yellow")
                elif flag == 'fix-debug':
                    args.fix_debug = True
                    print(f"  Recovered: --fix-debug")
                elif flag == 'fix-pcall':
                    args.fix_pcall = True
                    print(f"  Recovered: --fix-pcall")
                elif flag == 'experimental':
                    args.experimental = True
                    print(f"  Recovered: --experimental")
                elif flag.startswith('report'):
                    # try to get report filename
                    remaining = part[6:].strip()  # after "report"
                    if remaining:
                        args.report = remaining.split()[0].strip('"\'')
                        print(f"  Recovered: --report {args.report}")
            print("")
            
        clean_path = clean_path.strip('"\'').rstrip('/\\')
        mods_path = Path(clean_path)
    else:
        print("Anomaly Lua Auto Optimizer (ALAO)")
        print("=" * 55)
        user_input = input("\nEnter path to mods directory: ").strip()
        if not user_input:
            print("No path provided. Exiting.")
            sys.exit(1)
        clean_path = user_input.strip('"\'').rstrip('/\\')
        mods_path = Path(clean_path)

    if not mods_path.exists():
        print(f"Error: Path does not exist: {mods_path}")
        sys.exit(1)

    # Auto-detect single file and enable direct mode
    if mods_path.is_file():
        if mods_path.suffix.lower() in ('.script', '.lua'):
            if not args.direct:
                args.direct = True
                print(f"Single file detected, using direct mode")
        else:
            print(f"Error: File is not a Lua script (.script or .lua): {mods_path}")
            sys.exit(1)

    # discover mods and scripts
    print(f"\nScanning: {mods_path}")
    if args.direct:
        mods = discover_direct(mods_path)
    else:
        mods = discover_mods(mods_path)

    # resolve exclude file: honour --exclude when given, otherwise auto-detect
    # `alao_exclude.txt` next to this script. Auto-load matters for --revert,
    # --clean-backups, and --list-backups too (those run before --fix logic),
    # so an excluded mod's .alao-bak files stay untouched even on bulk revert.
    exclude_path: Optional[Path] = None
    if args.exclude:
        exclude_path = Path(args.exclude)
        if not exclude_path.exists():
            print(f"Warning: Exclude file not found: {exclude_path}")
            exclude_path = None
    else:
        default_exclude = Path(__file__).resolve().parent / 'alao_exclude.txt'
        if default_exclude.exists():
            exclude_path = default_exclude

    excluded_mods = set()
    if exclude_path is not None:
        try:
            with open(exclude_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        excluded_mods.add(line)

            if excluded_mods:
                before_count = len(mods)
                mods = {name: scripts for name, scripts in mods.items() if name not in excluded_mods}
                excluded_count = before_count - len(mods)
                if excluded_count > 0:
                    print(f"Excluded {excluded_count} mods from {exclude_path.name}")
        except Exception as e:
            print(f"Warning: Could not read exclude file: {e}")

    if not mods:
        if args.direct:
            print("No scripts found in path.")
        else:
            print("No mods with scripts found.")
        sys.exit(0)

    total_scripts = sum(len(scripts) for scripts in mods.values())
    if args.direct:
        print(f"Found {total_scripts} script files (direct mode)\n")
    else:
        print(f"Found {len(mods)} mods with {total_scripts} script files\n")

    # handle backup operations (only ALAO-created .alao-bak files)
    if args.list_backups or args.revert or args.clean_backups:
        backup_files = []
        for mod_name, scripts in mods.items():
            for script_path in scripts:
                bak_path = script_path.with_suffix(script_path.suffix + '.alao-bak')
                if bak_path.exists():
                    backup_files.append((script_path, bak_path, mod_name))

        if not backup_files:
            print("No backup files found.")
            sys.exit(0)

        if args.list_backups:
            print(f"Found {len(backup_files)} backup files:\n")
            by_mod = {}
            for script_path, bak_path, mod_name in backup_files:
                if mod_name not in by_mod:
                    by_mod[mod_name] = []
                by_mod[mod_name].append(bak_path)

            for mod_name, baks in sorted(by_mod.items()):
                print(f"  [{mod_name}]")
                for bak in baks:
                    print(f"    - {bak.name}")
            sys.exit(0)

        if args.clean_backups:
            confirm = input(f"Delete {len(backup_files)} backup files? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                sys.exit(0)

            deleted = 0
            for script_path, bak_path, mod_name in backup_files:
                try:
                    bak_path.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"  [ERROR] Could not delete {bak_path.name}: {e}")

            print(f"Deleted {deleted} backup files.")
            sys.exit(0)

        if args.revert:
            confirm = input(f"Restore {len(backup_files)} files from backups? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                sys.exit(0)

            restored = 0
            for script_path, bak_path, mod_name in backup_files:
                try:
                    shutil.copy2(bak_path, script_path)
                    bak_path.unlink()
                    restored += 1
                    if args.verbose:
                        print(f"  [RESTORED] {script_path.name}")
                except Exception as e:
                    print(f"  [ERROR] Could not restore {script_path.name}: {e}")

            print(f"Restored {restored} files from backups.")
            sys.exit(0)

    # handle extract-debug operation
    if args.extract_debug:
        
        # group files by mod
        files_by_mod = {}
        for mod_name, scripts in mods.items():
            for script_path in scripts:
                bak_path = script_path.with_suffix(script_path.suffix + '.alao-bak')
                if bak_path.exists():
                    if mod_name not in files_by_mod:
                        files_by_mod[mod_name] = []
                    files_by_mod[mod_name].append((script_path, bak_path))

        if not files_by_mod:
            print("No modified files found (no .alao-bak backups exist).")
            sys.exit(0)

        total_files = sum(len(files) for files in files_by_mod.values())
        total_mods = len(files_by_mod)
        
        zip_path = Path(args.extract_debug)
        if not zip_path.suffix:
            zip_path = zip_path.with_suffix('.zip')
        
        base_name = zip_path.stem
        base_dir = zip_path.parent
        
        # determine chunks
        mod_names = sorted(files_by_mod.keys())
        
        if args.split > 0:
            # split into multiple archives
            chunks = []
            for i in range(0, len(mod_names), args.split):
                chunks.append(mod_names[i:i + args.split])
            
            print(f"Extracting {total_files} modified files from {total_mods} mods into {len(chunks)} archives...")
            
            for chunk_idx, chunk_mods in enumerate(chunks, 1):
                chunk_zip = base_dir / f"{base_name}_{chunk_idx}.zip"
                chunk_file_count = 0
                
                with zipfile.ZipFile(chunk_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for mod_name in chunk_mods:
                        for script_path, bak_path in files_by_mod[mod_name]:
                            arcname_script = f"{mod_name}/{script_path.name}"
                            arcname_bak = f"{mod_name}/{bak_path.name}"
                            
                            zf.write(script_path, arcname_script)
                            zf.write(bak_path, arcname_bak)
                            chunk_file_count += 1
                            
                            if args.verbose:
                                print(f"  + {arcname_script}")
                                print(f"  + {arcname_bak}")
                
                print(f"  [{chunk_idx}/{len(chunks)}] {chunk_zip.name}: {len(chunk_mods)} mods, {chunk_file_count} file pairs")
            
            print(f"\nSaved {len(chunks)} archives to: {base_dir.absolute()}")
        else:
            # single archive
            print(f"Extracting {total_files} modified files to {zip_path}...")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for mod_name in mod_names:
                    for script_path, bak_path in files_by_mod[mod_name]:
                        arcname_script = f"{mod_name}/{script_path.name}"
                        arcname_bak = f"{mod_name}/{bak_path.name}"
                        
                        zf.write(script_path, arcname_script)
                        zf.write(bak_path, arcname_bak)
                        
                        if args.verbose:
                            print(f"  + {arcname_script}")
                            print(f"  + {arcname_bak}")
            
            print(f"\nExtracted {total_files} file pairs from {total_mods} mods:")
            for mod_name in mod_names:
                print(f"  [{mod_name}] {len(files_by_mod[mod_name])} files")
            
            print(f"\nSaved to: {zip_path.absolute()}")
        
        sys.exit(0)

    # analyze
    reporter = Reporter()
    files_analyzed = 0
    files_with_issues = 0
    files_skipped = 0
    parse_errors = 0

    # flatten for progress tracking
    all_files = []
    for mod_name, scripts in mods.items():
        for script_path in scripts:
            all_files.append((mod_name, script_path))

    # check if any fix flags are set
    fix_flags_set = (args.fix or args.fix_debug or args.fix_yellow or 
                     args.experimental or args.fix_nil or args.remove_dead_code
                     or args.fix_pcall)
    
    # auto-backup on first fix run (safety mechanism)
    if fix_flags_set and not args.no_first_time_auto_backup and not args.backup_all_scripts:
        backup_pattern = str(mods_path / "scripts-backup-*.zip")
        existing_backups = glob.glob(backup_pattern)
        
        if not existing_backups:
            if not args.quiet:
                print("=" * 60)
                print("SAFETY: No backup found. Creating automatic backup first...")
                print("        (use --no-first-time-auto-backup to skip this)")
                print("=" * 60)
            
            backup_path = backup_all_scripts(
                all_files,
                output_path='auto',
                mods_root=mods_path,
                quiet=args.quiet
            )
            if backup_path is None:
                print("\n[!] WARNING: Auto-backup failed!")
                print("    For safety, aborting fix operation.")
                print("    You can:")
                print("      1. Fix the backup issue and try again")
                print("      2. Use --no-first-time-auto-backup to skip (not recommended)")
                print("      3. Manually backup your scripts folder first")
                return
            if not args.quiet:
                print()  # blank line after backup

    # backup all scripts if explicitly requested (before any modifications)
    if args.backup_all_scripts:
        backup_path = backup_all_scripts(
            all_files,
            output_path=args.backup_all_scripts,
            mods_root=mods_path,
            quiet=args.quiet
        )
        if backup_path is None and not args.quiet:
            print("Warning: Backup failed, continuing anyway...")
        print()  # blank line after backup

    # validate and calculate number of workers
    if args.workers is not None:
        if args.workers < 1:
            print(f"Warning: Invalid worker count ({args.workers}), using default")
            num_workers = min(os.cpu_count() or 4, 8)
        else:
            num_workers = min(args.workers, 32)  # cap at reasonable max
    else:
        num_workers = min(os.cpu_count() or 4, 8)
    
    start_time = datetime.now()

    if not args.quiet:
        print(f"Analyzing with {num_workers} workers...")

    # prepare work items for parallel analysis
    work_items = [
        (mod_name, script_path, args.timeout, args.cache_threshold, args.experimental)
        for mod_name, script_path in all_files
    ]

    completed = 0
    pool_crashed = False
    interrupted = False
    processed_paths = set()  # track which files were processed

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(analyze_file_worker, item): item for item in work_items}

            for future in as_completed(futures):
                item = futures[future]
                
                try:
                    mod_name, script_path, findings, error = future.result()
                    processed_paths.add(script_path)
                    completed += 1
                except BrokenExecutor:
                    pool_crashed = True
                    break
                except Exception as e:
                    processed_paths.add(item[1])
                    completed += 1
                    files_skipped += 1
                    if args.verbose:
                        print(f"\n  [ERROR] {item[1].name}: {e}")
                    continue

                if not args.quiet:
                    total = len(all_files)
                    progress = (completed / total * 100) if total > 0 else 100
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = completed / elapsed if elapsed > 0.01 else 0
                    remaining = total - completed
                    eta = remaining / rate if rate > 0 and remaining > 0 else 0
                    print(
                        f"\r[{progress:5.1f}%] {completed}/{total} | ETA: {eta:.0f}s  ", end="", flush=True)

                if error:
                    if 'SyntaxError' in error or 'parse' in error.lower() or 'TimeoutError' in error:
                        parse_errors += 1
                        if args.verbose:
                            print(f"\n  [PARSE ERROR] {script_path.name}")
                    else:
                        files_skipped += 1
                        if args.verbose:
                            print(f"\n  [ERROR] {script_path.name}: {error}")
                else:
                    files_analyzed += 1
                    if findings:
                        files_with_issues += 1
                        for finding in findings:
                            reporter.add_finding(mod_name, script_path, finding)

                        if args.verbose and not args.quiet:
                            print(f"\n  [{len(findings):3d} issues] {script_path.name}")
    except BrokenExecutor:
        pool_crashed = True
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted! Finishing current tasks...")

    if interrupted:
        print(f"Processed {completed}/{len(all_files)} files before interruption.")
        # continue to show partial results

    if pool_crashed and not interrupted:
        if not args.quiet:
            print(f"\n\nWorker crashed. Falling back to single-threaded mode...")

        # process remaining files sequentially
        remaining = [(m, s) for m, s in all_files if s not in processed_paths]

        for mod_name, script_path in remaining:
            completed += 1
            if not args.quiet:
                progress = completed / len(all_files) * 100
                print(
                    f"\r[{progress:5.1f}%] {completed}/{len(all_files)} | {script_path.name[:30]:<30}", end="", flush=True)

            try:
                # honour --timeout in the single-thread fallback too - without
                # this a single bad file can hang the whole run after a worker
                # crash forced us off the process pool.
                if args.timeout and args.timeout > 0:
                    findings = analyze_file_with_timeout(
                        script_path, args.timeout,
                        cache_threshold=args.cache_threshold,
                        experimental=args.experimental,
                    )
                else:
                    findings = analyze_file(
                        script_path,
                        cache_threshold=args.cache_threshold,
                        experimental=args.experimental,
                    )
                files_analyzed += 1
                if findings:
                    files_with_issues += 1
                    for finding in findings:
                        reporter.add_finding(mod_name, script_path, finding)
            except TimeoutError:
                parse_errors += 1
                if args.verbose:
                    print(f"\n  [TIMEOUT] {script_path.name}")
            except Exception as e:
                files_skipped += 1
                if args.verbose:
                    print(f"\n  [ERROR] {script_path.name}: {e}")

    # clear progress line
    if not args.quiet:
        print("\r" + " " * 80 + "\r", end="")

    # apply fixes if requested
    files_modified = 0
    total_edits = 0

    if args.fix or args.fix_debug or args.fix_yellow or args.experimental or args.fix_nil or args.remove_dead_code or args.fix_pcall:
        # proceed with fixes (auto-backup already handled before analysis)
        fix_msg = "Applying fixes"
        fix_types = []
        if args.fix:
            fix_types.append("GREEN")
        if args.fix_yellow:
            fix_types.append("YELLOW")
        if args.fix_debug:
            fix_types.append("DEBUG")
        if args.experimental:
            fix_types.append("EXPERIMENTAL")
        if args.fix_nil:
            fix_types.append("NIL-GUARD")
        if args.remove_dead_code:
            fix_types.append("DEAD-CODE")
        if args.fix_pcall:
            fix_types.append("PCALL-HOIST")
        print(f"{fix_msg} ({', '.join(fix_types)}) with {num_workers} workers...")

        # prepare work items, skip files that already have .alao-bak (prevent double-fix)
        work_items = []
        skipped_has_backup = 0
        for mod_name, script_path in all_files:
            bak_path = script_path.with_suffix(script_path.suffix + '.alao-bak')
            if bak_path.exists():
                skipped_has_backup += 1
                if args.verbose:
                    print(f"  [SKIP] {script_path.name} - backup already exists")
            else:
                work_items.append(
                    (script_path, args.backup, args.fix_debug, args.fix_yellow, args.experimental, args.fix_nil, args.remove_dead_code, args.cache_threshold, args.fix_pcall)
                )
        
        if skipped_has_backup > 0 and not args.quiet:
            print(f"Skipping {skipped_has_backup} files with existing backups (already processed)")
            print(f"Tip: Use --revert first if you want to re-process, or --clean-backups to remove old backups\n")
        
        if not work_items:
            if not args.quiet:
                print("No files to process (all have existing backups).")
        elif args.single_thread:
            # single-threaded mode for debugging
            completed = 0
            if not args.quiet:
                print("Running in single-thread mode...")
            for item in work_items:
                script_path = item[0]
                completed += 1
                if not args.quiet:
                    progress = completed / len(work_items) * 100
                    print(f"\r[{progress:5.1f}%] Fixing {completed}/{len(work_items)}...", end="", flush=True)
                
                try:
                    script_path, modified, edit_count, error = transform_file_worker(item)
                    if error:
                        if args.verbose:
                            print(f"\n  [FIX ERROR] {script_path.name}: {error}")
                    elif modified:
                        files_modified += 1
                        total_edits += edit_count
                        if args.verbose:
                            print(f"\n  [FIXED] {script_path.name} ({edit_count} edits)")
                except Exception as e:
                    print(f"\n  [ERROR] {script_path.name}: {e}")
            
            if not args.quiet:
                print()
        else:
            completed = 0
            pool_crashed = False
            transform_interrupted = False
            processed_paths = set()

            try:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(
                            transform_file_worker,
                            item): item for item in work_items}

                    for future in as_completed(futures):
                        completed += 1
                        if not args.quiet:
                            progress = (completed / len(work_items) * 100) if work_items else 100
                            print(
                                f"\r[{progress:5.1f}%] Fixing {completed}/{len(work_items)}...", end="", flush=True)

                        try:
                            script_path, modified, edit_count, error = future.result()
                            processed_paths.add(futures[future][0])
                        except BrokenExecutor:
                            pool_crashed = True
                            break
                        except Exception as e:
                            if args.verbose:
                                print(f"\n  [ERROR] {e}")
                            continue

                        if error:
                            if args.verbose:
                                print(f"\n  [FIX ERROR] {script_path.name}: {error}")
                        elif modified:
                            files_modified += 1
                            total_edits += edit_count
                            if args.verbose:
                                print(f"\n  [FIXED] {script_path.name} ({edit_count} edits)")
            except BrokenExecutor:
                pool_crashed = True
            except KeyboardInterrupt:
                transform_interrupted = True
                print("\n\nInterrupted during fixes! Some files may have been modified.")

            if pool_crashed and not transform_interrupted:
                if not args.quiet:
                    print(f"\n\nWorker crashed. Falling back to single-threaded mode...")

                # process remaining files sequentially
                for item in work_items:
                    if item[0] in processed_paths:
                        continue
                    script_path = item[0]
                    completed += 1
                    if not args.quiet:
                        progress = (completed / len(work_items) * 100) if work_items else 100
                        print(
                            f"\r[{progress:5.1f}%] Fixing {completed}/{len(work_items)}...", end="", flush=True)

                    try:
                        modified, _, edit_count = transform_file(
                            script_path,
                            backup=args.backup,
                            fix_debug=args.fix_debug,
                            fix_yellow=args.fix_yellow,
                            experimental=args.experimental,
                            fix_nil=args.fix_nil,
                            remove_dead_code=args.remove_dead_code,
                            cache_threshold=args.cache_threshold,
                            fix_pcall=args.fix_pcall,
                        )
                        if modified:
                            files_modified += 1
                            total_edits += edit_count
                            if args.verbose:
                                print(f"\n  [FIXED] {script_path.name} ({edit_count} edits)")
                    except Exception as e:
                        if args.verbose:
                            print(f"\n  [FIX ERROR] {script_path.name}: {e}")

            if not args.quiet:
                print("\r" + " " * 60 + "\r", end="")

    # output results
    if not args.quiet:
        reporter.print_summary()

        if args.verbose:
            reporter.print_detailed()

    # save report if requested
    if args.report:
        report_path = Path(args.report)
        print(f"\nGenerating report: {report_path.name}...")
        reporter.save(report_path, verbose=not args.quiet)
        print(f"Report saved to: {report_path}")

    # final stats
    print(f"\n{'=' * 55}")
    print(f"Files analyzed: {files_analyzed}")
    print(f"Files with issues: {files_with_issues}")
    if files_skipped > 0:
        print(f"Files skipped (timeout/error): {files_skipped}")
    if parse_errors > 0:
        print(f"Files with parse errors: {parse_errors}")
    if (args.fix or args.fix_debug or args.fix_yellow or args.experimental or args.fix_nil or args.remove_dead_code or args.fix_pcall):
        print(f"Files modified: {files_modified}")
        print(f"Total edits applied: {total_edits}")

    green_count = reporter.count_by_severity("GREEN")
    yellow_count = reporter.count_by_severity("YELLOW")
    red_count = reporter.count_by_severity("RED")
    debug_count = reporter.count_by_severity("DEBUG")
    
    # count nil access findings
    nil_count = sum(1 for f in reporter.all_findings if f.pattern_name == 'potential_nil_access')
    nil_fixable = sum(1 for f in reporter.all_findings 
                     if f.pattern_name == 'potential_nil_access' and f.details.get('is_safe_to_fix'))
    
    # count dead code findings
    dead_code_count = sum(1 for f in reporter.all_findings if f.pattern_name.startswith('dead_code_'))
    dead_code_fixable = sum(1 for f in reporter.all_findings
                           if f.pattern_name.startswith('dead_code_') and f.details.get('is_safe_to_remove'))
    unused_count = sum(1 for f in reporter.all_findings if f.pattern_name.startswith('unused_'))

    pcall_hoist_count = sum(1 for f in reporter.all_findings if f.pattern_name == 'pcall_anon_hoist')
    pcall_skip_count = sum(1 for f in reporter.all_findings if f.pattern_name == 'pcall_anon_skip')
    pcall_count = pcall_hoist_count + pcall_skip_count

    findings_str = f"{green_count} GREEN (auto-fixable), {yellow_count} YELLOW (review), {red_count} RED (info)"
    if debug_count > 0:
        findings_str += f", {debug_count} DEBUG (logging)"
    if nil_count > 0:
        findings_str += f", {nil_count} NIL ({nil_fixable} fixable)"
    if dead_code_count > 0 or unused_count > 0:
        findings_str += f", {dead_code_count + unused_count} DEAD-CODE ({dead_code_fixable} removable)"
    if pcall_count > 0:
        findings_str += f", {pcall_count} PCALL ({pcall_hoist_count} hoistable)"
    print(f"\nFindings: {findings_str}")

    if green_count > 0 and not args.fix:
        print("\nTip: Run with --fix to automatically apply GREEN fixes")
    if yellow_count > 0 and not args.fix_yellow: 
        print("Tip: Run with --fix-yellow to also apply YELLOW fixes (unsafe)")
    if debug_count > 0 and not args.fix_debug:
        print("Tip: Run with --fix-debug to comment out DEBUG statements")
    if yellow_count > 0 and not args.experimental:
        print("Tip: Run with --experimental to fix string concat in loops (experimental)")
    if nil_fixable > 0 and not args.fix_nil:
        print("Tip: Run with --fix-nil to add nil guards for safe nil access patterns")
    if dead_code_fixable > 0 and not args.remove_dead_code:
        print("Tip: Run with --remove-dead-code to remove safe unreachable code")
    if pcall_hoist_count > 0 and not args.fix_pcall:
        print("Tip: Run with --fix-pcall to hoist pcall(function() ... end) to a named local")
    if (args.fix or args.fix_debug or args.fix_yellow or args.experimental or args.fix_nil or args.remove_dead_code or args.fix_pcall):
        print("Tip: Run with --revert to undo all changes using .alao-bak files")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
