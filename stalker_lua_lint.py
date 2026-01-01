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

    # IMPORTANT
    --revert          Restore all .bak backup files (undo fixes)
    --report [file]   Generate a comprehensive report (supports .txt, .html, .json)

    # MULTITHREAD processing
    --timeout [seconds]
                       Timeout per file in seconds (default: 10)
    --workers / -j    Number of parallel workers for fixes (default: CPU count)
    
    # USELESS things
    --backup / --no-backup
                       Create .bak files before modifying (default: True)
    --verbose / -v    Show detailed output
    --quiet / -q      Only show summary

    # DANGER ZONE
    --list-backups    List all .bak backup files without restoring
    --clean-backups   Remove all .bak backup files
"""

import sys
import os
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed, BrokenExecutor

from discovery import discover_mods
from ast_analyzer import analyze_file
from ast_transformer import transform_file
from reporter import Reporter
from models import Finding


def analyze_file_with_timeout(file_path: Path, timeout: float):
    """Analyze a file with a timeout to prevent hanging on problematic files."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(analyze_file, file_path)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise TimeoutError(f"Analysis timed out for {file_path.name}")


def analyze_file_worker(args_tuple):
    """Worker function for parallel analyze_file calls."""
    mod_name, script_path, timeout = args_tuple
    try:
        findings = analyze_file(script_path)
        return (mod_name, script_path, findings, None)
    except Exception as e:
        return (mod_name, script_path, [], str(e))


def transform_file_worker(args_tuple):
    """Worker function for parallel transform_file calls."""
    script_path, backup, fix_debug, fix_yellow, experimental = args_tuple
    try:
        modified, _, edit_count = transform_file(
            script_path,
            backup=backup,
            fix_debug=fix_debug,
            fix_yellow=fix_yellow,
            experimental=experimental,
        )
        return (script_path, modified, edit_count, None)
    except Exception as e:
        return (script_path, False, 0, str(e))


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
        "--experimental",
        action="store_true",
        help="Enable experimental fixes (string concat in loops)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create .bak files before modifying (default: True)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Don't create backup files"
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
        help="Number of parallel workers for fixes (default: CPU count)"
    )
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Restore all .bak backup files (undo fixes)"
    )
    parser.add_argument(
        "--list-backups",
        action="store_true",
        help="List all .bak backup files without restoring"
    )
    parser.add_argument(
        "--clean-backups",
        action="store_true",
        help="Remove all .bak backup files"
    )

    args = parser.parse_args()

    # tyr get path interactively if not provided
    if args.path:
        mods_path = Path(args.path)
    else:
        print("Anomaly Lua Auto Optimizer (ALAO)")
        print("=" * 55)
        user_input = input("\nEnter path to mods directory: ").strip()
        if not user_input:
            print("No path provided. Exiting.")
            sys.exit(1)
        mods_path = Path(user_input)

    if not mods_path.exists():
        print(f"Error: Path does not exist: {mods_path}")
        sys.exit(1)

    # discover mods and scripts
    print(f"\nScanning: {mods_path}")
    mods = discover_mods(mods_path)

    if not mods:
        print("No mods with scripts found.")
        sys.exit(0)

    total_scripts = sum(len(scripts) for scripts in mods.values())
    print(f"Found {len(mods)} mods with {total_scripts} script files\n")

    # handle backup operations
    if args.list_backups or args.revert or args.clean_backups:
        backup_files = []
        for mod_name, scripts in mods.items():
            for script_path in scripts:
                bak_path = script_path.with_suffix(script_path.suffix + '.bak')
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
            confirm = input(
                f"Restore {
                    len(backup_files)} files from backups? [y/N]: ").strip().lower()
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

    num_workers = args.workers or min(os.cpu_count() or 4, 8)  # cap at 8 by default
    start_time = datetime.now()

    if not args.quiet:
        print(f"Analyzing with {num_workers} workers...")

    # prepare work items for parallel analysis
    work_items = [
        (mod_name, script_path, args.timeout)
        for mod_name, script_path in all_files
    ]

    completed = 0
    pool_crashed = False

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(analyze_file_worker, item): item for item in work_items}

            for future in as_completed(futures):
                completed += 1

                try:
                    mod_name, script_path, findings, error = future.result()
                except BrokenExecutor:
                    pool_crashed = True
                    break
                except Exception as e:
                    files_skipped += 1
                    if args.verbose:
                        item = futures[future]
                        print(f"\n  [ERROR] {item[1].name}: {e}")
                    continue

                if not args.quiet:
                    progress = completed / len(all_files) * 100
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(all_files) - completed) / rate if rate > 0 else 0
                    print(
                        f"\r[{progress:5.1f}%] {completed}/{len(all_files)} | ETA: {eta:.0f}s  ", end="", flush=True)

                if error:
                    if 'SyntaxError' in error or 'parse' in error.lower():
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

    if pool_crashed:
        if not args.quiet:
            print(f"\n\nWorker crashed. Falling back to single-threaded mode...")

        # process remaining files sequentially
        remaining = [(m, s) for m, s in all_files if (m, s, args.timeout) not in
                     {futures[f] for f in futures if f.done()}] if 'futures' in dir() else all_files[completed:]

        for mod_name, script_path in remaining:
            completed += 1
            if not args.quiet:
                progress = completed / len(all_files) * 100
                print(
                    f"\r[{progress:5.1f}%] {completed}/{len(all_files)} | {script_path.name[:30]:<30}", end="", flush=True)

            try:
                findings = analyze_file(script_path)
                files_analyzed += 1
                if findings:
                    files_with_issues += 1
                    for finding in findings:
                        reporter.add_finding(mod_name, script_path, finding)
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

    if args.fix or args.fix_debug or args.fix_yellow or args.experimental:
        if True: # yes
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
            print(f"{fix_msg} ({', '.join(fix_types)}) with {num_workers} workers...")

            # prepare work items
            work_items = [
                (script_path, args.backup, args.fix_debug, args.fix_yellow, args.experimental)
                for mod_name, script_path in all_files
            ]

            completed = 0
            pool_crashed = False
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
                            progress = completed / len(all_files) * 100
                            print(
                                f"\r[{progress:5.1f}%] Fixing {completed}/{len(all_files)}...", end="", flush=True)

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

            if pool_crashed:
                if not args.quiet:
                    print(f"\n\nWorker crashed. Falling back to single-threaded mode...")

                # process remaining files sequentially
                for item in work_items:
                    if item[0] in processed_paths:
                        continue
                    script_path = item[0]
                    completed += 1
                    if not args.quiet:
                        progress = completed / len(all_files) * 100
                        print(
                            f"\r[{progress:5.1f}%] Fixing {completed}/{len(all_files)}...", end="", flush=True)

                    try:
                        modified, _, edit_count = transform_file(
                            script_path,
                            backup=args.backup,
                            fix_debug=args.fix_debug,
                            fix_yellow=args.fix_yellow,
                            experimental=args.experimental,
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
    if (args.fix or args.fix_debug or args.fix_yellow or args.experimental):
        print(f"Files modified: {files_modified}")
        print(f"Total edits applied: {total_edits}")

    green_count = reporter.count_by_severity("GREEN")
    yellow_count = reporter.count_by_severity("YELLOW")
    red_count = reporter.count_by_severity("RED")
    debug_count = reporter.count_by_severity("DEBUG")

    findings_str = f"{green_count} GREEN (auto-fixable), {yellow_count} YELLOW (review), {red_count} RED (info)"
    if debug_count > 0:
        findings_str += f", {debug_count} DEBUG (logging)"
    print(f"\nFindings: {findings_str}")

    if green_count > 0 and not args.fix:
        print("\nTip: Run with --fix to automatically apply GREEN fixes")
    if yellow_count > 0 and not args.fix_yellow: 
        print("Tip: Run with --fix-yellow to also apply YELLOW fixes (unsafe)")
    if debug_count > 0 and not args.fix_debug:
        print("Tip: Run with --fix-debug to comment out DEBUG statements")
    if yellow_count > 0 and not args.experimental:
        print("Tip: Run with --experimental to fix string concat in loops (experimental)")
    if (args.fix or args.fix_debug or args.fix_yellow or args.experimental):
        print("Tip: Run with --revert to undo all changes using .bak files")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
