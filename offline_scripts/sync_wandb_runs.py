#!/usr/bin/env python3
"""
Sync offline WandB runs in parallel from config save folders.

This script scans config files in a directory, extracts their save_folder paths,
and runs 'wandb sync' in parallel for each run's wandb directory.
By default, only the latest run per config is synced.

Usage:
    # Sync latest runs from all configs in a directory
    python sync_wandb_runs.py --config-dir configs/stu-ablations

    # Sync all runs (not just latest)
    python sync_wandb_runs.py --config-dir configs/stu-ablations --all-runs

    # Sync specific config files
    python sync_wandb_runs.py --configs configs/stu-ablations/OLMo-STU-All-150M.yaml configs/tiny/OLMo-20M.yaml

    # Dry run to see what would be synced
    python sync_wandb_runs.py --config-dir configs/stu-ablations --dry-run

    # Limit number of parallel syncs
    python sync_wandb_runs.py --config-dir configs/stu-ablations --max-workers 4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import yaml
import shutil

# Lock for printing to avoid interleaved output
print_lock = Lock()


def safe_print(msg, prefix="", color=None):
    """Thread-safe printing with optional color."""
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
    }
    
    with print_lock:
        if color and color in colors:
            print(f"{colors[color]}{prefix}{msg}{colors['reset']}")
        else:
            print(f"{prefix}{msg}")


def read_config_yaml(config_path):
    """Read the YAML config file and extract relevant information."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        safe_print(f"Warning: Could not parse config file {config_path}: {e}", color="yellow")
        return {}


def get_save_folder(config, config_path):
    """Extract save_folder from config, with variable interpolation."""
    save_folder = config.get("save_folder")
    if not save_folder:
        return None
    
    # Handle variable interpolation like ${run_name}
    run_name = config.get("run_name", Path(config_path).stem)
    save_folder = save_folder.replace("${run_name}", run_name)
    
    return save_folder


def find_wandb_runs(save_folder, latest_only=True):
    """Find wandb run directories in a save folder.
    
    Args:
        save_folder: Path to the checkpoint save folder
        latest_only: If True, only return the latest run (default: True)
    
    Returns:
        List of Path objects for wandb runs to sync
    """
    save_path = Path(save_folder)
    
    if not save_path.exists():
        return []
    
    wandb_runs = []
    
    # Look for wandb directory - WandB creates a nested wandb/wandb structure
    wandb_dir = save_path / "wandb" / "wandb"
    if not wandb_dir.exists():
        # Fallback to single level if nested doesn't exist
        wandb_dir = save_path / "wandb"
        if not wandb_dir.exists():
            return []
    
    # Check for latest-run symlink first (most reliable)
    latest_run = wandb_dir / "latest-run"
    if latest_run.exists() and latest_run.is_symlink():
        target = latest_run.resolve()
        if target.exists():
            if latest_only:
                return [target]
            else:
                wandb_runs.append(target)
    
    # Find all offline run directories (format: offline-run-TIMESTAMP-ID)
    all_runs = []
    for item in wandb_dir.iterdir():
        if item.is_dir() and (item.name.startswith("offline-run-") or item.name.startswith("run-")):
            all_runs.append(item)
    
    if not all_runs:
        return wandb_runs
    
    if latest_only:
        # If we didn't find latest-run symlink, use most recently modified
        latest = max(all_runs, key=lambda p: p.stat().st_mtime)
        return [latest]
    else:
        # Return all runs (excluding duplicates if latest-run was already added)
        for run in all_runs:
            if run not in wandb_runs:
                wandb_runs.append(run)
        return wandb_runs


def sync_wandb_run(run_dir, config_name):
    """Sync a single wandb run directory."""
    run_name = run_dir.name
    prefix = f"[{config_name}:{run_name}] "
    
    try:
        # Check if wandb command is available
        if not shutil.which("wandb"):
            safe_print("wandb command not found. Is wandb installed?", prefix=prefix, color="red")
            return False
        
        safe_print(f"Starting sync...", prefix=prefix, color="cyan")
        
        # Run wandb sync with real-time output
        # Use the current Python's environment (inherits venv)
        process = subprocess.Popen(
            ["wandb", "sync", str(run_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        
        # Stream output in real-time
        for line in process.stdout:
            line = line.rstrip()
            if line:
                safe_print(line, prefix=prefix, color="blue")
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            safe_print("Sync completed successfully!", prefix=prefix, color="green")
            return True
        else:
            safe_print(f"Sync failed with return code {return_code}", prefix=prefix, color="red")
            return False
            
    except Exception as e:
        safe_print(f"Error during sync: {e}", prefix=prefix, color="red")
        return False


def collect_sync_tasks(config_paths, latest_only=True):
    """Collect all wandb sync tasks from config files.
    
    Args:
        config_paths: List of config file paths to process
        latest_only: If True, only sync the latest run per config (default: True)
    
    Returns:
        List of (run_dir, config_name) tuples
    """
    tasks = []  # List of (run_dir, config_name) tuples
    
    for config_path in config_paths:
        config_path = Path(config_path)
        
        if not config_path.exists():
            safe_print(f"Config file not found: {config_path}", color="yellow")
            continue
        
        # Read config
        config = read_config_yaml(config_path)
        if not config:
            continue
        
        # Get save folder
        save_folder = get_save_folder(config, config_path)
        if not save_folder:
            safe_print(f"No save_folder found in config: {config_path}", color="yellow")
            continue
        
        # Find wandb runs
        wandb_runs = find_wandb_runs(save_folder, latest_only=latest_only)
        
        if not wandb_runs:
            safe_print(f"No wandb runs found in: {save_folder}", color="yellow")
            continue
        
        # Add tasks
        config_name = config.get("run_name", config_path.stem)
        for run_dir in wandb_runs:
            tasks.append((run_dir, config_name))
    
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Sync offline WandB runs in parallel from config save folders"
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config-dir",
        type=str,
        help="Directory containing config YAML files to scan",
    )
    group.add_argument(
        "--configs",
        type=str,
        nargs="+",
        help="Specific config file(s) to sync",
    )
    
    # Execution options
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel syncs (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without actually syncing",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.yaml",
        help="File pattern for config files when using --config-dir (default: *.yaml)",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Sync all runs instead of just the latest run per config (default: latest only)",
    )
    
    args = parser.parse_args()
    
    # Collect config files
    config_paths = []
    if args.config_dir:
        config_dir = Path(args.config_dir)
        if not config_dir.exists():
            print(f"Error: Config directory not found: {config_dir}")
            sys.exit(1)
        
        config_paths = sorted(config_dir.glob(args.pattern))
        if not config_paths:
            print(f"No config files matching '{args.pattern}' found in: {config_dir}")
            sys.exit(1)
    else:
        config_paths = [Path(p) for p in args.configs]
    
    print(f"Scanning {len(config_paths)} config file(s)...")
    
    # Collect sync tasks
    latest_only = not args.all_runs  # Default is latest only, unless --all-runs is specified
    tasks = collect_sync_tasks(config_paths, latest_only=latest_only)
    
    if not tasks:
        print("\nNo wandb runs found to sync!")
        return
    
    sync_mode = "all runs" if args.all_runs else "latest runs only"
    print(f"\nFound {len(tasks)} wandb run(s) to sync ({sync_mode}):")
    for run_dir, config_name in tasks:
        print(f"  - {config_name}: {run_dir}")
    
    if args.dry_run:
        print("\nDry run mode - no syncing performed")
        return
    
    # Check if wandb is available
    if not shutil.which("wandb"):
        print("\nError: 'wandb' command not found!")
        print("Please ensure wandb is installed and your virtual environment is activated.")
        sys.exit(1)
    
    # Sync runs in parallel
    print(f"\nStarting parallel sync with {args.max_workers} worker(s)...\n")
    print("=" * 80)
    
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(sync_wandb_run, run_dir, config_name): (run_dir, config_name)
            for run_dir, config_name in tasks
        }
        
        # Wait for completion
        for future in as_completed(future_to_task):
            run_dir, config_name = future_to_task[future]
            try:
                success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                safe_print(f"Unexpected error for {config_name}: {e}", color="red")
                failed += 1
    
    # Summary
    print("=" * 80)
    print("\nSync Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(tasks)}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

