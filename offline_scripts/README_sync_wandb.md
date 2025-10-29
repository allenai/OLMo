# Sync WandB Runs Script

This script automatically syncs offline WandB runs in parallel from your training configs' save folders.

## Overview

When running training with `WANDB_MODE=offline`, WandB stores runs locally that need to be manually synced to the cloud. This script automates that process by:

1. Reading config files to find their `save_folder` paths
2. Locating offline wandb runs in each save folder (latest run by default)
3. Running `wandb sync` in parallel for all runs
4. Showing real-time sync progress in the console

**By default, only the latest run per config is synced.** Use `--all-runs` to sync all runs.

## Prerequisites

- WandB must be installed: `pip install wandb`
- You must be logged in: `wandb login`
- Your virtual environment must be activated before running the script

## Usage

### Sync latest runs from all configs in a directory (default)

```bash
python offline_scripts/sync_wandb_runs.py --config-dir configs/stu-ablations
```

### Sync all runs (not just latest)

```bash
python offline_scripts/sync_wandb_runs.py --config-dir configs/stu-ablations --all-runs
```

### Sync specific config files

```bash
python offline_scripts/sync_wandb_runs.py --configs \
    configs/stu-ablations/OLMo-STU-All-150M.yaml \
    configs/tiny/OLMo-20M.yaml
```

### Dry run to see what would be synced

```bash
python offline_scripts/sync_wandb_runs.py --config-dir configs/stu-ablations --dry-run
```

### Control parallelism

```bash
# Use 8 parallel workers (default is 4)
python offline_scripts/sync_wandb_runs.py --config-dir configs/stu-ablations --max-workers 8

# Use only 1 worker (sequential syncing)
python offline_scripts/sync_wandb_runs.py --config-dir configs/stu-ablations --max-workers 1
```

### Custom file pattern

```bash
# Sync only configs matching a specific pattern
python offline_scripts/sync_wandb_runs.py --config-dir configs/stu-ablations --pattern "*-150M.yaml"
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config-dir DIR` | Directory containing config YAML files | Required (or use --configs) |
| `--configs FILE [FILE ...]` | Specific config file(s) to sync | Required (or use --config-dir) |
| `--max-workers N` | Number of parallel syncs | 4 |
| `--dry-run` | Show what would be synced without syncing | False |
| `--pattern PATTERN` | File pattern when using --config-dir | `*.yaml` |
| `--all-runs` | Sync all runs instead of just latest per config | False (latest only) |

## How It Works

1. **Config Scanning**: The script scans the specified configs and extracts the `save_folder` field
2. **Variable Interpolation**: Handles variable substitution like `${run_name}` in save paths
3. **WandB Detection**: Searches for wandb runs in `<save_folder>/wandb/wandb/` (WandB creates a nested structure)
4. **Latest Run Detection**: Uses the `latest-run` symlink if available, otherwise selects the most recently modified run
5. **Parallel Sync**: Uses ThreadPoolExecutor to sync multiple runs simultaneously
6. **Real-time Output**: Streams output from each sync process with color-coded prefixes

## Output Format

Each sync process outputs with a prefix showing which config and run it's syncing:

```
[OLMo-STU-All-150M:offline-run-20241029-123456] Starting sync...
[OLMo-STU-All-150M:offline-run-20241029-123456] View run at: https://wandb.ai/...
[OLMo-STU-All-150M:offline-run-20241029-123456] Sync completed successfully!
```

Colors indicate status:
- **Cyan**: Starting sync
- **Blue**: Sync progress/output
- **Green**: Success
- **Yellow**: Warnings
- **Red**: Errors

## Example Workflow

After running training jobs with offline wandb:

```bash
# 1. Activate your virtual environment
source OLMo-base/bin/activate

# 2. Login to wandb (if not already)
wandb login

# 3. Sync all runs from your ablation experiments
python offline_scripts/sync_wandb_runs.py --config-dir configs/stu-ablations

# 4. Check the summary to see if all synced successfully
```

## Troubleshooting

### "wandb command not found"
- Ensure WandB is installed: `pip install wandb`
- Verify your virtual environment is activated

### "No wandb runs found"
- Check that training has actually created runs in the save folder
- Verify the `save_folder` path in your config is correct
- Look for `<save_folder>/wandb/wandb/offline-run-*` directories (note the nested wandb structure)

### Sync failures
- Check your internet connection
- Verify you're logged in: `wandb login`
- Check WandB status at https://status.wandb.ai/

### Virtual environment issues
- The script inherits the current Python environment
- Always activate your venv before running the script
- The same venv used for training should have wandb installed

## Advanced Usage

### Sync all runs instead of just latest
By default, the script only syncs the latest run per config. To sync all runs:

```bash
python offline_scripts/sync_wandb_runs.py --config-dir configs/stu-ablations --all-runs
```

You can also use a dry-run to see what runs would be synced:

```bash
python offline_scripts/sync_wandb_runs.py --config-dir configs/stu-ablations --all-runs --dry-run
```

### Integration with SLURM
You can run this as a post-processing step after training:

```bash
# In your SLURM script, after training completes:
python offline_scripts/sync_wandb_runs.py --configs $CONFIG_PATH
```

Or submit as a separate job:

```bash
sbatch --wrap="source OLMo-base/bin/activate && python offline_scripts/sync_wandb_runs.py --config-dir configs/stu-ablations"
```

