# run_slurm_job_config.py

A Python script to automatically generate and submit SLURM jobs for OLMo training configs.

## Quick Start

```bash
# Submit a job for a single config file
python offline_scripts/run_slurm_job_config.py --config configs/stu-ablations/OLMo-STU-All-150M.yaml

# Submit jobs for all configs in a folder
python offline_scripts/run_slurm_job_config.py --config configs/stu-ablations/

# Preview what will be generated (dry run)
python offline_scripts/run_slurm_job_config.py --config configs/tiny/OLMo-STU-20M.yaml --dry-run

# Generate script without submitting
python offline_scripts/run_slurm_job_config.py --config configs/tiny/OLMo-STU-20M.yaml --no-submit
```

## Features

- **Single file or folder**: Submit jobs for a single config file or all YAML files in a folder
- **Automatic inference**: Job name, memory requirements, and other parameters are inferred from the config file
- **Flexible overrides**: All SLURM parameters can be overridden via command-line arguments
- **Safe defaults**: Based on the most recent SLURM job templates (run_stu_sandwich_all_layers.slurm)
- **Absolute paths**: Uses absolute paths for config files to avoid path issues
- **WandB offline**: Automatically configured for offline logging

## Usage Examples

### Basic submission (single config)
```bash
python offline_scripts/run_slurm_job_config.py --config configs/stu-ablations/OLMo-STU-All-150M.yaml
```

### Submit all configs in a folder
```bash
# Submit jobs for all YAML files in a folder
python offline_scripts/run_slurm_job_config.py --config configs/stu-ablations/

# With custom settings for all configs in the folder
python offline_scripts/run_slurm_job_config.py --config configs/stu-wide-depth-ablations/ --gpus 2 --time 48:00:00
```

### Multi-GPU training
```bash
python offline_scripts/run_slurm_job_config.py --config configs/stu-ablations/OLMo-STU-All-150M.yaml --gpus 2
```

### Override memory and time
```bash
python offline_scripts/run_slurm_job_config.py \
    --config configs/tiny/OLMo-STU-20M.yaml \
    --mem 48G \
    --time 48:00:00
```

### Custom job name
```bash
python offline_scripts/run_slurm_job_config.py \
    --config configs/tiny/OLMo-STU-20M.yaml \
    --job-name my-experiment
```

### Save script without submitting
```bash
python offline_scripts/run_slurm_job_config.py \
    --config configs/tiny/OLMo-STU-20M.yaml \
    --save-script offline_scripts/my_job.slurm \
    --no-submit
```

### Pass extra training arguments
```bash
python offline_scripts/run_slurm_job_config.py \
    --config configs/tiny/OLMo-STU-20M.yaml \
    --extra-args "--reset_optimizer_state --reset_trainer_state"
```

## Command-line Arguments

### Required
- `--config`: Path to a config YAML file or folder containing YAML files

### Optional SLURM Parameters
- `--job-name`: Override job name (default: inferred from config's `run_name`)
- `--time`: Max runtime (default: 72:00:00)
- `--gpus`: Number of GPUs (default: 1)
- `--nodes`: Number of nodes (default: 1)
- `--cpus-per-task`: CPUs per GPU (default: 12)
- `--mem`: Memory per node (default: auto-inferred from model size)
- `--partition`: SLURM partition (default: pli)
- `--account`: SLURM account (default: eladgroup)

### Other Options
- `--venv`: Virtual environment name (default: OLMo-base)
- `--extra-args`: Extra arguments to pass to train.py
- `--save-script`: Save generated SLURM script to this path
- `--dry-run`: Print the SLURM script without submitting
- `--no-submit`: Generate the script but don't submit

## Automatic Inference

The script automatically infers:

1. **Job name**: Extracted from `run_name` field in the config YAML
2. **Memory**: Based on model size (`d_model` in config):
   - `d_model >= 768`: 64G (150M+ models)
   - `d_model >= 512`: 48G (50M-150M models)
   - Otherwise: 32G (smaller models)

## Default Behavior

- Saves logs to `logs/{job_name}_{job_id}.out` and `logs/{job_name}_{job_id}.err`
- Activates the `OLMo-base` virtual environment
- Configures WandB in offline mode
- Automatically adds `--save_overwrite` flag to training command
- Uses absolute paths for config files to avoid path issues
- Sets unique rendezvous ports to avoid multi-job collisions

## Tips

1. **Always use dry-run first** to preview what will be generated:
   ```bash
   python offline_scripts/run_slurm_job_config.py --config <your-config> --dry-run
   ```

2. **Check logs directory** exists or will be created:
   ```bash
   mkdir -p logs
   ```

3. **For debugging**, save the script and inspect it:
   ```bash
   python offline_scripts/run_slurm_job_config.py \
       --config <your-config> \
       --save-script debug.slurm \
       --no-submit
   ```

4. **Monitor your job**:
   ```bash
   squeue -u $USER
   tail -f logs/{job_name}_{job_id}.out
   ```

