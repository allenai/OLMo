#!/usr/bin/env python3
"""
Automatically generate and submit SLURM jobs for OLMo training configs.

Usage:
    # Single config file
    python run_slurm_job_config.py --config configs/tiny/OLMo-STU-20M.yaml
    python run_slurm_job_config.py --config configs/stu-ablations/OLMo-STU-All-150M.yaml --gpus 2
    python run_slurm_job_config.py --config configs/tiny/OLMo-STU-20M.yaml --dry-run
    
    # All configs in a folder
    python run_slurm_job_config.py --config configs/stu-ablations/
    python run_slurm_job_config.py --config configs/stu-wide-depth-ablations/ --gpus 2
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM jobs for OLMo training configs"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a config YAML file or folder containing YAML files (e.g., configs/tiny/OLMo-STU-20M.yaml or configs/stu-ablations/)",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Override job name (default: inferred from config)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="72:00:00",
        help="Max runtime (default: 72:00:00)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=12,
        help="CPUs per GPU (default: 12)",
    )
    parser.add_argument(
        "--mem",
        type=str,
        default=None,
        help="Memory per node (e.g., 32G, 64G). Default: auto-inferred from model size",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="pli",
        help="SLURM partition (default: pli)",
    )
    parser.add_argument(
        "--account",
        type=str,
        default="eladgroup",
        help="SLURM account (default: eladgroup)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1)",
    )
    parser.add_argument(
        "--venv",
        type=str,
        default="OLMo-base",
        help="Virtual environment name (default: OLMo-base)",
    )
    parser.add_argument(
        "--save-script",
        type=str,
        default=None,
        help="Save generated SLURM script to this path (instead of temp file)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the SLURM script without submitting it",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Generate the script but don't submit (implies --save-script if not set)",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra arguments to pass to train.py (e.g., '--save_overwrite --reset_optimizer_state')",
    )
    
    return parser.parse_args()


def find_yaml_files(path):
    """Find all YAML files in a directory or return single file if path is a file."""
    path_obj = Path(path)
    
    if path_obj.is_file():
        return [path_obj]
    elif path_obj.is_dir():
        # Find all .yaml and .yml files in the directory (not recursive)
        yaml_files = list(path_obj.glob("*.yaml")) + list(path_obj.glob("*.yml"))
        yaml_files.sort()  # Sort for consistent ordering
        return yaml_files
    else:
        return []


def read_config_yaml(config_path):
    """Read the YAML config file and extract relevant information."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not parse config file: {e}")
        return {}


def infer_memory_from_config(config):
    """Infer memory requirements from model configuration."""
    model = config.get("model", {})
    d_model = model.get("d_model", 256)
    n_layers = model.get("n_layers", 8)
    
    # Rough heuristics for memory requirements
    if d_model >= 768:  # 150M+ models
        return "64G"
    elif d_model >= 512:  # 50M-150M models
        return "48G"
    else:  # Smaller models
        return "32G"


def infer_job_name(config_path, config):
    """Infer job name from config or file path."""
    # Try to get run_name from config
    run_name = config.get("run_name")
    if run_name:
        # Simplify the run name for job name (remove special chars, truncate)
        job_name = run_name.replace("_", "-").replace(".", "-")
        # Truncate to avoid SLURM job name limits
        if len(job_name) > 30:
            job_name = job_name[:30]
        return job_name
    
    # Fallback: use config filename without extension
    config_file = Path(config_path).stem
    job_name = config_file.replace("_", "-")
    if len(job_name) > 30:
        job_name = job_name[:30]
    return job_name


def generate_slurm_script(args, config):
    """Generate SLURM script content."""
    
    # Infer job name if not provided
    job_name = args.job_name or infer_job_name(args.config, config)
    
    # Infer memory if not provided
    mem = args.mem or infer_memory_from_config(config)
    
    # Get the absolute path to the config
    config_path = Path(args.config).resolve()
    
    # Get OLMo directory (parent of offline_scripts)
    olmo_dir = Path(__file__).parent.parent.resolve()
    
    # Get config description for logging
    config_desc = config.get("run_name", config_path.name)
    
    # Build extra training args
    train_args = args.extra_args.strip()
    if not train_args:
        train_args = "--save_overwrite"
    elif "--save_overwrite" not in train_args:
        train_args = f"--save_overwrite {train_args}"
    
    # Generate SLURM script
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --time={args.time}
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --partition={args.partition}
#SBATCH --account={args.account}

# Print job information
echo "Starting job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "Config: {config_desc}"
echo "Config path: {config_path}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to the OLMo directory
cd {olmo_dir} || exit 1

# Activate the virtual environment
echo "Activating virtual environment: {args.venv}"
source {args.venv}/bin/activate

# Weights & Biases offline mode setup
export WANDB_MODE=offline
export WANDB_DIR={olmo_dir}/wandb_offline
mkdir -p "$WANDB_DIR"
echo "WandB will log offline to: $WANDB_DIR"

# Print environment info for debugging
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set unique rendezvous address/port to avoid collisions
export MASTER_ADDR=${{MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${{MASTER_PORT:-$((15000 + (SLURM_JOB_ID % 20000)))}}
echo "Rendezvous: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# Run the training
echo "Starting training at $(date)"
torchrun --nproc_per_node={args.gpus} \\
    --master_addr "$MASTER_ADDR" \\
    --master_port "$MASTER_PORT" \\
    scripts/train.py \\
    {config_path} \\
    {train_args}

# Print completion info
echo "Job finished at $(date)"
"""
    
    return script


def process_single_config(config_path, args):
    """Process a single config file: generate and optionally submit SLURM job."""
    # Read config file
    print(f"\nProcessing config: {config_path}")
    config = read_config_yaml(config_path)
    
    # Create a copy of args with the specific config path
    single_args = argparse.Namespace(**vars(args))
    single_args.config = str(config_path)
    
    # Generate SLURM script
    script = generate_slurm_script(single_args, config)
    
    # Handle dry-run mode
    if single_args.dry_run:
        print("\n" + "="*80)
        print(f"Generated SLURM script for {config_path.name} (dry-run mode):")
        print("="*80)
        print(script)
        print("="*80)
        return True
    
    # Determine if we should save the script
    save_script_path = single_args.save_script
    if single_args.no_submit and not save_script_path:
        # Auto-generate a save path
        job_name = single_args.job_name or infer_job_name(str(config_path), config)
        save_script_path = f"offline_scripts/generated_{job_name}.slurm"
    
    # Save or submit the script
    if save_script_path:
        # Save to specified path
        save_path = Path(save_script_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(script)
        os.chmod(save_path, 0o755)  # Make executable
        print(f"SLURM script saved to: {save_path}")
        
        if not single_args.no_submit:
            # Submit the saved script
            print(f"Submitting job with sbatch...")
            result = subprocess.run(
                ["sbatch", str(save_path)],
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error submitting job: {result.stderr}", file=sys.stderr)
                return False
    else:
        # Create temporary file and submit
        with tempfile.NamedTemporaryFile(mode='w', suffix='.slurm', delete=False) as f:
            f.write(script)
            temp_script = f.name
        
        try:
            os.chmod(temp_script, 0o755)
            print(f"Submitting job with sbatch...")
            result = subprocess.run(
                ["sbatch", temp_script],
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"Error submitting job: {result.stderr}", file=sys.stderr)
                return False
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_script)
            except:
                pass
    
    return True


def main():
    args = parse_args()
    
    # Check if config path exists
    if not os.path.exists(args.config):
        print(f"Error: Config path not found: {args.config}")
        sys.exit(1)
    
    # Find all YAML files (single file or directory)
    yaml_files = find_yaml_files(args.config)
    
    if not yaml_files:
        print(f"Error: No YAML files found in: {args.config}")
        sys.exit(1)
    
    # Report what we found
    if len(yaml_files) == 1:
        print(f"Found 1 config file: {yaml_files[0].name}")
    else:
        print(f"Found {len(yaml_files)} config files in {args.config}:")
        for yaml_file in yaml_files:
            print(f"  - {yaml_file.name}")
    
    # Process each config file
    successful = 0
    failed = 0
    
    for yaml_file in yaml_files:
        success = process_single_config(yaml_file, args)
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    if len(yaml_files) > 1:
        print("\n" + "="*80)
        print(f"Summary: {successful} successful, {failed} failed out of {len(yaml_files)} total")
        print("="*80)
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

