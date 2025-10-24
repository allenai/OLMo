#!/bin/bash
#SBATCH --job-name=olmo-train          # Name of your job
#SBATCH --output=logs/train_%j.out     # Standard output log (%j = job ID)
#SBATCH --error=logs/train_%j.err      # Standard error log
#SBATCH --time=72:00:00                # Max runtime (72 hours -> gpu-medium QOS)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=12             # CPUs per GPU on 80GB nodes (max 12 per GPU)
#SBATCH --mem=200G                     # Memory (80GB nodes have 240GB per GPU)
#SBATCH --gres=gpu:1                   # Number of GPUs
#SBATCH --partition=pli-c              # Use PLI partition (H100 80GB GPUs)

# Print job information
echo "Starting job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to the OLMo directory
cd /home/tt6444/ssm-attention-routing/OLMo || exit 1

# Activate the uv environment
echo "Activating uv environment: OLMo-base"
source OLMo-base/bin/activate 
# If using uv to run, you could alternatively use: uv run --env OLMo-base <command>


# Note: Your config has num_workers=32 but we only requested 12 CPUs.
# The dataloader will use available CPUs. If you get CPU bottlenecks,
# you can request more CPUs (up to 12 per GPU on 80GB nodes) or reduce
# num_workers in your config to match SLURM_CPUS_PER_TASK.

# Weights & Biases offline mode setup
export WANDB_MODE=offline                           # Run in offline mode
export WANDB_DIR=/home/tt6444/ssm-attention-routing/OLMo/wandb_offline  # Where to store offline runs
mkdir -p $WANDB_DIR
echo "WandB will log offline to: $WANDB_DIR"

# Print environment info for debugging
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Run the training
echo "Starting training at $(date)"
torchrun --nproc_per_node=1 \
    scripts/train.py \
    configs/OLMo2-1B-10B-tokens-local.yaml \
    --save_overwrite

# Print completion info
echo "Job finished at $(date)"
