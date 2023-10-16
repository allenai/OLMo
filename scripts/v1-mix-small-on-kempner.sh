#!/bin/bash
#SBATCH --job-name=v1-mix-small
#SBATCH --account=kempner_pi_lab
#SBATCH --output=/n/home06/dgroeneveld/logs/%j.log
#SBATCH --nodes=2              # Total number of nodes 
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --time-min=24:00:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --partition=kempner

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export PYTHONPATH=.:${PYTHONPATH}

# Try playing with max_split_size_mb if you run into OOM errors.
# export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/run_with_environment.sh \
    conda run -n LLM --live-stream \
      python scripts/train.py configs/v1-mix-small-mcli.yaml --run_name=kempner_${SLURM_JOB_ID} ${@}
