#!/bin/bash
#SBATCH --job-name=llama7
#SBATCH --account=kempner_lab
#SBATCH --output=/n/holyscratch01/kempner_lab/Lab/logs/%j.log
#SBATCH --nodes=16              # Total number of nodes
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=16
#SBATCH --time=167:00:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --partition=kempner_project

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export PYTHONPATH=.:${PYTHONPATH}

# Try playing with max_split_size_mb if you run into OOM errors.
# export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

export DATA_PATH=/n/home06/dgroeneveld/data/preprocessed/olmo-mix
export EVAL_DATA_PATH=/n/home06/dgroeneveld/data/eval-data
export CHECKPOINTS_PATH=/n/home06/dgroeneveld/checkpoints

export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/run_with_environment.sh \
    $HOME/miniconda3/envs/LLM/bin/python -u scripts/train.py configs/llama7.yaml \
      --run_name=kempner_llama7_${SLURM_JOB_ID} \
      --save_folder=/n/holyscratch01/kempner_lab/Lab/checkpoints/${SLURM_JOB_ID}/ \
      --data.num_workers=4 \
      --device_train_microbatch_size=6 \
      --time_limit=$((167 * 60 * 60)) \
      --model.flash_attention=true \
      ${@}
