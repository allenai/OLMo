#!/bin/bash
#SBATCH --job-name=olmo-small
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
export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

export PYTHONPATH=.:${PYTHONPATH}

# check if LOAD_PATH is provided as an environment variable;
# if so, create an argument to pass to the training script
if [ -z ${LOAD_PATH+x} ]; then
  LOAD_PATH_ARG=""
else
  LOAD_PATH_ARG="--load_path=${LOAD_PATH}"
fi

# check if CONFIG_PATH is provided as an environment variable;
# if so, use that instead of olmo-small-ablation.yaml
if [ -z ${CONFIG_PATH+x} ]; then
  export CONFIG_PATH="configs/olmo-small-ablation.yaml"
else
  export CONFIG_PATH="${CONFIG_PATH}"
fi

if [ -z ${RUN_NAME+x} ]; then
  # get run name, we will postpend it with the job id of this slurm run
  export RUN_NAME=$(cat $CONFIG_PATH | grep -ohP "^run_name\:\w*(.+)$" | sed 's/run_name:\s*//')
else
  # override passed from the environment
  export RUN_NAME="${RUN_NAME}"
fi

# get W&B settings from the config file, then extract the project and group
WANDB_SETTINGS=$(cat $CONFIG_PATH |  tr '\n' '\r' | grep -ohP "\rwandb:\r.*?\r\r"  | tr '\r' '\n')
export WANDB_PROJECT=$(echo $WANDB_SETTINGS | grep -ohP "\w*project\:\s*\S+(\s|$)" | sed 's/project:\s*//')

# check if W&B is provided; if not, use the run name as the project name
# (the actual run rame with have slurm ID appended)
export WANDB_GROUP=$(echo $WANDB_SETTINGS | grep -ohP "\w*group\:\w*(.+)" | sed 's/group:\s*//')
if [[ -z "${WANDB_GROUP}" ]]; then
  export WANDB_GROUP="${RUN_NAME}"
fi

# actually run the training script
srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/run_with_environment.sh \
    python scripts/train.py $CONFIG_PATH \
      --run_name="${RUN_NAME}_${SLURM_JOB_ID}" \
      --wandb.project="${WANDB_PROJECT}" \
      --wandb.group="${WANDB_GROUP}" \
      --wandb.name="${RUN_NAME}_${SLURM_JOB_ID}" \
      $LOAD_PATH_ARG \
      ${@}
