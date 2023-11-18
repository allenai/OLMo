#!/bin/bash
#SBATCH --job-name=olmo-small
#SBATCH --account=project_462000229
#SBATCH --output=/pfs/lustref1/flash/project_462000229/logs/%j.log
#SBATCH --nodes=16              # Total number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --partition=standard-g

module load LUMI/22.08 partition/G

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

export OLMO_CONTAINER=llm-lumi_latest.sif

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

# We need to set this to avoid "Cassini Event Queue overflow detected." errors.
export FI_CXI_DEFAULT_CQ_SIZE=131072

#export NCCL_DEBUG=INFO
export PYTHONPATH=.:${PYTHONPATH}
export ROCM_PATH=/opt/rocm
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64

# Try playing with max_split_size_mb if you run into OOM errors.
# export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512


if [ -z ${RUN_NAME+x} ]; then
  # get run name, we will postpend it with the job id of this slurm run
  export RUN_NAME=$(cat $CONFIG_PATH | grep -ohP "^run_name\:\w*(.+)$" | sed 's/run_name:\s*//')
else
  # override passed from the environment
  export RUN_NAME="${RUN_NAME}"
fi

# get W&B settings from the config file, then extract the project and group
WANDB_SETTINGS=$(cat $CONFIG_PATH |  tr '\n' '\r' | grep -ohP "\rwandb:\r.*?\r\r"  | tr '\r' '\n')
export WANDB_PROJECT=$(echo $WANDB_SETTINGS | grep -ohP "\w*project\:\s*\S+\s" | sed 's/project:\s*//')

# check if W&B is provided; if not, use the run name as the project name
# (the actual run rame with have slurm ID appended)
export WANDB_GROUP=$(echo $WANDB_SETTINGS | grep -ohP "\w*group\:\w*(.+)" | sed 's/group:\s*//')
if [[ $WANDB_GROUP -eq "" ]]; then
  export WANDB_GROUP="${RUN_NAME}"
fi

# actually run the training script
srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/run_with_environment.sh \
    singularity exec \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    -B /opt/cray:/opt/cray \
    -B /usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjson-c.so.3:/usr/lib64/libjson-c.so.3 \
    $PROJECT_DIR/containers/$OLMO_CONTAINER \
    python scripts/train.py $CONFIG_PATH \
      --run_name="${RUN_NAME}_${SLURM_JOB_ID}" \
      --wandb.project="${WANDB_PROJECT}" \
      --wandb.group="${WANDB_GROUP}" \
      --wandb.name="${RUN_NAME}_${SLURM_JOB_ID}" \
      $LOAD_PATH_ARG \
      ${@}
