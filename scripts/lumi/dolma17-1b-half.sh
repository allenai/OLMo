#!/bin/bash
#SBATCH --job-name=dolma17-1b
#SBATCH --account=project_462000229
#SBATCH --output=/pfs/lustref1/flash/project_462000229/logs/%j.log
#SBATCH --nodes=16              # Total number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --partition=standard-g



export CACHED_PATH_CACHE_ROOT="${SCRATCH_DIR}/tmp"
mkdir -p "${CACHED_PATH_CACHE_ROOT}"

# check if CONFIG_PATH is provided as an environment variable;
# if not set, use a default value
if [ -z ${CONFIG_PATH+x} ]; then
  export CONFIG_PATH="configs/road-to-1_7/runs/r70b-baseline-sources-1b-150b.yaml"
else
  export CONFIG_PATH=${CONFIG_PATH}
fi

export OLMO_CONTAINER=llm-lumi-torch21_latest.sif

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

if [ -z "${HF_DATASETS_OFFLINE}" ]; then
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE}"
fi

# check if LOAD_PATH is provided as an environment variable;
# if so, create an argument to pass to the training script
if [ -z ${LOAD_PATH+x} ]; then
  LOAD_PATH_ARG=""
else
  LOAD_PATH_ARG="--load_path=${LOAD_PATH}"
fi

# We need to set this to avoid "Cassini Event Queue overflow detected." errors.
export FI_CXI_DEFAULT_CQ_SIZE=131072

#export NCCL_DEBUG=INFO
export PYTHONPATH=.:${PYTHONPATH}
export ROCM_PATH=/opt/rocm
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64

# Try playing with max_split_size_mb if you run into OOM errors.
#export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128

export DATA_PATH=$FLASH_DIR/preprocessed/olmo-mix
export CHECKPOINTS_PATH=$SCRATCH_DIR/checkpoints
export EVAL_DATA_PATH=$SCRATCH_DIR/eval-data

export RUN_NAME=$(cat $CONFIG_PATH | grep -ohP "^run_name\:\w*(.+)$" | sed 's/run_name:\s*//')

srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/run_with_environment.sh \
    singularity exec \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B /opt/cray:/opt/cray \
    -B /usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjson-c.so.3:/usr/lib64/libjson-c.so.3 \
    $PROJECT_DIR/containers/$OLMO_CONTAINER \
    python scripts/train.py $CONFIG_PATH \
      --run_name=${SLURM_JOB_ID} \
      --wandb.group="${RUN_NAME}-half" \
      --time_limit=$((11 * 60 * 60)) \
      --device_train_microbatch_size=8 \
      --global_train_batch_size=1024 \
      --fsdp.sharding_strategy=FULL_SHARD \
      --fsdp.wrapping_strategy=by_block \
      --save_interval=1000 \
      --save_interval_ephemeral=1000000 \
      --save_interval_unsharded=5000 \
      $LOAD_PATH_ARG \
      ${@}
