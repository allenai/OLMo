#!/usr/bin/env bash
#SBATCH --job-name=mup-lr-search
#SBATCH --account=project_462000229
#SBATCH --output=/scratch/project_462000229/logs/%j.log
#SBATCH --nodes=2              # Total number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --partition=standard-g

module load LUMI/24.03 partition/G

## Container-dependent settings
export OLMO_CONTAINER=$PROJECT_DIR/containers/lumi-torch25rc-rocm62-py312.sif
export ROCM_PATH=/opt/rocm
export CONDA_ENV=pytorch
export PYTHONPATH=.:${PYTHONPATH}

## General LUMI settings (these rarely change)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
export GPU_MAX_HW_QUEUES=8
# We need to set this to avoid "Cassini Event Queue overflow detected." errors.
export FI_CXI_DEFAULT_CQ_SIZE=131072

## Job settings
export CHECKPOINTS_PATH=$SCRATCH_DIR/checkpoints
export DATA_PATH=$SCRATCH_DIR
export HF_DATASETS_OFFLINE=1
export SINGULARITYENV_TORCH_DIST_INIT_BARRIER=1
# Try playing with max_split_size_mb if you run into OOM errors.
#export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128

## Debug settings
#export NCCL_DEBUG=INFO
#export FI_LOG_LEVEL=INFO

# Read mup params from script args
export WIDTH=$1
shift
export LR=$1
shift

srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/run_with_environment.sh \
    singularity exec \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B /var/spool/slurmd,/opt/cray/,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/usr/lib64/libjson-c.so.3 \
    $OLMO_CONTAINER \
    scripts/lumi/run-in-container.sh \
      python scripts/train.py configs/mup/peteish-1b-target.yaml \
      --run_name="mup_peteish_1b_${WIDTH}_${LR}" \
      --wandb.name="mup_peteish_1b_${WIDTH}_${LR}" \
      --wandb.group="mup_peteish_1b" \
      --wandb.project=olmo-mup \
      --scheduler.t_warmup=100 \
      --model.use_mup \
      --model.mup_query_zero_init=false \
      --model.mup_base_shapes=configs/mup/peteish-1b-target.bsh \
      --model.d_model=$WIDTH \
      --optimizer.learning_rate=$LR \
      --fsdp.sharding_strategy=HYBRID_SHARD \
      --save_interval_ephemeral=250 \
      --eval_interval=100 \
      --try_load_latest_save \
      --save_overwrite
      "${@}"
