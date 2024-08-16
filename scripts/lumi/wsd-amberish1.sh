#!/bin/bash
#SBATCH --job-name=const-lr-amberish1
#SBATCH --account=project_462000229
#SBATCH --output=/pfs/lustref1/flash/project_462000229/logs/%j.log
#SBATCH --nodes=64              # Total number of nodes 
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --partition=standard-g

module load LUMI/23.09 partition/G
# module load PyTorch/2.2.2-rocm-5.6.1-python-3.10-singularity-20240617

export OLMO_CONTAINER=llm-lumi-torch22_latest.sif

export CACHED_PATH_CACHE_ROOT=${SCRATCH_DIR}/tmp/${USER}
export SIF_CONTAINER=$PROJECT_DIR/containers/$OLMO_CONTAINER
# export SIF_CONTAINER=$SIF

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

#export NCCL_DEBUG=INFO
export PYTHONPATH=.:${PYTHONPATH}
export ROCM_PATH=/opt/rocm
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64
# export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64:/opt/rocm/lib
export SINGULARITYENV_TORCH_DIST_INIT_BARRIER=1

# Try playing with max_split_size_mb if you run into OOM errors.
#export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128

export HF_DATASETS_OFFLINE=1

export CHECKPOINTS_PATH=$SCRATCH_DIR/$USER/checkpoints
# export DATA_PATH=$FLASH_DIR/preprocessed/olmo-mix
# export EVAL_DATA_PATH=$SCRATCH_DIR/eval-data

srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/run_with_environment.sh \
    singularity exec \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B /usr/lib64/libjson-c.so.3:/usr/lib64/libjson-c.so.3 \
    $SIF_CONTAINER \
    python scripts/train.py configs/wsd-amberish1-s3.yaml \
      --run_name=wsd-amberish1_${SLURM_JOB_ID} \
      --wandb.name=wsd-amberish1_${SLURM_JOB_ID} \
      '--load_path=${path.last_checkpoint:${save_folder}}' \
      --data.num_workers=$SLURM_CPUS_PER_TASK \
      --data.prefetch_factor=2 \
      --save_overwrite \
      ${@}