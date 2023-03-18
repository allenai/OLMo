#!/bin/bash
#SBATCH --job-name=c4-1.2b
#SBATCH --account=project_462000229
#SBATCH --output=/pfs/lustref1/flash/project_462000229/logs/%j.log
#SBATCH --nodes=1               # Total number of nodes 
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=6       # 6 threads per ranks
#SBATCH --time=00:15:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --partition=small-g

module load LUMI/22.08 partition/G

CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
export NCCL_DEBUG=INFO
export PYTHONPATH=.:${PYTHONPATH}
export WANDB_PROJECT=lumi-${SLURM_JOB_PARTITION}

srun \
  --cpu-bind=${CPU_BIND} \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/run_with_environment.sh \
    singularity exec \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    $PROJECT_DIR/containers/llm-lumi_latest.sif \
    python scripts/train.py configs/1.2b-c4.yaml --run_name=${SLURM_JOB_ID}

