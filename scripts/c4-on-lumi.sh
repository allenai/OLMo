#!/bin/bash
#SBATCH --job-name=c4-1.2b
#SBATCH --account=project_462000229
#SBATCH --output=/users/dgroeneveld/logs/%j.log
#SBATCH --nodes=1               # Total number of nodes 
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=2       # 2 threads per ranks
#SBATCH --time=00:15:00
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH --mail-user=dirkg@allenai.org
#SBATCH --mem=64G
#SBATCH --partition=small-g

module load LUMI/22.08 partition/G

CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

export OMP_NUM_THREADS=2
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

srun \
  --cpu-bind=${CPU_BIND} \
  --distribution=block:block \
  --kill-on-bad-exit \
  singularity exec \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    $PROJECT_DIR/containers/llm-lumi_latest.sif \
    scripts/run_with_environment.sh \
    python scripts/train.py configs/1.2b-c4-lumi.yaml

