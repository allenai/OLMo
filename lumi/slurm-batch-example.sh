#!/bin/bash
#SBATCH --job-name=exampleJob
#SBATCH --account=project_462000229
#SBATCH --output=examplejob.o%j # Name of stdout output file
#SBATCH --error=examplejob.e%j  # Name of stderr error file
#SBATCH --nodes=1               # Total number of nodes 
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=2       # 2 threads per ranks
#SBATCH --time=00:15:00
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH --mail-user=dirkg@allenai.org
#SBATCH --mem=8G
#SBATCH --partition=small-g

module load LUMI/22.08 partition/G
module load singularity-bindings
module load aws-ofi-rccl

module load MyApp/1.2.3

CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

export OMP_NUM_THREADS=2
export MPICH_GPU_SUPPORT_ENABLED=1

srun --cpu-bind=${CPU_BIND} --distribution=block:block ./run_with_slurm_device.sh <executable> <args>

