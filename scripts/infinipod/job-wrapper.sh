#!/usr/bin/env bash
#SBATCH --job-name=peteish1-wsd-anneal
#SBATCH --output=/home/common/shanea/logs/%u_%j_%n.log
#SBATCH --nodes=16              # Total number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=180
#SBATCH --time=96:00:00
#SBATCH --mem=0			# All memory on the node

srun --mpi=pmi2 --nodes="$SLURM_NNODES" ${@}