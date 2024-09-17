#!/bin/bash
#SBATCH -J olmo-tulu           # Job name
#SBATCH -o slurm-outputs/olmo-tulu.o%j       # Name of stdout output file
#SBATCH -e slurm-outputs/olmo-tulu.e%j       # Name of stderr output file
#SBATCH -p gh          # Queue (partition) name
#SBATCH -N 8               # Total # of nodes
#SBATCH -n 1               # Total # of mpi tasks
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)
#SBATCH -A AST24021       # Allocation name (req'd if you have more than 1)

# conda init
# module purge
module load cuda/12.4
module list

path_to_data="/home1/09636/zyliu/work/OLMo/olmo_data/tulu"
# path_to_checkpoint="/home1/09636/zyliu/work/shared_resources/models/OLMo-7B-final"
# path_to_checkpoint="/home1/09636/zyliu/work/shared_resources/models/OLMo-7B"
# path_to_train_config="configs/official/OLMo-7B-tulu.yaml"
path_to_train_config="${path_to_checkpoint}/config.yaml"
path_to_checkpoint="https://olmo-checkpoints.org/ai2-llm/olmo-medium/p067ktg9/step558223-unsharded/"

# conda activate astro
export NCCL_DEBUG=WARN
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO

# torchrun --nproc_per_node=1 cdscripts/train.py ${path_to_train_config} \
#     --data.paths=[${path_to_data}/input_ids.npy] \
#     --data.label_mask_paths=[${path_to_data}/label_mask.npy] \
#     --load_path=${path_to_checkpoint} \
#     --reset_trainer_state
    
torchrun --nproc_per_node=1 \
    scripts/train.py \
    configs/mitchish7-local.yaml \
    --reset_trainer_state \
    --wandb=null \
    --remote_save_folder=null \
    --save_overwrite \
    --reset_optimizer_state \
    --data.paths=[${path_to_data}/input_ids.npy] \
    --data.label_mask_paths=[${path_to_data}/label_mask.npy] \
    --load_path=${path_to_checkpoint}



# python scripts/train.py ${path_to_train_config} \
#     --data.paths=${path_to_data}/input_ids.npy \
#     --data.label_mask_paths=${path_to_data}/label_mask.npy \
#     --load_path=${path_to_checkpoint} \
#     --reset_trainer_state