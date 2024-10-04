#!/bin/bash
#SBATCH -J olmo-tulu           # Job name
#SBATCH -o slurm-outputs/olmo-tulu.o%j       # Name of stdout output file
#SBATCH -e slurm-outputs/olmo-tulu.e%j       # Name of stderr output file
#SBATCH -p gh          # Queue (partition) name
#SBATCH -N 1              # Total # of nodes
##SBATCH --ntasks-per-node=1 
#SBATCH -t 00:30:00        # Run time (hh:mm:ss)
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
# path_to_checkpoint="https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step738020-unsharded/"
path_to_checkpoint=https://olmo-checkpoints.org/ai2-llm/olmo-medium/p067ktg9/step558223-unsharded

# conda activate astro
# export NCCL_DEBUG=WARN
export NCCL_DEBUG=INFO
export NODENAME=$(hostname -s)
# export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
# export MASTER_PORT=39591
# export WORLD_SIZE=$SLURM_NNODES
export RANK=$SLURM_PROCID
export FS_LOCAL_RANK=$SLURM_PROCID
export LOCAL_WORLD_SIZE=1 # $SLURM_NTASKS_PER_NODE
export LOCAL_RANK=0 # $SLURM_LOCALID
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NNODES
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************

# zoom zoom - recommended from lightning
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32


# configs/olmo-small-3T-lower-lr-local.yaml \
torchrun --nproc_per_node=1 \
    scripts/train.py \
    configs/mitchish-tulu-local.yaml \
    --reset_trainer_state \
    --remote_save_folder=null \
    --save_overwrite \
    --reset_optimizer_state \
    --data.paths=[${path_to_data}/input_ids.npy] \
    --data.label_mask_paths=[${path_to_data}/label_mask.npy] \
    --load_path=${path_to_checkpoint} \
    --max_duration=15 \
    --stop_at=15 \
    # --evaluators=null
    # --wandb=null \
# --rdzv_conf 'read_timeout=420' \
    # --nnodes="${WORLD_SIZE}" \
    # --rdzv_id 12349 \
    # --rdzv_backend static \
    # --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    # --master_addr ${MASTER_ADDR} \
    

# python scripts/train.py ${path_to_train_config} \
#     --data.paths=${path_to_data}/input_ids.npy \
#     --data.label_mask_paths=${path_to_data}/label_mask.npy \
#     --load_path=${path_to_checkpoint} \
#     --reset_trainer_state