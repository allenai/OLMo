#!/usr/bin/env bash
#SBATCH --job-name=peteish1-wsd-anneal
#SBATCH --output=/home/common/shanea/logs/%u_%j.log
#SBATCH --nodes=2              # Total number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --time-min=01:00:00
#SBATCH --mem=0			# All memory on the node

set -exuo pipefail

BASE_RUN_NAME=$1
shift

CONDA_ENV=$1
shift

LR=$1
shift

LOAD_STEP=$1
shift

ANNEAL_STEPS=$1
shift

# Read secrets from env file
set -a
set +x
source /home/common/shanea/.env
set -x
set +a

# Set up conda
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
pip freeze

# Infinipod specific environment
export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY="/dev/aperture_devices"
NCCL_LIB_DIR="/usr/local/nvidia/lib64" source /usr/local/nvidia/lib64/nccl-env-profile.sh
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:"$LD_LIBRARY_PATH"
export NCCL_SHIMNET_SHIM_LAYERS="unused"
export NCCL_TUNER_PLUGIN="none"
export NVIDIA_PYTORCH_VERSION=24.03
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Tell OLMo all ranks share the same filesystem for checkpoints.
export OLMO_SHARED_FS=1
# Redirect stdout and stderr so that we get a prefix with the node name
export NODENAME=$(hostname -s)
exec > >(trap "" INT TERM; sed -u "s/^/$NODENAME:$SLURM_LOCALID out: /")
exec 2> >(trap "" INT TERM; sed -u "s/^/$NODENAME:$SLURM_LOCALID err: /" >&2)

# Setup for multi-node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=34126 # This can be any free port
export GPUS_PER_NODE=8 # Please replace with number of GPUs per node

# Force processes to synchronize at init_process_group
export TORCH_DIST_INIT_BARRIER=1
# Better error handling from Python
export PYTHONFAULTHANDLER=1

# Job details
ANNEAL_NAME="${BASE_RUN_NAME}_anneal${LOAD_STEP}_${ANNEAL_STEPS}"
RUN_NAME=$ANNEAL_NAME-$(date -u +"%Y%m%d_%H%M%S")
SAVE_FOLDER=/mnt/checkpoints/shanea/checkpoints/OLMo-medium/$ANNEAL_NAME
mkdir -p $SAVE_FOLDER

MAX_STEPS=$(($ANNEAL_STEPS + $LOAD_STEP))

srun \
  --mpi=pmi2 \
  --nodes="$SLURM_NNODES" \
  torchrun \
    --nnodes $SLURM_NNODES \
    --nproc-per-node $GPUS_PER_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --node_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    scripts/train.py \
      configs/peteish1-infinipod.yaml \
        --run_name=$RUN_NAME \
        --wandb.group=$ANNEAL_NAME \
        --optimizer.learning_rate=$LR \
        --scheduler.t_warmup=$LOAD_STEP \
        --scheduler.name=linear_with_warmup \
        --scheduler.units=steps \
        --scheduler.alpha_f=0.0 \
        --scheduler.t_max=$MAX_STEPS \
        --stop_at=$(($MAX_STEPS + 101)) \
        --save_interval_ephemeral=250 \
        --eval_interval=1000 \
        --fsdp.sharding_strategy=HYBRID_SHARD \
        --fsdp.hybrid_sharding_num_model_replicas="${SLURM_NNODES}" \
        --fsdp.wrapping_strategy=by_block_and_size \
        --load_path="/mnt/checkpoints/shanea/checkpoints/OLMo-medium/$BASE_RUN_NAME/step$LOAD_STEP/" \
        --save_folder=$SAVE_FOLDER \
        --remote_save_folder=null \
        --try_load_latest_save \
        --save_overwrite \
        --sharded_checkpointer=olmo_core \
        --device_train_microbatch_size=4 \
        --device_eval_batch_size=8 \
        --compile.fullgraph=false \
        --fused_loss=false \
        --model.flash_attention=false \
        --data.num_workers=$SLURM_CPUS_PER_TASK \
        --optimizer.metrics_log_interval=10 \
        --data.prefetch_factor=8
