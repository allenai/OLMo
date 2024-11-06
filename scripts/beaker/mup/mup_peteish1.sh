#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

BEAKER_REPLICA_RANK=$1
shift

WIDTH=$1
shift

LR=$1
shift

# Setup Python environment.
conda shell.bash activate base

# Install flash-attn
#conda install -y -c nvidia cuda-python
pip install packaging ninja
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
pip install flash-attn==2.5.9.post1 --no-build-isolation
# pip install awscli
pip install '.[train]'
pip install '.[scaling]'

pip freeze

export HF_DATASETS_OFFLINE=1

# Move AWS credentials from env to relevant files
mkdir -p ~/.aws
printenv AWS_CONFIG > ~/.aws/config
printenv AWS_CREDENTIALS > ~/.aws/credentials

# Force processes to synchronize at init_process_group
export TORCH_DIST_INIT_BARRIER=1

# Tell OLMo all ranks share the same filesystem for checkpoints.
export OLMO_SHARED_FS=1

export NCCL_DEBUG=INFO
export NCCL_IB_HCA="^=mlx5_bond_0"
export NCCL_SOCKET_IFNAME=ib
# export NCCL_IB_GID_INDEX=0

export CHECKPOINTS_PATH=/weka/oe-training-default
export DATA_PATH=/weka/oe-training-default

export RUN_NAME="peteish1_${WIDTH}_${LR}"
export GROUP_NAME="peteish1_100Btokens"

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  scripts/train.py configs/peteish1.yaml \
    --run_name=$RUN_NAME \
    --wandb.name=$RUN_NAME \
    --wandb.group=$GROUP_NAME \
    --wandb.project=olmo-mup \
    --load_path="${CHECKPOINTS_PATH}/ai2-llm/checkpoints/OLMo-mup/peteish1_2048_1.56e-2/step0" \
    --save_folder="${CHECKPOINTS_PATH}/ai2-llm/checkpoints/OLMo-mup/${RUN_NAME}" \
    --model.use_mup \
    --model.mup_query_zero_init=false \
    --model.mup_base_shapes=configs/peteish1.bsh \
    --model.mup_base_n_heads=1 \
    --model.d_model=$WIDTH \
    --optimizer.learning_rate=$LR \
    --fsdp.sharding_strategy=SHARD_GRAD_OP \
    --save_interval_ephemeral=250 \
    --scheduler.t_warmup=1000 \
    --scheduler.t_max=50000 \
    --stop_at=50000 \
    --eval_interval=500 \
    --try_load_latest_save \
    --save_overwrite \
    "${@}"

# --load_path="${CHECKPOINTS_PATH}/ai2-llm/checkpoints/OLMo-mup/peteish1_2048_1.56e-2/step0" \
# --load_path="${CHECKPOINTS_PATH}/ai2-llm/checkpoints/OLMo-mup/peteish1_v2_512_2.44e-4/step0" \