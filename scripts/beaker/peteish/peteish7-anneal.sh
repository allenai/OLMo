#!/usr/bin/env bash

set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

BEAKER_REPLICA_RANK=$1
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
pip freeze

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

MAX_STEPS=12000 # a little over 50B tokens

torchrun \
  --nnodes "${NUM_NODES}:${NUM_NODES}" \
  --nproc-per-node 8 \
  --rdzv_id 12347 \
  --rdzv_backend static \
  --rdzv_endpoint "${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" \
  --node_rank "${BEAKER_REPLICA_RANK}" \
  --rdzv_conf 'read_timeout=420' \
  scripts/train.py \
    configs/peteish7-weka.yaml \
      --run_name="${GANTRY_TASK_NAME}" \
      --fsdp.sharding_strategy=HYBRID_SHARD \
      --fsdp.hybrid_sharding_num_model_replicas="${NUM_NODES}" \
      --activation_checkpointing=whole_layer \
      --load_path=/weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7/step928646 \
      --data.seed=23250 \
      --restore_dataloader=false \
      --optimizer.learning_rate=6.1852e-5 \
      --scheduler.name=linear_with_warmup \
      --scheduler.units=steps \
      --scheduler.t_warmup=0 \
      --scheduler.t_max=${MAX_STEPS} \
      --scheduler.alpha_f=0.0 \
      --max_duration=${MAX_STEPS} \
      --stop_at=${MAX_STEPS} \
      --save_interval_ephemeral=500 \
      --save_overwrite
