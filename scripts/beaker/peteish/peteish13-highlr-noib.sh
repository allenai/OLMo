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

# Force processes to synchronize at init_process_group
export TORCH_DIST_INIT_BARRIER=1

# Tell OLMo all ranks share the same filesystem for checkpoints.
export OLMO_SHARED_FS=1

export NCCL_DEBUG=INFO
export NCCL_IB_HCA="=mlx5_0"
#export NCCL_SOCKET_IFNAME=mlx5_bond_0
# export NCCL_IB_GID_INDEX=0

torchrun \
  --nnodes "${NUM_NODES}:${NUM_NODES}" \
  --nproc-per-node 8 \
  --rdzv_id 12349 \
  --rdzv_backend static \
  --rdzv_endpoint "${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" \
  --node_rank "${BEAKER_REPLICA_RANK}" \
  --rdzv_conf 'read_timeout=420' \
  scripts/train.py \
    configs/peteish13-weka.yaml \
      --run_name="${GANTRY_TASK_NAME}" \
      --save_interval_ephemeral=200 \
      --fsdp.sharding_strategy=FULL_SHARD \
      --save_folder="/weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/${GANTRY_TASK_NAME}" \
      --remote_save_folder=null \
      --wandb=null \
      --no_pre_train_checkpoint \
      --save_overwrite \
      --sharded_checkpointer=olmo_core \
      --device_train_microbatch_size=4 \
      --activation_checkpointing=whole_layer \
      --fused_loss=true \
      --model.flash_attention=true \
      --data.num_workers=8 \
      --optimizer.learning_rate=9.0e-4

     # '--load_path=${path.last_checkpoint:${save_folder}}' \
