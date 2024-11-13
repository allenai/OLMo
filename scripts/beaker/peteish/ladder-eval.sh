#!/usr/bin/env bash

set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift
NUM_NODES=$1
shift
NUM_GPUS=$1
shift
BEAKER_REPLICA_RANK=$1
shift
CHECKPOINT=$1
shift
SUFFIX=$1
shift
NUM_CHECKPOINTS=1
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

# # Move AWS credentials from env to relevant files
# mkdir -p ~/.aws
# printenv AWS_CONFIG > ~/.aws/config
# printenv AWS_CREDENTIALS > ~/.aws/credentials

# Force processes to synchronize at init_process_group
export TORCH_DIST_INIT_BARRIER=1

# Tell OLMo all ranks share the same filesystem for checkpoints.
export OLMO_SHARED_FS=1

export NCCL_DEBUG=INFO
export NCCL_IB_HCA="^=mlx5_bond_0"
export NCCL_SOCKET_IFNAME=ib
# export NCCL_IB_GID_INDEX=0

port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

torchrun \
  --nnodes "${NUM_NODES}:${NUM_NODES}" \
  --nproc-per-node "${NUM_GPUS}" \
  --rdzv_id 12347 \
  --rdzv_backend static \
  --rdzv_endpoint "${BEAKER_LEADER_REPLICA_HOSTNAME}:${port}" \
  --node_rank "${BEAKER_REPLICA_RANK}" \
  --rdzv_conf 'read_timeout=420' \
  scripts/eval.py \
    /weka/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/${CHECKPOINT}/step0-unsharded/config.yaml \
      --run_name="${GANTRY_TASK_NAME}" \
      --save_num_checkpoints_to_keep=$NUM_CHECKPOINTS \
      --save_folder="/weka/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/${CHECKPOINT}-${SUFFIX}" \
      --wandb.group="${CHECKPOINT}-${SUFFIX}" \
      --wandb.name="${CHECKPOINT}-${SUFFIX}" \
      --load_path="/weka/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/${CHECKPOINT}"