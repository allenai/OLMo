#!/usr/bin/env bash

set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1; shift
BEAKER_REPLICA_RANK=$1; shift
NUM_NODES=$1; shift
NUM_GPUS=$1; shift

CHECKPOINT=${CHECKPOINT:-""}
CONFIG=${CONFIG:-"${CHECKPOINT}/step0-unsharded/config.yaml"}
BACKFILL_SUFFIX=${BACKFILL_SUFFIX:-"backfill"}
NUM_CHECKPOINTS=${NUM_CHECKPOINTS:-"-1"}

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


function EPHEMERAL_PORT() {
    LOW_BOUND=49152
    RANGE=16384
    while true; do
        CANDIDATE=$[$LOW_BOUND + ($RANDOM % $RANGE)]
        (echo -n >/dev/tcp/127.0.0.1/${CANDIDATE}) >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo $CANDIDATE
            break
        fi
    done
}

if [ $NUM_NODES -eq 1 ]; then
    PORT=$(EPHEMERAL_PORT)
else
    PORT=29400
fi


torchrun \
  --nnodes "${NUM_NODES}:${NUM_NODES}" \
  --nproc-per-node "${NUM_GPUS}" \
  --rdzv_id 12347 \
  --rdzv_backend static \
  --rdzv_endpoint "${BEAKER_LEADER_REPLICA_HOSTNAME}:${PORT}" \
  --node_rank "${BEAKER_REPLICA_RANK}" \
  --rdzv_conf 'read_timeout=420' \
  scripts/eval.py \
    $CONFIG \
      --run_name="${GANTRY_TASK_NAME}" \
      --save_num_checkpoints_to_keep=$NUM_CHECKPOINTS \
      --save_folder="${CHECKPOINT}-${BACKFILL_SUFFIX}" \
      --load_path="${CHECKPOINT}" \
      --wandb.project="olmo-ladder" \
      --wandb.group="$(basename $CHECKPOINT)-${BACKFILL_SUFFIX}" \
      --wandb.name="$(basename $CHECKPOINT)-${BACKFILL_SUFFIX}" \
      --save_overwrite