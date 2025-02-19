#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

TASK_NAME=$1
shift

CONFIG_PATH=$1
shift

OPTIMIZER=$1
shift

MUON_LR=$1
shift

MUON_WEIGHT_DECAY=$1
shift

MUON_MOMENTUM=$1
shift

# Warm HF cache
mkdir -p /root/.cache
pushd /root/.cache
curl "https://storage.googleapis.com/hf-cache/huggingface_cache_v4.tar.gz" | tar --keep-newer-files -xzf -
popd
export HF_DATASETS_OFFLINE=1

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 4 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29418 \
  scripts/train.py \
    $CONFIG_PATH \
      --run_name=$TASK_NAME \
      --wandb.name=$TASK_NAME \
      --wandb.group=$TASK_NAME \
      --wandb.project=olmo-optimizers \
      --optimizer.name=$OPTIMIZER \
      --optimizer.learning_rate=3e-3 \
      --optimizer.muon_lr=$MUON_LR \
      --optimizer.muon_weight_decay=$MUON_WEIGHT_DECAY \
      --optimizer.muon_momentum=$MUON_MOMENTUM \
      --save_overwrite