#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

BEAKER_REPLICA_RANK=$1
shift

# Warm HF cache
mkdir -p /root/.cache
pushd /root/.cache
curl "https://storage.googleapis.com/hf-cache/huggingface_cache_v4.tar.gz" | tar --keep-newer-files -xzf -
popd
export HF_DATASETS_OFFLINE=1

# base-olmo
torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  scripts/train.py \
    configs/mup/base-olmo.yaml \
      --run_name="sp_olmo_128" \
      --wandb.name="sp_olmo_128" \
      --wandb.group="sp_olmo" \
      --wandb.project=olmo-mup \
      --model.width=128


torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  scripts/train.py \
    configs/mup/base-olmo.yaml \
      --run_name="mup_olmo_128" \
      --wandb.name="mup_olmo_128" \
      --wandb.group="mup_olmo" \
      --wandb.project=olmo-mup \
      --model.use_mup \
      --model.width=128