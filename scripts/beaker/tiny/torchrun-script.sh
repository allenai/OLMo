#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

TASK_NAME=$1
shift

# Warm HF cache
mkdir -p /root/.cache
pushd /root/.cache
curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -
popd
export HF_DATASETS_OFFLINE=1

# olmo-300M
torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  scripts/train.py \
    configs/tiny/OLMo-300M.yaml \
      --run_name=$TASK_NAME \
      --wandb.name=$TASK_NAME \
      --wandb.group=$TASK_NAME \
      --wandb.project=alt_arch \
      --device_train_microbatch_size=16 \
      --distributed_strategy=ddp \
      --ddp.grad_sync_mode=batch \
      --model.init_fn=normal \
      --save_overwrite