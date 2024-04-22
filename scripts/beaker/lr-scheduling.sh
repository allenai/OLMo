#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

# Warm HF cache
mkdir -p /root/.cache
pushd /root/.cache
curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -
popd
export HF_DATASETS_OFFLINE=1

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  scripts/train.py \
    configs/lr-scheduling-s3.yaml \
      --run_name=olmo-small-linear-decay-step446000-match40000steps \
      --fsdp.sharding_strategy=SHARD_GRAD_OP \
      --load_path=s3://ai2-llm/checkpoints/olmo-small-3T-lower-lr-tie/5076629/step446000-unsharded/ \
      --wandb.name=olmo-small-linear-decay-step446000-match40000steps \
      --wandb.group=olmo-small-linear-decay-step446000-match40000steps \
      --scheduler.name=linear_with_warmup \
      --scheduler.t_warmup=446000 \
      --scheduler.alpha_f=0.0 \
      --optimizer.learning_rate=1.633e-4 \
      --max_duration=472025 \
      --save_overwrite
