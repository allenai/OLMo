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
curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -
popd
export HF_DATASETS_OFFLINE=1

# Force processes to synchronize at init_process_group
export TORCH_DIST_INIT_BARRIER=1

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=12347 \
  --rdzv_backend=static \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  --rdzv_conf="read_timeout=420" \
  scripts/train.py \
    configs/lr-scheduling-s3.yaml \
      --run_name=lr-scheduling-low-const-lr \
      --fsdp.sharding_strategy=SHARD_GRAD_OP \
      --load_path=weka://oe-training-default/ai2-llm/checkpoints/unsorted/6746551/step440000-unsharded/ \
      --remote_save_folder=weka://oe-training-default/ai2-llm/checkpoints/1b/lr-scheduling-low-const-lr \
      --wandb.name=lr-scheduling-low-const-lr \
      --wandb.group=lr-scheduling-low-const-lr \
      --scheduler.name=linear_with_warmup \
      --scheduler.t_warmup=440000 \
      --scheduler.alpha_f=0.0 \
      --optimizer.learning_rate=5.07e-5 \
      --save_overwrite
