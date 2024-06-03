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

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=52346 \
  --rdzv_backend=static \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  --rdzv_conf="read_timeout=420" \
  scripts/train.py \
    configs/mitchish70-s3.yaml \
      --run_name=mitchish70-from160510 \
      '--wandb.group=${run_name}' \
      '--load_path=${path.last_checkpoint:${remote_save_folder}}' \
      --load_path_sharded_checkpointer=olmo_core \
      --sharded_checkpointer=olmo_core \
      --global_train_batch_size=3584 \
      --device_train_microbatch_size=4 \
      --fsdp.sharding_strategy=FULL_SHARD \
      --save_overwrite \
      --optimizer.learning_rate=3.0e-05 \
      --scheduler.alpha_f=1.0 \
      --scheduler.t_warmup=0
