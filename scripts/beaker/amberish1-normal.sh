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
  --rdzv_id=12347 \
  --rdzv_backend=static \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  --rdzv_conf="read_timeout=420" \
  scripts/train.py \
  configs/amberish1-s3.yaml \
    --model.flash_attention=true \
    --gen1_gc_interval=null \
    --save_folder=runs/ \
    --activation_checkpointing=fine_grained \
    --fused_loss=false \
    --save_interval=250 \
    --eval_interval=250 \
    --optimizer.metrics_log_interval=1 \
    --save_overwrite \
    --save_num_checkpoints_to_keep=3 \
    # '--load_path=${path.last_checkpoint:s3://ai2-llm/checkpoints/OLMo-small/llm-306-amber-data-repro-db-normal-init-2}'
