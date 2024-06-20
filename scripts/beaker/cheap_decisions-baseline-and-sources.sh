#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

BEAKER_REPLICA_RANK=$1
shift

source scripts/beaker/warm_hf_cache.sh

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=12347 \
  --rdzv_backend=static \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  --rdzv_conf="read_timeout=420" \
  scripts/train.py \
  configs/mitchish1-s3-cheap-decisions-r70b-baseline-sources.yaml \
    --load_path='${path.last_checkpoint:s3://ai2-llm/checkpoints/cheap_decisions/dolma-v1-6-and-sources-baseline-1B-N-1T-D-mitchish1-001}' \
    --model.flash_attention=true \
    # --save_folder=runs/ \
    # --save_overwrite \
    --gen1_gc_interval=10