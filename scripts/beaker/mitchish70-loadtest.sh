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
curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v2.tar.gz" | tar --keep-newer-files -xzf -
popd
export HF_DATASETS_OFFLINE=1

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  scripts/train.py \
  configs/mitchish70-s3.yaml \
    --run_name=mitchish70-loadtest \
    --wandb.name=mitchish70-loadtest \
    --model.flash_attention=true \
    --fsdp.wrapping_strategy=by_block_and_size \
    --save_folder=runs/ \
    --fused_loss=true \
    --device_train_microbatch_size=2 \
    --global_train_batch_size=512 \
    --save_overwrite \
    --remote_save_folder=null \
    --load_path=s3://ai2-llm/checkpoints/OLMo-large/mitchish70-002/step32300-unsharded