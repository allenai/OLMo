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
  configs/llamaish1-s3.yaml \
    --model.flash_attention=true \
    --fsdp.wrapping_strategy=by_block_and_size \
    --fsdp.sharding_strategy=SHARD_GRAD_OP \
    --gen1_gc_interval=null \
    --save_folder=runs/ \
    --activation_checkpointing=fine_grained \
    --fused_loss=false \
    --device_train_microbatch_size=4 \
    --global_train_batch_size=512 \
    --save_interval=250 \
    --eval_interval=250 \
    --optimizer.metrics_log_interval=1 \
    --save_overwrite \
    --model.init_fn=normal \
    --model.init_std=0.02 \
    --model.init_cutoff_factor=3 \
    --model.clip_qkv=null \
    --save_num_checkpoints_to_keep=3 \
    --scheduler.warmup_min_lr=0 \
    --scheduler.grad_clip_warmup_steps=null \
    --scheduler.units=steps \
    --scheduler.t_warmup=2000
    '--load_path=${path.last_checkpoint:s3://ai2-llm/checkpoints/OLMo-small/llm-306-amber-data-repro-db-normal-init-2}'