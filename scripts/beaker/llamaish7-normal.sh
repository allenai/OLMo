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

# Move AWS credentials from env to relevant files
mkdir -p ~/.aws
printenv AWS_CONFIG > ~/.aws/config
printenv AWS_CREDENTIALS > ~/.aws/credentials

export EXPERIMENT=llamaish7-normal-final

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=12347 \
  --rdzv_backend=static \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  --rdzv_conf="read_timeout=420" \
  scripts/train.py \
  configs/llamaish7-normal-s3.yaml \
    --run_name=$EXPERIMENT \
    --wandb.name=$EXPERIMENT \
    --wandb.group=$EXPERIMENT \
    --fsdp.wrapping_strategy=by_block_and_size \
    --fsdp.sharding_strategy=SHARD_GRAD_OP \
    --save_folder=runs/ \
    --activation_checkpointing=fine_grained \
    --device_train_microbatch_size=2 \
    --global_train_batch_size=1024 \
    --save_interval=250 \
    --eval_interval=250 \
    --optimizer.metrics_log_interval=1 \
    --save_overwrite \
    --save_num_checkpoints_to_keep=3 \
    --data.num_workers=64 \
    --load_path=s3://ai2-llm/checkpoints/OLMo-medium/llamaish7-normal/step2000
