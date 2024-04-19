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

# Disabling infiniband for jupiter as temp workaround
export NCCL_IB_DISABLE=1

mkdir -p /media/16TBNVME/data/shanea/mitchish7/step358000-unsharded
aws s3 sync s3://ai2-llm/checkpoints/OLMo-medium/mitchish7/step358000-unsharded /media/16TBNVME/data/shanea/mitchish7/step358000-unsharded

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  scripts/train.py \
  configs/lr-scheduling-7b-s3.yaml \
    --run_name=lr-schedule-const-lr-7B \
    --wandb.name=lr-schedule-const-lr-7B \
    --model.flash_attention=true \
    --fsdp.wrapping_strategy=by_block_and_size \
    --fsdp.sharding_strategy=SHARD_GRAD_OP \
    --save_folder=runs/ \
    --activation_checkpointing=fine_grained \
    --fused_loss=true \
    --device_train_microbatch_size=2 \
    --global_train_batch_size=1024 \
    --save_overwrite \
    '--load_path=/media/16TBNVME/data/shanea/mitchish7/step358000-unsharded'