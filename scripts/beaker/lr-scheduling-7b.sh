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
# export NCCL_IB_DISABLE=1

# # Install OLMo-core
# mkdir -p ~/.ssh
# printenv SSH_KEY > ~/.ssh/id_ed25519
# cat ~/.ssh/id_ed25519 | wc
# ssh -T git@github.com
# git clone git@github.com:allenai/OLMo-core.git
# cd OLMo-core
# pip install .[all]
# cd ..
# rm -rf OLMo-core

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=12345 \
  --rdzv_backend=static \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --rdzv_conf="read_timeout=420" \
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
    --time_limit=null \
    --max_duration=2ep \
    --save_overwrite \
    '--load_path=${path.last_checkpoint:s3://ai2-llm/checkpoints/7b/lr-schedule-const-lr-7B/}'