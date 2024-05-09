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

# PyTorch debug logs
export NCCL_P2P_DISABLE=1

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=12345 \
  --rdzv_backend=static \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  --rdzv_conf="read_timeout=420" \
  scripts/train.py \
  configs/lr-scheduling-7b-s3.yaml \
    --run_name=const-lr-linear-decay-match-50B \
    --wandb.name=const-lr-linear-decay-match-50B \
    --wandb.group=const-lr-linear-decay-match-50B \
    --model.flash_attention=true \
    --fsdp.wrapping_strategy=by_block_and_size \
    --fsdp.sharding_strategy=SHARD_GRAD_OP \
    --save_folder=runs/ \
    --activation_checkpointing=fine_grained \
    --fused_loss=true \
    --device_train_microbatch_size=2 \
    --global_train_batch_size=1024 \
    --time_limit=null \
    --scheduler.alpha_f=0.0 \
    --scheduler.t_warmup=1967128576000 \
    --scheduler.name=linear_with_warmup \
    --max_duration=2ep \
    --save_overwrite \
    --load_path=s3://ai2-llm/checkpoints/7b/lr-schedule-const-lr-7B/step469000/