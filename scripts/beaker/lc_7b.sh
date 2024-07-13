#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

export NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=ib NCCL_IB_HCA="^=mlx5_bond_0"


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

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=12347 \
  --rdzv_backend=static \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  scripts/train.py \
  configs/long_context_dolma_v0.5_cont_train.yaml \
    --gen1_gc_interval=null \
    --wandb.group=long_contexts_dolma_v0.5\
    --save_overwrite \
    --save_folder=runs/ \
    --save_num_checkpoints_to_keep=3 \
    --activation_checkpointing=fine_grained \
    --load_path=s3://ai2-llm/checkpoints/OLMo-medium/mitchish7/step477000-unsharded