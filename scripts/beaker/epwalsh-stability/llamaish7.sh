#!/usr/bin/env bash

set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

BEAKER_REPLICA_RANK=$1
shift

# Setup Python environment.
conda shell.bash activate base
pip install git+https://github.com/allenai/OLMo-core.git@main
pip install '.[train]'
pip freeze

# Warm HF cache
# mkdir -p /root/.cache
# pushd /root/.cache
# curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -
# popd
# export HF_DATASETS_OFFLINE=1

# Move AWS credentials from env to relevant files
mkdir -p ~/.aws
printenv AWS_CONFIG > ~/.aws/config
printenv AWS_CREDENTIALS > ~/.aws/credentials

# Force processes to synchronize at init_process_group
export TORCH_DIST_INIT_BARRIER=1

torchrun \
  --nnodes "${NUM_NODES}:${NUM_NODES}" \
  --nproc-per-node 8 \
  --rdzv_id 12347 \
  --rdzv_backend static \
  --rdzv_endpoint "${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" \
  --node_rank "${BEAKER_REPLICA_RANK}" \
  --rdzv_conf 'read_timeout=420' \
  scripts/train.py \
    configs/llamaish7-weka.yaml \
      --run_name="${GANTRY_TASK_NAME}" \
      --model.scale_emb_init=true \
      --model.layer_norm_type=rms \
      --scheduler.warmup_min_lr=0.0 \
      --stop_at=5000

# ALiBi:
#      --model.rope=false \
#      --model.alibi=true \
#
# Complex RoPE:
#      --model.complex_rope=true \
#
# Emb init fix:
#      --model.scale_emb_init=true \
#
# RMS norm:
#      --model.layer_norm_type=rms \
#
# Warmup from LR=0.0 \
#      --scheduler.warmup_min_lr=0.0 \
#
#'--load_path=${path.last_checkpoint:s3://ai2-llm/checkpoints/OLMo-medium/epwalsh-stability/}'
