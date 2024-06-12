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

# Install flash-attn
#conda install -y -c nvidia cuda-python
pip install packaging ninja
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
pip install flash-attn --no-build-isolation
# pip install awscli
pip install '.[train]'
pip freeze

# Warm HF cache
# mkdir -p /root/.cache
# pushd /root/.cache
# curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -
# popd
# export HF_DATASETS_OFFLINE=1

# Move AWS credentials from env to relevant files
# mkdir -p ~/.aws
# printenv AWS_CONFIG > ~/.aws/config
# printenv AWS_CREDENTIALS > ~/.aws/credentials

# mkdir /root/checkpoint-unsharded
# aws s3 cp --no-progress --recursive --profile=S3 \
#     s3://ai2-llm/checkpoints/OLMo-medium/llamaish7-EmbInitFix/step0-unsharded \
#     /root/checkpoint-unsharded

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
    configs/llm-360-amber-s3.yaml \
      --run_name="${GANTRY_TASK_NAME}" \
      --optimizer.metrics_log_interval=1 \
      '--load_path=${path.last_checkpoint:${remote_save_folder}}'

#      --activation_checkpointing=fine_grained \
