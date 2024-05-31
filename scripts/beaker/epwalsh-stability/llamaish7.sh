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
#pip install packaging ninja
#export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
#pip install flash-attn --no-build-isolation
pip install awscli
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
    configs/llamaish7-weka.yaml \
      --run_name="${GANTRY_TASK_NAME}" \
      --model.scale_emb_init=true \
      --model.complex_rope=true \
      --model.layer_norm_type=rms \
      --model.layer_norm_with_affine=true \
      --model.clip_qkv=null \
      --scheduler.warmup_min_lr=0.0 \
      --scheduler.t_warmup=8388608000 \
      --scheduler.t_max=2e12 \
      --seed=18898 \
      --stop_at=5000

# No data instance filtering:
#      --data.instance_filter=null \
#
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
# Non-parametric RMS norm:
#      --model.layer_norm_type=rms \
#
# (Parametric) RMS norm:
#      --model.layer_norm_type=rms \
#      --model.layer_norm_with_affine=true \
#
# No QKV clipping:
#      --model.clip_qkv=null \
#
# Initialize embeddings with std=1.0:
#      --model.emb_init_std=1.0 \
#
# Warmup from LR=0.0:
#      --scheduler.warmup_min_lr=0.0 \
#
# Llama 2 schedule:
#      --scheduler.t_warmup=8388608000 \
#      --scheduler.t_max=2e12 \
#
# Fused loss:
#      --fused_loss=true \
#
# Using torch's default epsilon=1e-8 with AdamW
#      --optimizer.eps=1e-8 \
#
#      '--load_path=${path.last_checkpoint:weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/epwalsh-stability/${run_name}}' \
#
#      '--load_path=weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/epwalsh-stability/${run_name}/step3000' \
#
#      --load_path=weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/epwalsh-stability/llamaish7-alibi-emb-init-fix-data-filter-wup0/step2500 \
#      --fast_forward_batches=300 \
#
#      --load_path=s3://ai2-llm/checkpoints/OLMo-medium/llamaish7-EmbInitFix/step0-unsharded \
