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
# pushd /root/.cache
# curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -
# popd
# export HF_DATASETS_OFFLINE=1

# Move AWS credentials from env to relevant files
mkdir -p ~/.aws
printenv AWS_CONFIG > ~/.aws/config
printenv AWS_CREDENTIALS > ~/.aws/credentials

# # Force processes to synchronize at init_process_group
# export TORCH_DIST_INIT_BARRIER=1

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=123412 \
  --rdzv_backend=static \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  --rdzv_conf="read_timeout=420" \
  scripts/train.py \
    configs/lr-scheduling-s3.yaml \
      --run_name=hf-cache-test-control \
      --fsdp.sharding_strategy=SHARD_GRAD_OP \
      '--load_path=${path.last_checkpoint:s3://ai2-llm/checkpoints/1b/lr-scheduling-high-const-lr/}' \
      --wandb.name=control \
      --wandb.group=hf-cache-test \
      --scheduler.name=linear_with_warmup \
      --scheduler.t_warmup=442000 \
      --scheduler.alpha_f=1.0 \
      --scheduler.warmup_min_lr=-5.09e-3 \
      --optimizer.learning_rate=4.0e-4 \
      --remote_save_folder=null \
      --eval_interval=10 \
      --save_overwrite



# --load_path=r2://olmo-checkpoints/unsorted/6720641/step430000-unsharded/ \
# --load_path=r2://olmo-checkpoints/unsorted/6655301/step422000-unsharded/ \
# --load_path=/net/nfs.cirrascale/allennlp/shanea/checkpoints/unsorted/6655301/step422000-unsharded/ \
# --load_path=/net/nfs.cirrascale/allennlp/shanea/checkpoints/unsorted/6574511/step402000-unsharded/ \
# --load_path=r2://olmo-checkpoints/unsorted/6574511/step402000-unsharded/ \
# --load_path=weka://oe-training-default/ai2-llm/checkpoints/1b/lr-scheduling-low-const-lr/step574000/ \
# '--load_path=${path.last_checkpoint:weka://oe-training-default/ai2-llm/checkpoints/1b/lr-scheduling-low-const-lr}' \
#'--load_path=${path.last_checkpoint:s3://ai2-llm/checkpoints/7b/const-lr-linear-decay-match-50B/}'
#--load_path=weka://oe-training-default/ai2-llm/checkpoints/unsorted/6746551/step440000-unsharded/

# --remote_save_folder=weka://oe-training-default/ai2-llm/checkpoints/1b/lr-linear-decay-step402000-40000steps \