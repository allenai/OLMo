#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

GPUS_PER_NODE=$1
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
  --nproc-per-node ${GPUS_PER_NODE} \
  --rdzv_id=12347 \
  --rdzv_backend=static \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  --rdzv_conf="read_timeout=420" \
  scripts/train.py \
  configs/llamaish7-weka.yaml \
    --run_name=llamaish7-detailed-no-fsdp-torch-clip-3 \
    --wandb.name=llamaish7-detailed-no-fsdp-torch-clip-3 \
    --wandb.group=llamaish7-detailed-no-fsdp \
    --model.flash_attention=true \
    --save_folder=/data/shanea/checkpoints/OLMo-medium/llamaish7-detailed-no-fsdp-torch-clip-3/ \
    --use_torch_clipping \
    --fused_loss=true \
    --fsdp.enabled=false \
    --device_train_microbatch_size=1 \
    --global_train_batch_size=1024 \
    --save_interval=50 \
    --eval_interval=50 \
    --optimizer.metrics_log_interval=1 \
    --model.init_device=cuda \
    --model.parallelize_model \
    --save_overwrite \
    --load_path=weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/llamaish7-detailed/step2750

# --load_path=/net/nfs.cirrascale/allennlp/shanea/checkpoints/llamaish7-detailed/step2800-unsharded
