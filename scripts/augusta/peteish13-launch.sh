#!/bin/bash

set -euxo pipefail

HOSTFILE=$1
shift

NUM_NODES=64
RUN_NAME=peteish13-$(date -u +"%Y%m%d_%H%M%S")
SAVE_FOLDER=/mnt/localssd/runs/$RUN_NAME
mkdir -p $SAVE_FOLDER

./scripts/augusta/launch_train.sh $HOSTFILE $NUM_NODES \
  configs/peteish13-google.yaml \
    --run_name=$RUN_NAME \
    --wandb.group=peteish13-lumi \
    --save_interval_ephemeral=500 \
    --eval_interval=500 \
    --fsdp.sharding_strategy=HYBRID_SHARD \
    --fsdp.hybrid_sharding_num_model_replicas=$NUM_NODES \
    --save_folder=$SAVE_FOLDER \
    --remote_save_folder="gs://ai2-llm/checkpoints/OLMo-medium/peteish13-lumi/" \
    --save_overwrite \
    '--load_path=${path.last_checkpoint:${remote_save_folder}}' \
    --load_path=gs://ai2-llm/checkpoints/OLMo-medium/peteish13-lumi/step12500 \
    --sharded_checkpointer=olmo_core \
    --device_train_microbatch_size=4 \
    --activation_checkpointing=whole_layer \
    --compile.fullgraph=false \
    --fused_loss=true \
    --model.flash_attention=true \
    --data.num_workers=8 \
    --optimizer.learning_rate=3.0e-4 \
    --optimizer.metrics_log_interval=10 \
    --data.prefetch_factor=8 2>&1 | tee $SAVE_FOLDER/log.txt
