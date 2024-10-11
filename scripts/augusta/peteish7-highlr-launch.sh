#!/bin/bash

set -euxo pipefail

HOSTFILE=$1
shift

NUM_NODES=32
RUN_NAME=peteish7-highlr-$(date -u +"%Y%m%d_%H%M%S")
SAVE_FOLDER=/mnt/localssd/runs/$RUN_NAME
mkdir -p $SAVE_FOLDER

./scripts/augusta/launch_train.sh $HOSTFILE $NUM_NODES \
  configs/peteish7-google.yaml \
    --run_name=$RUN_NAME \
    --wandb.group=peteish7-highlr \
    --save_interval_ephemeral=1000 \
    --eval_interval=1000 \
    --fsdp.sharding_strategy=HYBRID_SHARD \
    --fsdp.hybrid_sharding_num_model_replicas=$NUM_NODES \
    --save_folder=$SAVE_FOLDER \
    --remote_save_folder="gs://ai2-llm/checkpoints/OLMo-medium/peteish7-highlr/" \
    --save_overwrite \
    '--load_path=${path.last_checkpoint:${remote_save_folder}}' \
    --load_path=gs://ai2-llm/checkpoints/OLMo-medium/peteish7-highlr/step47000 \
    --sharded_checkpointer=olmo_core \
    --device_train_microbatch_size=2 \
    --activation_checkpointing=one_in_four \
    --compile.fullgraph=false \
    --fused_loss=true \
    --model.flash_attention=true \
    --data.num_workers=8 \
    --optimizer.learning_rate=9.0e-4 2>&1 | tee $SAVE_FOLDER/log.txt
