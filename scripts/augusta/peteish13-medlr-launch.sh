#!/bin/bash

set -euxo pipefail

NUM_NODES=128
RUN_NAME=peteish13-medlr-$(date -u +"%Y%m%d_%H%M%S")

./scripts/augusta/launch_train.sh $NUM_NODES \
  configs/peteish13-google.yaml \
    --run_name=$RUN_NAME \
    --wandb.group=peteish13-medlr \
    --save_interval_ephemeral=1000 \
    --eval_interval=200 \
    --fsdp.sharding_strategy=HYBRID_SHARD \
    --fsdp.hybrid_sharding_num_model_replicas=$NUM_NODES \
    --save_folder=/mnt/localssd/runs/$RUN_NAME \
    --remote_save_folder="s3://ai2-llm/checkpoints/OLMo-medium/peteish13-medlr/" \
    --save_overwrite \
    '--load_path=${path.last_checkpoint:${save_folder}}' \
    --sharded_checkpointer=olmo_core \
    --device_train_microbatch_size=2 \
    --activation_checkpointing=three_in_four \
    --compile=null \
    --fused_loss=true \
    --model.flash_attention=true \
    --data.num_workers=16 \
    --optimizer.learning_rate=6.0e-4 \
    --data.prefetch_factor=64 2>&1 | tee /mnt/localssd/runs/$RUN_NAME/log.txt
