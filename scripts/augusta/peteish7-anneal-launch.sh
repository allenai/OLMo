#!/bin/bash

set -euxo pipefail

NAME=$1
shift

HOSTPATTERN=$1
shift

NUM_NODES=$1
shift

HOSTS=$(
  grep -E $HOSTPATTERN ~/hostfiles/hosts | \
  fgrep -hv \# | \
  parallel 'echo {} $(ssh {} curl -s http://metadata.google.internal/computeMetadata/v1/instance/attributes/physical_host -H \"Metadata-Flavor: Google\")' | \
  sort -k 2 | \
  head -$NUM_NODES | \
  cut -f 1 -d" " | \
  paste -sd,
)

RUN_NAME=$NAME-$(date -u +"%Y%m%d_%H%M%S")
SAVE_FOLDER=/mnt/localssd/runs/$RUN_NAME
mkdir -p $SAVE_FOLDER

./scripts/augusta/launch_train.sh $HOSTS \
  configs/annealing/$NAME-google.yaml \
    --run_name=$RUN_NAME \
    --wandb.group=$NAME \
    --save_interval_ephemeral=1000 \
    --eval_interval=1000 \
    --fsdp.sharding_strategy=HYBRID_SHARD \
    --fsdp.hybrid_sharding_num_model_replicas=$NUM_NODES \
    --save_folder=$SAVE_FOLDER \
    --remote_save_folder="gs://ai2-llm/checkpoints/OLMo-medium/$NAME/" \
    --save_overwrite \
    --restore_dataloader=false \
    --sharded_checkpointer=olmo_core \
    --device_train_microbatch_size=2 \
    --activation_checkpointing=one_in_four \
    --compile.fullgraph=false \
    --fused_loss=true \
    --model.flash_attention=true \
    --data.num_workers=8 \
    --optimizer.metrics_log_interval=10 \
    --data.prefetch_factor=8 2>&1 | tee $SAVE_FOLDER/log.txt
