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
  configs/annealing/$NAME.yaml \
    --fsdp.sharding_strategy=HYBRID_SHARD \
    --fsdp.hybrid_sharding_num_model_replicas=$NUM_NODES \
    --save_folder=$SAVE_FOLDER \
    --try_load_latest_save=true \
    --restore_dataloader=true \
    --device_train_microbatch_size=4 \
    --device_eval_batch_size=8 \
    --activation_checkpointing=whole_layer \
    --compile.fullgraph=false \
    --fused_loss=true \
    --model.flash_attention=true \
    --data.num_workers=8 \
    --optimizer.metrics_log_interval=10 \
    --data.prefetch_factor=8 2>&1 | tee $SAVE_FOLDER/log.txt
