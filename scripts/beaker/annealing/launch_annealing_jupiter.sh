#!/usr/bin/env bash
# Kicks off anneal runs on the Jupiter cluster.

set -ex

CONFIG_NAME=$1
NUM_NODES=$2
CLUSTER=$3
PRIORITY=$4

CONFIG_DIR=configs/annealing
CONFIG_PATH=${CONFIG_DIR}/${CONFIG_NAME}.yaml

gantry run \
  --preemptible \
  --allow-dirty \
  --workspace ai2/davidw-oe-annealing \
  --task-name ${CONFIG_NAME} \
  --description ${CONFIG_NAME} \
  --priority $PRIORITY \
  --beaker-image shanea/olmo-torch2.2-weka-gantry \
  --cluster $CLUSTER \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --propagate-failure \
  --synchronized-start-timeout "30m" \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --env-secret R2_ACCESS_KEY_ID=R2_ACCESS_KEY_ID \
  --env-secret R2_SECRET_ACCESS_KEY=R2_SECRET_ACCESS_KEY \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --env-secret WEKA_ENDPOINT_URL=WEKA_ENDPOINT_URL \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c "source scripts/beaker/warm_hf_cache.sh && torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH} --model.flash_attention=true --fsdp.wrapping_strategy=by_block_and_size --fsdp.sharding_strategy=SHARD_GRAD_OP --activation_checkpointing=fine_grained --fused_loss=true --device_train_microbatch_size=2 --global_train_batch_size=1024 --gen1_gc_interval=8 --save_num_checkpoints_to_keep=2"
