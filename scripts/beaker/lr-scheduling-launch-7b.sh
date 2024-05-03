#!/usr/bin/env bash

set -ex

NUM_NODES=16

gantry run \
  --workspace ai2/shanea \
  --task-name lr-schedule-const-lr-7B \
  --description "Const learning rate schedule experiment on OLMo 7B" \
  --priority high \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --cluster ai2/jupiter-cirrascale \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --synchronized-start-timeout "15m" \
  --no-nfs \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env NCCL_IB_HCA=^mlx5_bond_0 \
  --env-secret SSH_KEY=SSH_KEY \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --env-secret R2_ACCESS_KEY_ID=R2_ACCESS_KEY_ID \
  --env-secret R2_SECRET_ACCESS_KEY=R2_SECRET_ACCESS_KEY \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --env-secret WEKA_ACCESS_KEY_ID=WEKA_ACCESS_KEY_ID \
  --env-secret WEKA_SECRET_ACCESS_KEY=WEKA_SECRET_ACCESS_KEY \
  --env-secret WEKA_ENDPOINT_URL=WEKA_ENDPOINT_URL \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/lr-scheduling-7b.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"
