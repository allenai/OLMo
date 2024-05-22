#!/usr/bin/env bash

set -ex

NUM_NODES=2

gantry run \
  --workspace ai2/OLMo-lr-scheduling \
  --task-name lr-schedule-const-lr-1B \
  --description "Const learning rate schedule experiment on OLMo 1B" \
  --priority normal \
  --preemptible \
  --beaker-image shanea/olmo-torch2.2-weka-gantry \
  --cluster ai2/jupiter-cirrascale \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --propagate-failure \
  --synchronized-start-timeout "30m" \
  --no-nfs \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env R2_PROFILE=R2 \
  --env S3_PROFILE=S3 \
  --env WEKA_PROFILE=WEKA \
  --env-secret AWS_CONFIG=AWS_CONFIG \
  --env-secret AWS_CREDENTIALS=AWS_CREDENTIALS \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --env-secret WEKA_ENDPOINT_URL=WEKA_ENDPOINT_URL \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/lr-scheduling.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"
