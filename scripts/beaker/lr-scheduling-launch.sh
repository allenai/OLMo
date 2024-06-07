#!/usr/bin/env bash

set -ex

NUM_NODES=4

gantry run \
  --workspace ai2/OLMo-lr-scheduling \
  --task-name lr-schedule-const-lr-1B \
  --description "Const learning rate schedule experiment on OLMo 1B" \
  --priority normal \
  --preemptible \
  --beaker-image shanea/olmo-torch2.2-weka-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --propagate-failure \
  --synchronized-start-timeout "30m" \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env R2_PROFILE=R2 \
  --env S3_PROFILE=S3 \
  --env WEKA_PROFILE=WEKA \
  --env-secret AWS_CONFIG=AWS_CONFIG \
  --env-secret AWS_CREDENTIALS=AWS_CREDENTIALS \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --env-secret R2_ACCESS_KEY_ID=R2_ACCESS_KEY_ID \
  --env-secret R2_SECRET_ACCESS_KEY=R2_SECRET_ACCESS_KEY \
  --env-secret WEKA_ENDPOINT_URL=WEKA_ENDPOINT_URL \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/lr-scheduling.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"
