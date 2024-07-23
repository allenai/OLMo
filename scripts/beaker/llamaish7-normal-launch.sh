#!/usr/bin/env bash

set -ex

NUM_NODES=64

gantry run \
  --workspace ai2/OLMo-training \
  --task-name llamaish7-normal-qk-norm-reorder-zloss \
  --description "OLMo medium - 7B - Llamaish Normal" \
  --priority urgent \
  --preemptible \
  --beaker-image shanea/olmo-torch2.3-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --propagate-failure \
  --synchronized-start-timeout 15m \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=AKSHITAB_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AKSHITAB_AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AKSHITAB_AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/llamaish7-normal.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"
