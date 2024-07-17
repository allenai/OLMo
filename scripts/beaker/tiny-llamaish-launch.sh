#!/usr/bin/env bash

set -ex

NUM_NODES=1

gantry run \
  --workspace ai2/OLMo-training \
  --task-name tiny-llamaish \
  --description "OLMo tiny-llamaish test" \
  --priority high \
  --preemptible \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --cluster ai2/jupiter-cirrascale \
  --gpus 2 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --propagate-failure \
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
  --allow-dirty \
  -- /bin/bash -c "scripts/beaker/tiny-llamaish.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"
  #--synchronized-start-timeout 600m
