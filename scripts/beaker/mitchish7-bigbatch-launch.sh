#!/usr/bin/env bash

set -ex

NUM_NODES=4

gantry run \
  --allow-dirty \
  --workspace ai2/dirkg \
  --task-name mitchish7-bigbatch \
  --description "OLMo medium - 7B - bigbatch" \
  --priority high \
  --beaker-image michaelw/olmo-torch2.2-gantry-static \
  --cluster ai2/jupiter-cirrascale \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --venv base \
  --propagate-failure \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/mitchish7-bigbatch.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES}"
