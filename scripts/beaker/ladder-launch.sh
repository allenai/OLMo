#!/usr/bin/env bash

set -ex

NUM_NODES=$1
shift

gantry run \
  --workspace ai2/OLMo-training \
  --task-name ladder \
  --description "OLMo ladder with $*" \
  --priority normal \
  --preemptible \
  --beaker-image shanea/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --weka=oe-training-default:/weka/oe-training-default \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --propagate-failure \
  --synchronized-start-timeout 10m \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=DIRKG_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/ladder.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK $*"
