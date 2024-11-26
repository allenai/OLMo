#!/usr/bin/env bash

set -ex

NUM_NODES=8

gantry run \
  --workspace ai2/OLMo-tiny \
  --task-name peteish13-highlr-eval \
  --description "Peteish13" \
  --priority urgent \
  --preemptible \
  --beaker-image dirkg/OLMo \
  --cluster ai2/augusta-google-1 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --propagate-failure \
  --propagate-preemption \
  --synchronized-start-timeout 25m \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=JIACHENGL_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=JIACHENGL_AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=JIACHENGL_AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  --allow-dirty \
  -- /bin/bash -c "scripts/augusta/peteish13-eval.sh \$BEAKER_LEADER_REPLICA_HOSTNAME \$BEAKER_REPLICA_RANK"
