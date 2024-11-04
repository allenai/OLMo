#!/usr/bin/env bash

set -ex

NUM_NODES=16

gantry run \
  --allow-dirty \
  --workspace ai2/alexw \
  --task-name peteish1-eval \
  --description "Pete-ish 1B eval test" \
  --priority normal \
  --preemptible \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 1 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --propagate-failure \
  --propagate-preemption \
  --synchronized-start-timeout 90m \
  --budget ai2/oe-training \
  --no-nfs \
  --weka oe-training-default:/weka/oe-training-default \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/peteish/peteish1-eval.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"