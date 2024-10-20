#!/usr/bin/env bash

set -ex

NUM_NODES=1

gantry run \
  --allow-dirty \
  --workspace ai2/hb-wolf-olmo \
  --task-name peteish1-eval \
  --description "Pete-ish 1B" \
  --priority high \
  --preemptible \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --budget ai2/oe-training \
  --no-nfs \
  --weka oe-training-default:/weka/oe-training-default \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/peteish/peteish1.sh ${NUM_NODES}"
  # -- /bin/bash -c "scripts/beaker/peteish/peteish1.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"

  # --leader-selection \
  # --host-networking \
  # --propagate-failure \
  # --propagate-preemption \
  # --synchronized-start-timeout 90m \
