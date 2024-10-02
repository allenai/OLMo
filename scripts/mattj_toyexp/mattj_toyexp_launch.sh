#!/usr/bin/env bash

set -ex

NUM_NODES=2
gantry run \
  --workspace ai2/mattj \
  --task-name mattj_toyexp \
  --description "mattj's toy experiment" \
  --priority normal \
  --preemptible \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/saturn-cirrascale \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-data \
  --no-nfs \
  --weka oe-data-default:/weka/oe-data-default \
  --propagate-failure \
  --propagate-preemption \
  --synchronized-start-timeout 90m \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env R2_PROFILE=R2 \
  --env S3_PROFILE=S3 \
  --env WEKA_PROFILE=WEKA \
  --env-secret AWS_CONFIG=MATTJ_AWS_CONFIG \
  --env-secret AWS_CREDENTIALS=MATTJ_AWS_CREDENTIALS \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --env-secret WEKA_ENDPOINT_URL=WEKA_ENDPOINT_URL \
  --env-secret WANDB_API_KEY=MATTJ_WANDB_API_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/mattj_toyexp/mattj_toyexp_setup.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"