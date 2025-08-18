#!/usr/bin/env bash

set -ex

NUM_NODES=2

gantry run \
  --workspace ai2/oe-data-model-based-cleanup \
  --allow-dirty \
  --task-name refine1-mixed-cx5-20240822 \
  --description "OLMo refine 1B" \
  --priority urgent \
  --preemptible \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-data \
  --no-nfs \
  --weka oe-training-default:/weka/oe-training-default \
  --propagate-failure \
  --propagate-preemption \
  --no-python \
  --synchronized-start-timeout 20m \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env R2_PROFILE=R2 \
  --env S3_PROFILE=S3 \
  --env WEKA_PROFILE=WEKA \
  --env-secret AWS_CONFIG=TCM_AWS_CONFIG \
  --env-secret AWS_CREDENTIALS=TCM_AWS_CREDENTIALS \
  --env-secret WANDB_API_KEY=TCM_WANDB_API_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/refine/refine1-mixed-Cx5.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"
