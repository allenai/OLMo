#!/usr/bin/env bash

# Launcher for DCLM:7b1x runs on Jupiter.

CONFIG_NAME=$1
NUM_NODES=$2
PRIORITY=$3

CONFIG_DIR=configs/annealing
CONFIG_PATH=${CONFIG_DIR}/${CONFIG_NAME}.yaml



gantry run \
  --workspace ai2/oe-data \
  --task-name ${CONFIG_NAME} \
  --description ${CONFIG_NAME} \
  --priority $PRIORITY \
  --preemptible \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --weka oe-training-default:/weka/oe-training-default \
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
  --env-secret AWS_ACCESS_KEY=MATTJ_AWS_ACCESS_KEY \
  --env-secret AWS_SECRET=MATTJ_AWS_SECRET \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --env-secret WEKA_ENDPOINT_URL=WEKA_ENDPOINT_URL \
  --env-secret WANDB_API_KEY=PETEW_WANDB_API_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/dclm/7b1x.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK ${CONFIG_PATH}"

