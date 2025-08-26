#!/usr/bin/env bash

# set -ex

CONFIG_PATH=${1:-"configs/peteish1-control.yaml"}
GPUS=${GPUS:-8}

NODES=1
# For multinode training, add:
#   --replicas "${NODES}" \
#   --synchronized-start-timeout 90m \

gantry run \
  --allow-dirty \
  --workspace ai2/willm-ppt2 \
  --budget ai2/oe-base \
  --task-name ppt2-control \
  --description "PPT2 control" \
  --priority normal \
  --cluster ai2/neptune-cirrascale \
  --preemptible \
  --beaker-image petew/olmo-torch23-gantry \
  --gpus $GPUS \
  --leader-selection \
  --host-networking \
  --weka oe-training-default:/weka/oe-training-default \
  --propagate-failure \
  --propagate-preemption \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env R2_PROFILE=R2 \
  --env S3_PROFILE=S3 \
  --env WEKA_PROFILE=WEKA \
  --env-secret AWS_CONFIG=WILLM_AWS_CONFIG \
  --env-secret AWS_CREDENTIALS=WILLM_AWS_CREDENTIALS \
  --env-secret WANDB_API_KEY=WILLM_WANDB_API_KEY \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --env-secret WEKA_ENDPOINT_URL=WEKA_ENDPOINT_URL \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "ppt2/scripts/peteish1.sh ${NODES} ${GPUS} ${CONFIG_PATH} \$BEAKER_LEADER_REPLICA_HOSTNAME \$BEAKER_REPLICA_RANK"