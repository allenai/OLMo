#!/usr/bin/env bash

set -ex

# CONFIG_NAME="peteish7-anneal"
# CONFIG_NAME="peteish7-anneal-B34v0x10"
# CONFIG_NAME="peteish7-anneal-B34v0.1x50"

# bash sewon-scripts/peteish7-anneal-launch.ch peteish7-anneal-from-1T-scitech
# bash sewon-scripts/peteish7-anneal-launch.ch peteish7-anneal-from-1T-edu
# bash sewon-scripts/peteish7-anneal-launch.ch peteish7-anneal-from-1T-history
# bash sewon-scripts/peteish7-anneal-launch.ch peteish7-anneal-from-1T-health
# bash sewon-scripts/peteish7-anneal-launch.ch peteish7-anneal-from-1T-edu


CONFIG_NAME=$1
NUM_NODES=$2

gantry run \
  --workspace ai2/ds-olmo \
  --task-name ${CONFIG_NAME} \
  --description ${CONFIG_NAME} \
  --priority high \
  --preemptible \
  --beaker-image petew/olmo-torch23-gantry \
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
  --env-secret AWS_CONFIG=SEWONM_AWS_CONFIG \
  --env-secret AWS_CREDENTIALS=SEWONM_AWS_CREDENTIALS \
  --env-secret WANDB_API_KEY=SEWONM_WANDB_API_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "sewon-scripts/peteish7.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK ${CONFIG_PATH}"
