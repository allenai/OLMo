#!/usr/bin/env bash

set -ex

SOCKET=29400
NUM_NODES=1
TASK_NAME=olmo-150M-optimizer-schedule-free-adamw-lr-6e-4-wd-0
CONFIG_PATH=configs/optimizers/OLMo-150M.yaml

OPTIMIZER=schedule_free_adamw
LR=6e-4
WD=0.1

gantry run \
  --allow-dirty \
  --workspace ai2/OLMo-tiny \
  --task-name ${TASK_NAME} \
  --description "OLMo optimizer runs" \
  --priority high \
  --preemptible \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 4 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --propagate-failure \
  --propagate-preemption \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=ANANYA_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=ANANYA_AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=ANANYA_AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/optimizer/torchrun-script.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${SOCKET} ${NUM_NODES} ${TASK_NAME} ${CONFIG_PATH} ${OPTIMIZER} ${LR} ${WD}"