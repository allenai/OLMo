#!/usr/bin/env bash

set -ex

SOCKET=29401
NUM_NODES=1
TASK_NAME=olmo-150M-optimizer-muon-lr-1.2e-3-wd-0.1-cosine
CONFIG_PATH=configs/optimizers/OLMo-150M.yaml

OPTIMIZER=muon
MUON_LR=1.2e-3
MUON_WEIGHT_DECAY=0.1

gantry run \
  --allow-dirty \
  --workspace ai2/OLMo-tiny \
  --task-name ${TASK_NAME} \
  --description "OLMo optimizer runs" \
  --priority high \
  --preemptible \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --cluster ai2/ceres-cirrascale \
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
  -- /bin/bash -c "scripts/beaker/optimizer/muon-torchrun-script.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${SOCKET} ${NUM_NODES} ${TASK_NAME} ${CONFIG_PATH} ${OPTIMIZER} ${MUON_LR} ${MUON_WEIGHT_DECAY}"