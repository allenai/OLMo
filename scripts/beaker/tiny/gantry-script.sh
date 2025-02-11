#!/usr/bin/env bash

set -ex

NUM_NODES=1
TASK_NAME=olmo-150M-optimizer-adamw-lr-6e-4-beta1-0.85-beta2-0.95
CONFIG_PATH=configs/optimizers/OLMo-150M.yaml

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
  -- /bin/bash -c "scripts/beaker/tiny/torchrun-script.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} ${TASK_NAME} ${CONFIG_PATH}"