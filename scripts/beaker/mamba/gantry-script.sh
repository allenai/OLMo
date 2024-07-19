#!/usr/bin/env bash

set -ex

NUM_NODES=4
TASK_NAME=vanilla-mamba2-60M-d_state-64-mbsz_16

gantry run \
  --workspace ai2/OLMo-training \
  --task-name ${TASK_NAME} \
  --description "Tiny mamba runs" \
  --priority normal \
  --preemptible \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-adapt \
  --no-nfs \
  --propagate-failure \
  --synchronized-start-timeout 30m \
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
  -- /bin/bash -c "scripts/beaker/mamba/torchrun-script.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} ${TASK_NAME}"