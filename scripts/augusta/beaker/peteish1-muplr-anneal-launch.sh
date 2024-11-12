#!/usr/bin/env bash

set -ex

NUM_NODES=$1
shift

ORIGINAL_WANDB_RUN_ID=$1
shift

START_STEP=$1
shift

LENGTH=$1
shift

NAME=$(python ./scripts/group_name_from_wandb.py $ORIGINAL_WANDB_RUN_ID)-from$START_STEP-$LENGTH

gantry run \
  --workspace ai2/OLMo-mup \
  --task-name $NAME \
  --description "Peteish1 muP LR annealing: $NAME" \
  --priority high \
  --preemptible \
  --beaker-image michalg/cuda11.8-ubuntu20.04-arb \
  --cluster ai2/augusta-google-1 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --propagate-failure \
  --propagate-preemption \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=SHANEA_WANDB_API_KEY \
  --env-secret AWS_CONFIG=SHANEA_AWS_CONFIG \
  --env-secret AWS_CREDENTIALS=SHANEA_AWS_CREDENTIALS \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  --allow-dirty \
  --retries 10 \
  -- /bin/bash -c "scripts/augusta/beaker/peteish1-muplr.sh \$BEAKER_LEADER_REPLICA_HOSTNAME \$BEAKER_REPLICA_RANK $ORIGINAL_WANDB_RUN_ID $START_STEP $LENGTH"
