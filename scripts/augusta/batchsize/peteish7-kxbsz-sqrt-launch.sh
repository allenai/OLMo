#!/usr/bin/env bash

set -ex

NUM_NODES=$1
shift

K=$1
echo "Increasing batch size by factor of $K..."
shift

TOTAL_STEPS=${TOTAL_STEPS:-512}

# Compute as function of k.
BSIZE=$((K * 1024))
NSTEPS=$((TOTAL_STEPS/K))
LR=$(echo $K | awk '{print sqrt($1) * 0.0003}')

# Get the checkpoint that we should load from.
LOAD_PATH=${LOAD_PATH:-"gs://ai2-llm/checkpoints/OLMo-medium/peteish7/step477000/"}
step=$(echo $LOAD_PATH | grep -oP 'step\K\d+')
NAME="peteish7-${K}xbsz-sqrt-from$step"

echo $NAME | gantry run \
  --workspace ai2/13B \
  --task-name $NAME \
  --description "Peteish7 with ${K}x batch size and sqrt(${K})x LR from $START_STEP" \
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
  --synchronized-start-timeout 15m \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env LOAD_PATH=$LOAD_PATH \
  --env BSIZE=$BSIZE \
  --env NSTEPS=$NSTEPS \
  --env LR=$LR \
  --env-secret WANDB_API_KEY=WILLM_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=WILLM_AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=WILLM_AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  --allow-dirty \
  --retries 10 \
  -- /bin/bash -c "scripts/augusta/batchsize/peteish7-branch.sh \$BEAKER_LEADER_REPLICA_HOSTNAME \$BEAKER_REPLICA_RANK"
