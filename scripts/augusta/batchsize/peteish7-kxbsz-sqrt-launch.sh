#!/usr/bin/env bash

set -ex

NUM_NODES=$1
shift

K=$1
echo "Increasing batch size by factor of $K..."
NAME="peteish7-${K}xbsz-sqrt"
shift

# Get start step.
START_STEP=${START_STEP:-"477000"}
if [ "$START_STEP" != "477000" ]; then
  NAME="${NAME}-from${START_STEP}"
fi

# Compute as function of k.
BSIZE=$((K * 1024))
NSTEPS=$((512/K))
LR=$(echo $K | awk '{print sqrt($1) * 0.0003}')

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
  --env START_STEP=$START_STEP \
  --env BSIZE=$BSIZE \
  --env NSTEPS=$NSTEPS \
  --env LR=$LR \
  --env-secret WANDB_API_KEY=DIRKG_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=DIRKG_AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=DIRKG_AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  --allow-dirty \
  --retries 10 \
  -- /bin/bash -c "scripts/augusta/batchsize/peteish7-branch.sh \$BEAKER_LEADER_REPLICA_HOSTNAME \$BEAKER_REPLICA_RANK"
