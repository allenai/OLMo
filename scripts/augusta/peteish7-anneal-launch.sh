#!/usr/bin/env bash

set -ex

NUM_NODES=$1
shift

NAME=$1
shift

if [ -z $1 ]; then
  PRIORITY='high'
else
  PRIORITY=$1
fi


gantry run \
  --workspace ai2/13B \
  --task-name $NAME \
  --description "Peteish7 annealing: $NAME" \
  --priority ${PRIORITY} \
  --preemptible \
  --beaker-image dirkg/OLMo \
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
  --env-secret WANDB_API_KEY=DIRKG_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=DIRKG_AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=DIRKG_AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  --allow-dirty \
  --retries 10 \
  -- /bin/bash -c "scripts/augusta/peteish7-anneal.sh \$BEAKER_LEADER_REPLICA_HOSTNAME \$BEAKER_REPLICA_RANK"
