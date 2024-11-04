#!/usr/bin/env bash

set -ex

NUM_NODES=$1
shift
PRIORITY=$1
shift
NUM_GPUS = 4

if [[ $NUM_NODES -eq 1 ]]; then
  MULTI_NODE_ARGS=""
  COMMAND="scripts/beaker/ladder.sh localhost ${NUM_NODES} 0 ${NUM_GPUS} $*"
else
  MULTI_NODE_ARGS="--replicas ${NUM_NODES} --leader-selection --host-networking --propagate-failure --propagate-preemption --synchronized-start-timeout 10m"
  COMMAND="scripts/beaker/ladder.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK ${NUM_GPUS} $*"
fi

gantry run \
  --workspace ai2/ptqa \
  --task-name mcqa \
  --description "PTQA with $*" \
  --priority $PRIORITY \
  --preemptible \
  --beaker-image shanea/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --cluster ai2/pluto-cirrascale \
  --cluster ai2/saturn-cirrascale \
  --gpus $NUM_GPUS \
  $MULTI_NODE_ARGS \
  --budget ai2/aristo \
  --no-nfs \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=BENB_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=0 \
  --allow-dirty \
  -- /bin/bash -c "${COMMAND}"
