#!/usr/bin/env bash

set -ex

NUM_NODES=8

gantry run \
  --workspace ai2/cheap_decisions  \
  --task-name cheap-decisions-dolma-v1-6-and-sources-baseline-docspara-dedup-1B-N-1T-D-mitchish1\
  --description "OLMO 1B LLM 1T tokens on dolma 1.6 with baseline sources document and paragraph deduplicated" \
  --priority high \
  --preemptible \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-eval \
  --no-nfs \
  --propagate-failure \
  --synchronized-start-timeout 600m \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --venv base \
  --weka oe-eval-default:/weka/oe-eval-default \
  --weka oe-training-default:/weka/oe-training-default \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/cheap_decisions-baseline-and-sources-docspara-dedup.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"