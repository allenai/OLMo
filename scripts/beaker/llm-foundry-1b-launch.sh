#!/usr/bin/env bash

set -ex

NUM_NODES=2

gantry run \
  --workspace ai2/OLMo-pretraining-stability \
  --task-name llm-foundry-mpt-1B \
  --description "LLM Foundry - MPT 1B" \
  --priority normal \
  --preemptible \
  --beaker-image shanea/llm-foundry-torch2.3 \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --propagate-failure \
  --synchronized-start-timeout 10m \
  --weka oe-training-default:/weka/oe-training-default \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env-secret WANDB_API_KEY=SHANEA_WANDB_API_KEY \
  --env-secret AWS_CONFIG=SHANEA_AWS_CONFIG \
  --env-secret AWS_CREDENTIALS=SHANEA_AWS_CREDENTIALS \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/llm-foundry-1b.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"
