#!/usr/bin/env bash

set -ex

NUM_NODES=4

gantry run \
  --workspace ai2/OLMo-training  \
  --task-name llm-360-amber-baseline \
  --description "LLM 360 Amber 7B in the OLMo codebase" \
  --priority urgent \
  --preemptible \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --propagate-failure \
  --synchronized-start-timeout 30m \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret AWS_ACCESS_KEY_ID=DUSTINS_AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=DUSTINS_AWS_SECRET_ACCESS_KEY \
  --env-secret WANDB_API_KEY=PETEW_WANDB_API_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/epwalsh-stability/llm-360-amber7.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"

  # --env R2_PROFILE=R2 \
  # --env S3_PROFILE=S3 \
  # --env WEKA_PROFILE=WEKA \
  # --env-secret AWS_CONFIG=PETEW_AWS_CONFIG \
  # --env-secret AWS_CREDENTIALS=PETEW_AWS_CREDENTIALS \
  # --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  # --env-secret WEKA_ENDPOINT_URL=WEKA_ENDPOINT_URL \
