#!/usr/bin/env bash

set -ex

NUM_NODES=1
GPUS_PER_NODE=2

gantry run \
  --workspace ai2/OLMo-training \
  --task-name llama-detailed-no-fdsp \
  --description "OLMo training stability run with no FSDP" \
  --priority normal \
  --beaker-image shanea/olmo-torch2.2-weka-gantry \
  --cluster ai2/pluto-cirrascale \
  --gpus "${GPUS_PER_NODE}" \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env R2_PROFILE=R2 \
  --env S3_PROFILE=S3 \
  --env WEKA_PROFILE=WEKA \
  --env-secret WANDB_API_KEY=SHANEA_WANDB_API_KEY \
  --env-secret AWS_CONFIG=SHANEA_AWS_CONFIG \
  --env-secret AWS_CREDENTIALS=SHANEA_AWS_CREDENTIALS \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --env-secret WEKA_ENDPOINT_URL=WEKA_ENDPOINT_URL \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/llamaish7-no-fsdp.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} ${GPUS_PER_NODE} \$BEAKER_REPLICA_RANK"
