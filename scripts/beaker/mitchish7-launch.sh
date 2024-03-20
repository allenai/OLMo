#!/usr/bin/env bash

set -ex

NUM_NODES=8

gantry run \
  --workspace ai2/dirkg \
  --task-name mitchish7 \
  --description "OLMo medium - 7B" \
  --priority high \
  --stop-preemptible \
  --beaker-image petew/olmo-torch2-gantry \
  --cluster ai2/pluto-cirrascale \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --nfs \
  --mount /net/nfs.cirrascale/allennlp/petew/cache:/root/.cache \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  scripts/beaker/mitchish7.sh ${NUM_NODES}
