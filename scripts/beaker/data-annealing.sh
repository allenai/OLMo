#!/usr/bin/env bash

set -ex

CONFIG_PATH=configs/annealing/OLMo-7B.yaml
NUM_NODES=4  # TODO: update an needed
RUN_NAME="data-annealing-001"  # TODO: update
ARGS="--run_name=${RUN_NAME} --device_train_microbatch_size=4"

gantry run \
  --allow-dirty \
  --workspace ai2/data-annealing \  # TODO: update
  --task-name OLMo-7B-train \
  --description "OLMo large - 70B" \
  --priority high \
  --beaker-image petew/olmo-torch2-gantry \
  --cluster ai2/general-cirrascale-a100-80g-ib \  # TODO: update to pluto cluster
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --nfs \
  --mount /net/nfs.cirrascale/allennlp/petew/cache:/root/.cache \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \  # TODO: 'beaker secret write ...'
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \  # TODO: 'beaker secret write ...'
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \  # TODO: 'beaker secret write ...'
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH} ${ARGS}"
