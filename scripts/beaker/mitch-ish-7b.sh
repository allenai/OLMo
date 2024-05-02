#!/usr/bin/env bash

set -ex

CONFIG_PATH=configs/v1_5-mix-medium-mitch-ish-s3.yaml
NUM_NODES=4
ARGS='--activation_checkpointing=fine_grained wandb.name=v1_5-mix-mitch-ish-mcli-final --epoch=1 --optimizer.learning_rate=0.000023 --scheduler.t_warmup=556000 --scheduler.t_max=557000 --scheduler.alpha_f=0.001 --stop_at=557000'

gantry run \
  --allow-dirty \
  --workspace ai2/llm-testing \
  --task-name mitchish-mcli-final \
  --description mitchish-mcli-final \
  --priority high \
  --beaker-image olmo-torch2-gantry \
  --cluster ai2/general-cirrascale-a100-80g-ib \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
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
  -- /bin/bash -c "torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH} ${ARGS}"
