#!/usr/bin/env bash

set -ex

CONFIG_PATH=configs/olmoe/OLMo-1B-NOSHARD.yml
NUM_NODES=1
# BEAKER_NODE_HOSTNAME
# BEAKER_LEADER_REPLICA_HOSTNAME
#ARGS='--activation_checkpointing=fine_grained --epoch=1 --optimizer.learning_rate=0.000023 --scheduler.t_warmup=556000 --scheduler.t_max=557000 --scheduler.alpha_f=0.001 --stop_at=557000'
#  --env-secret WANDB_API_KEY=WANDB_API_KEY \
#  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
#  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
#petew/olmo-torch2-gantry
#  --cluster ai2/pluto-cirrascale \
gantry run \
  --allow-dirty \
  --workspace ai2/olmoe \
  --task-name mitchish-mcli-final \
  --description mitchish-mcli-final \
  --priority normal \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --budget ai2/oe-training \
  --cluster ai2/jupiter-cirrascale \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --nfs \
  --mount /net/nfs.cirrascale/allennlp/petew/cache:/root/.cache \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c "torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\$BEAKER_NODE_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH}"
# ${ARGS}"
