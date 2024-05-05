#!/usr/bin/env bash

set -ex

CONFIG_PATH=configs/olmoe/OLMo-1B-NOSHARD.yml
NUM_NODES=1
BEAKER_REPLICA_RANK=0


gantry run \
  --allow-dirty \
  --workspace ai2/olmoe \
  --task-name mitchish-mcli-final \
  --description mitchish-mcli-final \
  --priority normal \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --budget ai2/oe-training \
  --cluster ai2/pluto-cirrascale \
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
  -- /bin/bash -c "torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --node_rank=$BEAKER_REPLICA_RANK --rdzv_id=12347 --rdzv_backend=static --rdzv_conf='read_timeout=420' --rdzv_endpoint=\$BEAKER_NODE_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH}"
# ${ARGS}"
