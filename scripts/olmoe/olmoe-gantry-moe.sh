#!/usr/bin/env bash
set -ex

#CONFIG_PATH=configs/olmoe/OLMoE-8x1B-NOSHARD-S3.yml
CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu.yml
NUM_NODES=8
BEAKER_REPLICA_RANK=0

# Warm HF cache
#mkdir -p /root/.cache
#pushd /root/.cache
#curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -
#popd
#export HF_DATASETS_OFFLINE=1

#shanea/olmo-torch2.3-gantry
#shanea/olmo-torch2.2-gantry
#petew/olmo-torch2-gantry

gantry run \
  --allow-dirty \
  --workspace ai2/olmoe \
  --task-name mitchish-mcli-final \
  --description mitchish-mcli-final \
  --priority normal \
  --beaker-image shanea/olmo-torch2.3-gantry \
  --budget ai2/oe-training \
  --cluster ai2/jupiter-cirrascale \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --nfs \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --leader-selection \
  --host-networking \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c "pip install --upgrade torch; pip install megablocks; mkdir -p /root/.cache; pushd /root/.cache; curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -; popd; export HF_DATASETS_OFFLINE=1; torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=12347 --rdzv_backend=c10d --rdzv_conf='read_timeout=420' --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH}"

#--no-deps # -> Does not work
#export NCCL_DEBUG=INFO;
# ${ARGS}"

# Single node:
#--rdzv_endpoint=\$BEAKER_NODE_HOSTNAME:29400
# Multinode:
#--rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400
#  --mount /net/nfs.cirrascale/allennlp/petew/cache:/root/.cache \
#pip install megablocks;
#--node_rank=$BEAKER_REPLICA_RANK
