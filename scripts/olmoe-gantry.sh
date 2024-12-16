#!/usr/bin/env bash
set -ex

CONFIG_PATH=configs/moe-dolmino-anneal/olmoe-8x1b-newhp-newds-dolmino-seed-42.yml
# ARGS='--save-overwrite --fsdp.sharding_strategy=FULL_SHARD --device_train_microbatch_size=4 --canceled_check_interval=9999999'
ARGS='--fsdp.sharding_strategy=FULL_SHARD --device_train_microbatch_size=4 --canceled_check_interval=9999999'

#NUM_NODES=1
NUM_NODES=4
#NUM_NODES=16
# NUM_NODES=32
BEAKER_REPLICA_RANK=0

gantry run \
  --weka oe-training-default:/weka/oe-training-default \
  --allow-dirty \
  --preemptible \
  --priority high \
  --workspace ai2/olmoe \
  --description "${CONFIG_PATH}" \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --budget ai2/oe-training \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --env-secret WANDB_API_KEY=lucas_WANDB_API_KEY \
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
  --synchronized-start-timeout 60m \
  -- /bin/bash -c "pip install --upgrade torch==2.3.0; pip install --upgrade flash-attn --no-build-isolation; pip install git+https://github.com/Muennighoff/megablocks.git@zloss; mkdir -p /root/.cache; pushd /root/.cache; curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -; popd; export HF_DATASETS_OFFLINE=1; export NCCL_IB_HCA=^=mlx5_bond_0; SLURM_JOB_ID=${BEAKER_JOB_ID} torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --node_rank ${BEAKER_REPLICA_RANK} --nproc-per-node 8 --rdzv_id=12347 --rdzv_backend=c10d --rdzv_conf='read_timeout=420' --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH} ${ARGS}"

# Single node:
#--rdzv_endpoint=\$BEAKER_NODE_HOSTNAME:29400
# Multinode:
#--rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400
#  --mount /net/nfs.cirrascale/allennlp/petew/cache:/root/.cache \
#--node_rank=$BEAKER_REPLICA_RANK
#  --nfs \
