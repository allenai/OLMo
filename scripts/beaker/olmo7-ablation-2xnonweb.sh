#!/usr/bin/env bash

set -ex

CONFIG_PATH=configs/olmo7-ablation-2xnonweb.yaml
NUM_NODES=8
ARGS='--run_name=olmo7-ablation-2xnonweb --wandb.name=2xnonweb --model.flash_attention=true --fsdp.wrapping_strategy=by_block_and_size --fsdp.sharding_strategy=SHARD_GRAD_OP --save_folder=runs/ --device_train_microbatch_size=3 --global_train_batch_size=6144 --wandb.group=2xnonweb --remote_save_folder=s3://ai2-llm/checkpoints/olmo7-ablation/2xnonweb'

gantry run \
  --allow-dirty \
  --workspace ai2/llm-testing \
  --task-name olmo7-ablation-2xnonweb \
  --description olmo7-ablation-2xnonweb \
  --priority high \
  --beaker-image olmo-torch2-gantry \
  --cluster ai2/pluto-cirrascale \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --nfs \
  --mount /net/nfs.cirrascale/allennlp/petew/cache:/root/.cache \
  --budget ai2/oe-training \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --env-secret R2_ACCESS_KEY_ID=R2_ACCESS_KEY_ID \
  --env-secret R2_SECRET_ACCESS_KEY=R2_SECRET_ACCESS_KEY \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c "source scripts/beaker/warm_hf_cache.sh && torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH} ${ARGS}"
