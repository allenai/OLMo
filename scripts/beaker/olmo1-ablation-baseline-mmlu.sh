#!/usr/bin/env bash

set -ex

CONFIG_PATH=configs/road-to-1_7/runs/v1_5-1b-150b-mmlu.yml
LOAD_PATH='s3://ai2-llm/checkpoints/1b/v1_5-1b-150b_mmlu/6368542/step35000-unsharded'
TASK_NAME=$(basename ${CONFIG_PATH} .yml)
NUM_NODES=2
ARGS=' --wandb.name="\${run_name}-${BEAKER_JOB_ID}" --wandb.group="\${run_name}" --model.flash_attention=true --fsdp.sharding_strategy=FULL_SHARD --fsdp.wrapping_strategy=by_block --save_folder=runs/ --device_train_microbatch_size=12 --global_train_batch_size=2304 --load_path="${LOAD_PATH}"'

gantry run \
  --allow-dirty \
  --workspace ai2/llm-testing \
  --task-name "${TASK_NAME}" \
  --description "${TASK_NAME}" \
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
  --env LOAD_PATH="${LOAD_PATH}" \
  --env-secret WANDB_API_KEY=LUCAS_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --env-secret R2_ACCESS_KEY_ID=R2_ACCESS_KEY_ID \
  --env-secret R2_SECRET_ACCESS_KEY=R2_SECRET_ACCESS_KEY \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c "source scripts/beaker/warm_hf_cache.sh && torchrun --nnodes ${NUM_NODES} --nproc-per-node 8 --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29401 scripts/train.py ${CONFIG_PATH} ${ARGS}"
