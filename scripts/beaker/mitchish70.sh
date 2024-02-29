#!/usr/bin/env bash

set -ex

CONFIG_PATH=configs/mitchish70-s3.yaml
NUM_NODES=4
SEED=3423
INIT=fan_in
RUN_NAME="fan-in-init-${SEED}"
ARGS="--run_name=${RUN_NAME} --data.seed=6198 --seed=${SEED} --model.init_fn=${INIT} --model.init_std=0.006 --model.init_cutoff_factor=3 --device_train_microbatch_size=4 --model.flash_attention=true --fused_loss=true --evaluators=[] --stop_at=500 --wandb.group=mitchish70-ablate-init --save_interval_ephemeral=100"

gantry run \
  --allow-dirty \
  --workspace ai2/llm-testing \
  --task-name mitchish70 \
  --description "OLMo mitchish 70B, model init ablations" \
  --priority high \
  --stop-preemptible \
  --beaker-image olmo-torch2-gantry \
  --cluster ai2/general-cirrascale-a100-80g-ib \
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
  -- /bin/bash -c "torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH} ${ARGS}"
