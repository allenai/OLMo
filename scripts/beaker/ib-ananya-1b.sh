#!/bin/bash

set -ex

export LOAD_PATH_ARG=""
export CONFIG_PATH=scripts/ananya-1b-ib.yaml
export NCCL_DEBUG=INFO

# get run name, we will use this as task name in gantry
RUN_NAME=$(cat $CONFIG_PATH | grep -ohP "^run_name\:\w*(.+)$" | sed 's/run_name:\s*//')

# get a hash of the load path and config path; take the first 8 characters
RUN_HASH=$(echo "${LOAD_PATH_ARG}-${CONFIG_PATH}" | md5sum | cut -c 1-8)

# compose the two
FULL_RUN_NAME="${RUN_NAME}-${RUN_HASH}"

# check if there is an env var called 'WANDB_API_KEY' and if so, create a flag
# to pass to gantry
if [ -z ${WANDB_API_KEY+x} ]; then
  WANDB_API_KEY_ARG="--env-secret WANDB_API_KEY=WANDB_API_KEY"
else
  WANDB_API_KEY_ARG="--env WANDB_API_KEY=${WANDB_API_KEY}"
fi

# check if there is an env var called 'AWS_ACCESS_KEY_ID' and 'AWS_SECRET_ACCESS_KEY' and if so, create a flag
# to pass to gantry
if [ -z ${WANDB_API_KEY+x} ]; then
  AWS_ACCESS_KEY_ID_ARG="--env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID"
  AWS_SECRET_ACCESS_KEY_ARG="--env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY"
else
  AWS_ACCESS_KEY_ID_ARG="--env AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}"
  AWS_SECRET_ACCESS_KEY_ARG="--env AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}"
fi

NUM_NODES=2

gantry run \
  --workspace ai2/llm-testing \
  --task-name "${FULL_RUN_NAME}" \
  --description "${FULL_RUN_NAME}" \
  --priority "high" \
  --beaker-image olmo-torch2-gantry \
  --cluster ai2/general-cirrascale-a100-80g-ib \
  --gpus 8 \
  --replicas ${NUM_NODES} \
  --leader-selection  \
  --host-networking \
  --nfs \
  ${WANDB_API_KEY_ARG} \
  ${AWS_ACCESS_KEY_ID_ARG} \
  ${AWS_SECRET_ACCESS_KEY_ARG} \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c "torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME scripts/train.py ${CONFIG_PATH} --model.flash_attention=true"
