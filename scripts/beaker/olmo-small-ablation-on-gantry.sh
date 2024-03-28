#!/bin/bash

set -ex

# check if LOAD_PATH is provided as an environment variable; if so, create an argument
# to pass to the training script
if [ -z ${LOAD_PATH+x} ]; then
  LOAD_PATH_ARG=""
else
  LOAD_PATH_ARG="--load_path=${LOAD_PATH}"
fi


# check if CONFIG PATH is provided as an environment variable;
# if so, use that instead of olmo-small-ablation.yaml
if [ -z ${CONFIG_PATH+x} ]; then
  export CONFIG_PATH=configs/olmo-small-ablation.yaml
else
  export CONFIG_PATH="${CONFIG_PATH}"
fi

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

NUM_NODES=4

gantry run \
  --workspace ai2/llm-testing \
  --task-name "${FULL_RUN_NAME}" \
  --description "${FULL_RUN_NAME}" \
  --priority "normal" \
  --beaker-image olmo-torch2-gantry \
  --cluster ai2/general-cirrascale-a100-80g-ib \
  --gpus 8 \
  --replicas ${NUM_NODES} \
  --leader-selection  \
  --host-networking \
  --nfs \
  ${WANDB_API_KEY_ARG} \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c "torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH} --run_name=${FULL_RUN_NAME} ${LOAD_PATH_ARG} --device_train_microbatch_size=8  --model.flash_attention=true ${@}"
