#!/bin/bash

set -ex

# check if LOAD_PATH is provided as an environment variable; if so, create an argument
# to pass to the training script
if [ -z ${LOAD_PATH+x} ]; then
  LOAD_PATH_ARG=""
else
  LOAD_PATH_ARG="--load_path=${LOAD_PATH}"
fi

EXTRA_ARGS="${@}"

# check if CONFIG PATH is provided as an environment variable;
# if so, use that instead of olmo-small-ablation.yaml
if [ -z ${CONFIG_PATH+x} ]; then
  export CONFIG_PATH=configs/olmo-small-ablation.yaml
else
  export CONFIG_PATH="${CONFIG_PATH}"
fi

if [ -z ${BEAKER_CLUSTER+x} ]; then
  export BEAKER_CLUSTER=ai2/general-cirrascale-a100-80g-ib
else
  export BEAKER_CLUSTER="${BEAKER_CLUSTER}"
fi

if [ -z ${BEAKER_NODES+x} ]; then
  export BEAKER_NODES=4
else
  export BEAKER_NODES="${BEAKER_NODES}"
fi

if [ -z ${BEAKER_GPUS+x} ]; then
  export BEAKER_GPUS=8
else
  export BEAKER_GPUS="${BEAKER_GPUS}"
fi

if [ -z ${BEAKER_PRIORITY+x} ]; then
  export BEAKER_PRIORITY="normal"
else
  export BEAKER_PRIORITY="${BEAKER_PRIORITY}"
fi

# get run name, we will use this as task name in gantry
if [-z ${RUN_NAME+x} ]; then
  export RUN_NAME=$(cat $CONFIG_PATH | grep -ohP "^run_name\:\w*(.+)$" | sed 's/run_name:\s*//')
else
  export RUN_NAME="${RUN_NAME}"
fi

# get a hash of the load path and config path; take the first 8 characters
RUN_HASH=$(echo "${LOAD_PATH_ARG}-${CONFIG_PATH}-${EXTRA_ARGS}$" | md5sum | cut -c 1-8)

# compose the two
FULL_RUN_NAME="${RUN_NAME}-${RUN_HASH}"

# check if there is an env var called 'WANDB_API_KEY' and if so, create a flag
# to pass to gantry
if [ -z ${WANDB_API_KEY+x} ]; then
  WANDB_API_KEY_ARG="--env-secret WANDB_API_KEY=WANDB_API_KEY"
else
  WANDB_API_KEY_ARG="--env WANDB_API_KEY=${WANDB_API_KEY}"
fi

# check if number of nodes in > 1, if so, use leader selection
if [ ${BEAKER_NODES} -gt 1 ]; then
  NETWORK_CONFIG="--replicas ${BEAKER_NODES} --leader-selection"
  TORCHRUN_CONFIG="--rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400"
else
  NETWORK_CONFIG=""
  TORCHRUN_CONFIG=""
fi

gantry run \
  --workspace ai2/llm-testing \
  --task-name "${FULL_RUN_NAME}" \
  --description "${FULL_RUN_NAME}" \
  --priority "normal" \
  --beaker-image olmo-torch2-gantry \
  --cluster ${BEAKER_CLUSTER} \
  --priority ${BEAKER_PRIORITY} \
  --gpus ${BEAKER_GPUS} \
  --host-networking \
  ${NETWORK_CONFIG} \
  --nfs \
  ${WANDB_API_KEY_ARG} \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=${BEAKER_NODES} \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c "torchrun --nnodes ${BEAKER_NODES}:${BEAKER_NODES} --nproc-per-node ${BEAKER_NODES} ${TORCHRUN_CONFIG} scripts/train.py ${CONFIG_PATH} --run_name=${FULL_RUN_NAME} --wandb.group=${RUN_NAME} ${LOAD_PATH_ARG} ${EXTRA_ARGS}"
