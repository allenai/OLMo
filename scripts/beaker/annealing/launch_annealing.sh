#!/usr/bin/env bash
# Invoke from project root like `bash scripts/beaker/annealing/launch_annealing.sh`.

set -ex

function launch {
  CONFIG_NAME=$1
  NUM_NODES=$2
  CLUSTER=$3

  CONFIG_DIR=configs/annealing
  CONFIG_PATH=${CONFIG_DIR}/${CONFIG_NAME}.yaml

  echo $CONFIG_NAME
  echo $NUM_NODES
  echo $CLUSTER

  gantry run \
    --allow-dirty \
    --workspace ai2/davidw-oe-annealing \
    --task-name ${CONFIG_NAME} \
    --description ${CONFIG_NAME} \
    --priority high \
    --beaker-image petew/olmo-torch2-gantry \
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
    -- /bin/bash -c "source scripts/beaker/warm_hf_cache.sh && torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH}"
}


################################################################################

# Launch runs.

launch v1.7-step_2T-resume_optimizer-steps_50B 8 ai2/pluto-cirrascale
