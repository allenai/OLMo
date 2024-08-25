#!/usr/bin/env bash

set -ex

# CONFIG_NAME="peteish1"                # This is for a baseline trained with OLMoE-mix: DCLM (including OpenWebText, Algebraic-stack, Starcoder) + Pes2o + Wikipedia
# CONFIG_NAME="peteish1-dclm-only"      # This is for a baseline trained with DCLM (including OpenWebText, Algebraic-stack, Starcoder) 
# CONFIG_NAME="peteish1-B34v0"            # This is for a model trained on OLMoE-mix + B34v0
CONFIG_NAME=$1
NUM_NODES=1

gantry run \
  --workspace ai2/sewonm \
  --task-name ${CONFIG_NAME} \
  --description ${CONFIG_NAME} \
  --priority normal \
  --preemptible \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 1 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --weka oe-training-default:/weka/oe-training-default \
  --propagate-failure \
  --propagate-preemption \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env R2_PROFILE=R2 \
  --env S3_PROFILE=S3 \
  --env WEKA_PROFILE=WEKA \
  --env-secret AWS_CONFIG=SEWONM_AWS_CONFIG \
  --env-secret AWS_CREDENTIALS=SEWONM_AWS_CREDENTIALS \
  --env-secret WANDB_API_KEY=SEWONM_WANDB_API_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "sewon-scripts/peteish1.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"
