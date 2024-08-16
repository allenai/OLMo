#!/usr/bin/env bash

set -ex

export NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=ib NCCL_IB_HCA="^=mlx5_bond_0"

NUM_NODES=4

gantry run \
  --workspace ai2/long-contexts \
  --task-name long_contexts_amberish7B_cont_training \
  --description "OLMo - Amberish 7B - continued training" \
  --priority high \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2\
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --preemptible \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --propagate-failure \
  --propagate-preemption \
  --synchronized-start-timeout 10m \
  --weka oe-training-default:/weka/oe-training-default \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=DUSTINS_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=DUSTINS_AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=DUSTINS_AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c " scripts/beaker/lc_7b.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"
