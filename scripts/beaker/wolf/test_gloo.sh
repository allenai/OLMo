#!/usr/bin/env bash

set -ex

NUM_NODES=2
RUN_NAME="test_gloo"

gantry run \
  --allow-dirty \
  --name ${RUN_NAME} \
  --task-name ${RUN_NAME} \
  --description ${RUN_NAME} \
  --workspace ai2/infini-llm \
  --budget ai2/oe-training \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --priority high \
  --no-nfs \
  --weka oe-training-default:/weka/oe-training-default \
  --gpus 2 \
  --shared-memory 10GiB \
  --replicas "${NUM_NODES}" \
  --host-networking \
  --leader-selection \
  --propagate-failure \
  --propagate-preemption \
  --synchronized-start-timeout 48h \
  --no-python \
  --yes \
  -- /bin/bash -c "\
    set -exuo pipefail; \
    IFS=$'\n\t'; \
    conda shell.bash activate base; \
    pip install '.[train]'; \
    export TORCH_DIST_INIT_BARRIER=1; \
    export PYTHONPATH=.; \
    torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 2 --rdzv_id=20310 --rdzv_backend=static --rdzv_conf='read_timeout=1200' --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 --node_rank=\$BEAKER_REPLICA_RANK \
        scripts/test_gloo.py \
  "
