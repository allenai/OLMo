#!/usr/bin/env bash

set -ex

MODEL_NAME=$1

gantry run \
  --allow-dirty \
  --workspace ai2/OLMo-tiny \
  --task-name eval-bpb-mc \
  --description "Evaluate bpb and mc for ${MODEL_NAME}" \
  --priority high \
  --preemptible \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 1 \
  --budget ai2/oe-training \
  --no-nfs \
  --weka oe-training-default:/weka/oe-training-default \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env HF_TOKEN_DOWNLOAD=JIACHENGL_HF_TOKEN \
  --shared-memory 10GiB \
  --yes \
  --timeout=0 \
  -- /bin/bash -c "\
    set -exuo pipefail; \
    IFS=$'\n\t'; \
    conda shell.bash activate base; \
    pip install '.[train]'; \
    torchrun --nproc-per-node 1 scripts/eval_hf.py configs/peteish1-weka.yaml ${MODEL_NAME}; \
    "
