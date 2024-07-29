#!/usr/bin/env bash

set -ex

CHECKPOINT_DIR=$1

gantry run \
  --workspace ai2/cheap_decisions  \
  --task-name convert-cheap-decisions-checkpoint\
  --description "hf OLMO convert 1B LLM 1T tokens on weka" \
  --priority high \
  --preemptible \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 1 \
  --budget ai2/oe-eval \
  --no-nfs \
  --venv base \
  --weka oe-eval-default:/weka/oe-eval-default \
  --yes \
  --timeout=-1 \
  -- /bin/bash scripts/convert_all_checkpoints.sh ${CHECKPOINT_DIR}