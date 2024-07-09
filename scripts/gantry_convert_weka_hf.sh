#!/usr/bin/env bash

set -ex

INPUT_DIR=$1

OUTPUT_DIR=$2

gantry run \
  --workspace ai2/cheap_decisions  \
  --task-name convert cheap decisions checkpoint\
  --description "hf OLMO convert 1B LLM 1T tokens on weka" \
  --priority high \
  --preemptible \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 0 \
  --budget ai2/oe-eval \
  --no-nfs \
  --venv base \
  --weka oe-eval-default:/weka/oe-eval-default \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "python scripts/convert_olmo_to_hf_new.py --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} --tokenizer_json_path tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"