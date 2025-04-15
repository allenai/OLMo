#!/usr/bin/env bash

RUN_NAME="index_v5_dolmino-mix-1124-7b-50b_dolma2"

gantry run \
  --allow-dirty \
  --name ${RUN_NAME} \
  --task-name ${RUN_NAME} \
  --description ${RUN_NAME} \
  --workspace ai2/infini-llm \
  --budget ai2/oe-training \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --cluster ai2/ceres-cirrascale \
  --priority high \
  --no-nfs \
  --weka oe-training-default:/weka/oe-training-default \
  --cpus 186 \
  --memory 1912GiB \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c "cd infini_gram_indexing ; \
    python indexing_v5_u32.py --cpus 186 --mem 1912 \
      --data_paths_file ./datapaths_dolmino-mix-1124-7b-50b.json \
      --save_dir /weka/oe-training-default/jiachengl/hb-wolf/index/v5_dolmino-mix-1124-7b-50b_dolma2 ; \
    "
