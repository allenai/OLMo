#!/usr/bin/env bash

RUN_NAME="index_v5_olmoe-mix-0924-s0-flan-rulebased_dolma2"

gantry run \
  --allow-dirty \
  --name ${RUN_NAME} \
  --task-name ${RUN_NAME} \
  --description ${RUN_NAME} \
  --workspace ai2/hb-wolf-olmo \
  --budget ai2/oe-training \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --priority high \
  --preemptible \
  --no-nfs \
  --weka oe-training-default:/weka/oe-training-default \
  --cpus 186 \
  --memory 1912GiB \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c "cd infini_gram_indexing ; \
    python indexing_v5_u32.py --cpus 186 --mem 1912 \
      --data_paths \
        '/weka/oe-training-default/ai2-llm/preprocessed/olmo-mix/danyh-compiled-v1_7/documents/wiki/allenai/dolma2-tokenizer/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/allenai/dolma2-tokenizer/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/open-web-math/train/allenai/dolma2-tokenizer/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/pes2o/allenai/dolma2-tokenizer/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v1-decon-100_to_20k-2star-top_token_030/allenai/dolma2-tokenizer/*.npy' \
        '/weka/oe-training-default/ai2-llm/preprocessed/tulu_flan/v1-decontaminated-60M-shots_all-upweight_1-dialog_false-sep_rulebased/train/allenai_dolma2/*.npy' \
      --save_dir /weka/oe-training-default/wolf/index/v5_olmoe-mix-0924-s0-flan-rulebased_dolma2 ; \
    "
