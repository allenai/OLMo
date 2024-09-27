#!/usr/bin/env bash

RUN_NAME="index_v5_olmoe-mix-0924_dolma2"

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
  --replicas 20 \
  --host-networking \
  --leader-selection \
  --synchronized-start-timeout 48h \
  --cpus 186 \
  --memory 1912GiB \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  -- /bin/bash -c "cd infini_gram_indexing ; \
    if [ \$BEAKER_REPLICA_RANK -eq 0 ]; then \
      python u32_reversed_indexing_for_wolf.py --cpus 186 --mem 1912 \
        --data_paths \
          '/weka/oe-training-default/ai2-llm/preprocessed/olmo-mix/danyh-compiled-v1_7/documents/wiki/allenai/dolma2-tokenizer/*.npy' \
          '/weka/oe-training-default/ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/allenai/dolma2-tokenizer/*.npy' \
          '/weka/oe-training-default/ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/arxiv/train/allenai/dolma2-tokenizer/*.npy' \
          '/weka/oe-training-default/ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/open-web-math/train/allenai/dolma2-tokenizer/*.npy' \
          '/weka/oe-training-default/ai2-llm/preprocessed/pes2o/allenai/dolma2-tokenizer/*.npy' \
          '/weka/oe-training-default/ai2-llm/preprocessed/starcoder/v1-decon-100_to_20k-2star-top_token_030/allenai/dolma2-tokenizer/*.npy' \
        --save_dir /weka/oe-training-default/wolf/index/v5_olmoe-mix-0924_dolma2/0 ; \
    else \
      export PREFIX=\$(printf '%02d' \$((10#\$BEAKER_REPLICA_RANK - 1))) ; \
      python u32_reversed_indexing_for_wolf.py --cpus 186 --mem 1912 \
        --data_paths \
          \"/weka/oe-training-default/ai2-llm/preprocessed/dclm/text_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/allenai/dolma2-tokenizer/part-\${PREFIX}*.npy\" \
        --save_dir /weka/oe-training-default/wolf/index/v5_olmoe-mix-0924_dolma2/\$BEAKER_REPLICA_RANK ; \
    fi ; \
    "
